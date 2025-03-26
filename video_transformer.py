import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import video_loader
from torch.utils.checkpoint import checkpoint_sequential
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.checkpoint import checkpoint
from torchsummary import summary
from prettytable import PrettyTable
from vit_pytorch.vit_3d import ViT
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, precision_score, accuracy_score, f1_score, recall_score
from torchcam.methods import LayerCAM
from einops import rearrange
from matplotlib.lines import Line2D

# import torch.distributed as dist
import os

class VideoModel(nn.Module):
    def __init__(self):
        super(VideoModel, self).__init__()
        print("init model")
        self.frame_patch_size = 5
        self.image_patch_size = 20
        
        self.batch_size = 2
        self.epoch = 0
        self.v = ViT(
            image_size = 200,          # image size
            frames = 30,               # number of frames
            image_patch_size = self.image_patch_size,     # image patch size
            frame_patch_size = self.frame_patch_size,      # frame patch size
            num_classes = 7,
            dim = 512,
            depth = 6,
            heads = 10,
            mlp_dim = 1024,
            dropout = 0.3,
            emb_dropout = 0.1
        )
        
        # self.cross_entropy_weights = torch.tensor([0.55460489, 0.35252472, 0.13050915, 0.89940325, 2.26344285, 2.2887799, 0.51073523])
        self.cross_entropy_weights = torch.tensor([0.18673272, 0.11610441, 0.04197427, 0.3129362, 0.91943635, 0.93728948, 0.17071762])
        
        # summary(self, (1, 1025, 400))
        print("init data loader", flush=True)
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.count_parameters()
        print("parameters:", pytorch_total_params)
        self.data_loader = video_loader.VideoLoader(self.batch_size)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0005)
    
    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def forward(self, x):
        return self.v(x)
    

    def train(self): 
        epochs = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cross_entropy_weights = self.cross_entropy_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=self.cross_entropy_weights)
        self.losses = []
        accuracy = 0
        total_items = 0
        self.accuracys = []
        
        
        for i, (data, target) in enumerate(self.data_loader):
            
            data, target = data.to(device), target.to(device)
            # print(target)
            self.optimizer.zero_grad()
            output = self(data).to(device)
            t = target.argmax(dim=1).long().to(device)            
            loss = criterion(output, t)
            

            if i % 100 == 0:
                print("----------", i, "-------------", flush=True)
                print("Pred:",output, flush=True)
                print("Targ:",target, flush=True)
            loss.backward()
            if i % 100 == 0:
                self.plot_grad_flow(self.named_parameters())
                plt.savefig(f"debug/meld_video_t_grad{i * (self.epoch + 1)}.png")
            self.optimizer.step()
            pred = output.detach()
            predictions = np.argmax(pred.cpu().numpy(), axis=1)

            targets = np.argmax(target.cpu().numpy(), axis=1)
                
            correct_predictions = np.sum(predictions == targets)
            total_items += self.batch_size
            if i % 100 == 0:
                print("Preds:",predictions, flush=True)
                print("Targs:",targets, flush=True)
                print("corr_preds:", predictions == targets)
    
            accuracy += correct_predictions
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                self.epoch, i, len(self.data_loader),
                100. * i / len(self.data_loader), loss.item(), (accuracy.item() / total_items) * 100), flush=True)
            
            self.losses.append(loss.detach().cpu().numpy())
            self.accuracys.append((accuracy.item() / total_items) * 100)
        
        self.validate()

    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                if p.grad == None:
                    print(n, "has no gradient", flush = True)
                    continue
                layers.append(n)
                ave_grads.append(p.grad.cpu().abs().mean())
                max_grads.append(p.grad.cpu().abs().max())
        plt.figure()
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    def validate(self):
        tmp_data_loader = video_loader.VideoLoader(1, folder="./sets/validation")
        num_classes = 7
        total_items = 0
        total_correct = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for i, (data, target) in enumerate(tmp_data_loader):
                data, target = data.to(device), target.to(device)
                output = self(data).to(device)
                t = target.argmax(dim=1).long().to(device)
                loss = criterion(output, t)
                pred = F.softmax(output.detach())
                predictions = np.argmax(pred.cpu().numpy(), axis=1)
                targets = np.argmax(target.cpu().numpy(), axis=1)
                correct_predictions = np.sum(predictions == targets)
                total_items += 1
                total_correct += correct_predictions
                accuracy = (correct_predictions / total_items) * 100
                print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                    self.epoch, i, len(tmp_data_loader),
                    100. * i / len(tmp_data_loader), loss.item(), (accuracy.item() / total_items) * 100), flush
                    = True)


    def test(self):
        self.data_loader = video_loader.VideoLoader(1, folder="./sets/testing")
        num_classes = 7
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            all_labels = []
            all_predictions = []
            all_logits =  []
            num_classes = 7

            
            for i, batch in enumerate(self.data_loader):
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                logits = self(inputs)
                predictions = np.argmax(logits.cpu(), axis=1)
                all_labels.extend(np.argmax(labels.cpu(), axis=1))
                all_predictions.extend(predictions)
                # tmp = precision_recall_curve(np.array(all_labels), np.array(all_predictions))
                print(i, "/", 1000, flush= True)
                all_logits.extend(logits.cpu())

            
            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)
            all_logits = np.array(all_logits)

            plt.figure()
            lookup = ["anger", "happy", "neutral", "sadness", "disgust", "fear", "surprise"]
            for j in range(num_classes):
                precision, recall, _ = precision_recall_curve(all_labels == j, all_logits[:, j])
                auc_score = auc(recall, precision)
                plt.plot(recall, precision, marker='.', label=f'Class {lookup[j]} (AUC = {auc_score:.2f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.savefig("plots/meld_video_t_pcr.png")
            plt.savefig("plots/meld_video_t_pcr.svg")
            plt.show()
            num_classes = 7

            
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=lookup, yticklabels=lookup)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig("plots/meld_video_t_hm.png")
            plt.savefig("plots/meld_video_t_hm.svg")
            plt.show()

            # Berechne Metriken
            precision_score_value = precision_score(all_labels, all_predictions, average='weighted')
            accuracy_score_value = accuracy_score(all_labels, all_predictions)
            f1_score_value = f1_score(all_labels, all_predictions, average='weighted')
            recall_score_value = recall_score(all_labels, all_predictions, average='weighted')

            print(f"Precision: {precision_score_value:.4f}", flush=True)
            print(f"Accuracy: {accuracy_score_value:.4f}", flush=True)
            print(f"F1 Score: {f1_score_value:.4f}", flush=True)
            print(f"Recall: {recall_score_value:.4f}", flush=True)
        
    def save(self):
        snapshot = dict(
             EPOCHS=self.epoch,
             MODEL_STATE=self.state_dict(),
             OPTIMIZER=self.optimizer.state_dict(),
             LOSSES=self.losses
         )

        torch.save(snapshot, 'meld_video_tt.pt')

        

    def load(self):
        snapshot = torch.load("meld_video_tt.pt")
       
        self.load_state_dict(snapshot["MODEL_STATE"])
        self.losses = snapshot["LOSSES"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.epoch = snapshot["EPOCHS"]
    
def create_activation_map(model, i):
    tmp_data_loader = video_loader.VideoLoader(1)
    data, input_tensor = None, None
    for j, (data, target) in enumerate(tmp_data_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data, target = data.to(device), target.to(device)
        print(f"Getting {data[0][0][0]}", flush = True)
        input_tensor = data.clone()  # Shape: (1, 3, 30, 200, 200)
        break  # Nur einen Batch verwenden

    def extract_attention(model, x):
        """Extrahiert die Attention Map aus der letzten Transformer-Schicht."""
        with torch.no_grad():
            tokens = model.v.to_patch_embedding(x)  # (1, Tokens, Dim)
            
            # Forward-Pass
            for attn, ff in model.v.transformer.layers:
                tokens = attn(tokens) + tokens
                tokens = ff(tokens) + tokens

            last_attn_layer = model.v.transformer.layers[-1][0]  # Letzte Attention-Schicht
            qkv = last_attn_layer.to_qkv(last_attn_layer.norm(tokens)).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=last_attn_layer.heads), qkv)
            attn_maps = torch.matmul(q, k.transpose(-1, -2)) * last_attn_layer.scale
            attn_maps = last_attn_layer.attend(attn_maps)  # Softmax über Attention Scores
        return attn_maps.mean(dim=1).squeeze(0)  # Mittelwert über alle Attention Heads

    def extract_patch_embeddings(model, x):
        """Extrahiert die Patch-Embeddings und formatiert sie als Feature Map."""
        with torch.no_grad():
            embeddings = model.v.to_patch_embedding(x)  # (1, Tokens, Dim)
        num_patches_t = (30 // model.frame_patch_size)  # Zeitliche Patches
        num_patches_s = (200 // model.image_patch_size)  # Räumliche Patches
        return embeddings.view(1, num_patches_t, num_patches_s, num_patches_s, -1).squeeze(0).mean(dim=0)

    attn_map_avg = extract_attention(model, input_tensor)
    feature_map = extract_patch_embeddings(model, input_tensor)

    
    frame_index = 5
    original_frame = input_tensor[0, :, frame_index, :, :].reshape((200, 200, 3)).cpu().numpy()  # (200, 200, 3)


    print(f"Feature Map Shape: {feature_map.shape}", flush = "True")
    heatmap = feature_map[frame_index]  # Shape: (Num_Patches_H, Num_Patches_W)
    heatmap = torch.nn.functional.interpolate(
        heatmap.unsqueeze(0).unsqueeze(0), size=(200, 200), mode="bilinear", align_corners=False
    ).squeeze().cpu().numpy()

    # Originalbild + Heatmap
    plt.figure(figsize=(12, 6))

    # Normierung für richtige Farben
    # print(f"Plotting {data[0][0][0]}", flush = True)
    # frames = data[0, :, :, :, :].reshape((30, 200, 200, 3)).cpu().numpy()
    # for j, frame in enumerate(frames):
    #     plt.imshow(frame)
    #     plt.savefig(f"debug/meld_video_t_am_{i}_{j}.png")
    # plt.savefig(f"plots/meld_video_t_am_{i}_{j}.svg")

    plt.imshow(original_frame)  # Originalbild
    plt.imshow(heatmap, cmap="jet", alpha=0.1)  # Heatmap halbtransparent
    
    plt.colorbar()
    # plt.title(f"Feature Map mit Heatmap-Overlay (Frame {frame_index})")
    plt.savefig(f"debug/meld_video_t_am_{i}.png")
    plt.savefig(f"debug/meld_video_t_am_{i}.svg")



if __name__ == "__main__":
    model = VideoModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)
    model.to(device)
    if os.path.exists("meld_video_tt.pt"):
        model.load()
        print("loading existing snapshot")

    while model.epoch < 20:
        print(model.epoch)
        model.train()
        model.epoch += 1
        model.save()
    else:
        for i in range(1):
            create_activation_map(model, i)
            print("Map", i, flush = True)
        model.test()
        

