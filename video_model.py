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
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, precision_score, accuracy_score, f1_score, recall_score

# import torch.distributed as dist
import os

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, k):
        super(UpscaleBlock, self).__init__()
        self.model = nn.Sequential(
            # nn.LayerNorm(200),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels * k, kernel_size=3, stride= 2, padding=1),
          #  nn.ReLU()
            # nn.MaxPool3d(kernel_size=4)
            )
        
    def forward(self, x):
        segments =  1
        modules = [module for k, module in self.model._modules.items()]
        return checkpoint_sequential(modules, segments, x, use_reentrant=True)

class DownscaleBlock(nn.Module):
    def __init__(self, in_channels, k):
        super(DownscaleBlock, self).__init__()
        self.model = nn.Sequential(
            #nn.LayerNorm(200),
            nn.Conv3d(in_channels=in_channels, out_channels=int(in_channels // k), kernel_size=3, stride= 2, padding=1),
         #   nn.ReLU()
            # nn.MaxPool3d(kernel_size=4)
            )
        
    def forward(self, x):
        segments =  1
        modules = [module for k, module in self.model._modules.items()]
        return checkpoint_sequential(modules, segments, x, use_reentrant=True)

class VideoModel(nn.Module):
    def __init__(self):
        super(VideoModel, self).__init__()
        print("init model")
        k = 10
        num_batch = 16
        batch_size = 32 # * (32768 * 2)
        size = 3
        self.batch_size = num_batch
        self.epoch = 0
        
        # (batch size, 3, 30, 200, 200)
        # self.f4 = nn.Linear(50000, 10000)
        

        self.conv1 = UpscaleBlock(size, k)
        self.conv2 = UpscaleBlock(size * k, k)
        # self.conv3 = UpscaleBlock(size * k * k, k)
        # self.conv4 = UpscaleBlock(size * k * k * k, k)
        # self.conv5 = DownscaleBlock(size * k * k * k, k)
        # self.conv6 = DownscaleBlock(size * k * k * k, k)
        self.conv7 = DownscaleBlock(size * k * k, k)
        self.conv8 = DownscaleBlock(size * k, k)
        self.lin1 = nn.Linear(1014, 1600)
        self.lin2 = nn.Linear(1600, 100)
        self.lin3 = nn.Linear(100, 7)
        
        
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        torch.nn.init.xavier_uniform_(self.lin3.weight)
        
        self.cross_entropy_weights = torch.tensor([0.55460489, 0.35252472, 0.13050915, 0.89940325, 2.26344285, 2.2887799, 0.51073523])

        # summary(self, (1, 1025, 400))
        print("init data loader", flush=True)
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.count_parameters()
        print("parameters:", pytorch_total_params)
        self.data_loader = video_loader.VideoLoader(num_batch)
        self.optimizer = optim.Adam(self.parameters(), lr=0.00005)
    
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
        x = self.conv1(x)
        # x = F.relu(x)
        x = self.conv2(x)
        # x = F.relu(x)
        #x = self.conv3(x)
        #x = F.relu(x)
        # x = self.conv4(x)
        # print("here", flush = True)
        # x = F.relu(x)
        # x = self.conv5(x)
        # x = F.relu(x)
        #x = self.conv6(x)
        # print("here", flush = True)
        #x = F.relu(x)
        x = self.conv7(x)
        #x = F.relu(x)
        x = self.conv8(x)
        #x = F.relu(x)
        # print("here", flush = True)
        # print(x.shape, flush = True)
        x = x.reshape((self.batch_size, 1014))
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        # print(x.shape, flush = True)
        return x

    

    def train(self): 
        criterion = nn.CrossEntropyLoss(weight = self.cross_entropy_weights, reduction='mean')
        self.losses = []
        accuracy = 0
        total_items = 0
        self.accuracys = []
        
        
    
        for i, (data, target) in enumerate(self.data_loader):
            # print(target)
            self.optimizer.zero_grad()
            output = self(data)
            t = target.clone().detach().float()
            
            loss = criterion(output, t)
            # print(loss)
            
            if i % 100 == 0:
                print("----------", i, "-------------", flush=True)
                print("Pred:",output, flush=True)
                print("Targ:",target, flush=True)
            loss.backward()
            self.optimizer.step()
            pred = output.detach().clone()
            predictions = np.argmax(pred.numpy(), axis=1)
    
            # Umformen der One-Hot-Ziele zu Klassenindizes
            targets = np.argmax(target.numpy(), axis=1)
    
    # Vergleichen mit den tatsächlichen Klassen
            correct_predictions = np.sum(predictions == targets)
            total_items += self.batch_size
            if i % 10 == 0:
                print("Preds:",predictions, flush=True)
                print("Targs:",targets, flush=True)
                print("corr_preds:", predictions == targets)
    
    # Berechnen der Genauigkeit
            accuracy += correct_predictions
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%'.format(
                self.epoch, i, len(self.data_loader),
                100. * i / len(self.data_loader), loss.item(), (accuracy.item() / total_items) * 100), flush=True)
            
            self.losses.append(loss.detach().numpy())
            self.accuracys.append((accuracy.item() / total_items) * 100)


    def test(self):
        self.data_loader.toggle_test()
        num_classes = 7
        with torch.no_grad():
            all_labels = []
            all_predictions = []
            all_logits =  []
            num_classes = 7

            # Iteriere über den Dateniterator, um Eingabedaten und Labels zu erhalten
            for i, batch in enumerate(self.data_loader):
                inputs, labels = batch
                logits = self(inputs)
                predictions = np.argmax(logits, axis=1)
                all_labels.extend(np.argmax(labels, axis=1))
                all_predictions.extend(predictions)
                # tmp = precision_recall_curve(np.array(all_labels), np.array(all_predictions))
                print(i, "/", 1000, flush= True)
                all_logits.extend(logits)

            # Konvertiere Listen in NumPy-Arrays
            all_labels = np.array(all_labels)
            all_predictions = np.array(all_predictions)
            all_logits = np.array(all_logits)

            # Berechne Präzision und Recall für die Precision-Recall-Kurve pro Klasse
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
            plt.savefig("plots/meld_me_video_c_pcr.png")
            plt.savefig("plots/meld_me_video_c_pcr.svg")
            plt.show()
            num_classes = 7

            # Erstellen und Plotten der Verwechslungs-Matrix
            conf_matrix = confusion_matrix(all_labels, all_predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=lookup, yticklabels=lookup)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig("plots/meld_me_video_c_hm.png")
            plt.savefig("plots/meld_me_video_c_hm.svg")
            plt.show()

            # Berechne und drucke die Metriken
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

        torch.save(snapshot, 'snapshot_ovweight.pt')

        

    def load(self):
        snapshot = torch.load("crema_video_c.pt")
       
        self.load_state_dict(snapshot["MODEL_STATE"])
        self.losses = snapshot["LOSSES"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
        self.epoch = snapshot["EPOCHS"]
    

if __name__ == "__main__":
    model = VideoModel()
    if os.path.exists("crema_video_ccc.pt"):
        model.load()
        print("loading existing snapshot")

    while model.epoch < 24:
        print(model.epoch)
        model.train()
        model.epoch += 1
        model.save()
    else:
        model.test()

# def ddp_setup_torchrun():
#     dist.init_process_group(backend="nccl")


# class TrainerDDPTorchrun(torch.distributed.TrainerDDP):
#     def __init__(
#         self,
#         model: nn.Module,
#         trainloader,
#         testloader
#     ) -> None:
#         self.cpu_id = int(os.environ["LOCAL_RANK"])
#         self.epochs_run = 0
#         super().__init__(self.cpu_id, model, trainloader)

#     def _save_snapshot(self, epoch: int):
#         snapshot = dict(
#             EPOCHS=epoch,
#             MODEL_STATE=self.model.state_dict(),
#             OPTIMIZER=self.optimizer.state_dict()
#         )
#         model_path = self.const["trained_models"] / f"snapshot.pt"
#         torch.save(snapshot, model_path)

#     def _load_snapshot(self, path: str):
#         snapshot = torch.load(path, map_location="cpu")
#         self.epochs_run = snapshot["EPOCHS"] + 1
#         self.model.load_state_dict(snapshot["MODEL_STATE"])
#         self.optimizer.load_state_dict(snapshot["OPTIMIZER"])
#         print(
#             f"[CPU{self.gpu_id}] Resuming training from snapshot at Epoch {snapshot['EPOCHS']}"
#         )

#     def train(self, max_epochs: int, snapshot_path: str):
#         if os.path.exists(snapshot_path):
#             print("Loading snapshot")
#             self._load_snapshot(snapshot_path)

#         self.model.train()
#         for epoch in range(self.epochs_run, max_epochs):
#             # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
            
#             self._run_epoch(epoch)
#             # only save once on master gpu
#             if self.cpu_id == 0 and epoch % self.const["save_every"] == 0:
#                 self._save_snapshot(epoch)
#         # save last epoch
#         self._save_checkpoint(max_epochs - 1)

# def main_ddp_torchrun(
#     snapshot_path: str,
#     final_model_path: str,
#     ):
#     ddp_setup_torchrun()

#     const = torch.distributed.prepare_const()
#     train_dataloader= audio_data_loader.AudioLoader(16)
#     model = AudioModel()
#     trainer = TrainerDDPTorchrun(
#         model=model,
#         trainloader=train_dataloader,
#         testloader=None,
#     )
#     trainer.train(const["total_epochs"], snapshot_path=snapshot_path)
#     trainer.test(final_model_path)

#     torch.distributed.destroy_process_group()  # clean up

# if __name__ == '__main__':
#     ddp_setup_torchrun()
#     snapshot_path = "snapshot.pt"
#     final_model_path = "model.pt"
#     main_ddp_torchrun(snapshot_path, final_model_path)