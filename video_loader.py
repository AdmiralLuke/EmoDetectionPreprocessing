import numpy as np
import torch
import os
import sys
import pandas as pd
from video2numpy import video2numpy
from video2numpy.frame_reader import FrameReader
from matplotlib import pyplot as plt

from sklearn.preprocessing import normalize

class VideoLoader(torch.utils.data.Dataset):
    def __init__(self, batch_size, folder = './videos_n'):
        self.folder = folder
        self.folder = os.path.abspath(self.folder)
        print(self.folder, flush=True)
        self.data = np.array([])
        self.target = np.array([])
        self.len = 9988
        # self.annos = pd.read_csv("train_sent_emo.csv")["Emotion"]
        self.test = False
        self.batch_size = batch_size
        self.rng = np.random.default_rng()
        self.indx = list(range(len(self.data)))
        self.noise = False
        np.random.shuffle(self.indx)


    def load_data(self):
        VIDS = glob.glob(self.folder + "/*.mp4")
        print(len(VIDS), "a")
        self.data = []
         
        reader = FrameReader(VIDS, take_every_nth=5, resize_size=500, batch_size=self.batch_size, workers=1)
        reader.start_reading()  

        i = 0

        for vid_frames, info_dict in reader:
            i += self.batch_size
            self.data.append(vid_frames)  
            if i % 1000 == 0:
                print(f"Loaded {i} files", flush=True)        
            if i >= 7441:
                print(f"Loaded {i} files", flush=True)
                break

        self.data = np.array(self.data)
        torch.save({"min": np.min(self.data), "max": np.max(self.data)}, "video_min_max.pt")

    def toggle_test(self):
        self.test = not self.test

    def toggle_noise(self):
        self.noise = not self.noise 

    def label_to_index(self, label):
        lookup = {'anger':      [1, 0, 0, 0, 0, 0, 0], 
                  'joy':        [0, 1, 0, 0, 0, 0, 0], 
                  'neutral':    [0, 0, 1, 0, 0, 0, 0], 
                  'sadness':    [0, 0, 0, 1, 0, 0, 0],
                  'disgust':    [0, 0, 0, 0, 1, 0, 0], 
                  'fear':       [0, 0, 0, 0, 0, 1, 0],
                  'surprise':   [0, 0, 0, 0, 0, 0, 1]}
        return lookup.get(label, [0, 0, 0, 0, 0, 0, 0])

    def normalize_dataframes(self, path):
        # return np.array(self.data, dtype=np.float32) / 255
        norms = torch.load(path)
        # # print(norms)
        self.data = np.array(self.data)
        return (self.data - float(norms["min"])) / (float(norms["max"]) - float(norms["min"])) 
        
    def __len__(self):
        if not self.test:
            return int(self.len // 1.1) // self.batch_size
        else:
            return (self.len - int(self.len // 1.1)) // self.batch_size

    def __iter__(self):
        
        self.files = glob(self.folder, ".mp4")
        print(len(self.files), "b", flush = True)
        self.data = []
        self.indx = list(range(len(self.files)))
        np.random.shuffle(self.indx)
        
        self.files = np.array(self.files)[self.indx]

        # print(self.files.shape, "c")
        self.files = np.random.permutation(self.files)
        # reader = FrameReader(self.files, resize_size=200, batch_size=1, workers=1)
        # reader.start_reading() 
        start_indx = 0
        end_index = int(self.len // 1.1)
        
        if self.test:
            start_indx = int(self.len // 1.1)
            end_index = self.len
        self.data = []
        self.target = []
        for i, file in enumerate(self.files):
            if self.test:
                if i >= 2500:
                    break
            # file = self.files[i]

            blockPrint()
            tmp_reader = FrameReader(np.array([file]), resize_size=200, batch_size=1)
            tmp_reader.start_reading()
            batch = np.array([None])
            for batch_j, l_t in tmp_reader:
                batch = batch_j
            
            enablePrint()
            emo = file.split("/")[-1].split("_")[0]
            # batch = pad(batch)
            if np.all(batch[0] == None):
                continue
            #print(batch.shape, batch.dtype)
            batch = np.array(batch)
            try:
                batch = pick_frames(batch)
                plot_debug_image(batch.reshape((30, 200, 200, 3))[0], "after-pick")
                if self.noise: 
                    batch = apply_noise(batch)
            except:
                if not batch.shape == (3, 30, 200, 200):
                    print("[ERROR] while picking frames from " + emo + str(batch.shape))
                    os.remove(file)
                    continue
            #print(batch.shape, batch.dtype)
            self.data.append(batch)
            self.target.append(self.label_to_index(emo))
            
            if (np.array(self.data).shape[0] == self.batch_size):
                self.data = self.normalize_dataframes("video_min_max.pt")
                # check if data is normalized (between 0 and 1)
                if np.max(self.data) > 1 or np.min(self.data) < -1:
                    print("[ERROR] Data not normalized")
                    print(np.max(self.data), np.min(self.data))
                    return

                
                # self.data = self.data.reshape((30,3, 200, 200, self.batch_size))
                # print(f"Yielding {self.data[0][0][0]}", flush = True)
                yield torch.tensor(self.data, dtype = torch.float32), torch.tensor(np.array(self.target))
                
                self.data = []
                self.target = []    

def pad(d):
    # shape: (16, 32, 224, 224, 3)
    MAX_LENGTH = 128
    d = np.array(d)
    if MAX_LENGTH - d.shape[0] < 0:
        print("[ERROR] Negative values while padding")
        return np.array([None])
    elif MAX_LENGTH - d.shape[0] > 0:
        d = np.pad(d, [(0, MAX_LENGTH - d.shape[0]), (0,0), (0,0), (0,0), (0,0)])
    
    return d.astype(np.float32)

def pick_frames(d, n = 30):
    idxs = np.random.randint(low=0, high=d.shape[0], size=n)
    d = np.squeeze(d)
    plot_debug_image(d[0], "pick")
    return d[idxs].reshape((3, 30, 200, 200))


def glob(path, type):
    files = os.listdir(path)
    f = []
    for file in files:
        if str(file).endswith(type):
            f.append(path + "/" + file)
    return f

def apply_noise(frames, mean = 0, std_deviation = 10):
    gaussian_noise = np.random.normal(mean, std_deviation, frames.shape)
    return frames + gaussian_noise


def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

DEBUG_COUNTER = 0
def plot_debug_image(image, stage):
    pass
    # global DEBUG_COUNTER
    # DEBUG_COUNTER += 1
    # plt.figure()
    # plt.imshow(image)
    # plt.savefig(f"debug/debug_{stage}_{DEBUG_COUNTER}.png")
    # plt.close()
    

if __name__ == '__main__':
   l = VideoLoader(1)
   for t, e in l:
       print(t)
       print(e)
       plot_debug_image(t[0].reshape((30, 200, 200, 3))[0], "final")
       break
       
