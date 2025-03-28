import os
import pandas as pd 
import sklearn
import sklearn.utils
import numpy as np
from matplotlib import pyplot as plt
import cv2
from tqdm import tqdm

def rename_all(path):
    dataset = pd.read_csv("train_sent_emo.csv")

    files = os.listdir(path)
    for file in files:
        f = file.split("_") # dia0 utt0.mp4
        ending = f[1].split(".")[1]
        f[1] = f[1].split(".")[0].replace("utt", "") # 0
        f[0] = f[0].replace("dia", "") # 0

        emo = dataset[(dataset["Dialogue_ID"] == int(f[0])) & (dataset["Utterance_ID"] == int(f[1]))]["Emotion"].values[0]
        os.rename(path + file, f"{path}{emo}_{f[0]}_{f[1]}.{ending}")
        print(f"renamed {path}{file} -> {path}{emo}_{f[0]}_{f[1]}.{ending}")

def emo_to_num(emo):
    lookup = {  'anger':      0, 
                'joy':        1, 
                'neutral':    2, 
                'sadness':    3,
                'disgust':    4, 
                'fear':       5,
                'surprise':   6}
    return lookup.get(emo, -1)

def calc_loss_weights(path):
    files = os.listdir(path)
    emos = []
    n = 0
    for file in files:
        emo = file.split("_")[0]
        emos.append(emo_to_num(emo))
    
    return sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(emos), y=np.array(emos))


def evaluate_pipeline(path):
    b_frames, b_total = count_black_frames(path)
    print(f"Blackframes: {b_frames/b_total*100:.2f}% with {b_frames}:{b_total}")
    f_frames, f_total = count_face_frames(path)
    assert b_total == f_total
    plt.pie([b_frames, f_frames, abs(b_total - b_frames - f_frames)], labels=[f"Blackframes: {b_frames/b_total*100:.2f}%", f"Faces: {f_frames/b_total*100:.2f}%", 
                                                                       f"Other: {(f_total - b_frames - f_frames)/f_total*100:.2f}%"])
    plt.savefig("pie.png")
    plt.savefig("pie.svg")
    plt.show()

def count_black_frames(path):
    files = os.listdir(path)
    total = 0
    black_frames = 0
    for file in tqdm(files):
        cap = cv2.VideoCapture(path + file)
        total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if np.mean(frame) < 10:
                black_frames += 1
        cap.release()
    return black_frames, total

def count_face_frames(path):
    files = os.listdir(path)
    total = 0
    face_frames = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for file in tqdm(files):
        cap = cv2.VideoCapture(path + file)
        total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                face_frames += 1
        cap.release()
    return face_frames, total

def remove_frames_without_faces(path, n_path):
    files = os.listdir(path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for file in tqdm(files):
        cap = cv2.VideoCapture(path + file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{n_path}{file}", fourcc, 30.0, (200, 200))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                out.write(frame)
        cap.release()
        out.release()

def create_still_frame_by_diff(path_from, path_to):
    files = os.listdir(path_from)
    for file in tqdm(files):
        file_name = file.split(".")[0]
        cap = cv2.VideoCapture(path_from + file)
        ret, frame1 = cap.read()
        if not ret:
            continue
        ret, frame2 = cap.read()
        diff = frame1.copy()
        while(cap.isOpened()):
            if not ret:
                break
            diff = cv2.absdiff(diff, frame2)
            ret, frame2 = cap.read()


        if np.any(diff == None):
            continue        
        
        cv2.imwrite(f"{path_to}{file_name}.png", diff)
        cap.release()

def look_if_face_in_image(path):
    files = os.listdir(path)
    faces_c = 0
    total = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for file in tqdm(files):
        img = cv2.imread(path + file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            faces_c += 1
        total += 1
    
    plt.pie([faces_c, total - faces_c], labels=[f"Faces: {faces_c/total*100:.2f}%", f"No Faces: {(total - faces_c)/total*100:.2f}%"])
    plt.savefig("pie_face.png")
    plt.savefig("pie_face.svg")
        

def pick_equal_amount(path_from, path_to, amount):
    """
    select equal amount of videos per emotion and move them to path_to
    """
    files = os.listdir(path_from)
    emo_count = {"anger": 0, "joy": 0, "neutral": 0, "sadness": 0, "disgust": 0, "fear": 0, "surprise": 0}
    for file in files:
        emo = file.split("_")[0]
        if emo_count[emo] < amount:
            emo_count[emo] += 1
            os.rename(path_from + file, path_to + file)
            print(f"moved {file} to {path_to} {emo_count[emo]}/{amount}")


if __name__ == "__main__":
    # pick_equal_amount("videos_n/", "sets/testing/", 40)
    weights = calc_loss_weights("./videos_n/")
    print(weights / 7 )
    