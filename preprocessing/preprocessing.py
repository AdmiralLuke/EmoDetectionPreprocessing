# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm

class VideoObject():
    def __init__(self, path):
        # open video object
        self.path = path

        self.cap = None
        self.fps = None
        self.width = None
        self.height = None
        self.total_frames = None
        self.frames = []
        # load the video
        self.is_open = self.open()

        self.prev_gray = None
        self.prev_mouths = []
        self.mouth_movements = []
        

    def process(self, width_to: int, height_to: int):
        # process the video
        pass

    def open(self):
        # open video file with cv2
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            print("Error opening video stream or file")
            return False
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)
        
        return True

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.is_open = False

    def get_faces(self):
        faces = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        for i, frame in enumerate(self.frames):
            # detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_f = face_cascade.detectMultiScale(gray, 1.3, 5)
            # print(f"Found {len(faces_f)} faces in frame {i} \t {self.path}")
            faces.append(faces_f)
        return faces

    def get_active_speaker(self, faces):
        active_speakers = []
        
        for i, frame in enumerate(self.frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.prev_gray is None:
                self.prev_gray = gray
                active_speakers.append(None)
                self.mouth_movements.append([])
                continue

            frame_speaker = None
            max_movement = 0
            frame_movements = []
            
            if len(faces[i]) == 1:
                active_speakers.append(0)  # Falls nur ein Gesicht erkannt wird, ist es der Sprecher
                self.mouth_movements.append([1])  # Dummy-Wert für Bewegung
                self.prev_gray = gray
                continue

            for idx, face in enumerate(faces[i]):
                
                x, y, w, h = face  # Gesichtskoordinaten
                mouth_roi = gray[y + int(h * 0.6): y + h, x:x + w]  # Mundregion
                
                if mouth_roi.size == 0:
                    frame_movements.append(0)
                    continue
                
                mouth_flow = self._calculate_optical_flow(mouth_roi)
                frame_movements.append(mouth_flow)
                if mouth_flow > max_movement:
                    max_movement = mouth_flow
                    frame_speaker = idx
            
            self.mouth_movements.append(frame_movements)
            active_speakers.append(frame_speaker)
            self.prev_gray = gray
        
        return active_speakers

    def _calculate_optical_flow(self, mouth_roi):
        if self.prev_mouths:
            prev_mouth = self.prev_mouths[-1]
            
            # Resize, falls die Größen nicht übereinstimmen
            if prev_mouth.shape != mouth_roi.shape:
                mouth_roi = cv2.resize(mouth_roi, (prev_mouth.shape[1], prev_mouth.shape[0]))
            
            flow = cv2.calcOpticalFlowFarneback(prev_mouth, mouth_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            movement = np.mean(np.abs(flow))
        else:
            movement = 0
        
        self.prev_mouths.append(mouth_roi)
        return movement

    def clean_active_speaker_list(self, active_speakers, faces):
        active_speakers = [256 if speaker is None else speaker for speaker in active_speakers]
        most_common = np.bincount(np.array(active_speakers)).argmax()
        li = [most_common if most_common != 256 else 0 for _ in active_speakers]
        for i, face_w in enumerate(faces):
            if not len(face_w) > 0:
                li[i] = -1
        return li

    def visualize_speaker(self):
        if not self.mouth_movements:
            print("Keine Daten zum Plotten vorhanden.")
            return

        num_faces = max(len(frame) for frame in self.mouth_movements if frame)
        fig, axes = plt.subplots(num_faces, 1, figsize=(10, 5 * num_faces), sharex=True)
        
        if num_faces == 1:
            axes = [axes]
        
        for i in range(num_faces):
            movements = [frame[i] if i < len(frame) else 0 for frame in self.mouth_movements]
            axes[i].plot(movements, label=f'Face {i}')
            axes[i].set_ylabel("Mundbewegung")
            axes[i].legend()
        
        plt.xlabel("Frame Index")
        plt.suptitle("Mundbewegung über die Zeit")
        plt.savefig("./../debug/mouth_movements.png")
        # plt.show()

    def scale(self, with_to: int, height_to: int):
        pass

    def plot_frame(self, frame, faces = None, active_speaker = -1):
        if faces is not None:
            faces = faces[frame]
            for i, (x, y, w, h) in enumerate(faces):
                if i == active_speaker:
                    cv2.rectangle(self.frames[frame], (x, y), (x+w, y+h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(self.frames[frame], (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.imwrite(f'./../debug/frame{frame}{active_speaker}.png', self.frames[frame])
        

    def get_landmarks(self):
        pass


class VideoPipeline():
    def __init__(self,path):
        self.path = path
        pass

    def run(self):
        files = os.listdir(self.path)
        i = 0
        for file in tqdm(files):
            video = VideoObject(f"{self.path}/{file}")
            if not video.is_open:
                continue
            faces = video.get_faces()
            if any([len(face) > 1 for face in faces]):
                video.plot_frame(i, faces = faces)
                active_speakers = video.get_active_speaker(faces)
                active_speakers = video.clean_active_speaker_list(active_speakers, faces)
                # print(active_speakers)
                video.visualize_speaker()
                video.plot_frame(i, faces = faces, active_speaker = active_speakers[i])
                video.close()
                if i >= 26:
                    break
                i += 1
            

if __name__ == '__main__':
    vp = VideoPipeline("./../original_videos")
    vp.run()