

import os 
from PIL import Image 
import numpy as np
import cv2 as cv 
import pickle 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"Bilder")

face_cascade = cv.CascadeClassifier(r"C:\Users\simon\AppData\Local\Programs\Python\Python_CODE\Diplomarbeit_opencv\cascades\data\haarcascade_frontalface_alt2.xml")

current_id = 0 
label_ids = {}
y_labels = []
x_train =[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg") :
            path = os.path.join(root, file) 
            label = os.path.basename(root).replace(" ","_").lower()
            print(label, path)

            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1 

            id_ = label_ids[label]
            print(label_ids)

            pil_image = Image.open(path).convert("L") #grayscale 
            image_array = np.array(pil_image,"uint8")    #Bilder in Zahlen (numpy) verwandeln 
            print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5 )

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+h]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
#print(x_train)

