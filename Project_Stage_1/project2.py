# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:36:45 2020

@author: adity
"""

from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras.layers import Dropout
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import cv2
import os
import json
import numpy as np


class Dataset():
    def __init__(self, image_dir, anno_dir):
        self.images = list()
        self.annotations = list()
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        
    def load_images(self):
        count = 0
        if os.path.exists(self.image_dir):
            if os.path.exists(self.anno_dir):
                for image_path in os.listdir(self.image_dir):
                    image_path_temp = os.path.join(self.image_dir,image_path)
                    img = cv2.imread(image_path_temp,0).astype('float32')/255
                    anno_path = os.path.join(self.anno_dir,image_path.replace('.jpeg','.json'))
                    with open(anno_path, 'r') as f:
                        annotated = json.load(f)
                    self.annotations.append(self.oneHotVector(annotated['bbox']))
                    # self.images.append(img.flatten())
                    self.images.append(img)
                    count+=1
                    if count == 100:
                        break
        return self.images,self.annotations
    
    def oneHotVector(self,bbox):
        vector = np.zeros(4)
        mask = np.arange(640).reshape(4,160)
        for box in bbox:
            x1 = box[0]
            x2 = box[0]+box[2]
            for i in range(0,4):
                if x1 in mask[i]:
                    vector[i] = 1
                if x2 in mask[i]:
                    vector[i] = 1
        return vector

def baseline_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (9, 9), activation='relu', input_shape=(512, 640, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    model.summary()
    return model
            
                
# humanData = Dataset('./human_images','./labels')
# images,anno = humanData.load_images()
# X_train, X_test, y_train, y_test = train_test_split(images, anno, test_size=0.33, random_state=42)
model = baseline_model()



                    
                    