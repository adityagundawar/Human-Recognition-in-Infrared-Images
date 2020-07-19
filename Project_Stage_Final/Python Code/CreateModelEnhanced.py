# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:36:45 2020

@author: adity
"""
from tensorflow import lite
from keras.models import Sequential
from keras.layers import Dense
from keras import layers
# from keras.layers import Dropout
# from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import cv2
import os
import json
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model
from numpy import zeros
from numpy import asarray
import tensorflow as tf


class Dataset():
    def __init__(self, image_dir, anno_dir):
        self.images = []
        self.annotations = list()
        self.image_dir = image_dir
        self.anno_dir = anno_dir
        
    def load_images(self):
        if os.path.exists(self.image_dir):
            if os.path.exists(self.anno_dir):
                count=0
                for image_path in os.listdir(self.image_dir):
                    tempImgList = []
                    tempAnnoList = []
                    image_path_temp = os.path.join(self.image_dir,image_path)
                    img = cv2.imread(image_path_temp,0).astype('float32')/255
                    anno_path = os.path.join(self.anno_dir,image_path.replace('.jpeg','.json'))
                    with open(anno_path, 'r') as f:
                        annotated = json.load(f)
                    bbox = annotated['bbox']
                    mask = load_mask(bbox)
                    a=0
                    b=0
                    c = 0
                    while a<img.shape[0]:
                        tempImg = img[a:a+128,b:b+160]
                        tempMask = mask[a:a+128,b:b+160]
                        if tempImg.shape[1] == 0:
                            break
                        if len(np.where(tempMask == 1)[0]) >= 1:
                            self.annotations.append(1)
                            c+=1
                            self.images.append(tempImg)
                        else:
                            tempAnnoList.append(0)
                            tempImgList.append(tempImg)
                        b+=160
                        if a >= 512 and b>=640:
                            break
                        if a <= 512 and b>=640:
                            a+=128
                            b=0
                    # count+=1
                    # if count == 100:
                    #     break
                    countOf1 = c
                    countOf0 = len(tempAnnoList)
                    val =countOf0-1
                    while(countOf0>countOf1+2 and val > 0):
                        del tempAnnoList[val]
                        del tempImgList[val]
                        val-=1
                        countOf0-=1
                    self.images.extend(tempImgList)
                    self.annotations.extend(tempAnnoList)             
        return self.images,self.annotations
    
    def oneHotVector(self,bbox):
        vector = np.zeros(4)
        mask = np.arange(640).reshape(4,40)
        for box in bbox:
            x1 = box[0]
            x2 = box[0]+box[2]
            for i in range(0,4):
                if x1 in mask[i]:
                    vector[i] = 1
                if x2 in mask[i]:
                    vector[i] = 1
        return vector

def load_mask(boxes):
	# create one array for all masks, each on a different channel
	masks = zeros([512, 640], dtype='uint8')
	# create masks
	class_ids = list()
	for i in range(len(boxes)):
		box = boxes[i]
		row_s, row_e = box[1],box[3]+box[1] 
		col_s, col_e = box[0],box[2]+box[0]
		masks[row_s:row_e, col_s:col_e] = 1
	return masks

def baseline_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 160, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))   
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    model.summary()
    return model
            
def imageEvaluate(imagePath,modelPath):
    model = load_model(modelPath)
    img = cv2.imread(imagePath,0).astype('float32')/255
    a=0
    b=0
    image=[]
    while a<img.shape[0]:
        tempImg = img[a:a+128,b:b+160]
        if tempImg.shape[1] != 0:
            tempImg = tempImg.reshape((1, 128, 160, 1))
            prediction = model.predict(tempImg)
            image.append(prediction.argmax())
        else:
            break
        b+=160
        if a >= 512 and b>=640:
            break
        if a <= 512 and b>=640:
            a+=128
            b=0
    print(image.reshape(4,4))
    return image
           
def createModel(modelPath):     
    # humanData = Dataset('./hmi','./labels')
    humanData = Dataset('./human_images','./labels')
    images,anno = humanData.load_images()
    
    images=np.asarray(images)
    anno=np.asarray(anno)
    X_train, X_test, y_train, y_test = train_test_split(images, anno, test_size=0.33, random_state=42)
    model = baseline_model()
    Xtrain_images = X_train.reshape((X_train.shape[0], 128, 160, 1))
    X_test = X_test.reshape((X_test.shape[0], 128, 160, 1))
    train_labels = to_categorical(y_train)
    test_labels = to_categorical(y_test)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(Xtrain_images, train_labels, epochs=35, batch_size=256)
    model.save(modelPath)
    test_loss, test_acc = model.evaluate(X_test, test_labels)

def convertH5toTflite(modelPath):
    converter = lite.TFLiteConverter.from_keras_model_file(modelPath)
    tfmodel = converter.convert()
    open("model.tflite","wb").write(tfmodel)
    
modelPath = './model.h5'
# createModel(modelPath)
imageEvaluate('./FLIR_00002.jpeg',modelPath)
# convertH5toTflite(modelPath)
                    
                    