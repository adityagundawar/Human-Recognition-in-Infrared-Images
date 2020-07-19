import cv2
import numpy as np
import os
import pandas as pd

vector_size = 32

def extract(img, kaze):
    img = cv2.imread(img)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kps = kaze.detect(gray,None)

    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    kps, dsc = kaze.compute(img, kps)
    dsc = dsc.flatten()
    needed_size = (vector_size * 64)
    if dsc.size < needed_size:
        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    dsc = np.append(dsc,["1"])
    return dsc

def main():
    files = []
    dsc = []
    header = ["vec"+str(i) for i in range(0,vector_size*64)]
    header.append("class")
    data ={hd:[] for hd in header}
    
    files_path = './human_images'
    if os.path.exists(files_path):
        for file in os.listdir(files_path):
            files.append(os.path.join(files_path,file))
    # create the feature using KAZE algo
    kaze = cv2.KAZE_create()
    for img in files:
        dsc = extract(img,kaze)
        for i in range(0,len(dsc)-1):
            data["vec"+str(i)].append(dsc[i])
        data['class'].append(dsc[-1])


    featureDf = pd.DataFrame(data)
    featureDf.to_csv('features.csv')
        



if __name__ == "__main__":
    main()