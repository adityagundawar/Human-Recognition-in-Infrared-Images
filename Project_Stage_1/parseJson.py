import json
import os
import shutil

path = "./other_images"

def parse():
    personId = 1

    with open('./thermal_annotations.json', 'r') as f:
        annotated = json.load(f)

    images = annotated['images']
    categories = annotated['categories']
    annotations = annotated['annotations']
    print(len(annotations))
    images = annotated['images']
    imagesWithHumans = []
    counter = 0

    for anno in annotations:
        if anno['category_id'] in [2,3,4] and counter <= 888:
            counter = counter+1
            imageId = anno['image_id']
            for image in images:
                if image['id'] == imageId:
                    imagesWithHumans.append(image['file_name'])

    # print(imagesWithHumans)
    if not os.path.exists(path):
        os.mkdir(path)
        for i in imagesWithHumans:
            shutil.copy(i,path)

if __name__ == "__main__":
    parse()





