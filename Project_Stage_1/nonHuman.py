import json
import os
import shutil

def parse(currentPath,labels,imagesWOHumans):
    print(currentPath)
    print(os.listdir(currentPath))

    for tar in os.listdir(currentPath):
        if tar != 'tar02':
            temp = os.path.join(currentPath,tar)
            print(temp)
            interested = os.path.join(temp,'interested')
            imagesDir = os.path.join(interested,'thermal_8_bit')
            jsonFile = os.path.join(interested,'thermal_annotations.json')
    
    
            with open(jsonFile, 'r') as f:
                annotated = json.load(f)
            
            images = annotated['images']
            categories = annotated['categories']
            annotations = annotated['annotations']
            print(len(annotations))
            images = annotated['images']
            imagesWithHumans = []
    
            for anno in annotations:
                tempJson = {}
                if anno['category_id'] == 1:
                    imageId = anno['image_id']
                    for image in images:
                        if image['id'] == imageId:
                            # imagesWithHumans.append(image['file_name'])
                            tempJson.update({'bbox': anno['bbox']})
                            tempJson.update({'category_id': anno['category_id']})
                            tempJson.update({'file_name': image['file_name'].split('/')[-1]})
                            shutil.copy(os.path.join(interested,image['file_name']),imagesWOHumans)
                            with open(os.path.join(labels,image['file_name'].split('/')[-1].replace('jpeg','json')), 'w') as fp:
                                json.dump(tempJson, fp)


if __name__ == "__main__":
    currentPath = os.getcwd()
    labels = os.path.join(currentPath,'labels')
    if not os.path.exists(labels):
        os.mkdir(labels)
    imagesWOHumans = os.path.join(currentPath,'imagesWOHumans')
    if not os.path.exists(imagesWOHumans):
        os.mkdir(imagesWOHumans)
    parse(os.path.join(currentPath,'Dataset'),labels,imagesWOHumans)