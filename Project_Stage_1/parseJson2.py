import json
import os
import shutil
import json

def parse():
    personId = 1

    with open("./Dataset/tar01/interested/thermal_annotations.json", 'r') as f:
        annotated = json.load(f)
    
    images = annotated['images']
    categories = annotated['categories']
    annotations = annotated['annotations']
    images = annotated['images']
    imagesWithHumans = []
    fullresult = []
    for anno in annotations:
        if anno['category_id'] == 1:
            result = {}
            imageId = anno['image_id']
            for image in images:
                if image['id'] == imageId:
                    res = [ sub['fileName'] for sub in fullresult ]
                    if image['file_name'] in res:                      
                        row = next(item for item in fullresult if item["fileName"] == image['file_name'])
                        # row['bbox'] = [anno['bbox'],row['bbox']]
                        row['bbox'].append(anno['bbox'])
                    else:
                        result['bbox']=[anno['bbox']]
                        result['category_id']=1
                        result['fileName']=image['file_name']
                        fullresult.append(result)  
                    imagesWithHumans.append(image['file_name']) 
            
   # print(imagesWithHumans)
    if not os.path.exists('./labels'):
       os.mkdir('./labels')
    for result in fullresult:
        if len(result) > 0:
           filename = result['fileName'].replace('thermal_8_bit','').replace('.jpeg','.json')
           with open('./labels'+filename, "w+") as outfile: 
               json.dump(result, outfile)         
    if not os.path.exists('./human_images'):
        os.mkdir('./human_images')
    for i in imagesWithHumans:
        shutil.copy("./Dataset/tar01/interested/"+i,'./human_images')    
     
if __name__ == "__main__":
    parse()
