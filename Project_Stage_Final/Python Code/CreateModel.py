import json
from mrcnn.config import Config
from os import listdir
from mrcnn.utils import Dataset
import numpy as np
from numpy import expand_dims
from numpy import zeros
from numpy import mean
from numpy import asarray
from matplotlib import pyplot
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from matplotlib.patches import Rectangle
import cv2 
import skimage.io
from matplotlib import image

class Human(Dataset):
    
    def extract_boxes(self,filename):
        boxes = list()
        f = open(filename.replace('./labels/','./labels/FLIR_'))
        data = json.load(f)
        f.close()  
        boxes = data['bbox']
        # print('initial box- '+str(boxes))
        category_id = data['category_id']
        return boxes,category_id
    
    def load_dataset(self, dataset_dir, is_train=True):
        count=0
        self.add_class("dataset", 1, "human")
        images_dir = dataset_dir + 'human_images/'
        annotations_dir = dataset_dir + 'labels/'
        for filename in listdir(images_dir):
    		# extract image id
            image_id = filename.replace('.jpeg','').replace('FLIR_','')
            image_num = int(image_id)
            # all images after 150 if we are building the train set
            if is_train and image_num <= 1455:
                img_path = images_dir + filename
                ann_path = annotations_dir + image_id + '.json'
        		# add to dataset
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
 			# skip all images before 150 if we are building the test/val set
            if not is_train and image_num > 1455 and count <=20:
                count+=1
                img_path = images_dir + filename
                ann_path = annotations_dir + image_id + '.json'
        		# add to dataset
                self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)
    
    def load_mask(self, image_id):
    	# get details of image
    	info = self.image_info[image_id]
    	# define box file location
    	path = info['annotation']
    	boxes,category_id  = self.extract_boxes(path)
    	# create one array for all masks, each on a different channel
    	masks = zeros([512, 640, len(boxes)], dtype='uint8')
    	# create masks
    	class_ids = list()
    	for i in range(len(boxes)):
    		box = boxes[i]
    		row_s, row_e = box[1],box[3]+box[1] 
    		col_s, col_e = box[0],box[2]+box[0]
    		masks[row_s:row_e, col_s:col_e, i] = 1
    		class_ids.append(self.class_names.index('human'))
    	return masks, asarray(class_ids, dtype='int32')
    
    # load an image reference
    def image_reference(self, image_id):
    	info = self.image_info[image_id]
    	return info['path']

# load an image
def displayImage(self):
    image_id = '00010'
    image = train_set.load_image(int(image_id))
    print(image.shape)
    # load image mask
    mask, class_ids = train_set.load_mask(int(image_id))
    bbox = extract_bboxes(mask)
    print('final box- '+str(bbox))
    # display image with masks and bounding boxes
    display_instances(image, bbox, mask, class_ids, train_set.class_names)
 
# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=2):
	# load image and mask
	for i in range(n_images):
		# load the image and mask
		image = dataset.load_image(i)
		mask, _ = dataset.load_mask(i)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)[0]
		# define subplot
		pyplot.subplot(n_images, 2, i*2+1)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Actual')
		# plot masks
		for j in range(mask.shape[2]):
			pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
		# get the context for drawing boxes
		pyplot.subplot(n_images, 2, i*2+2)
		# plot raw pixel data
		pyplot.imshow(image)
		pyplot.title('Predicted')
		ax = pyplot.gca()
		# plot each box
		for box in yhat['rois']:
			# get coordinates
			y1, x1, y2, x2 = box
			# calculate width and height of the box
			width, height = x2 - x1, y2 - y1
			# create the shape
			rect = Rectangle((x1, y1), width, height, fill=False, color='red')
			# draw the box
			ax.add_patch(rect)
	# show the figure
	pyplot.show()     

# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
    NAME = "model_cfg"
	# number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
	# simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM=512
    IMAGE_MAX_DIM=640
    DETECTION_MIN_CONFIDENCE=0.1

# define a configuration for the model
class HumanConfig(Config):
    GPU_COUNT=2
    IMAGES_PER_GPU=1
	# Give the configuration a recognizable name
    NAME = "model_cfg"
	# Number of classes (background + human)
    NUM_CLASSES = 1 + 1
	# Number of training steps per epoch
    STEPS_PER_EPOCH = 120
    IMAGE_MIN_DIM=512
    IMAGE_MAX_DIM=640
    DETECTION_MIN_CONFIDENCE=0.1
    # GPU_COUNT=2
    
def createModel(layer):
    # prepare config
    config = HumanConfig()
    config.display()
    # define the model
    model = MaskRCNN(mode='training', model_dir='./model', config=config)
    # load weights (mscoco) and exclude the output layers
    model.load_weights('./mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask","conv1"])
    for lay in layer:
        # train weights (output layers or 'heads')
        model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=35, layers=lay)

def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
   		# convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
   		# convert image into one sample
        sample = expand_dims(scaled_image, 0)
   		# make prediction
        yhat = model.detect(sample, verbose=0)
   		# extract results for first sample
        r = yhat[0]
   		# calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
   		# store
        APs.append(AP)
    mAP = mean(APs)
    return mAP

def predict(train_set,test_set,modelPath):
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./model', config=cfg)
    # load model weights
    model.load_weights(modelPath, by_name=True)
    # evaluate model on training dataset
    train_mAP = evaluate_model(train_set, model, cfg)
    print("Train mAP: %.3f" % train_mAP)
    # evaluate model on test dataset
    test_mAP = evaluate_model(test_set, model, cfg)
    print("Test mAP: %.3f" % test_mAP)
  
def predictImage(train_set,test_set,modelPath):
    # create config
    cfg = PredictionConfig()
    # define the model
    model = MaskRCNN(mode='inference', model_dir='./model', config=cfg)
    # load model weights
    model_path = modelPath
    model.load_weights(model_path, by_name=True)
    # plot predictions for train dataset
    plot_actual_vs_predicted(train_set, model, cfg)
    # plot predictions for test dataset
    plot_actual_vs_predicted(test_set, model, cfg)

# train set
train_set = Human()
train_set.load_dataset('./', is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
 
# test/val set
test_set = Human()
test_set.load_dataset('./', is_train=False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))

modelPath='./model/model_cfg20200420T1751/mask_rcnn_model_cfg_0031.h5'

# create model'heads', ,'4+','all'
# models=['heads']
# createModel(models)

# #evaluate model
# predict(train_set,test_set,modelPath)

#predict Image
predictImage(train_set,test_set,modelPath)

#predict image
# cfg = PredictionConfig()
# plotIndividualImage(train_set,cfg,modelPath);


