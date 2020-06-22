#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:

# --- SETTINGS ---
MEMORY_LIMIT_IN_MB = 6*1024
VISUALIZE = False
BACKGROUND_GRAY_LEVEL = 50
OBJECT_CLASS_FILTER = ['apple', 'cake', 'donut', 'frisbee', 'orange', 'sports ball']

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import time
import skimage.transform
import cv2 as cv
import datetime
import tensorflow as tf
from typing import Tuple, Dict

# Root directory of the project
# ROOT_DIR = os.path.abspath("../")
ROOT_DIR = os.path.abspath(".")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# --- helper functions ---
def skimage_to_opencv(im: np.ndarray) -> np.ndarray:
    #im *= 255
    im = im.astype(np.uint8)
    im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
    return im

def opencv_to_skimage(im: np.ndarray) -> np.ndarray:
    #im = im.astype(np.float32)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    #im /= 255.0
    return im


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 

# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# --- GPU config ---
# for limiting GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=math.floor(MEMORY_LIMIT_IN_MB))])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #input("Press [ENTER] to continue")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)



# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:
def draw_text(im: np.ndarray, text: str, position: Tuple[int, int]) -> None:
    cv.putText(im, text, position, cv.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,0), 2, cv.LINE_AA)
    return

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Run Object Detection

# In[5]:


# Load all images from the images folder
time_elapsed = []
# use timestamp as session name
session_name = datetime.datetime.now().strftime("session_%m-%d-%H_%M_%S_") + ("%.6f"%time.time())[-6:]
if VISUALIZE:
    os.makedirs(f"./results/{session_name}_visualized", exist_ok=True)

os.makedirs(f"./results/{session_name}_rois", exist_ok=True)
os.makedirs(f"./results/{session_name}_masks", exist_ok=True)

image_count = len(os.listdir(IMAGE_DIR))
for image_index, filename in enumerate(os.listdir(IMAGE_DIR)):
    time_start = time.time()
    #image_raw = skimage.io.imread(os.path.join(IMAGE_DIR, filename))
    image = cv.imread(os.path.join(IMAGE_DIR, filename))

    # re-scale to 1/2 of original image size
    #image = skimage.transform.rescale(image_raw, 0.5)
    # image = cv.resize(image, None, None, 0.5, 0.5)  # re-scale is not needed now
    
    # convert to skimage format
    image = opencv_to_skimage(image)

    # Run detection
    results = model.detect([image], verbose=0) # 'verbose' was 1

    # Visualize results
    r = results[0]
    if VISUALIZE:
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'], output_path=f"./results/{session_name}_visualized/{filename}")
    time_elapsed.append(time.time() - time_start)
    print(f"time elapsed: {time_elapsed[-1]:.3f}s", flush=True)
    print(f"{image_index+1}/{image_count} images processed")

    if len(r['rois']) > 0:
        rois: Tuple[dict] = []
        for roi_index, raw_roi in enumerate(r['rois']):
            y1, x1, y2, x2 = tuple(raw_roi)
            area = abs((x2-x1)*(y2-y1))
            class_name = class_names[r['class_ids'][roi_index]]
            # data type defined by ourselves
            roi = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'area': area, 'class_name': class_name, 'original_index': roi_index}
            if (class_name in OBJECT_CLASS_FILTER) or (len(OBJECT_CLASS_FILTER) == 0):
                rois.append(roi)
        
        if len(rois) == 0:
            # use original image as 'masked image' and 'ROI image'
            cv.imwrite(f"./results/{session_name}_masks/{filename}", skimage_to_opencv(image))
            cv.imwrite(f"./results/{session_name}_rois/{filename}", skimage_to_opencv(image))
            continue
        rois = sorted(rois, key=lambda x:1.0/x['area'])
        
        roi = rois[0]
        x1, y1, x2, y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']
        roi_image = image[y1:y2, x1:x2, :]
        # RGB -> BGR
        roi_image = np.stack((roi_image[:, :, 2], roi_image[:, :, 1], roi_image[:, :, 0]), axis=2)
        cv.imwrite(f"./results/{session_name}_rois/{filename}", roi_image)

        # applying mask
        # input("Press [ENTER] to continue")
        mask = r['masks'][:,:,roi['original_index']]
        mask = np.stack((mask, mask, mask), axis=2)
        bg = np.full(image.shape, BACKGROUND_GRAY_LEVEL)
        mask_image = np.where(mask == 1, image, bg)
        mask_image = mask_image[y1:y2, x1:x2, :]
        # RGB -> BGR
        mask_image = np.stack((mask_image[:, :, 2], mask_image[:, :, 1], mask_image[:, :, 0]), axis=2)
        # add text (object class)
        class_name = class_names[r['class_ids'][roi['original_index']]]
        #draw_text(mask_image, class_name, (30,30))
        cv.imwrite(f"./results/{session_name}_masks/{filename}", mask_image)
    

print(f"\n{'-'*30}")
print(f"{image_count} images processed.")
print(f"average processing time: {(sum(time_elapsed)/image_count):.3f}s")

# In[ ]:




