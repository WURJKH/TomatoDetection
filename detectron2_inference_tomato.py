#!/usr/bin/env python3


############################################################
#Main settings
############################################################
train = False
evaluate = True
workdir = "weights/tomaat1" #Used for storing + reading weightsfile

nrclasses = 1 #How many classes are in annotations / model

#Training
traindataset = "Tomaat" #options: "Tomaat", "Potato"


#Evaluation
testfolder = "/media/Tomato_Annotationset 1_subset_200_40/real_test/img"

weightsfile = "model_final.pth" #Name of weightsfile, should be in workdir


############################################################
#Basics
############################################################


# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os
import cv2
import random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.engine import DefaultTrainer
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 10,10

def imshow(img):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.show()
    
# run on gpu 0 (NVIDIA Geforce GTX 1080Ti) and gpu 1 (NVIDIA Geforce GTX 1070Ti)
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

############################################################
#Register datasets
############################################################

from detectron2.data.datasets import register_coco_instances

#Tomato
if traindataset == "Tomaat":
    register_coco_instances("train", {}, "/media/Tomato_Annotationset 1_subset_200_40/train/train_adapted.json", "/media/Tomato_Annotationset 1_subset_200_40/train/img")
    register_coco_instances("val", {}, "/media/Tomato_Annotationset 1_subset_200_40/val/val_adapted.json", "/media/Tomato_Annotationset 1_subset_200_40/val/img")

#Pop3Poot
elif traindataset == "other":
    register_coco_instances("train", {}, "own_datasets/")
    register_coco_instances("train2", {}, "own_datasets/")
    register_coco_instances("train3", {}, "own_datasets/")
    register_coco_instances("val", {}, "own_datasets/")

else:
    raise Exception("No dataset selected")

#Get metadata
train_metadata = MetadataCatalog.get("train")
print(train_metadata)

# train_metadata2 = MetadataCatalog.get("train2")
# print(train_metadata2)

# train_metadata3 = MetadataCatalog.get("train3")
# print(train_metadata3)

val_metadata = MetadataCatalog.get("val")
print(val_metadata)


dataset_dicts_train = DatasetCatalog.get("train")
dataset_dicts_val = DatasetCatalog.get("val")


#Display 3 images for validation of masks from annotations 
#for d in random.sample(dataset_dicts_train, 3):
#    img = cv2.imread(d["file_name"])
#    visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
#    vis = visualizer.draw_dataset_dict(d)
#    imshow(vis.get_image()[:, :, ::-1])


cfg = get_cfg()
cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")

if traindataset == "Tomaat":
    cfg.DATASETS.TRAIN = ("train",)

elif traindataset == "Potato":
    cfg.DATASETS.TRAIN = ("train", "train2", "train3",)

cfg.DATASETS.TEST = ("val",)

cfg.NUM_GPUS = 1
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")

# solver file settings extracted from: https://github.com/facebookresearch/Detectron/blob/master/configs/04_2018_gn_baselines/scratch_e2e_mask_rcnn_R-101-FPN_3x_gn.yaml
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.LR_POLICY = 'steps_with_decay'
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = (0, 7000, 11000)
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.DETECTIONS_PER_IMAGE = 300

cfg.MODEL.ROI_HEADS.NUM_CLASSES = nrclasses

# https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
cfg.OUTPUT_DIR = workdir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 

############################################################
#Training
############################################################


if train:
    trainer.resume_or_load(resume=False)
    trainer.train()

############################################################
#Evaluation
############################################################

if evaluate:
    #trainer.resume_or_load(resume=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weightsfile)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5 #0.01
    predictor = DefaultPredictor(cfg)



    for imgname in os.listdir(testfolder):
        img = cv2.imread(testfolder + "/" + imgname)
        outputs = predictor(img)

        instances = outputs["instances"].to("cpu")
        count = len(instances)
        print(instances)

        visualizer = Visualizer(img[:, :, ::-1], metadata=val_metadata, scale=0.8)
        vis = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
        imshow(vis.get_image()[:, :, ::-1])
        #print(countfolder + "/" + imgname[:-4]+"_"+repr(count)+".png")
        #cv2.imwrite(countfolder + "/" + imgname[:-4]+"_"+repr(count)+".png",img)