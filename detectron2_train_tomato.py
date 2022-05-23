#!/usr/bin/env python3

############################################################
#Main settings
############################################################
workdir = "weights/tomaat1" #Used for storing weightsfile

nrclasses = 1 #How many classes are in annotations / model

############################################################
#Basics
############################################################

# import some common libraries
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog,MetadataCatalog
from detectron2.engine import DefaultTrainer

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()
   
# Use this setting to only expose certain GPUs in machine if present
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

############################################################
#Register datasets
############################################################

from detectron2.data.datasets import register_coco_instances

register_coco_instances("train", {}, "/media/Tomato_Annotationset 1_subset_200_40/train/train_adapted.json", "/media/Tomato_Annotationset 1_subset_200_40/train/img")
register_coco_instances("val", {}, "/media//Tomato_Annotationset 1_subset_200_40/val/val_adapted.json", "/media/Tomato_Annotationset 1_subset_200_40/val/img")


#Get metadata
# train_metadata = MetadataCatalog.get("train")
# print(train_metadata)

# train_metadata2 = MetadataCatalog.get("train2")
# print(train_metadata2)

# train_metadata3 = MetadataCatalog.get("train3")
# print(train_metadata3)

# train_metadata4 = MetadataCatalog.get("train4")
# print(train_metadata4)

# val_metadata = MetadataCatalog.get("val")
# print(val_metadata)

train_metadata = MetadataCatalog.get("train")
print(train_metadata)

val_metadata = MetadataCatalog.get("val")
print(val_metadata)

dataset_dicts_train = DatasetCatalog.get("train")
dataset_dicts_val = DatasetCatalog.get("val")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"))
#cfg.DATASETS.TRAIN = ("train", "train2", "train3","train4",)
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("val",)
cfg.NUM_GPUS = 1
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")

# solver file settings extracted from: https://github.com/facebookresearch/Detectron/blob/master/configs/04_2018_gn_baselines/scratch_e2e_mask_rcnn_R-101-FPN_3x_gn.yaml
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.LR_POLICY = 'steps_with_decay'
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 15000
cfg.SOLVER.STEPS = (1, 7000, 11000)
cfg.SOLVER.CHECKPOINT_PERIOD = 1000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = nrclasses

# https://medium.com/@apofeniaco/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e
cfg.OUTPUT_DIR = workdir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 

############################################################
#Training
############################################################

trainer.resume_or_load(resume=False)
trainer.train()

