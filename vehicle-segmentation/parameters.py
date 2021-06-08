import torch 
import os

# Model Parameters
LEARNING_RATE = 0.00001
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 16
NUM_EPOCHS    = 500
NUM_WORKERS   = 2
IMAGE_SIZE    = 320

# Path Parameters
TRAIN_IMG_DIR       = os.path.dirname(os.getcwd()) + "/data/training/images/"
TRAIN_MASK_IMG_DIR 	= os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels/"

VAL_IMG_DIR        	= os.path.dirname(os.getcwd()) + "/data/validation/images/"
VAL_MASK_IMG_DIR 	= os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels/"

TRAIN_MASK_SIDEWALK_DIR  = os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels_sidewalk/"
TRAIN_MASK_ROAD_DIR  	 = os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels_road/"
TRAIN_MASK_LANELINE_DIR  = os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels_laneline/"
TRAIN_MASK_LANEPOINT_DIR = os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels_lanepoint/"
TRAIN_MASK_BOUNDARY_DIR  = os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels_boundary/"

VAL_MASK_SIDEWALK_DIR  = os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels_sidewalk/"
VAL_MASK_ROAD_DIR  	 = os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels_road/"
VAL_MASK_LANELINE_DIR  = os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels_laneline/"
VAL_MASK_LANEPOINT_DIR = os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels_lanepoint/"
VAL_MASK_BOUNDARY_DIR  = os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels_boundary/"

VIDEO_DIR 				= "input videos/"

# Save Path
SAVE_MODEL_DIR     = "saved models/" 
SAVE_VIDEO_DIR     = "output videos/"
SAVE_IMAGE_DIR     = "output images/"

# Model
COLOR_CHANNEL = 3
GRAY_CHANNEL  = 1
N_CLASSES     = 4

# Saved Models

UNET_MODEL_SAVE_DIR = "unet-streetinfo"

# Colors
SIDEWALK_COLOR   = [244, 35, 232]
LANELINE_COLOR   = [255, 255, 255]
POINTSLINE_COLOR = [128, 128, 128]
BOUNDARY_COLOR   = [196, 196, 196]
ROAD_COLOR       = [128, 64, 128]