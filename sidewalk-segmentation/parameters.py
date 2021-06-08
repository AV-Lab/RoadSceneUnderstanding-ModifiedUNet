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
TRAIN_IMG_DIR      		= os.path.dirname(os.getcwd()) + "/data/training/images/"
TRAIN_MASK_IMG_DIR 		= os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels/"
TRAIN_MASK_VEHICLE_DIR 	= os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels_vehicle/"
VAL_IMG_DIR        		= os.path.dirname(os.getcwd()) + "/data/validation/images/"
VAL_MASK_IMG_DIR 		= os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels/"
VAL_MASK_VEHICLE_DIR    = os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels_vehicle/"
VIDEO_DIR 				= "input videos/"

# Save Path
SAVE_MODEL_DIR     = "saved models/" 
SAVE_VIDEO_DIR     = "output videos/"
SAVE_IMAGE_DIR     = "output images/"

# Model
IN_CHANNELS_COLORED  = 3
OUT_CHANNELS_COLORED = 3 
IN_CHANNELS_GRAY  = 1
OUT_CHANNELS_GRAY = 1


# Saved Models

UNET_MODEL = "unet_vehicle"
UNET_MODEL_PRETRAINED = "unet_vehicle_pretrained"