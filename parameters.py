import torch 
import os


# Model Parameters
LEARNING_RATE = 1e-4
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 16
NUM_EPOCHS    = 1
NUM_WORKERS   = 1
IMAGE_SIZE    = 225

# Path Parameters
TRAIN_IMG_DIR      		= os.path.dirname(os.getcwd()) + "/data/training/images/"
TRAIN_MASK_IMG_DIR 		= os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels/"
TRAIN_MASK_VEHICLE_DIR 	= os.path.dirname(os.getcwd()) + "/data/training/v2.0/labels_vehicle/"
VAL_IMG_DIR        		= os.path.dirname(os.getcwd()) + "/data/validation/images/"
VAL_MASK_IMG_DIR 		= os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels/"
VAL_MASK_VEHICLE_DIR    = os.path.dirname(os.getcwd()) + "/data/validation/v2.0/labels_vehicle/"

SAVE_MODEL_DIR     = "saved models/" 
# Model
IN_CHANNELS_COLORED  = 3
OUT_CHANNELS_COLORED = 3 
IN_CHANNELS_GRAY  = 1
OUT_CHANNELS_GRAY = 1
