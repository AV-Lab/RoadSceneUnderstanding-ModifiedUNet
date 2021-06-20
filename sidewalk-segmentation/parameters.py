import torch 
import os

# Model Parameters
LEARNING_RATE = 0.00001
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 8
NUM_EPOCHS    = 500
NUM_WORKERS   = 2
IMAGE_SIZE    = 320

# Path Parameters
TRAIN_IMG_DIR       = os.path.dirname(os.getcwd()) + "/data/training/images/"
TRAIN_MASK_IMG_DIR 	= os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels/"

VAL_IMG_DIR        	= os.path.dirname(os.getcwd()) + "/data/validation/images/"
VAL_MASK_IMG_DIR 	= os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels/"

TRAIN_MASK_SIDEWALK_DIR  = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_sidewalk/"

TRAIN_MASK_LANELINE_DIR  = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_laneline/"
TRAIN_MASK_LANEPOINT_DIR = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_lanepoint/"
TRAIN_MASK_BOUNDARY_DIR  = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_boundary/"

TRAIN_MASK_TRAFFIC_DIR   = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_traffic/"

TRAIN_MASK_ROAD_DIR  	 = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_road/"
TRAIN_MASK_ROAD_SUDDEN_DIR = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_road_sudden_issue/"
TRAIN_MASK_ROAD_BICYCLE_DIR = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_road_bicycle/"
TRAIN_MASK_ROAD_PARKING     = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_road_parking/"

TRAIN_MASK_CROSSWALK = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_crosswalk/"
TRAIN_MASK_STREET_SINK = os.path.dirname(os.getcwd()) + "/data/training/v1.2/labels_street_sink/"


VAL_MASK_SIDEWALK_DIR  = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_sidewalk/"

VAL_MASK_LANELINE_DIR  = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_laneline/"
VAL_MASK_LANEPOINT_DIR = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_lanepoint/"
VAL_MASK_BOUNDARY_DIR  = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_boundary/"

VAL_MASK_TRAFFIC_DIR   = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_traffic/"

VAL_MASK_ROAD_DIR  	 = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_road/"
VAL_MASK_ROAD_SUDDEN_DIR = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_road_sudden_issue/"
VAL_MASK_ROAD_BICYCLE_DIR = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_road_bicycle/"
VAL_MASK_ROAD_PARKING     = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_road_parking/"

VAL_MASK_CROSSWALK = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_crosswalk/"
VAL_MASK_STREET_SINK = os.path.dirname(os.getcwd()) + "/data/validation/v1.2/labels_street_sink/"



VIDEO_DIR 				= "input videos/"

# Save Path
SAVE_MODEL_DIR     = "saved models/" 
SAVE_VIDEO_DIR     = "output videos/"
SAVE_IMAGE_DIR     = "output images/"

# Model
COLOR_CHANNEL = 3
GRAY_CHANNEL  = 1
N_CLASSES     = 3

# Saved Models

UNET_MODEL_SAVE_DIR = "unet-streetinfo"

# Colors
# Sidewalk
SIDEWALK_COLOR        = [244, 35, 232]
# Laneline
LANELINE_COLOR        = [255, 255, 255]
LANELINE_COLOR_MARKERS= [90, 120, 150]
# Kerb
KERB_COLOR_1     = [196, 196, 196]
KERB_COLOR_2     = [170, 170, 170]
# Traffic
TRAFFIC_LIGHT    = [250, 170, 30]
# Road
ROAD_COLOR       = [128, 64, 128]
ROAD_SUDDEN_ISSUE= [110, 110, 110]
PARKING          = [250, 170, 160]
ROAD_BICYCLE     = [128, 64, 255]
CROSSWALK_1      = [200, 128, 128]
CROSSWALK_2      = [140, 140, 200]

# Objects
STREET_SINK      = [100, 128, 160]
