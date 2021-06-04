import torch 
import os



LEARNING_RATE = 1e-4
DEVICE        = "cude" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 16
NUM_EPOCHS    = 1
NUM_WORKERS   = 2
IMAGE_SIZE    = 225
TRAIN_IMG_DIR = os.path.dirname(os.getcwd()) + "\\data\\training\\images\\"
MASK_IMG_DIR  = os.path.dirname(os.getcwd()) + "\\data\\training\\v2.0\\label\\"
VAL_IMG_DIR   = os.path.dirname(os.getcwd()) + "\\data\\validation\\images\\"
VAL_IMG_DIR   = os.path.dirname(os.getcwd()) + "\\data\\validation\\v2.0\\labels\\"
