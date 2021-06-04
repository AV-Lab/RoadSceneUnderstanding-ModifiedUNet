import torch
import torch.nn as nn
import torch.optim as optim

from parameters import *
from models import *
from utils import (
	getAugmentationTransform,
	getDatasetLoader, 
	saveParameters,
	loadParameters
	)


# Transform
train_aug_transform = getAugmentationTransform(type = 'train')
test_aug_transoform = getAugmentationTransform(type = 'test')

# Dataset
train_loader = getDatasetLoader(
	dir_data    = TRAIN_IMG_DIR,
	dir_mask    = TRAIN_MASK_IMG_DIR,
	transform   = train_aug_transform,
	batch_size  = BATCH_SIZE,
	num_workers = NUM_WORKERS,
	shuffle     = True
)

validation_loader  = getDatasetLoader(
	dir_data    = VAL_IMG_DIR,
	dir_mask    = VAL_MASK_IMG_DIR,
	transform   = test_aug_transoform,
	batch_size  = BATCH_SIZE,
	num_workers = NUM_WORKERS,
	shuffle     = False
)


model     = UNET(in_channels= IN_CHANNELS_COLORED, out_channels= OUT_CHANNELS_COLORED).to(DEVICE)
loss_fun  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters, lr= LEARNING_RATE)

 