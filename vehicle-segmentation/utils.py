from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from dataset import *
from parameters import *

import albumentations as Aug
import torch

def saveParameters(model, optimizer, name, epoch = None):
	state = {
		'epoch': epoch,
		'state_dict': model.state_dict(),
		'opt_dict': optimizer.state_dict()
	} if epoch is not None else {
		'state_dict': model.state_dict(),
		'opt_dict': optimizer.state_dict()
	}

	torch.save(state, SAVE_MODEL_DIR + name)

def loadParameters(model, optimizer, name):
	state = torch.load(SAVE_MODEL_DIR + name)

	assert state is not None

	model.load_state_dict(state['state_dict'])
	if optimizer is not None:
		optimizer.load_state_dict(state['opt_dict'])

		return model, optimizer
	return model

def getAugmentationTransform(type):
	return Aug.Compose([
			Aug.Resize(height=IMAGE_SIZE, width= IMAGE_SIZE),
			Aug.Rotate(limit=20, p=0.8),
			Aug.HorizontalFlip(p=0.5),
			Aug.Normalize(
				mean=[0.0, 0.0, 0.0],
				std= [1.0, 1.0, 1.0],
				max_pixel_value = 255.0
				),
			ToTensorV2()
		]) if type=='train' else Aug.Compose([
			Aug.Resize(height=IMAGE_SIZE, width= IMAGE_SIZE),
			Aug.Normalize(
				mean=[0.0, 0.0, 0.0],
				std= [1.0, 1.0, 1.0],
				max_pixel_value = 255.0
				),
			ToTensorV2()
		])

def getDatasetLoader(dir_data, dir_mask, transform, batch_size, 
	num_workers, pin_memory= True, shuffle = False):
	dataset = MapillaryDataset(
		image_dir = dir_data, 
		mask_dir  = dir_mask , 
		transform = transform,
		multiple_classes = True
	)

	dataloader = DataLoader(
		dataset,
		batch_size  = batch_size,
		num_workers = num_workers,
		pin_memory  = pin_memory,
		shuffle     = shuffle
		)

	return dataloader
