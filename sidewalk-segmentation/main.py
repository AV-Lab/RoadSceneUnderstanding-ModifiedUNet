import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np 

from tqdm import tqdm
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
	dir_mask    = [TRAIN_MASK_BOUNDARY_DIR, TRAIN_MASK_LANELINE_DIR, TRAIN_MASK_SIDEWALK_DIR, TRAIN_MASK_ROAD_DIR],
	transform   = train_aug_transform,
	batch_size  = BATCH_SIZE,
	num_workers = NUM_WORKERS,
	shuffle     = True
)

validation_loader  = getDatasetLoader(
	dir_data    = VAL_IMG_DIR,
	dir_mask    = [VAL_MASK_BOUNDARY_DIR, VAL_MASK_LANELINE_DIR, VAL_MASK_SIDEWALK_DIR, VAL_MASK_ROAD_DIR],
	transform   = test_aug_transoform,
	batch_size  = BATCH_SIZE,
	num_workers = NUM_WORKERS,
	shuffle     = False
)
# Model Setup
model     = UNET(in_channels= COLOR_CHANNEL, out_channels= N_CLASSES).cuda()
loss_fun  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)

scalar = torch.cuda.amp.GradScaler()
model, optimizer = loadParameters(model, optimizer, name= "side_walk_bg")

for epoch in range(NUM_EPOCHS):
	'''
	loop = tqdm(train_loader, leave = False)

	for batch_idx, (data, target) in enumerate(loop):

		data, target = data.cuda(), target.long().cuda()
		# Forward
		with torch.cuda.amp.autocast():
			pred = model(data)
			loss = loss_fun(pred, target)

		# Backward
		optimizer.zero_grad()
		scalar.scale(loss).backward()
		scalar.step(optimizer)
		scalar.update()

		loop.set_postfix(loss=loss.item())
	'''
	# Validation
	num_correct_boundary = 0
	num_correct_sidewalk = 0
	num_correct_laneline = 0
	num_correct_road = 0
	num_correct = 0
	num_pixels  = 0
	loss_total  = 0
	prev        = np.inf
	model.eval()

	with torch.no_grad():
		looper = tqdm(validation_loader, leave=False)
		for data, target in looper:
			data, target = data.float().cuda(), target.long().cuda()
			preds  = model(data)
			loss_total += loss_fun(preds, target)
			preds  = torch.softmax(preds, dim=1)
			preds  = torch.argmax(preds, dim = 1)
			# Score for each category
			num_correct_boundary = ((preds == 1) & (target == 1)).sum()

			num_correct_sidewalk = ((preds == 3) & (target == 3)).sum()
			num_correct_laneline = ((preds == 2) & (target == 2)).sum()
			num_correct_road = ((preds == 4) & (target == 4)).sum()
			# Assign class for each pixel

			num_pixels  += torch.numel(preds)
			num_correct += (preds == target).sum()
			print(f'sidewalk: {num_correct_sidewalk/(target == 3).sum() * 100:.2f}%, ')
			print(f'road: {num_correct_road/(target == 4).sum() * 100:.2f}%, ')
			print(f'lane: {num_correct_laneline/(target == 2).sum() * 100:.2f}%, ')
			print(f'curb: {num_correct_boundary/(target == 1).sum()*100:.2f}%, ')
			#dice_score  += (2 * (preds * target).sum())/ ((preds + target).sum() + 1e-8)


	print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
	print(f"loss {loss_total/len(validation_loader)}")

	if prev > loss_total/len(validation_loader):
		saveParameters(model = model, optimizer = optimizer, name= "side_walk_bg")
		prev = loss_total/len(validation_loader)

	model.train()

