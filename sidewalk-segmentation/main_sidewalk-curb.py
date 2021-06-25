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
	dir_mask    = [TRAIN_MASK_SIDEWALK_DIR, TRAIN_MASK_LANELINE_DIR,
				   TRAIN_MASK_BOUNDARY_DIR, TRAIN_MASK_TRAFFIC_DIR ,TRAIN_MASK_ROAD_DIR, TRAIN_MASK_ROAD_SUDDEN_DIR,
				   TRAIN_MASK_ROAD_BICYCLE_DIR, TRAIN_MASK_ROAD_PARKING, TRAIN_MASK_CROSSWALK,TRAIN_MASK_STREET_SINK],
	transform   = train_aug_transform,
	batch_size  = BATCH_SIZE,
	num_workers = NUM_WORKERS,
	shuffle     = True
)

validation_loader  = getDatasetLoader(
	dir_data    = VAL_IMG_DIR,
	dir_mask    = [VAL_MASK_SIDEWALK_DIR, VAL_MASK_LANELINE_DIR, VAL_MASK_BOUNDARY_DIR, VAL_MASK_TRAFFIC_DIR ,
				   VAL_MASK_ROAD_DIR, VAL_MASK_ROAD_SUDDEN_DIR, VAL_MASK_ROAD_BICYCLE_DIR, VAL_MASK_ROAD_PARKING,
				   VAL_MASK_CROSSWALK,VAL_MASK_STREET_SINK],
	transform   = test_aug_transoform,
	batch_size  = BATCH_SIZE,
	num_workers = NUM_WORKERS,
	shuffle     = False
)
# sidewalk, lanline, boundary, traffic, road, road_sudden, road_bicycle, road_parking, crosswalk, street_sink
# Model Setup
model     = UNET(in_channels= COLOR_CHANNEL, out_channels= N_CLASSES).cuda()
loss_fun  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)

scalar = torch.cuda.amp.GradScaler()
model, optimizer = loadParameters(model, optimizer, name= "full_model")
prev        = 0

for epoch in range(NUM_EPOCHS):

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
	
	# Validation
	num_correct_boundary = 0
	num_correct_sidewalk = 0
	num_correct = 0
	num_pixels  = 0
	loss_total  = 0
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
			sidewalk = ((preds == 1) & (target == 1)).sum()
			lanline = ((preds == 2) & (target == 2)).sum()
			boundary = ((preds == 3) & (target == 3)).sum()
			traffic = ((preds == 4) & (target == 4)).sum()
			road = ((preds == 5) & (target == 5)).sum()
			road_sudden = ((preds == 6) & (target == 6)).sum()
			road_bicycle = ((preds == 7) & (target == 7)).sum()
			road_parking = ((preds == 8) & (target == 8)).sum()
			crosswalk = ((preds == 9) & (target == 9)).sum()
			street_sink = ((preds == 10) & (target == 10)).sum()

			# Assign class for each pixel

			num_pixels  += torch.numel(preds)
			num_correct += (preds == target).sum()

			print(f'sidewalk: {sidewalk / (target == 1).sum() * 100:.2f}%, ')
			print(f'lanline: {lanline / (target == 2).sum() * 100:.2f}%, ')
			print(f'boundary: {boundary / (target == 3).sum() * 100:.2f}%, ')
			print(f'traffic: {traffic / (target == 4).sum() * 100:.2f}%, ')
			print(f'road: {road / (target == 5).sum() * 100:.2f}%, ')
			print(f'road_sudden: {road_sudden / (target == 6).sum() * 100:.2f}%, ')
			print(f'road_bicycle: {road_bicycle / (target == 7).sum() * 100:.2f}%, ')
			print(f'road_parking: {road_parking / (target == 8).sum() * 100:.2f}%, ')
			print(f'crosswalk: {crosswalk / (target == 9).sum() * 100:.2f}%, ')
			print(f'street_sink: {street_sink / (target == 10).sum() * 100:.2f}%, ')


	print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
	print(f"loss {loss_total/len(validation_loader)}")

	if prev < num_correct:
		saveParameters(model = model, optimizer = optimizer, name= "full_model")
		prev = num_correct

	model.train()

