import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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
	dir_mask    = TRAIN_MASK_VEHICLE_DIR,
	transform   = train_aug_transform,
	batch_size  = BATCH_SIZE,
	num_workers = NUM_WORKERS,
	shuffle     = True
)

validation_loader  = getDatasetLoader(
	dir_data    = VAL_IMG_DIR,
	dir_mask    = VAL_MASK_VEHICLE_DIR,
	transform   = test_aug_transoform,
	batch_size  = BATCH_SIZE,
	num_workers = NUM_WORKERS,
	shuffle     = False
)

# Model Setup
model     = UNET(in_channels= IN_CHANNELS_COLORED, out_channels= OUT_CHANNELS_GRAY).cuda()
loss_fun  = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)

scalar = torch.cuda.amp.GradScaler()

for epoch in range(NUM_EPOCHS):
	loop = tqdm(train_loader)

	for batch_idx, (data, target) in enumerate(loop):

		data, target = data.cuda(), target.float().cuda()

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
	num_correct = 0
	num_pixels  = 0
	dice_score  = 0
	prev        = np.inf
	model.eval()

	with torch.no_grad():
		for data, target in validation_loader:
			data, target = data.cuda(), target.float().cuda()
			
			preds = torch.sigmoid(model(data))
			preds[preds >= 0.5] = 1.0

			num_correct += (preds == target).sum()
			num_pixels  += torch.numel(preds)
			dice_score  += (2 * (preds * target).sum())/ ((preds + target).sum + 1e-8)


	print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
	print(f"Dice score: {dice_score/len(validation_loader)}")

	if prev > (num_correct/num_pixels*100):
		saveParameters(model = model, optimizer = optimizer, name= 'unet_vehicle')

	model.train()
