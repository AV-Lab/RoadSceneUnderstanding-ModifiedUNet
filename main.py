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

# Model Setup
model     = UNET(in_channels= IN_CHANNELS_COLORED, out_channels= OUT_CHANNELS_COLORED).to(DEVICE)
loss_fun  = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= LEARNING_RATE)

scalar = torch.cuda.amp.GradScaler()

for epoch in range(NUM_EPOCHS):
	loop = tqdm(train_loader)

	for batch_idx, (data, target) in enumerate(loop):
		data, target = data.to(DEVICE), target.to(DEVICE)


		# Forward
		with torch.cuda.amp.autocast():
			pred = model(data)
			loss = loss_fun(pred, target)

		# Backward
		optimizer.zerograd()
		scalar.scale(loss).backward()
		scale.step(optimizer)
		scale.update()

		loop.set_postfix(loss=loss.item())
	
	# Validation
	num_correct = 0
	num_pixels  = 0

	model.eval()

	with torch.no_grad():
		for data, target in validation_loader:
			data, target = data.to(DEVICE), target.to(DEVICE)
			preds = model(data)
			print('target', target[0])
			print('preds', preds[0])
			plt.subplot(2,1,1)
			plt.imshow(target[0])
			plt.subplot(2,1,1)
			plt.imshow(preds[0])
