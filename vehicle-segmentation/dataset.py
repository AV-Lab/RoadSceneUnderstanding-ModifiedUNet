import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MapillaryDataset(Dataset):
	def __init__(self, image_dir, mask_dir, transform = None, multiple_classes = False):
		self.image_dir 			= image_dir
		self.mask_dir  			= mask_dir
		self.transform 			= transform
		self.multiple_classes 	= multiple_classes
		self.images    			= os.listdir(self.image_dir)


	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.images[index])
		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		if not self.multiple_classes:
			mask_path = os.path.join(self.mask_dir, self.images[index]).replace('.jpg', '.png')
			mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
			mask /= 255.

			if self.transform is not None:
				augmentation = self.transform(image=image, mask=mask)
				mask = augmentation['mask']
				image = augmentation['image']

			return image, mask.unsqueeze(0)
		else:
			output_mask = np.zeros((image.shape[0], image.shape[1]))
			for idx, mask_path in enumerate(self.mask_dir):
				mask_path_item = os.path.join(mask_path, self.images[index]).replace('.jpg', '.png')
				mask = cv2.imread(mask_path_item, cv2.IMREAD_GRAYSCALE)
				# Process
				output_mask[mask > 0] = idx + 1

			if self.transform is not None:
				augmentation = self.transform(image= image, mask=output_mask)
				output_mask = augmentation['mask']
				image = augmentation['image']

			return image, output_mask

