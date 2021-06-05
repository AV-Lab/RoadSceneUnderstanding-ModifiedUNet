import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class MapillaryDataset(Dataset):
	def __init__(self, image_dir, mask_dir, transform = None):
		self.image_dir = image_dir
		self.mask_dir  = mask_dir
		self.transform = transform
		self.images    = os.listdir(self.image_dir)


	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_path  = os.path.join(self.image_dir, self.images[index])
		mask_path = os.path.join(self.mask_dir, self.images[index]).replace('.jpg', '.png')

		image = cv2.imread(img_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		mask  = mask/ 255.

		if self.transform is not None:
			augmentation = self.transform(image = image, mask=mask)
			image = augmentation['image']
			mask  = augmentation['mask']

		
		return image, mask.unsqueeze(0)


def test_backdir(dir_):
	target = os.listdir(dir_)[0]
	img_path = os.path.join(dir_, target)
	print(img_path)
	img = cv2.imread(img_path)

	cv2.imshow('t',img)
	cv2.waitKey()

#test_backdir(os.path.dirname(os.getcwd()) + "\\data\\training\\images")

