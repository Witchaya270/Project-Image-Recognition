import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import imageio
from torch import Tensor

"""
Loads the train/test set. 
Every image in the dataset is 110x220 pixels and the labels are numbered from 1-10

Set root to point to the Train/Test folders.
"""

# Creating a sub class of torch.utils.data.dataset.Dataset
class notMNIST(Dataset):

	# The init method is called when this class will be instantiated.
	def __init__(self, root):
		Images, Y = [], []
		folders = os.listdir(root)

		for folder in folders:
			folder_path = os.path.join(root, folder)
			for ims in os.listdir(folder_path):
				try:
					img_path = os.path.join(folder_path, ims)
					Images.append(np.array(Image.open(img_path).convert('L')))
					Y.append(int(folder.split('_')[0])) 
				except:
					# Some images in the dataset are damaged
					print("File {}/{} is broken".format(folder, ims))
		data = [(x, y) for x, y in zip(Images, Y)]
		self.data = data

	# The number of items in the dataset
	def __len__(self):
		return len(self.data)

	# The Dataloader is a generator that repeatedly calls the getitem method.
	# getitem is supposed to return (X, Y) for the specified index.
	def __getitem__(self, index):
		img = self.data[index][0]

		# 8 bit images. Scale between [0,1]. This helps speed up our training
		img = img.reshape(110, 220) / 255.0

		# Input for Conv2D should be Channels x Height x Width
		img_tensor = Tensor(img).view(1, 110, 220).float()
		label = self.data[index][1]
		return (img_tensor, label)
