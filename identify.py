#!/usr/bin/env python

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

data_dir = './data'
test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), ])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('network_model.pth')
model.eval()

data = datasets.ImageFolder(data_dir, transform=test_transforms)
classes = data.classes



def get_single_images(idx):
	idx = [idx]
	from torch.utils.data.sampler import SubsetRandomSampler
	sampler = SubsetRandomSampler(idx)
	loader = torch.utils.data.DataLoader(data, sampler=sampler, batch_size=1)
	dataiter = iter(loader)
	images, labels = dataiter.next()
	return images, labels

def get_random_images(num):
	indices = list(range(len(data)))
	np.random.shuffle(indices)
	idx = indices[:num]

	from torch.utils.data.sampler import SubsetRandomSampler
	sampler = SubsetRandomSampler(idx)
	loader = torch.utils.data.DataLoader(data, 
	               sampler=sampler, batch_size=num)
	dataiter = iter(loader)
	images, labels = dataiter.next()
	return images, labels

def predict_image(image):
	image_tensor = test_transforms(image).float()
	image_tensor = image_tensor.unsqueeze_(0)
	input = Variable(image_tensor)
	input = input.to(device)
	output = model(input)
	index = output.data.cpu().numpy().argmax()
	return index



def show_random_training_results():
	to_pil = transforms.ToPILImage()
	images, labels = get_random_images(5)
	fig=plt.figure(figsize=(10,4))
	for ii in range(len(images)):
	    image = to_pil(images[ii])
	    index = predict_image(image)
	    sub = fig.add_subplot(1, len(images), ii+1)
	    res = int(labels[ii]) == index
	    sub.set_title(str(classes[index]) + ":" + str(res))
	    plt.axis('off')
	    plt.imshow(image)
	plt.show()

def get_result(idx):
	to_pil = transforms.ToPILImage()
	images, labels = get_single_images(idx)
	fig=plt.figure(figsize=(6,6))
	for ii in range(len(images)):
	    image = to_pil(images[ii])
	    index = predict_image(image)
	    sub = fig.add_subplot(1, len(images), ii+1)
	    sub.set_title(data.samples[idx][0] + '\nclass : ' + str(classes[index]))
	    plt.axis('off')
	    plt.imshow(image)
	plt.show()


show_random_training_results()
#get_result(409)	# input id of the image
