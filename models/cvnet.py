import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models  
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


import torch.nn as nn
import torch.utils.model_zoo as model_zoo



model_urls = {  
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class cvnet(nn.Module):  

	def __init__(self, block, layers, num_classes=3):  
		self.inplanes = 64  
		super(cvnet, self).__init__()  
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,  
							   bias=False)  
		self.bn1 = nn.BatchNorm2d(64)  
		self.relu = nn.ReLU(inplace=True)  
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  
		self.layer1 = self._make_layer(block, 64, layers[0])  
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  
		self.avgpool = nn.AvgPool2d((1,1))  
		# delete old fc layerl; create our new fc layer 
		self.fcnew = nn.Linear(512*block.expansion, num_classes)  
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []  
		layers.append(block(self.inplanes, planes, stride, downsample))  
		self.inplanes = planes * block.expansion  
		for i in range(1, blocks):  
			layers.append(block(self.inplanes, planes))  

		return nn.Sequential(*layers)  

	def forward(self, x):  
		x = self.conv1(x)  
		x = self.bn1(x)  
		x = self.relu(x)  
		x = self.maxpool(x)  

		x = self.layer1(x)  
		x = self.layer2(x)  
		x = self.layer3(x)  
		x = self.layer4(x)  

		x = self.avgpool(x)  
 
		x = x.view(x.size(0), -1)  
		x = self.fcnew(x)  

		return x  

		
def load_pretrained_model():
	#load model
	resnet50 = models.resnet50(pretrained=True)  
	cvnet = cvnet(Bottleneck, [3, 4, 6, 3])  
	#read parameters  
	# pretrained_parameters= resnet50.state_dict() 
	# print(pretrained_parameters) 
	curr_parameters = cvnet.state_dict()  
	# delete key 
	pretrained_parameters =  {k: v for k, v in pretrained_parameters.items() if k in curr_parameters}  
	# update key 
	curr_parameters.update(pretrained_parameters)  
	# load state_dict  
	cvnet.load_state_dict(curr_parameters)   
	# print(cvnet)  
