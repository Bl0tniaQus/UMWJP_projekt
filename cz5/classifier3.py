import pickle
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import numpy as np
import warnings
import os
from PIL import Image
from torchvision import transforms, models
warnings.filterwarnings('ignore') 
class classifier(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.network = models.mobilenet_v2(pretrained = True)
		for param in self.network.parameters():
			param.requires_grad = False	
		self.network.classifier = torch.nn.Sequential(torch.nn.Linear(1280,142))
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
		self.augment = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomResizedCrop(224), transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees = 45), transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
		self.losses = []
	def forward(self, x):
		return self.network(x)
	def predict(self, x):
		output = self.forward(x)
		pred = np.argmax(output.detach().numpy(), axis = 1)
		return pred
	def loadBatch(self, ids, data):
		if data=="val":
			x = []
			y = self.yval[ids]
			for path in self.xval[ids]:
				image = Image.open(path).convert("RGB")
				image = self.transform(image)
				x.append(image)
		if data=="train":
			x = []
			y = []
			for path in range(len(self.xtrain[ids])):
				or_image = Image.open(self.xtrain[ids][path]).convert("RGB")
				image = or_image.copy()
				image = self.transform(image)
				x.append(image)
				y.append(self.ytrain[ids][path])
				for i in range(5):
					image = or_image.copy()
					image = self.augment(image)
					x.append(image)
					y.append(self.ytrain[ids][path]) 
			return torch.stack(x), torch.from_numpy(np.array(y))
		return torch.stack(x).to(torch.float32), torch.from_numpy(np.array(y))
	def fit(self, x, y, xval, yval, n_iter):
		batch_size = 15
		batch_size_val = 83
		self.losses = []
		self.losses_val = []
		loss_function = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(self.parameters())
		self.xtrain = np.array(x)
		self.ytrain = np.array(y)
		self.xval = xval
		self.yval = yval
		for e in range(n_iter):
			print(f"Epoch: {e+1}/{n_iter}")
			ids = np.arange(self.xtrain.shape[0])
			np.random.shuffle(ids)
			self.xtrain = self.xtrain[ids]
			self.ytrain = self.ytrain[ids]
			batch_losses = []
			batch_losses_val = []
			for i in range(0, len(self.xtrain) // batch_size):
				x, y = self.loadBatch(slice(batch_size*i , (batch_size * i) + (batch_size)), "train")
				output = self.forward(x)
				loss = loss_function(output, y)
				batch_losses.append(loss.detach().numpy())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			for j in range(0, len(self.xval) // batch_size_val):
				x, y = self.loadBatch(slice(batch_size_val*j , (batch_size_val * j) + (batch_size_val)), "val")
				output = self.forward(x)
				loss = loss_function(output, y)
				batch_losses_val.append(loss.detach().numpy())
			self.losses.append(np.mean(batch_losses))
			self.losses_val.append(np.mean(batch_losses_val))
			
def Train():
	labels_file = open('labels_pickle', 'rb')
	X_train_file = open('X_train_pickle', 'rb')
	Y_train_file = open('Y_train_pickle', 'rb')
	X_val_file = open('X_val_pickle', 'rb')
	Y_val_file = open('Y_val_pickle', 'rb')
	
	X_train = pickle.load(X_train_file)
	Y_train = pickle.load(Y_train_file)
	X_val = pickle.load(X_val_file)
	Y_val = pickle.load(Y_val_file)
	labels = pickle.load(labels_file)
	X_train_file.close()
	Y_train_file.close()
	X_val_file.close()
	Y_val_file.close()
	labels_file.close()

	encoder = LabelEncoder()
	encoder.fit(labels)
	Y_val = encoder.transform(Y_val)
	Y_train = encoder.transform(Y_train)
	model = classifier()
	model.fit(X_train, Y_train, X_val, Y_val, 3)
	
	model_pickle = open("model3_pickle", "wb")

	pickle.dump(model, model_pickle)

	model_pickle.close()
# ~ Train()





