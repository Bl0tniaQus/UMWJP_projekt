import pickle
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
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
		print(self.network)
		self.network.classifier = torch.nn.Sequential(torch.nn.Identity())
		self.model = DecisionTreeClassifier()
		self.scaler = MinMaxScaler()
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		self.transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
		self.augment = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomResizedCrop(224), transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(), transforms.RandomRotation(degrees = 45), transforms.ColorJitter(), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
		self.losses = []
	def forward(self, x):
		return self.network(x)
	def predict(self, x):
		x_features = self.forward(x).detach().numpy()
		x_features = self.scaler.transform(x_features)
		pred = self.model.predict(x_features)
		return pred
	def loadBatch(self, ids):
		x = []
		y = []
		for path in range(len(self.xtrain[ids])):
			or_image = Image.open(self.xtrain[ids][path]).convert("RGB")
			image = or_image.copy()
			image = self.transform(image)
			x.append(image)
			y.append(self.ytrain[ids][path])
			for i in range(2):
				image = or_image.copy()
				image = self.augment(image)
				x.append(image)
				y.append(self.ytrain[ids][path]) 
		return torch.stack(x), torch.from_numpy(np.array(y))
		return torch.stack(x).to(torch.float32), torch.from_numpy(np.array(y))
	def fit(self, x, y):
		self.xtrain = x
		self.ytrain = y
		batch_size = 15
		x_all = []
		y_all = []
		print("extracting features")
		for i in range(0, len(self.xtrain) // batch_size):
			x, y = self.loadBatch(slice(batch_size*i , (batch_size * i) + (batch_size)))
			x = self.forward(x)
			if len(x_all) == 0:
				x_all = x.detach().numpy()
				y_all = y.detach().numpy()
			else:
				x_all = np.vstack((x_all, x.detach().numpy()))
				y_all = np.hstack((y_all, y.detach().numpy()))
		print("training classifier")
		self.scaler.fit(x_all)
		x_all = self.scaler.transform(x_all)
		
		self.model.fit(x_all, y_all)
			
def Train():
	labels_file = open('labels_pickle', 'rb')
	X_train_file = open('X_train_pickle', 'rb')
	Y_train_file = open('Y_train_pickle', 'rb')
	
	X_train = pickle.load(X_train_file)
	Y_train = pickle.load(Y_train_file)
	labels = pickle.load(labels_file)
	X_train_file.close()
	Y_train_file.close()
	labels_file.close()

	encoder = LabelEncoder()
	encoder.fit(labels)
	Y_train = encoder.transform(Y_train)
	model = classifier()
	model.fit(X_train, Y_train)
	
	model_pickle = open("model5_pickle", "wb")

	pickle.dump(model, model_pickle)

	model_pickle.close()






