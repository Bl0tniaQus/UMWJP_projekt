import torch
import data_loader as dl
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore') 
class classifier(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.network= torch.nn.Sequential(
		torch.nn.Linear(9, 83),
		#torch.nn.ReLU(),
		torch.nn.Linear(83, 166),
		#torch.nn.ReLU(),
		torch.nn.Linear(166, 332),
		#torch.nn.ReLU(),
		torch.nn.Linear(332, 166),
		)
		self.losses = []
	def forward(self, x):
		return self.network(x)
	def predict(self, x):
		y = self.forward(x)
		pred = np.argmax(y.detach().numpy(), axis = 1)
		return pred
	def fit(self, x, y, n_iter):
		self.losses = []
		loss_function = torch.nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(self.parameters())
		
		Y = np.zeros((len(y), len(np.unique(y["Name"].values))))
		for i in range(len(y)):
			Y[i][y["Name"].values[i]] = 1
		X = torch.from_numpy(x.values)
		Y = torch.from_numpy(Y)
		X = X.to(torch.float32)
		Y = Y.to(torch.float32)
		
		for e in range(n_iter):
			output = self.forward(X)
			loss = loss_function(output, Y)
			self.losses.append(loss.detach().numpy())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

originalTest = dl.readData("./test_data.csv")
train = dl.readData("./train_data.csv")
scaler = dl.trainStandardScaler(train)
X_train, Y_train = dl.prepareDataset(train, scaler)
test = originalTest.copy()
X_test, Y_test = dl.prepareDataset(test, scaler)
model = classifier()
model.fit(X_train,Y_train,200)
X_test = torch.from_numpy(X_test.values)
X_test = X_test.to(torch.float32)
Y_pred = model.predict(X_test)
ACC = accuracy_score(Y_pred, Y_test)
F1 = f1_score(Y_pred, Y_test, average="weighted")
print(f" Pytorch -  ACC: {ACC}; F1: {F1}")
# ~ plt.plot(model.losses)
# ~ plt.show()
scaler = dl.trainStandardScaler(train)
X_train, Y_train = dl.prepareDataset(train, scaler)
test = originalTest.copy()
X_test, Y_test = dl.prepareDataset(test, scaler)
model = MLPClassifier()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
ACC = accuracy_score(Y_pred, Y_test)
F1 = f1_score(Y_pred, Y_test, average="weighted")
print(f" Default MLP - ACC: {ACC}; F1: {F1}")

