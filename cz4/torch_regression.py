import torch
import data_loader as dl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 
class regressor(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.network= torch.nn.Sequential(
		torch.nn.Linear(8, 100),
		torch.nn.ReLU(),
		torch.nn.Linear(100, 85),
		torch.nn.ReLU(),
		torch.nn.Linear(85, 70),
		torch.nn.ReLU(),
		torch.nn.Linear(70, 55),
		torch.nn.ReLU(),
		torch.nn.Linear(55, 40),
		torch.nn.ReLU(),
		torch.nn.Linear(40, 25),
		torch.nn.ReLU(),
		torch.nn.Linear(25, 10),
		torch.nn.ReLU(),
		torch.nn.Linear(10, 1)
		)
		self.losses = []
	def forward(self, x):
		return self.network(x)
	def predict(self, x):
		return torch.round(self.forward(x))
	def fit(self, x, y, n_iter):
		self.losses = []
		loss_function = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(self.parameters())
		X = torch.from_numpy(x.values)
		Y = torch.from_numpy(y.values)
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
scaler = dl.trainStandardScalerHPRegression(train)
X_train, Y_train = dl.prepareDatasetHPRegression(train, scaler)
test = originalTest.copy()
X_test, Y_test = dl.prepareDatasetHPRegression(test, scaler)	
model = regressor()
model.fit(X_train,Y_train,10000)
X_test = torch.from_numpy(X_test.values)
X_test = X_test.to(torch.float32)
Y_pred = model.predict(X_test).data.numpy()
MSE = mean_squared_error(Y_pred, Y_test)
R2 = r2_score(Y_pred, Y_test)
print(f"Pytorch -  MSE: {MSE}; R2: {R2}")
# ~ plt.plot(model.losses)
# ~ plt.show()

