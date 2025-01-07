import data_loader as dl
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from timeit import default_timer as timer
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
warnings.filterwarnings('ignore') 

originalTest = dl.readData("./test_data.csv")
train = dl.readData("./train_data.csv")
scaler = dl.trainStandardScalerHPRegression(train)
X_train_original, Y_train = dl.prepareDatasetHPRegression(train, scaler)
test = originalTest.copy()
X_test_original, Y_test = dl.prepareDatasetHPRegression(test, scaler)
models = [LinearRegression(), Lasso(), Ridge()]
names = ["LinR", "Lasso", "Ridge"]
results_dicts = []

dic = {}
for i in range(len(models)):
	model = models[i]
	train_start = timer()
	model.fit(X_train_original, Y_train)
	train_end = timer()
	test_start = timer()
	Y_pred = np.round(model.predict(X_test_original))
	test_end = timer()
	mse = mean_squared_error(Y_test, Y_pred)
	r2 = r2_score(Y_test, Y_pred)
	print(f"{names[i]}: MSE: {mse}; R2: {r2}; training_time: {(train_end - train_start):.3f}; testing_time: {(test_end - test_start):.3f}")
	dic.update({"method": "", f"{names[i]}_MSE" : mse, f"{names[i]}_R2" : r2 })
dic["method"] = "No reduction"
results_dicts.append(dic)

for dim in range(1, len(X_train_original.columns)):
	X_train = X_train_original.copy()
	X_test = X_test_original.copy()
	pca = PCA(dim)
	X_train = pca.fit_transform(X_train)
	X_test = pca.transform(X_test)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	dic = {}
	for i in range(len(models)):
		model = models[i]
		train_start = timer()
		model.fit(X_train, Y_train)
		train_end = timer()
		test_start = timer()
		Y_pred = np.round(model.predict(X_test))
		test_end = timer()
		mse = mean_squared_error(Y_test, Y_pred)
		r2 = r2_score(Y_test, Y_pred)
		print(f"{names[i]}: MSE: {mse}; R2: {r2}; training_time: {(train_end - train_start):.3f}; testing_time: {(test_end - test_start):.3f}")
		dic.update({"method": "", f"{names[i]}_MSE" : mse, f"{names[i]}_R2" : r2 })
	dic["method"] = f"PCA {dim}"
	results_dicts.append(dic)
	

for dim in range(1, len(X_train_original.columns)):
	print(f"Gaussian Random Projection ({dim})")
	X_train = X_train_original.copy()
	X_test = X_test_original.copy()
	rp = GaussianRandomProjection(dim, random_state = 0)
	X_train = rp.fit_transform(X_train)
	X_test = rp.transform(X_test)
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)
	dic = {}
	for i in range(len(models)):
		model = models[i]
		train_start = timer()
		model.fit(X_train, Y_train)
		train_end = timer()
		test_start = timer()
		Y_pred = np.round(model.predict(X_test))
		test_end = timer()
		mse = mean_squared_error(Y_test, Y_pred)
		r2 = r2_score(Y_test, Y_pred)
		print(f"{names[i]}: MSE: {mse}; R2: {r2}; training_time: {(train_end - train_start):.3f}; testing_time: {(test_end - test_start):.3f}")
		dic.update({"method": "", f"{names[i]}_MSE" : mse, f"{names[i]}_R2" : r2 })
	dic["method"] = f"GRP {dim}"
	results_dicts.append(dic)
	
df = pd.DataFrame(results_dicts)
df.to_csv("hp_regression.csv", index = False)



