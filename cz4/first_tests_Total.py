import data_loader as dl
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore') 

originalTest = dl.readData("./test_data.csv")
train = dl.readData("./train_data.csv")
scaler = dl.trainStandardScaler(train)
X_train, Y_train = dl.prepareDatasetTotalRegression(train, scaler)
test = originalTest.copy()
X_test, Y_test = dl.prepareDatasetTotalRegression(test, scaler)
models = [LinearRegression(), Lasso(), Ridge()]
names = ["LinR", "Lasso", "Ridge"]
for i in range(len(models)):
	model = models[i]
	train_start = timer()
	model.fit(X_train, Y_train)
	train_end = timer()
	test_start = timer()
	Y_pred = model.predict(X_test)
	test_end = timer()
	mse = mean_squared_error(Y_test, Y_pred)
	r2 = r2_score(Y_test, Y_pred)
	print(f"{names[i]}: MSE: {mse}; R2: {r2}; training_time: {(train_end - train_start):.3f}; testing_time: {(test_end - test_start):.3f}")


