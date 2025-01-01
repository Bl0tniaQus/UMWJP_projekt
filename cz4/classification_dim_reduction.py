import data_loader as dl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from timeit import default_timer as timer
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore') 

originalTest = dl.readData("./test_data.csv")
train = dl.readData("./train_data.csv")
scaler = dl.trainStandardScaler(train)
X_train_original, Y_train = dl.prepareDataset(train, scaler)
test = originalTest.copy()
X_test_original, Y_test = dl.prepareDataset(test, scaler)
models = [RandomForestClassifier(), SVC(), GaussianNB()]
names = ["RF", "SVM", "GNB"]
results_dicts = []
dic = {}
for i in range(len(models)):
	model = models[i]
	train_start = timer()
	model.fit(X_train_original, Y_train)
	train_end = timer()
	test_start = timer()
	Y_pred = model.predict(X_test_original)
	test_end = timer()
	accuracy = accuracy_score(Y_pred, Y_test) * 100
	f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
	print(f"{names[i]}: acc: {accuracy:.2f}, f1: {f1:.2f}, training_time: {(train_end - train_start):.3f}, testing_time: {(test_end - test_start):.3f}")
	dic.update({"method": "", f"{names[i]}_ACC" : accuracy, f"{names[i]}_f1" : f1 })
dic["method"] = f"No reduction"
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
		Y_pred = model.predict(X_test)
		test_end = timer()
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		print(f"{names[i]}: acc: {accuracy:.2f}, f1: {f1:.2f}, training_time: {(train_end - train_start):.3f}, testing_time: {(test_end - test_start):.3f}")
		dic.update({"method": "", f"{names[i]}_ACC" : accuracy, f"{names[i]}_f1" : f1 })
	dic["method"] = f"PCA {dim}"
	results_dicts.append(dic)
	

for dim in range(1, len(X_train_original.columns)):
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
		Y_pred = model.predict(X_test)
		test_end = timer()
		accuracy = accuracy_score(Y_pred, Y_test) * 100
		f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
		print(f"{names[i]}: acc: {accuracy:.2f}, f1: {f1:.2f}, training_time: {(train_end - train_start):.3f}, testing_time: {(test_end - test_start):.3f}")
		dic.update({"method": "", f"{names[i]}_ACC" : accuracy, f"{names[i]}_f1" : f1 })
	dic["method"] = f"GRP {dim}"
	results_dicts.append(dic)
	
df = pd.DataFrame(results_dicts)
df.to_csv("classification.csv", index = False)




