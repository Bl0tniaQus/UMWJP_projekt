import data_loader as dl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore') 
X_train, Y_train, X_test, Y_test = dl.getData()
X = pd.concat([X_train, X_test])
Y = pd.concat([Y_train, Y_test])
X = X.reset_index(drop = True)
Y = Y.reset_index(drop = True)
n = 20
fold = StratifiedKFold(n_splits = n)
folds = fold.split(X,Y)
sets = []
for i, (train_index, test_index) in enumerate(folds):
	X_train_new = X.iloc[train_index]
	X_test_new = X.iloc[test_index]
	Y_train_new = Y.iloc[train_index]
	Y_test_new = Y.iloc[test_index]
	X_train_new, Y_train_new = dl.DataAugmenter(X_train_new, Y_train_new,5,5,7)
	scaler = dl.trainScaler(X_train_new)
	train = pd.concat([X_train_new, Y_train_new], axis = 1)
	test = pd.concat([X_test_new, Y_test_new], axis = 1)
	X_train_new, Y_train_new = dl.prepareDataset(train, scaler)
	X_test_new, Y_test_new = dl.prepareDataset(test, scaler)
	set_ = {"X_train":X_train_new.copy(), "Y_train": Y_train_new.copy(), "X_test":X_test_new.copy(), "Y_test": Y_test_new.copy()}
	sets.append(set_)
for i in range(len(sets)):
	res = ""
	X_train = sets[i]["X_train"]
	X_test = sets[i]["X_test"]
	Y_train = sets[i]["Y_train"]
	Y_test = sets[i]["Y_test"]
	RF = RandomForestClassifier()
	RF.fit(X_train, Y_train)
	Y_pred = RF.predict(X_test)
	accuracy = accuracy_score(Y_pred, Y_test) * 100
	f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
	res = res + f"(Fold {i+1}) RF - acc: {accuracy:.2f}, f1: {f1:.2f}; "
	GNB = GaussianNB()
	GNB.fit(X_train, Y_train)
	Y_pred = GNB.predict(X_test)
	accuracy = accuracy_score(Y_pred, Y_test) * 100
	f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
	res = res + f"GNB - acc: {accuracy:.2f}, f1: {f1:.2f};"
	print(res)


