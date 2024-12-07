import data_loader as dl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore') 

originalTest = dl.readData("./augmented_datasets/test.csv")




datasets = ["./augmented_datasets/train.csv", "./augmented_datasets/augment_train1.csv", "./augmented_datasets/augment_train2.csv"]

for dataset in datasets:
	print(dataset)
	train = dl.readData(dataset)
	test = originalTest.copy()
	scaler = dl.trainScaler(train)
	X_train, Y_train = dl.prepareDataset(train, scaler)
	X_test, Y_test = dl.prepareDataset(test, scaler)
	
	RF = RandomForestClassifier()
	RF.fit(X_train, Y_train)
	Y_pred = RF.predict(X_test)
	accuracy = accuracy_score(Y_pred, Y_test) * 100
	precision = precision_score(Y_pred, Y_test, average="weighted") * 100
	recall = recall_score(Y_pred, Y_test, average="weighted") * 100
	f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
	print(f"RF: acc: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}")
	
	GNB = GaussianNB()
	GNB.fit(X_train, Y_train)
	Y_pred = GNB.predict(X_test)
	accuracy = accuracy_score(Y_pred, Y_test) * 100
	precision = precision_score(Y_pred, Y_test, average="weighted") * 100
	recall = recall_score(Y_pred, Y_test, average="weighted") * 100
	f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
	print(f"GNB: acc: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}")


