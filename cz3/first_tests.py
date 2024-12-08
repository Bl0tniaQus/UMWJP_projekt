import data_loader as dl
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore') 

originalTest = dl.readData("./augmented_datasets/test.csv")
train = dl.readData("./augmented_datasets/train.csv")
scaler = dl.trainScaler(train)
X_train, Y_train = dl.prepareDataset(train, scaler)
test = originalTest.copy()
X_test, Y_test = dl.prepareDataset(test, scaler)
models = [DecisionTreeClassifier(), KNeighborsClassifier(), LogisticRegression(), RandomForestClassifier(), GradientBoostingClassifier(), GaussianNB(), SVC()]
names = ["DT", "KNN", "LR", "RF", "GBC", "GNB", "SVM"]
for i in range(len(models)):
	model = models[i]
	train_start = timer()
	model.fit(X_train, Y_train)
	train_end = timer()
	test_start = timer()
	Y_pred = model.predict(X_test)
	test_end = timer()
	accuracy = accuracy_score(Y_pred, Y_test) * 100
	precision = precision_score(Y_pred, Y_test, average="weighted") * 100
	recall = recall_score(Y_pred, Y_test, average="weighted") * 100
	f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
	print(f"{names[i]}: acc: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}, training_time: {(train_end - train_start):.3f}, testing_time: {(test_end - test_start):.3f}")


