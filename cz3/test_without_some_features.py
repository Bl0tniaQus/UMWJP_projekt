import data_loader as dl
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore') 

originalTest = dl.readData("./augmented_datasets/test.csv")

train = dl.readData("./augmented_datasets/augment_train1.csv")
test = originalTest.copy()
scaler = dl.trainScaler(train)
X_train, Y_train = dl.prepareDataset(train, scaler)
X_test, Y_test = dl.prepareDataset(test, scaler)


X_train1 = X_train.copy()[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1", "Type 2"]]
X_test1 = X_test.copy()[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1", "Type 2"]]
X_train2 = X_train.copy()[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1"]]
X_test2 = X_test.copy()[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1"]]
	
RF = RandomForestClassifier()
train_start = timer()
RF.fit(X_train1, Y_train)
train_end = timer()
test_start = timer()
Y_pred = RF.predict(X_test1)
test_end = timer()
accuracy = accuracy_score(Y_pred, Y_test) * 100
precision = precision_score(Y_pred, Y_test, average="weighted") * 100
recall = recall_score(Y_pred, Y_test, average="weighted") * 100
f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
print(f"RF (without legendary): acc: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}, training_time: {(train_end - train_start):.3f}, testing_time: {(test_end - test_start):.3f}")

GNB = GaussianNB()
train_start = timer()
GNB.fit(X_train1, Y_train)
train_end = timer()
test_start = timer()
Y_pred = GNB.predict(X_test1)
test_end = timer()
accuracy = accuracy_score(Y_pred, Y_test) * 100
precision = precision_score(Y_pred, Y_test, average="weighted") * 100
recall = recall_score(Y_pred, Y_test, average="weighted") * 100
f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
print(f"GNB (without legendary): acc: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}, training_time: {(train_end - train_start):.3f}, testing_time: {(test_end - test_start):.3f}")

RF = RandomForestClassifier()
train_start = timer()
RF.fit(X_train2, Y_train)
train_end = timer()
test_start = timer()
Y_pred = RF.predict(X_test2)
test_end = timer()
accuracy = accuracy_score(Y_pred, Y_test) * 100
precision = precision_score(Y_pred, Y_test, average="weighted") * 100
recall = recall_score(Y_pred, Y_test, average="weighted") * 100
f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
print(f"RF (without legendary and t2): acc: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}, training_time: {(train_end - train_start):.3f}, testing_time: {(test_end - test_start):.3f}")

GNB = GaussianNB()
train_start = timer()
GNB.fit(X_train2, Y_train)
train_end = timer()
test_start = timer()
Y_pred = GNB.predict(X_test2)
test_end = timer()
accuracy = accuracy_score(Y_pred, Y_test) * 100
precision = precision_score(Y_pred, Y_test, average="weighted") * 100
recall = recall_score(Y_pred, Y_test, average="weighted") * 100
f1 = f1_score(Y_pred, Y_test, average="weighted") * 100
print(f"GNB (without legendary and t2): acc: {accuracy:.2f}, precision: {precision:.2f}, recall: {recall:.2f}, f1: {f1:.2f}, training_time: {(train_end - train_start):.3f}, testing_time: {(test_end - test_start):.3f}")


