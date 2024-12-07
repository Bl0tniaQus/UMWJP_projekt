import data_loader as dl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore') 

originalTest = dl.readData("./augmented_datasets/test.csv")

train = dl.readData("./augmented_datasets/augment_train1.csv")
scaler = dl.trainScaler(train)
X_train, Y_train = dl.prepareDataset(train, scaler)
test = originalTest.copy()
X_test, Y_test = dl.prepareDataset(test, scaler)
models = [RandomForestClassifier(), GaussianNB()]
names = ["RF", "GNB"]

for i in range(len(models)):
	model = models[i]
	model.fit(X_train, Y_train)
	Y_pred = model.predict(X_test)
	cm = pd.DataFrame(confusion_matrix(Y_test, Y_pred))
	
	labels = pd.DataFrame(model.classes_)
	labels.columns = ["Name"]
	labels = dl.decodeName(labels)
	cm.columns = labels["Name"].values
	cm.insert(0, " ", labels["Name"].values)
	cm.to_csv(f"./{names[i]}.csv", index = False)
	
	


