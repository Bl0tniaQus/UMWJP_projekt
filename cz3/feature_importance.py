import data_loader as dl
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore') 
originalTest = dl.readData("./augmented_datasets/test.csv")
train = dl.readData("./augmented_datasets/augment_train1.csv")
scaler = dl.trainScaler(train)
X_train, Y_train = dl.prepareDataset(train, scaler)
test = originalTest.copy()
X_test, Y_test = dl.prepareDataset(test, scaler)

Features = X_train.columns

model = RandomForestClassifier()
model.fit(X_train, Y_train)
FI = model.feature_importances_
print("RANDOM FOREST")
print("BUILT-IN")
for i in range(len(FI)):
	print(f"{Features[i]}: {FI[i]:.3f}")
r = permutation_importance(model, X_train, Y_train,n_repeats=30, random_state=0)
means = r["importances_mean"]
print("PERMUTATION IMPORTANCE")
for i in range(len(r["importances_mean"])):
	print(f"{Features[i]}: {means[i]:.3f}")

print("GAUSSIAN NAIVE BAYES")
model = GaussianNB()
model.fit(X_train, Y_train)
r = permutation_importance(model, X_train, Y_train,n_repeats=30, random_state=0)
means = r["importances_mean"]
print("PERMUTATION IMPORTANCE")
for i in range(len(r["importances_mean"])):
	print(f"{Features[i]}: {means[i]:.3f}")
	
	


