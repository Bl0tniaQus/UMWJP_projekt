import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

def getX(dane):
	dane = dane[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
	x = dane.values
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	X = pd.DataFrame(x_scaled)
	X.columns = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
	return X
def getY_type1(dane):
	return dane["Type 1"]
def getY_name(dane):
	return dane["Name"]
def getData(target, test_size):
	dane = pd.read_csv("./dane.csv")
	
	#zamiana danych kategorycznych na liczby, niepotrzebne w zadaniu
	#dane['Type 1'] = dane['Type 1'].astype('category')
	#dane['Type 1'] = dane['Type 1'].cat.codes
	#dane['Type 2'] = dane['Type 2'].astype('category')
	#dane['Type 2'] = dane['Type 2'].cat.codes
	#dane['Legendary'] = dane['Legendary'].astype('category')
	#dane['Legendary'] = dane['Legendary'].cat.codes
	X = getX(dane)
	if target=="Name":
		Y = getY_name(dane)
	elif target=="Type 1":
		Y = getY_type1(dane)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y,stratify=Y, test_size=test_size, random_state = 0)
	return X_train, Y_train, X_test, Y_test
	
