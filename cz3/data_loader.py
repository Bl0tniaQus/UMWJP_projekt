import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os

def getX(dane):
	dane = dane[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1", "Type 2", "Legendary"]]
	
	return X
def getY_type1(dane):
	return dane["Type 1"]
def encodeTypes(X):
	dane = pd.read_csv("./train_data.csv")
	t1_enc = LabelEncoder()
	t2_enc = LabelEncoder()
	t1_enc.fit(dane["Type 1"])
	t2_enc.fit(dane["Type 2"])
	X["Type 1"] = t1_enc.transform(X['Type 1'])
	X["Type 2"] = t2_enc.transform(X['Type 2'])
	return X
def encodeName(Y):
	dane = pd.read_csv("./train_data.csv")
	name_enc = LabelEncoder()
	name_enc.fit(dane["Name"])
	Y["Name"] = name_enc.transform(Y['Name'])
	return Y
def decodeName(Y):
	dane = pd.read_csv("./train_data.csv")
	name_enc = LabelEncoder()
	name_enc.fit(dane["Name"])
	Y["Name"] = name_enc.inverse_transform(Y['Name'])
	return Y
def getData():
	dane = pd.read_csv("./train_data.csv")
	dane["Legendary"].astype(int)
	X = dane.copy()[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1", "Type 2", "Legendary"]]
	Y = dane.copy()[["Name"]]
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y,stratify=Y, test_size=0.2, random_state = 0)
	X_train = X_train.reset_index(drop=True)
	X_test = X_test.reset_index(drop=True)
	Y_train = Y_train.reset_index(drop=True)
	Y_test = Y_test.reset_index(drop=True)
	return X_train, Y_train, X_test, Y_test

def DataAugmenter(X,Y, n_min,n_max,v):
	rows = []
	rows_y = []
	new_X = X.copy()
	new_Y = Y.copy()
	for i in range(X.shape[0]):
		if n_min!=n_max:
			n = random.randint(n_min,n_max-1)
		else:
			n = n_min
		row_y = {"Name": Y.iloc[i]["Name"]}
		for new in range(n):
			t1 = X.iloc[i]["Type 1"]
			t2 = X.iloc[i]["Type 2"]
			l = X.iloc[i]["Legendary"]
			atk = X.iloc[i]["Attack"] + random.randint(-v,v-1)
			if atk <= 0:
				atk = 1
			d = X.iloc[i]["Defense"] + random.randint(-v,v-1)
			if d <= 0:
				d = 1
			spdef = X.iloc[i]["Sp. Def"] + random.randint(-v,v-1)
			if spdef <= 0:
				spdef = 1
			spatk = X.iloc[i]["Sp. Atk"] + random.randint(-v,v-1)
			if spatk <= 0:
				spatk = 1
			spd = X.iloc[i]["Speed"] + random.randint(-v,v-1)
			if spd <= 0:
				spd = 1
			hp = X.iloc[i]["HP"] + random.randint(-v,v-1)
			if hp <= 0:
				hp = 1
			row = {}
			row["HP"] = hp
			row["Attack"] = atk
			row["Defense"] = d
			row["Sp. Atk"] = spatk
			row["Sp. Def"] = spdef
			row["Speed"] = spd
			row["Type 1"] = t1
			row["Type 2"] = t2
			row["Legendary"] = l
			rows.append(row)
			rows_y.append(row_y)
	new_X = pd.concat([new_X, pd.DataFrame(rows)])
	new_X = new_X.reset_index(drop = True)
	new_Y = pd.concat([new_Y, pd.DataFrame(rows_y)])
	new_Y = new_Y.reset_index(drop = True)
	return new_X, new_Y

def prepareDataset(filename):
	dane = pd.read_csv(filename)
	dane["Legendary"].astype(int)
	X = dane.copy()[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1", "Type 2", "Legendary"]]
	Y = dane.copy()[["Name"]]
	X = encodeTypes(X)
	Y = encodeName(Y)
	x = X.values
	min_max_scaler = MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	X = pd.DataFrame(x_scaled)
	X.columns = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1", "Type 2", "Legendary"]
	return X, Y

	
	

