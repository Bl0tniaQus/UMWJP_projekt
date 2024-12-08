import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os

def encodeTypes(X):
	dane = pd.read_csv("./train_data.csv")
	new_X = X.copy()
	t1_enc = LabelEncoder()
	t2_enc = LabelEncoder()
	t1_enc.fit(dane["Type 1"])
	t2_enc.fit(dane["Type 2"])
	new_X["Type 1"] = t1_enc.transform(X['Type 1'])
	new_X["Type 2"] = t2_enc.transform(X['Type 2'])
	return new_X
def encodeName(Y):
	dane = pd.read_csv("./train_data.csv")
	new_Y = Y.copy()
	name_enc = LabelEncoder()
	name_enc.fit(dane["Name"])
	new_Y["Name"] = name_enc.transform(Y['Name'])
	return new_Y
def decodeName(Y):
	dane = pd.read_csv("./train_data.csv")
	name_enc = LabelEncoder()
	name_enc.fit(dane["Name"])
	new_Y = Y.copy()
	new_Y["Name"] = name_enc.inverse_transform(Y['Name'])
	return new_Y
def getData():
	dane_oryginalne = pd.read_csv("./train_data.csv")
	dane = dane_oryginalne.copy()
	dane["Legendary"].astype(int)
	dane.loc[dane["Attack"] <= 0, "Attack"] = 1
	dane.loc[dane["HP"] <= 0, "HP"] = 1
	dane.loc[dane["Total"] <= 0, "Total"] = 1
	dane.loc[dane["Defense"] <= 0, "Defense"] = 1
	dane.loc[dane["Speed"] <= 0, "Speed"] = 1
	dane.loc[dane["Sp. Atk"] <= 0, "Sp. Atk"] = 1
	dane.loc[dane["Sp. Def"] <= 0, "Sp. Def"] = 1
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

def trainScaler(dane):
	X = dane.copy()[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1", "Type 2", "Legendary"]]
	X = encodeTypes(X)
	scaler = MinMaxScaler()
	scaler.fit(X.values)
	return scaler

def readData(filename):
	dane = pd.read_csv(filename)
	dane["Legendary"].astype(int)
	return dane
def prepareDataset(dane, scaler):
	X = dane.copy()[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1", "Type 2", "Legendary"]]
	if "Name" in dane.columns:
		Y = dane.copy()[["Name"]]
		Y = encodeName(Y)
	else:
		Y = pd.DataFrame()
	X = encodeTypes(X)
	x_scaled = scaler.transform(X.values)
	X = pd.DataFrame(x_scaled)
	X.columns = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Type 1", "Type 2", "Legendary"]
	return X, Y

def predictionToDict(pred):
	Y = pd.DataFrame(pred)
	Y.columns = ["Name"]
	return Y

	
	

