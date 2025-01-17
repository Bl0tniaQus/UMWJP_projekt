import os
import cv2
import pickle
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch
from PIL import Image
def getLabels():
	return [label.name for label in os.scandir("data")]
	

labels = getLabels()
images = []
Y = []

for label in labels:
	species_count = 0
	for img in os.scandir(os.path.join("data", label)):
		try:
			image = Image.open(os.path.join("data", label, img.name))
			images.append(os.path.join("data", label, img.name))
			species_count = species_count + 1
		except:
			pass
	Y = Y + [label for _ in range(species_count)]

print(len(Y))
print(len(images))

X_train, X_rest, Y_train, Y_rest = train_test_split(images, Y,stratify=Y, test_size=0.25, random_state = 0)
X_test, X_val, Y_test, Y_val = train_test_split(X_rest, Y_rest,stratify=Y_rest, test_size=0.5, random_state = 0)

print(len(X_train))
print(len(X_test))
print(len(X_val))

labels_file = open("labels_pickle", "wb")
X_train_file = open("X_train_pickle", "wb")
Y_train_file = open("Y_train_pickle", "wb")
X_test_file = open("X_test_pickle", "wb")
Y_test_file = open("Y_test_pickle", "wb")
X_val_file = open("X_val_pickle", "wb")
Y_val_file = open("Y_val_pickle", "wb")

pickle.dump(X_train, X_train_file)
pickle.dump(Y_train, Y_train_file)
pickle.dump(X_test, X_test_file)
pickle.dump(Y_test, Y_test_file)
pickle.dump(X_val, X_val_file)
pickle.dump(Y_val, Y_val_file)
pickle.dump(Y, labels_file)
labels_file.close()
X_train_file.close()
Y_train_file.close()
X_test_file.close()
Y_test_file.close()
X_val_file.close()
Y_val_file.close()
	
