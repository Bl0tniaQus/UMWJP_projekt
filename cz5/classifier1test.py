import pickle
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import numpy as np
from classifier1 import classifier, Train
import warnings
from PIL import Image
from sklearn.metrics import confusion_matrix
import pandas as pd
warnings.filterwarnings('ignore')
# ~ Train()
x_file = open('X_test_pickle', 'rb')
y_file = open('Y_test_pickle', 'rb')
model_file = open('model1_pickle', 'rb')
labels_file = open('labels_pickle', 'rb')
X_test = pickle.load(x_file)
Y_test = pickle.load(y_file)
model = pickle.load(model_file)
labels = pickle.load(labels_file)
x_file.close()
y_file.close()
model_file.close()
labels_file.close()



encoder = LabelEncoder()
encoder.fit(labels)
Y_test = encoder.transform(Y_test)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
y = Y_test
batch_size = 83
for i in range(0, len(y) // batch_size):
	x = []
	for path in X_test[slice(i*batch_size, i*batch_size + batch_size)]:
		image = Image.open(path).convert("RGB")
		image = transform(image)
		x.append(image)
	pred = model.predict(torch.stack(x))
	if i == 0:
		Y_pred = pred
	else:
		Y_pred = np.hstack((Y_pred, pred))
print(Y_pred)
print(len(Y_pred))
print(len(y))

#Y_pred = model.predict(x)
accuracy = accuracy_score(y, Y_pred)
f1 = f1_score(Y_test, Y_pred, average="weighted")
recall = recall_score(Y_test, Y_pred, average="weighted")
precision = precision_score(Y_test, Y_pred, average="weighted")
print(f"Accuracy: {(accuracy * 100):.3f}")
print(f"f1: {(f1 * 100):.3f}")
print(f"Recall: {(recall * 100):.3f}")
print(f"Precision: {(precision * 100):.3f}")

cm = pd.DataFrame(confusion_matrix(Y_test, Y_pred))

labels_frame = pd.DataFrame(encoder.inverse_transform(np.unique(y)))
labels_frame.columns = ["Name"]
cm.columns = labels_frame["Name"].values
cm.insert(0, " ", labels_frame["Name"].values)
cm.to_csv(f"./cm.csv", index = False)
plt.subplot(1, 2, 1)
plt.plot(model.losses)
plt.title("Wykres średniego błędu serii w czasie [dane uczące]")
plt.xlabel("Czas [epoka]")
plt.ylabel("Cross Entropy Loss")
plt.xticks(range(0,20), labels = range(1,21))
plt.subplot(1, 2, 2)
plt.plot(model.losses_val)
plt.title("Wykres średniego błędu serii w czasie [dane walidacyjne]")
plt.xlabel("Czas [epoka]")
plt.ylabel("Cross Entropy Loss")
plt.xticks(range(0,20), labels = range(1,21))
plt.show()
plt.plot(model.accuracies)
plt.title("Zależność skuteczności klasyfikacji danych walidacyjnych od czasu")
plt.xlabel("Czas [epoka]")
plt.ylabel("Skuteczność [%]")
plt.xticks(range(0,20), labels = range(1,21))
plt.show()
