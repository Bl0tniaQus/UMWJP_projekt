import data_loader as dl
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt


train = dl.readData("./train_data.csv")
scaler = dl.trainStandardScaler(train)
X, Y = dl.prepareDataset(train, scaler)

tsne = TSNE(n_components=2,).fit_transform(X)
tsne_df = pd.DataFrame({"tsne_1": tsne[:,0], "tsne_2": tsne[:,1]})
tsne_df["Name"] = Y["Name"]
colors = []
names = np.unique(Y.values)
for i in range(len(names)):
	while True:
		c = (random.uniform(0,1), random.uniform(0,1), random.uniform(0,1))
		if c not in colors:
			colors.append(c)
			break

for i in range(len(names)):
	color = colors[i]
	dane_wykres = tsne_df[tsne_df["Name"] == names[i]]
	plt.scatter(x = dane_wykres[["tsne_1"]], y = dane_wykres[["tsne_2"]], c = [color for i in range(len(dane_wykres))])
plt.show()





tsne = TSNE(n_components=3,).fit_transform(X)
tsne_df = pd.DataFrame({"tsne_1": tsne[:,0], "tsne_2": tsne[:,1],  "tsne_3": tsne[:,2]})
tsne_df["Name"] = Y["Name"]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(len(names)):
	color = colors[i]
	dane_wykres = tsne_df[tsne_df["Name"] == names[i]]
	ax.scatter(xs = dane_wykres[["tsne_1"]], ys = dane_wykres[["tsne_2"]], zs = dane_wykres[["tsne_3"]], c = [color for i in range(len(dane_wykres))])
plt.show()
plt.clf()
