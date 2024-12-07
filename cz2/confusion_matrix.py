import data_loader
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

targets = ["Type 1", "Name"]
for t in targets:
	X_train, Y_train, X_test, Y_test = data_loader.getData(t, 0.25)
	model = SVC()
	model.fit(X_train, Y_train)
	Y_pred = model.predict(X_test)
	cm = confusion_matrix(Y_test, Y_pred)
	if (t=="Type 1"):
		disp = ConfusionMatrixDisplay(cm, display_labels = model.classes_)
	else:
		disp = ConfusionMatrixDisplay(cm)
	disp.plot()
	plt.show()
	

