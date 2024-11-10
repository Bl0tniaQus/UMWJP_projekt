import data_loader
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from timeit import default_timer as timer

targets = ["Type 1", "Name"]
test_sizes = [0.1,0.25,0.5,0.75]

for t in targets:
	for ts in test_sizes:
		print(f"\n{t.upper()}, TEST {ts*100}%\n")
		X_train, Y_train, X_test, Y_test = data_loader.getData(t, ts)
		model = DecisionTreeClassifier()
		train_start = timer()
		model.fit(X_train, Y_train)
		train_end = timer()
		train_time = train_end - train_start
		test_start = timer()
		Y_pred = model.predict(X_test)
		test_end = timer()
		test_time = test_end - test_start
		accuracy = balanced_accuracy_score(Y_pred, Y_test)
		precision = precision_score(Y_pred, Y_test, average="weighted")
		recall = recall_score(Y_pred, Y_test, average="weighted")
		f1 = f1_score(Y_pred, Y_test, average="weighted")
		print(f"Czas uczenia: {train_time}")
		print(f"Czas sprawdzania: {test_time}")
		print(f"Dokładność: {accuracy}")
		print(f"Precyzja: {precision}")
		print(f"Czułość: {recall}")
		print(f"Wskaźnik f1: {f1}")
		print("Nierozpoznane: " + str(set(Y_test).difference(set(Y_pred))))
