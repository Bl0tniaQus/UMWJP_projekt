import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

predicted = pd.read_csv("./test_predicted.csv")

real = pd.read_csv("./test_predicted.csv")

f1 = f1_score(real["Name"].values,predicted["Name"].values, average="weighted")
acc = accuracy_score(real["Name"].values,predicted["Name"].values)

print(f"acc: {acc:.4f}, f1: {f1:.4f}")

