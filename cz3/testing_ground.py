import data_loader as dl
import pandas as pd
X_train, Y_train, X_test, Y_test = dl.getData()

print(Y_train)
Y_train = dl.encodeName(Y_train)
print(Y_train)
