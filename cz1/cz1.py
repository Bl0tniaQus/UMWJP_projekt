import pandas as pd

dane = pd.read_csv("../train_data.csv")
dane.info()
print("-------")
dane.describe()
