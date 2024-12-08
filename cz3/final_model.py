import data_loader as dl
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore') 

test = dl.readData("./test_data.csv")
train = dl.readData("./augmented_datasets/augment_train1.csv")

scaler = dl.trainScaler(train)
X_train, Y_train = dl.prepareDataset(train, scaler)
X_test, _ = dl.prepareDataset(test, scaler)


GNB = GaussianNB()
GNB.fit(X_train, Y_train)
Y_pred = GNB.predict(X_test)

pred_df = pd.DataFrame(Y_pred)
pred_df.columns = ["Name"]
pred_df = dl.decodeName(pred_df)

pred_df.to_csv("./test_predicted.csv", index = False)


