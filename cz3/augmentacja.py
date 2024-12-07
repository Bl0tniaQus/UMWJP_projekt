import data_loader as dl
import pandas as pd
X_train, Y_train, X_test, Y_test = dl.getData()

new_data = pd.concat([X_train, Y_train], axis = 1).sample(frac = 1)
new_data.to_csv("./augmented_datasets/train.csv", index = False)

augmented_X, augmented_Y = dl.DataAugmenter(X_train, Y_train, 5, 5, 7)
new_data = pd.concat([augmented_X, augmented_Y], axis = 1).sample(frac = 1)
new_data.to_csv("./augmented_datasets/augment_train1.csv", index = False)

augmented_X, augmented_Y = dl.DataAugmenter(X_train, Y_train, 2, 10, 15)
new_data = pd.concat([augmented_X, augmented_Y], axis = 1).sample(frac = 1)
new_data.to_csv("./augmented_datasets/augment_train2.csv", index = False)

new_data = pd.concat([X_test, Y_test], axis = 1).sample(frac = 1)
new_data.to_csv("./augmented_datasets/test.csv", index = False)
