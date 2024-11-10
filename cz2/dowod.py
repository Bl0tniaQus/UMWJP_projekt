import pandas as pd
import os

dane_z_bledami = pd.read_csv("../train_data.csv")
bledy = dane_z_bledami[(dane_z_bledami["Total"] <= 0) | (dane_z_bledami["Attack"] <= 0) | (dane_z_bledami["Defense"] <= 0) | (dane_z_bledami["Sp. Atk"] <= 0) | (dane_z_bledami["Sp. Def"] <= 0) | (dane_z_bledami["HP"] <= 0) | (dane_z_bledami["Speed"] <= 0)]

print(bledy)

