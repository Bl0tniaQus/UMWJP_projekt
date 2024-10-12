import pandas as pd
import os

if os.path.exists("analiza_result.txt"):
  os.remove("analiza_result.txt")

plik = open("analiza_result.txt", "a")

dane = pd.read_csv("./przetwarzanie_zamienbledne.csv")
#analiza statystyk według typu
plik.write("** ANALIZA STATYSTYK KAŻDEGO TYPU **")
typy = dane["Type 1"].unique()
for typ in typy:
	plik.write("\n\n"+typ.upper())
	dane_typ = dane[(dane["Type 1"] == typ) | (dane["Type 2"] == typ)]
	plik.write("\n\n"+dane_typ.describe().to_string())

#zliczanie liczby Pokemon'ów każdego gatunku
plik.write("\n\n** ZLICZANIE LICZEBNOŚCI GATUNKÓW **")
plik.write("\n\n"+dane.groupby(["Name"]).size().to_string())
#analiza statystyk według gatunku
plik.write("**ANALIZA STATYSTYK KAŻDEGO GATUNKU**")
gatunki = dane.sort_values(by=["Name"])["Name"].unique()
for gatunek in gatunki:
	plik.write("\n\n"+gatunek.upper())
	dane_gatunek = dane[dane["Name"] == gatunek]
	plik.write("\n\n"+dane_gatunek.describe().to_string())

#zliczanie błędnych danych (statystyk mniejszych lub równych 0)
dane_z_bledami = pd.read_csv("../train_data.csv")
bledy = len(dane_z_bledami[(dane_z_bledami["Total"] <= 0) | (dane_z_bledami["Attack"] <= 0) | (dane_z_bledami["Defense"] <= 0) | (dane_z_bledami["Sp. Atk"] <= 0) | (dane_z_bledami["Sp. Def"] <= 0) | (dane_z_bledami["HP"] <= 0) | (dane_z_bledami["Speed"] <= 0)])
plik.write("\n\n** LICZBA REKORDÓW Z BŁĘDAMI: " + str(bledy)+" **")
