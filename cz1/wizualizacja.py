import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE

if not os.path.exists("./wykresy"):
  os.mkdir("wykresy")

dane = pd.read_csv("./przetwarzanie_zamienbledne.csv")

#wykresy rozkładu prawdopodobieństwa każdej z bojowych statystyk
stats = ["HP","Attack","Defense","Sp. Atk", "Sp. Def", "Speed"]
for stat in stats:
	dane[stat].plot.kde()
	plt.xlabel("Wartość statystyki \"" + stat + "\"")
	plt.ylabel("Gęstość prawdopodobieństwa")
	plt.title("Gęstość prawdopodobieństwa statystyki \""+ stat+"\"")
	plt.savefig("./wykresy/"+stat+".jpg")
	plt.clf()

#wykres zależności ataku i specjalnego ataku od obrony
dane_wykres = dane.sort_values(by=["Defense"])[["Attack","Defense","Sp. Atk"]].groupby(dane["Defense"]).median()
plt.plot(dane_wykres["Defense"],dane_wykres["Sp. Atk"])
plt.plot(dane_wykres["Defense"],dane_wykres["Attack"])
plt.xlabel("Wartość obrony")
plt.ylabel("Wartość ataku")
plt.legend(['atak fizyczny', 'atak specjalny'])
plt.title("Wykres zależności ataku od fizycznej obrony")
plt.savefig("./wykresy/atk_od_obrony.jpg")
plt.clf()

#wykres zależności specjalnego ataku od fizycznego ataku
dane_wykres = dane.sort_values(by=["Attack"])[["Sp. Atk","Attack"]].groupby(dane["Attack"]).median()
plt.plot(dane_wykres["Attack"],dane_wykres["Sp. Atk"])
plt.xlabel("Atak fizyczny")
plt.ylabel("Atak specjalny")
plt.title("Wykres zależności ataku fizycznego od ataku specjalnego")
plt.savefig("./wykresy/atk_od_spatk.jpg")
plt.clf()

colors = {"Fire" : "orangered", "Water" : "aqua", "Grass" : "green", "Normal" : "lightsalmon", "Electric" : "yellow", "Bug" : "lightgreen", "Fighting" : "red", "Ground" : "khaki", "Rock" : "goldenrod", "Steel" : "silver", "Poison" : "purple", "Psychic" : "palevioletred", "Ghost" : "rebeccapurple", "Dark" : "darkslategray", "Ice" : "cyan", "Dragon" : "dodgerblue", "Fairy" : "violet", "Flying" : "azure"}

#wykres kołowy rozkładu głównych typów
dane_wykres = dane[["Type 1", "#"]].groupby(dane["Type 1"]).count()
plt.pie(dane_wykres["#"], labels=dane_wykres.index, autopct="%1.1f%%", colors=[colors[key] for key in dane_wykres.index])
plt.gcf().set_size_inches(8, 8)
plt.title("Rozkład głównych typów")
plt.savefig("./wykresy/rozklad_glownych_typow.jpg")
plt.clf()
#wykres kołowy rozkładu drugorzędnych typów
dane_wykres = dane[["Type 2", "#"]].groupby(dane["Type 2"]).count()
plt.pie(dane_wykres["#"], labels=dane_wykres.index, autopct="%1.1f%%", colors=[colors[key] for key in dane_wykres.index])
plt.gcf().set_size_inches(8, 8)
plt.title("Rozkład drugorzędnych typów")
plt.savefig("./wykresy/rozklad_drugorzednych_typow.jpg")
plt.clf()
#wykres kołowy rozkładu wszystkich typów
typy1 = dane[["Type 1", "#"]]
typy2 = dane[["Type 2", "#"]]
typy2 = typy2[~(typy2["Type 2"].isna())]
typy2["Type 1"] = typy2["Type 2"]
typy2.drop(labels=["Type 2"],axis=1)
typy = pd.concat([typy1,typy2])
typy = typy.groupby(typy["Type 1"]).count()
plt.pie(typy["#"], labels=typy.index, autopct="%1.1f%%", colors=[colors[key] for key in typy.index])
plt.gcf().set_size_inches(8,8)
plt.title("Rozkład typów z uwzględnieniem drugorzędnych")
plt.savefig("./wykresy/rozklad_typow.jpg")
plt.clf()

#wykres słupkowy liczbie pokemonów o jednym lub dwóch typach
podwojne_typy = dane.dropna()
pojedyncze_typy = dane[dane["Type 2"].isna()]
n_podwojnych = len(podwojne_typy)
n_pojedynczych = len(pojedyncze_typy)

plt.bar(["Pojedyncze typy", "Podwójne typy"], [n_pojedynczych, n_podwojnych])
plt.title("Liczba Pokemon'ów o pojedynczym i podwójnym typie")
plt.savefig("./wykresy/liczba_pkmn_o_liczbie_typow.jpg")
plt.clf()

#wykres TSNE dla statystyk bojowych przyjmując za etykiety typ główny
tsne = TSNE(n_components=2,).fit_transform(dane[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]])
tsne_df = pd.DataFrame({'tsne_1': tsne[:,0], 'tsne_2': tsne[:,1]})
tsne_df['type'] = dane[["Type 1"]]
types = tsne_df.type.unique()
for typ in types:
	color = colors[typ]
	dane_wykres = tsne_df[tsne_df["type"] == typ]
	plt.scatter(x = dane_wykres[["tsne_1"]], y = dane_wykres[["tsne_2"]], c = color)
plt.legend(types)
plt.title("Wykres TSNE dla statystyk bojowych z etykietowaniem na podstawie głównego typu")
plt.savefig("./wykresy/tsne.jpg")
plt.clf()

