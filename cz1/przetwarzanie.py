import pandas as pd
import numpy as np
import os

if os.path.exists("przetwarzanie_result.txt"):
  os.remove("przetwarzanie_result.txt")
if os.path.exists("przetwarzanie_usunpuste.csv"):
  os.remove("przetwarzanie_usunpuste.csv")
if os.path.exists("przetwarzanie_usunduplikaty.csv"):
  os.remove("przetwarzanie_usunduplikaty.csv")
if os.path.exists("przetwarzanie_zamienbledne.csv"):
  os.remove("przetwarzanie_zamienbledne.csv")
if os.path.exists("przetwarzanie_sortowanie.csv"):
  os.remove("przetwarzanie_sortowanie.csv")
if os.path.exists("przetwarzanie_sortowanie.csv"):
  os.remove("przetwarzanie_sortowanie.csv")
if os.path.exists("gen1_data.csv"):
  os.remove("gen1_data.csv")


plik = open("przetwarzanie_result.txt", "a")
  
dane = pd.read_csv("../train_data.csv")
#sprawdzenie pustych danych
dane_bez_t2 = dane.loc[:, dane.columns != 'Type 2']
dane_puste = dane_bez_t2[dane_bez_t2.isna().any(axis=1)]
plik.write(dane_puste.to_string())
plik.write("\n\nPOZA DRUGORZĘDNYM TYPEM, KTÓRY W PRZYPADKU POKEMONÓW MOŻE BYĆ PUSTY, W ZBIORZE NIE MA INNYCH PUSTYCH DANYCH")
#zapis danych bez danych pustych
dane_bez_pustych = dane.dropna()
dane_bez_pustych.to_csv("przetwarzanie_usunpuste.csv", index=False)
#zapis danych bez duplikatów
dane_bez_duplikatow = dane.drop_duplicates()
dane_bez_duplikatow.to_csv("przetwarzanie_usunduplikaty.csv", index=False)
#sprawdzenie czy w ogóle były duplikaty
if (len(dane) == len(dane_bez_duplikatow)):
	plik.write("\n\nW ZBIORZE DANYCH NIE MA DUPLIKATÓW, TJ. POWTARZAJĄCYCH SIĘ REKORDÓW")
	
#zamiana brakujących typów na "None"
dane_wypelnij_puste_typy = dane.fillna("None")
#zamiana błędnych, mniejszych od 1 statystyk
dane_wypelnij_puste_typy.loc[dane_wypelnij_puste_typy["Attack"] <= 0, "Attack"] = 1
dane_wypelnij_puste_typy.loc[dane_wypelnij_puste_typy["HP"] <= 0, "HP"] = 1
dane_wypelnij_puste_typy.loc[dane_wypelnij_puste_typy["Total"] <= 0, "Total"] = 1
dane_wypelnij_puste_typy.loc[dane_wypelnij_puste_typy["Defense"] <= 0, "Defense"] = 1
dane_wypelnij_puste_typy.loc[dane_wypelnij_puste_typy["Speed"] <= 0, "Speed"] = 1
dane_wypelnij_puste_typy.loc[dane_wypelnij_puste_typy["Sp. Atk"] <= 0, "Sp. Atk"] = 1
dane_wypelnij_puste_typy.loc[dane_wypelnij_puste_typy["Sp. Def"] <= 0, "Sp. Def"] = 1

dane_wypelnij_puste_typy.to_csv("przetwarzanie_zamienbledne.csv", index=False)


#sortowanie danych według numeru
dane_posortowane = dane.sort_values(by=["#"])
dane_posortowane.to_csv("przetwarzanie_sortowanie.csv", index=False)

#dodanie nowych danych
dane2 = pd.read_csv("../pokemon.csv")
dane2 = dane2[dane2['#'] > 151] #wybór pokemonów, których nie ma w oryginalnym zbiorze
dane_nowe = pd.concat([dane, dane2])
dane_nowe.to_csv("przetwarzanie_dodawanie.csv", index=False)

#dodatkowe zadanie, utworzenie zbioru danych dla oryginalnej pierwszej generacji
#bez mega form oraz typów Fairy, Dark i Steel, a także zamiana statystyk Sp. Atk i Sp. Def jedną statystyką "Special"
dane_gen1 = dane.copy()
#usunięcie mega form
dane_gen1 = dane_gen1[~(dane["Name"].str.contains("Mega "))]
#przywrócenie starych typów
dane_gen1.loc[dane_gen1["Type 2"] == "Fairy", "Type 2"] = np.nan #Jigglypuff, Wigglytuff, Mr. Mime
dane_gen1.loc[dane_gen1["Type 2"] == "Steel", "Type 2"] = np.nan #Magnemite, Magneton
dane_gen1.loc[dane_gen1["Type 2"] == "Dark", "Type 2"] = np.nan #Mega Gyarados, którego już nie ma w bazie, ale usuwam dla formalności
dane_gen1.loc[dane_gen1["Type 1"] == "Fairy", "Type 1"] = "Normal" #Clefairy, Clefable
#Przywrócenie starego atrubutu Special, w tym przypadku wyznaczonego jako średnia Sp. Atk i Sp. Def
dane_gen1["Special"] = dane_gen1[["Sp. Atk", "Sp. Def"]].mean(axis=1).round().astype(int)
#usunięcie niepotrzebnych kolumn
dane_gen1 = dane_gen1.drop(labels=["Generation", "Sp. Atk", "Sp. Def"],axis=1)
#przeniesienie kolumny Special w odpowiednie miejsce
dane_gen1 = dane_gen1[["#", "Name", "Type 1", "Type 2", "Total", "HP", "Attack", "Defense", "Special", "Speed", "Legendary"]]
dane_gen1.to_csv("gen1_data.csv", index=False)

plik.close()


	



