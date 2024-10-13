import pandas as pd
import os
dane = pd.read_csv("../train_data.csv")
if os.path.exists("eksploracja_result.txt"):
  os.remove("eksploracja_result.txt")
plik = open("eksploracja_result.txt","a")
#zapisz w pliku strukturę źródłą danych
plik.write("**ANALIZA STRUKTURY DANYCH**")
dane.info(buf=plik)
#zapisz w pliku statystyczny opis każdej cechy
plik.write("\n\n**PODSUMOWANIE STATYSTYCZNE POSZCZEGÓLNYCH CECH**")
plik.write("\n\n"+dane.describe().to_string())
plik.write("\n\n"+dane.describe(include=["bool"]).to_string())
plik.write("\n\n"+dane.describe(include=["object"]).to_string())
plik.close()

