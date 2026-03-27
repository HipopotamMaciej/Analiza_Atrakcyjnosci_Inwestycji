import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster
import matplotlib.pyplot as plt


df = pd.read_csv('dane.csv', sep=';', decimal=',', encoding='utf-8')

cechy = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6',]

for col in cechy:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

print((df[cechy]))

scaler = MinMaxScaler()

df_scalowane = df.copy()

df_scalowane[cechy] = scaler.fit_transform(df[cechy])

print(df_scalowane[cechy])

df_scalowane['X1'] = 1 - df_scalowane['X1']
df_scalowane['X4'] = 1 - df_scalowane['X4']

print(df_scalowane[cechy])


# 1. Obliczamy powiązania metodą Warda

Z = linkage(df_scalowane[cechy], method='ward')

plt.figure(figsize=(12,7))
plt.title("Dendrogram - Metoda Warda (Województwa)")
plt.xlabel("Wskaźnik odległości (podobieństwa)")
plt.ylabel("Województwa")

dendrogram(
    Z,
    labels=df['Wojewodztwo'].values,
    orientation='top', # drzewo od góry
    leaf_rotation=90   # pionowe napisy województw
)

plt.axhline(y=1.5, color='r', linestyle='--') # Przykładowa linia cięcia
plt.show()


# Z analizy wyszło że najlepiej odciac grupy na [delta]ESS = 1.5

df['Grupa_Ward'] = fcluster(Z, t=1.5, criterion='distance')

print("\n--- LICZBA WOJEWÓDZTW W GRUPACH (WARD) ---")
print(df['Grupa_Ward'].value_counts())

# Profilowanie klastrów -> Srednie wartości X1..X6 w grupach
analiza_ward = df.groupby('Grupa_Ward')[cechy].mean()

analiza_ward['Liczba_Wojewodztw'] = df.groupby('Grupa_Ward')['Wojewodztwo'].count()

print("\n--- PROFIL ŚREDNIEGO WOJEWÓDZTWA W GRUPIE (METODA WARDA) ---")
print(analiza_ward)









