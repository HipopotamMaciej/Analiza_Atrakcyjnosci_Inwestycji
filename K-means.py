import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('dane.csv', sep=';', decimal=',', encoding='utf-8')

cechy = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']

# Zamiana ',' na '.', ; Konwersja z str na float64
for col in cechy:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.') , errors='coerce')

# Normalizacja danych
scaler = MinMaxScaler()

df_scalowane = df.copy()

df_scalowane[cechy] = scaler.fit_transform(df[cechy])

print(df_scalowane.dtypes)
print(df_scalowane.head(16))
# Przeskalowanie Desymulantow na stymulanty
df_scalowane['X1'] = 1 - df_scalowane['X1']
df_scalowane['X4'] = 1 - df_scalowane['X4']

print(df_scalowane.head(16))

#Analiza
X = df_scalowane[(cechy)]

sse = []
for k in range(1,11):
    kmeans = KMeans(n_clusters=k, random_state=18271, n_init=10)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

#Wykres Analizy
plt.figure(figsize = (10,6))
plt.plot(range(1,11), sse, marker='o', linestyle='--', color='b')
plt.title('Metoda Łokcia')
plt.xlabel('Liczba klastrów')
plt.ylabel('SSE (Interia)')
plt.xticks(range(1,11))
plt.grid(True)
plt.show()

#Decyzja: 3 lub 4 klastry: 4

ostateczny_kmeans = KMeans(n_clusters=4, random_state=18271, n_init=10)

df_scalowane['Grupa'] = ostateczny_kmeans.fit_predict(X)

print(df_scalowane[['Wojewodztwo', 'Grupa']].sort_values(by='Grupa', ascending=False).head(16))

# Profilowanie klastrów

analiza_grup = df_scalowane.groupby('Grupa')[cechy].mean()
analiza_grup['Liczba_Wojewodztw'] = df_scalowane.groupby('Grupa')['Wojewodztwo'].count()

print("--- PROFIL ŚREDNIEGO WOJEWÓDZTWA W GRUPIE ---")
print(analiza_grup)


#Zrobic Metode Warda











