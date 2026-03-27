import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from
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



