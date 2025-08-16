import streamlit as st
import pandas as pd

import numpy as np

from plotly import express as px
from prophet import Prophet
from prophet import plot
from plotly import graph_objs as go


st.title("Previsão do preço do petróleo tipo Brent em USD$")
st.image('D:\Pós Tech - Data Analytics\TechChallenge4\DataAnalysis\Brent-Crude-Oil.jpg')
df = pd.read_csv('D:\Pós Tech - Data Analytics\TechChallenge4\DataAnalysis\Dados_IPEA.csv')
df = df.dropna()
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df['Preco'] = df['Preco'].str.replace(',', '.')
df['Preco'] = df['Preco'].astype(np.float64)
df=df.set_index('Data')

#st.write(df.head())


st.line_chart(df)

prophet_df = df.reset_index('Data')
prophet_df['Data'] = pd.to_datetime(prophet_df['Data'])
prophet_df[['ds','y']] = prophet_df[['Data','Preco']]
prophet_df = prophet_df.drop(['Data', 'Preco'], axis=1)

#st.write(prophet_df.head())



train = prophet_df.sample(frac=0.8, random_state=0)
test = prophet_df.drop(train.index)
print(f'training data size : {train.shape}')
print(f'testing data size : {test.shape}')

model = Prophet(daily_seasonality=True)
model.fit(train)
dataFramefuture = model.make_future_dataframe(periods=365, freq='D')
pred = model.predict(dataFramefuture)

ano_pred = st.slider('Ano',1987,2026,2025)
df_futuro = pred[pred['ds'].dt.year == ano_pred]
#st.write(pred.tail())

st.line_chart(data=df_futuro, x='ds', y='yhat', x_label='Anos', y_label='Preço')


