import streamlit as st
import pandas as pd

import numpy as np


from prophet import Prophet
from prophet import plot

df = pd.read_csv('DataAnalysis/Dados_IPEA.csv')
df = df.dropna()
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
df['Preco'] = df['Preco'].str.replace(',', '.')
df['Preco'] = df['Preco'].astype(np.float64)
#df=df.set_index('Data')

st.title("Previsão do preço do petróleo tipo Brent em USD$")
st.image('DataAnalysis/Brent-Crude-Oil.jpg')

st.header('O que é o petróleo Brent?')
st.write('Brent é o nome dado ao petróleo cru extraido do Mar do Norte, e é uma das prinicipais referências para a precificação do petróleo no mundo.')
st.write('No gráfico abaixo podemos observar a flutuação dos preços do barril do petróleo Brent entre os anos de 1987 e 2025.')
st.line_chart(data=df, x='Data', y='Preco', x_label='Anos', y_label='Preço')

st.write('Ao analisar os dados, podemos notar que os preços estão em constante flutuação, mesmo que com o tempo surjam tendências fortes de alta ou queda. Podemos perceber que a tendência de alta mais proeminente aconteceu no início de 2002, logo após uma breve tendência de queda, explicada pelos ataques terroristas de Setembro daquele ano, que colocaram dúvidas sobre o preço do petróleo do Oriente Médio.')

st.line_chart(data=df[df['Data'].dt.year == 2002 ], x='Data', y='Preco', x_label='Meses - 2002', y_label='Preço')

st.write('A tendência, à seguir, é de alta nos preços, e então uma queda brusca, coincidindo com o ano de 2008 em que a crise econômica norte americana afetou a economia de diversos países. Após esse momento, o preço mais baixo registrado foi por volta de Março de 2020. Podemos formular a hipótese de que as incertezas de uma pandemia global e a baixa procura por combustível durante os anos de isolamento social afetaram o preço dos barris de petróleo.')

st.line_chart(data=df[df['Data'].dt.year == 2008 ], x='Data', y='Preco', x_label='Meses - 2008', y_label='Preço')

st.line_chart(data=df[df['Data'].dt.year == 2020 ], x='Data', y='Preco', x_label='Meses - 2020', y_label='Preço')


prophet_df = df
prophet_df['Data'] = pd.to_datetime(prophet_df['Data'])
prophet_df[['ds','y']] = prophet_df[['Data','Preco']]
prophet_df = prophet_df.drop(['Data', 'Preco'], axis=1)

#st.write(prophet_df.head())

st.write('A seguir podemos utilizar o modelo de forecast do Prophet para prever o preço diário do petróleo Brent em até 10 anos.')

train = prophet_df.sample(frac=0.8, random_state=0)
test = prophet_df.drop(train.index)
print(f'training data size : {train.shape}')
print(f'testing data size : {test.shape}')

anos_forecast=st.number_input('Quantidade de anos para o forecast',1,10)*365

model = Prophet(daily_seasonality=True)
model.fit(train)
dataFramefuture = model.make_future_dataframe(periods=anos_forecast, freq='D')
pred = model.predict(dataFramefuture)

#st.write(pred.tail())

st.line_chart(data=pred[pred['ds'].dt.year > 2024], x='ds', y='yhat', x_label='Anos', y_label='Preço')



