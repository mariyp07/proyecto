#########################################################
#     Predicción de la Tasa de Inflación Anual de US    #
#     Realizado por: Mariela Perdomo                    #
#########################################################

#####################
#    Bibliotecas:   #
#####################

import streamlit as st
import numpy as np
import pandas as pd
import plotly_express as px
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)
import time
from time import sleep
import datetime
from streamlit_lottie import st_lottie

#######################
#      Funciones      #
#######################

# Cargando el archivo CSV en un DataFrame
def readdata():
    df = pd.read_csv('df.csv', dtype={'Year':object})
    return(df)

#Agregando una columna de fecha completa al dataframe
def fulldate(df):
    df['Date']= df.Year + '-' + df.Period + '-' +'01'
    return (df)

#Colocándole formato a la columna de date
def formatdate(df):
    df['Date'] = df['Date'].replace({'M':''}, regex=True)
    return df

# Seleccionando las columnas CPI y Date
def selectingcolumns(df):
    df= df.loc[:, ['CPI', 'Date']]
    return df

# Cambiando el tipo de formato de la serie de datos
def formatdataCPI(data):
    data["CPI"] = pd.to_numeric(data["CPI"])
    return data

# Convirtiendo la columna 'date' en formato datetime
def formatdatetime(data):
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Colocando la columna 'Date' column como el índice
def indexdate(data):
    data.set_index('Date', inplace=True)
    return data

#Determinando la Tasa de Inflación Anual de US
def inflationrate(data,start_date,end_date):
    data2 = data.loc[start_date:end_date].pct_change(12)*100 
    return data2

# Seleccionando la data que no contiene NaN
def transformation(data2):
    data2 = data2.dropna()
    return data2

#Creando la columna Inflation
def transformation2(data2):
    data2['Inflation']= data2.CPI 
    return data2

#Colocando en el dataframe nuevo la fecha y la columnna de inflacion 
def transformation3(data2):
    datainflation = data2[['Inflation']]
    return datainflation

# Creando un gráfico de línea interactivo
def graphCPI(datainflation):

    # Definiendo la animación de Lottie URL
    lottie_animation_url = "https://lottie.host/89f1f8df-aa47-4771-9441-91da251470e2/qGrHDGTqFH.json"

    #Headers:
    st.markdown("# Gráficos y predicción de la :blue[Tasa de Inflación Anual de US]")

    # Mostrando la animación de Lottie usando st_lottie
    st_lottie(lottie_animation_url,height=200)

    st.divider()  # 👈 Línea horizontal
    
    st.subheader('1. Gráfico de la Tasa de Inflación Anual de US:')

    st.sidebar.markdown("# Tasa de Inflación Anual de US:")

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    for i in range(1, 101):
       status_text.text("Gráfico 1: %i%%" % i)
       progress_bar.progress(i)
       time.sleep(0.05)

   
    fig = px.line(datainflation, y=datainflation.Inflation, title='Tasa de Inflación Anual de US',
                 labels={'Date': 'Year',  'Inflation': 'Inflation'},
                 template='plotly', line_shape='linear')
    fig.update_xaxes(tickformat='%B<br>%Y')    
         
    st.plotly_chart(fig, use_container_width=False)

    
def reshapeinflation(datainflation):
    y= datainflation.Inflation
    y = y.values.reshape(-1, 1)
    return (y)

# Escalando la data
def scalerInflation(y):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)
    return (y, scaler) 

# Generando los input y los output de las sequencias
def sequences(y):
    n_lookback = 60  # longitud del input  de las sequencias (lookback period)
    n_forecast = 6 # longitud del output de las sequencias (forecast period)
    
    X = []
    Y = []
    
    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    return (X,Y, n_lookback, n_forecast)

# Ajuste del modelo y predicción

def fitmodel(X,Y, n_lookback, n_forecast):
    
    # Creando un texto y una barra para saber cuando la data o el gráfico está cargado.   
    st.sidebar.markdown("# Modelo y Predicción:")

    st.divider()  # 👈 Línea Horizontal

    st.subheader('2. Creando el modelo y generando la predicción 👀...')

    progress_bar2 = st.sidebar.progress(0)
    status_text2 = st.sidebar.empty()


    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=50, batch_size=32, verbose=0)  
    
    #model.save('my_model.keras')

    for i in range(1, 101):
        status_text2.text("Predicción: %i%%" % i)
        progress_bar2.progress(i)
        time.sleep(0.05)
      
    return(model, n_lookback)

def forecast(y, model, n_lookback, scaler):
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)
    
    Y_ = model.predict(X_).reshape(-1, 1) 
    Y_ = scaler.inverse_transform(Y_) 
    return (X_, Y_)

# Organizando los resultados en un dataframe

def datapast(datainflation):
    data_past = datainflation[['Inflation']].reset_index()
    return (data_past)

def datapastrename(data_past):
    data_past.rename(columns={'index': 'Date', 'Inflation': 'Actual'}, inplace=True)
    return(data_past)

def datapast2(data_past):
    data_past['Date'] = pd.to_datetime(data_past['Date'])
    return(data_past)

def datapastforecast(data_past):
    data_past['Forecast'] = np.nan
    data_past['Forecast'].iloc[-1] = data_past['Actual'].iloc[-1]
    return(data_past)

def datafuture(data_past, n_forecast):
    data_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    data_future['Date'] = pd.date_range(start=data_past['Date'].iloc[-1] + pd.Timedelta(days=30), periods=n_forecast, freq="MS")
    return(data_future)

def datafuture2(data_future,Y_):
    data_future['Forecast'] = Y_.flatten()
    data_future['Actual'] = np.nan
    return (data_future)

def resultfinal(data_past,data_future):
     results= pd.concat([data_past, data_future])
     results.set_index('Date', inplace=True)
     return (results)

def finalprediction(data_future):
    forecast2 = data_future
    forecast2.set_index('Date', inplace=True)

    st.write(
    """_Predicción de la Tasa Anual de la Inflación de US_:""")    
 
    st.table(forecast2['Forecast'])

# Creando un gráfico de línea interactivo
def graphCPI2(results):

    st.sidebar.markdown("# Data Histórica y Predicción de US:")

    progress_bar3 = st.sidebar.progress(0)
    status_text3 = st.sidebar.empty()

    st.divider()  # 👈 Línea horizontal

    st.subheader('3. Gráfico de la data histórica y la predicción de la Tasa de Inflación Anual de US')

    
    fig2 = px.line(results, y=[results.Actual,results.Forecast], title='Predicción de la Tasa Anual de Inflación con su data histórica',
                    labels={'Date': 'Year',  'Inflation': 'Inflation'},
                    template='plotly', line_shape='linear')
    fig2.update_xaxes(tickformat='%B<br>%Y')

    for i in range(1, 101):
        status_text3.text("Gráfico 2: %i%%" % i)
        progress_bar3.progress(i)
        time.sleep(0.05)

    st.plotly_chart(fig2, use_container_width=False)


def prediction(start_date, end_date):
    df = readdata()
    df = fulldate(df)
    df = formatdate(df)
    data = selectingcolumns(df)
    data = formatdataCPI(data)
    data = formatdatetime(data)
    data = indexdate(data)
    data2 = inflationrate(data,start_date,end_date)
    data2 = transformation(data2) 
    data2 = transformation2(data2)
    datainflation = transformation3(data2)
    graph1= graphCPI(datainflation)
    y = reshapeinflation(datainflation)
    [y, scaler] = scalerInflation(y)
    [X,Y, n_lookback, n_forecast] = sequences(y) 
    [model, n_lookback]= fitmodel(X,Y, n_lookback, n_forecast)
    [X_, Y_] = forecast(y, model, n_lookback, scaler)
    data_past = datapast(datainflation)
    data_past=datapastrename(data_past)
    data_past=datapast2(data_past)
    data_past=datapastforecast(data_past)
    data_future = datafuture(data_past, n_forecast)
    data_future = datafuture2(data_future,Y_)
    results = resultfinal(data_past,data_future)
    forecast2 = finalprediction(data_future)
    graph2=graphCPI2(results)
        
    return (graph1,forecast2,graph2)
      

if __name__ == "__main__":
    pass
