#####################
#    Bibliotecas:   #
#####################

import streamlit as st
import pydeck as pdk
import time
#from streamlit_extras.app_logo import add_logo 



st.set_page_config(
    page_title="Welcome!",
    page_icon="📈",
)

# add logo
#add_logo("logo4.jpg", height=103)

st.write("# Predicción de la Tasa de Inflación Anual de US 📈")
st.divider()  # 👈 Another horizontal rule


#US Map: 
st.pydeck_chart(pdk.Deck(
    map_style= None, #None
    initial_view_state=pdk.ViewState(
        latitude=37.09024,
        longitude=-95.712891,
        zoom=3.9,
        pitch=50,
    )
))


# Contenido: 
st.markdown(
    """
    ### Introducción

    La [inflación](https://www.imf.org/en/Publications/fandd/issues/Series/Back-to-Basics/Inflation#:~:text=Inflation%20is%20the%20rate%20of,of%20living%20in%20a%20country.) 
    es una medida de la tasa de aumento de los precios durante un período de tiempo. 
    Es una preocupación económica importante, ya que puede erosionar el poder adquisitivo de los 
    consumidores y las empresas. Pronosticar la inflación es fundamental para la política gubernamental 
    y la planificación empresarial. Los gobiernos deben monitorear cuidadosamente la inflación para 
    establecer políticas monetarias apropiadas. Los cambios inesperados pueden perjudicar la 
    planificación a largo plazo. Las empresas por su parte, deben tener en cuenta la inflación 
    al elaborar presupuestos, fijar precios, negociar salarios, etc. La inflación inesperada afecta las ganancias.

    ### Problema a tratar:
    💡 Pronosticar la Tasa de Inflación Anual de Estados Unidos.

    ### ¿Cómo abordaremos el problema?
    
    ✅ Utilizaremos un modelo de memoria a largo plazo 
    ([LSTM](https://www.researchgate.net/publication/13853244_Long_Short-term_Memory)) para pronosticar 
    la Tasa Anual de Inflación. Los LSTM son un tipo de red neuronal recurrente que es muy 
    adecuada para pronosticar datos de series temporales.

    ✅ Se empleará los datos históricos del [Consumer Prices Index (CPI)](https://www.bls.gov/cpi/) de Estados Unidos 
    (desde Enero de 1913 a Noviembre de 2023) para obtener la Tasa Anual de Inflación de Estados Unidos. 
    Con la Tasa Anual se entrenará el modelo LSTM. El modelo aprenderá los patrones de los datos de la Inflación 
    y utilizará este conocimiento para pronosticar los valores futuros.

    ✅ También visualizaremos las predicciones y los valores reales para comprender mejor el rendimiento del modelo.

    ✅ El modelo LSTM puede capturar patrones históricos complejos como estacionalidad, tendencias, 
    cambios repentinos, etc. Esta información se puede utilizar para anticipar cambios futuros.

    ✅ Los datos del CPI, lo encontramos en [U.S. Bureau of Labor Statistics](https://www.bls.gov/)

    ### Pasos para el desarrollo de este proyecto:

    🔷 Instalación e importación de las bibliotecas necesarias.

    🔷 Obtención de los datos del CPI a través de la API de la página Bureau of Labor Statistics.

    🔷 Exploración de los datos para comprender su distribución y características.

    🔷 Normalización de los datos para que sean adecuados para entrenar el modelo LSTM.

    🔷 División de los datos en conjuntos de train y test.

    🔷 Creación de secuencias para el modelo LSTM. 

    🔷 Construcción y entrenamiento del modelo LSTM. 

    🔷 Predicciones y evaluación del desempeño del modelo. 

    🔷 Visualización de los resultados de las predicciones y los valores reales.
    
"""
)


st.sidebar.markdown("# 🔎 Introducción ")
st.sidebar.markdown("# 💡 Problema a tratar ")
st.sidebar.markdown("# ❔¿Cómo abordaremos el problema? ")
st.sidebar.markdown("# 🪜 Pasos para el desarrollo de este proyecto ")

