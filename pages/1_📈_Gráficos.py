#####################
#    Bibliotecas:   #
#####################


import streamlit as st
import datetime
from pages.lib.forecast import prediction
#from streamlit_extras.app_logo import add_logo 

st.set_page_config(page_title="Graficos", page_icon="📈")

# Add logo
#add_logo("logo2.jpg", height=103)

###########
# sidebar #
###########

# create default date range
start = datetime.datetime(1913, 1, 1)
end = datetime.datetime(2023, 11, 1)
# ask user for his date
start_date = st.sidebar.date_input('Fecha de inicio:', start, min_value=start, max_value=end)
end_date = st.sidebar.date_input('Fecha final:', end,  min_value=start, max_value=end)



if st.sidebar.button("Predicción"):
    if (start_date < end_date) and 9 < (end_date.year - start_date.year):
        st.sidebar.write('Fecha de inicio: `%s`\n\nFecha final: `%s`' % (start_date, end_date))
        [graph1,forecast2,graph2]= prediction(start_date, end_date)
    else:
        st.sidebar.error('Error: El intervalo de fechas debe ser mayor o igual a 10')

################################################################################

def forecast(start_date, end_date):
    graph1= prediction(start_date, end_date) 
    return (graph1,forecast2, graph2)

if __name__ == "__main__":
     #prediction(start_date, end_date)
    pass

################################################################################

