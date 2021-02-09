import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

def main():

# Set configs
    st.set_page_config(
	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="expanded",  # Can be "auto", "expanded", "collapsed"
	page_title='V and B App',  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
    )
    

# Load Data
    beer = pd.read_csv('clean_beer.csv')
    wine = pd.read_csv('clean_wine.csv')
    whisky = pd.read_csv('clean_whisky.csv')
    rhum = pd.read_csv('clean_rhum.csv')

# Set Sidebar
    st.sidebar.title('Navigation onglet')
    page = st.sidebar.selectbox("Choisir une page", ["Page d'accueil", "Données", "Indicateurs"])
    sel_boisson = st.sidebar.selectbox('Boisson', ['Bière', 'Vin', 'Whisky', 'Rhum'])
    
    if sel_boisson == 'Bière':
        df = beer
        sel_type = st.sidebar.multiselect('Type de bière', sorted(df['Type de bière'].unique()))
        sel_type_2 = st.sidebar.multiselect('Type de bière précision', sorted(df['Type de bière précision'].unique()))
        df_type = multi_filter(df, sel_type, 'Type de bière')
        df_type_2 = multi_filter(df, sel_type_2, 'Type de bière précision')
    if sel_boisson == 'Vin':
        df = wine
        sel_type = st.sidebar.multiselect('Type de vin', sorted(df['Type de vin'].unique()))
        sel_type_2 = st.sidebar.multiselect('Type de vin précision', sorted(df['Type de vin précision'].unique()))
        df_type = multi_filter(df, sel_type, 'Type de vin')
        df_type_2 = multi_filter(df, sel_type_2, 'Type de vin précision')
    if sel_boisson == 'Whisky':
        df = whisky
        sel_type = st.sidebar.multiselect('Type de whisky', sorted(df['Type de whisky'].unique()))
        df_type = multi_filter(df, sel_type, 'Type de whisky')
    if sel_boisson == 'Rhum':
        df = rhum
        sel_type = st.sidebar.multiselect('Type de rhum', sorted(df['Type de rhum'].unique()))
        sel_type_2 = st.sidebar.multiselect('Type de rhum précision', sorted(df['Type de rhum précision'].unique()))
        df_type = multi_filter(df, sel_type, 'Type de rhum')
        df_type_2 = multi_filter(df, sel_type_2, 'Type de rhum précision')

    slider_prix = st.sidebar.slider('Prix (€)', float(df['Prix'].min()), float(df['Prix'].max()), (float(df['Prix'].min()), float(df['Prix'].max())))
    slider_degre = st.sidebar.slider("Degré d'alcool (%)", float(df["Degré d'alcool"].min()), float(df["Degré d'alcool"].max()), (float(df["Degré d'alcool"].min()), float(df["Degré d'alcool"].max())))
    sel_nom = st.sidebar.multiselect('Nom', sorted(df['Nom'].unique()))
    sel_pays = st.sidebar.multiselect('Pays', sorted(df['Pays'].unique()))

# Configure generals filters

    df_prix = df[df['Prix'].between(slider_prix[0],slider_prix[1])]
    df_degre = df[df["Degré d'alcool"].between(slider_degre[0],slider_degre[1])]
    df_pays = multi_filter(df, sel_pays, 'Pays')
    df_nom = multi_filter(df, sel_nom, 'Nom')

    if sel_boisson != 'Whisky':
        df_select = df[df.isin(df_type) & df.isin(df_type_2) & df.isin(df_prix) & df.isin(df_degre) & df.isin(df_pays) & df.isin(df_nom)].dropna()
    else:
        df_select = df[df.isin(df_type) & df.isin(df_prix) & df.isin(df_degre) & df.isin(df_pays) & df.isin(df_nom)].dropna()

### Main Page
# Page d'accueil
    if page == "Page d'accueil":
        st.title('Title')
        st.markdown("---")

# Données   
    if page == 'Données':
        st.dataframe(df_select)

# Indicateurs
    if page == 'Indicateurs':
        if sel_boisson == 'Bière':
            gauges = ['Amertume', 'Acidité', 'Vivacité', 'Puissance', 'Rondeur', 'Fruité']
            for gaug in gauges:
                gauge(df, df_select, gaug)

        if sel_boisson == 'Vin':
            gauges = ['Fruité', 'Vivacité', 'Puissance', 'Epicé/boisé']
            for gaug in gauges:
                gauge(df, df_select, gaug)

        if sel_boisson == 'Whisky':
            gauges = ['Epicé/Boisé', 'Tourbé/Fumé', 'Iodé', 'Fruité', 'Floral/Végétal', 'Malté', 'Organique', 'Vivacité', 'Pâtissier']
            for gaug in gauges:
                gauge(df, df_select, gaug)

# Bottom page
    st.write("\n") 
    st.write("\n")
    st.info("""By : Ligue des Datas [Instagram](https://www.instagram.com/ligueddatas/) / [Twitter](https://twitter.com/ligueddatas) | Data source : [V and B](https://vandb.fr/)""")


def multi_filter(df, sel, var):
    if len(sel) == 0:
        df_sel = df
    elif len(sel) != 0:
        df_sel = df[df[var].isin(sel)]
    return df_sel

def gauge(df_base, df_after, var_gauge):
    fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = df_after[var_gauge].mean(),
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': var_gauge, 'font': {'size': 28}},
    delta = {'reference': df_base[var_gauge].mean(), 'increasing': {'color': "RebeccaPurple"}},
    gauge = {
        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "darkblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': df_base[var_gauge].mean()}}))

    fig.update_layout(paper_bgcolor = "white", font = {'color': "darkblue", 'family': "Trebuchet MS"})
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
