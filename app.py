import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial

def main():

# Set configs
    st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="collapsed",  # Can be "auto", "expanded", "collapsed"
	page_title='V and B App',  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
    )
    

# Set Sidebar
    st.sidebar.title('Navigation onglet')
    page = st.sidebar.selectbox("Choisir une page", ["Page d'accueil", "Données", "Indicateurs", "Recommandateur"])
    if page != "Page d'accueil":
        sel_boisson = st.sidebar.selectbox('Boisson', ['Bière', 'Vin', 'Whisky', 'Rhum'])
        
        if sel_boisson != 'Whisky':
            df = pd.read_csv('clean_' + sel_boisson.lower() + '.csv')
            sel_type = st.sidebar.multiselect('Type de ' + sel_boisson.lower(), sorted(df['Type de ' + sel_boisson.lower()].unique()))
            sel_type_2 = st.sidebar.multiselect('Type de ' + sel_boisson.lower() + ' précision', sorted(df['Type de ' + sel_boisson.lower() + ' précision'].unique()))
            df_type = multi_filter(df, sel_type, 'Type de ' + sel_boisson.lower())
            df_type_2 = multi_filter(df, sel_type_2, 'Type de ' + sel_boisson.lower() + ' précision')

        if sel_boisson == 'Whisky':
            df = pd.read_csv('clean_' + sel_boisson.lower() + '.csv')
            sel_type = st.sidebar.multiselect('Type de whisky', sorted(df['Type de whisky'].unique()))
            df_type = multi_filter(df, sel_type, 'Type de whisky')

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
        col1, col2, col3 = st.beta_columns((1,3,1))
        with col2:
            st.title('Title')
            st.markdown("---")
            st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum")

# Données   
    if page == 'Données':
        st.dataframe(df_select)

# Indicateurs
    if page == 'Indicateurs':
        
        #nb_bouteilles(df)
        #nb_bouteilles(df_select)

        if sel_boisson == 'Bière':
            col1, col2, col3= st.beta_columns(3)
            with col1:
                gauges = ['Amertume', 'Acidité']
                for gaug in gauges:
                    gauge(df, df_select, gaug)
            with col2:
                gauges = ['Vivacité', 'Puissance']
                for gaug in gauges:
                    gauge(df, df_select, gaug)
            with col3:
                gauges = ['Rondeur', 'Fruité']
                for gaug in gauges:
                    gauge(df, df_select, gaug)

        if sel_boisson == 'Vin':
            col1, col2= st.beta_columns(2)
            with col1:
                gauges = ['Fruité', 'Vivacité']
                for gaug in gauges:
                    gauge(df, df_select, gaug)
            with col2:
                gauges = ['Puissance', 'Epicé/boisé']
                for gaug in gauges:
                    gauge(df, df_select, gaug)

        if sel_boisson == 'Whisky':
            col1, col2, col3= st.beta_columns(3)
            with col1:
                gauges = ['Epicé/Boisé', 'Tourbé/Fumé', 'Iodé']
                for gaug in gauges:
                    gauge(df, df_select, gaug)
            with col2:
                gauges = ['Fruité', 'Floral/Végétal', 'Malté']
                for gaug in gauges:
                    gauge(df, df_select, gaug)
            with col3:
                gauges = ['Organique', 'Vivacité', 'Pâtissier']
                for gaug in gauges:
                    gauge(df, df_select, gaug)

        if sel_boisson == 'Rhum':
            col1, col2, col3= st.beta_columns(3)
            with col1:
                gauges = ['Epicé/Boisé', 'Tourbé/Fumé', 'Fruité']
                for gaug in gauges:
                    gauge(df, df_select, gaug)
            with col2:
                gauges = ['Floral/Végétal', 'Vivacité']
                for gaug in gauges:
                    gauge(df, df_select, gaug)
            with col3:
                gauges = ['Sucrosité', 'Pâtissier']
                for gaug in gauges:
                    gauge(df, df_select, gaug)

# Recommandateur
    if page == 'Recommandateur':
        col1, col2 = st.beta_columns(2)
        with col1:
            st.title('Choix des variables')
            if sel_boisson != 'Whisky':
                y_var = st.selectbox('Variable à expliquer', list(df_select.select_dtypes(include=np.object).iloc[:,:5].columns))
            else :
                y_var = st.selectbox('Variable à expliquer', list(df_select.select_dtypes(include=np.object).iloc[:,:4].columns))
            X_var = st.multiselect('Variables expliquatives', list(df_select.select_dtypes(include=np.number).columns), default=list(df_select.select_dtypes(include=np.number).columns))

            df_acp, n, p, acp_, coord, eigval = acp(df = df_select, X = X_var, y = y_var)

        if n>=p:  
            with col2:
                st.title('Recommandateur')
                sel_simi = st.selectbox(sel_boisson + ' que tu aimes', sorted(df_acp.index))
                nb_simi = st.number_input("Nombre de " + sel_boisson.lower() + "s les plus ressemblants", min_value=1, max_value=n-1, value=3)
                df_near = get_indices_of_nearest_neighbours(df_acp, coord, nb_simi+1)
                same_reco(df_near, sel_simi, sel_boisson)

        else:
            st.error("Nombre de " + sel_boisson.lower() + "s sélectionnées inférieur au nombre de variables explicatives.")


# Bottom page
    st.write("\n") 
    st.write("\n")
    st.info("""By : Ligue des Datas [Instagram](https://www.instagram.com/ligueddatas/) | Data source : [V and B](https://vandb.fr/)""")






### define functions
def multi_filter(df, sel, var):
    if len(sel) == 0:
        df_sel = df
    elif len(sel) != 0:
        df_sel = df[df[var].isin(sel)]
    return df_sel

def nb_bouteilles(data):
    fig = go.Figure(go.Indicator(
    mode = "number",
    value = data.shape[0],
    title = {"text": "Nombre de bouteilles", 'font': {'size': 28}},
    domain = {'x': [0, 1], 'y': [0, 1]}))

    fig.update_layout(paper_bgcolor = "white", font = {'color': "darkblue", 'family': "Trebuchet MS"})
    st.plotly_chart(fig)


def gauge(df_base, df_after, var_gauge):
    fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = df_after[var_gauge].mean(),
    
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

    fig.update_layout(paper_bgcolor = 'rgba(0,0,0,0)', font = {'color': "darkblue", 'family': "Trebuchet MS"},
                    autosize=True,
                  width=370, height=370,
                  margin=dict(l=40, r=40, b=40, t=40))
    st.plotly_chart(fig)

# ACP & Recommandations
def acp(df, X, y):
    X.append(y)
    df_acp = df[X].groupby(y).mean()
    
    n = df_acp.shape[0]
    p = df_acp.shape[1]
    sc = StandardScaler()
    Z = sc.fit_transform(df_acp)
    
    acp_ = PCA(svd_solver='full')
    coord = acp_.fit_transform(Z)
    
    eigval = (n-1)/n*acp_.explained_variance_
    
    return df_acp, n, p, acp_, coord, eigval

def get_indices_of_nearest_neighbours(df,Coords, n):
    indice = np.array(df.index)
    tree=spatial.cKDTree(Coords)
    res=tree.query(Coords, k=n)[1][:,0:]
    res = pd.DataFrame(indice[res])
    return res


def same_reco(df, ind, type_boisson):
    res = []
    for i in df.columns:
        res.append(df[df[0] == ind].iloc[:,i].to_string()[5:])
    res = res[1:]
    for i in range(len(res)):
        if len(res[i]) <= 43:
            st.write(type_boisson + ' ressemblant n°' + str(i+1) +' : ' + res[i])
        else:
            st.write(type_boisson + ' ressemblant n°' + str(i+1) +' : ' + res[i][:43])
            st.write(res[i][43:])
    return res

if __name__ == "__main__":
    main()
