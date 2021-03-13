import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import spatial
import fr_core_news_md

def main():

# Set configs
    st.set_page_config(
	layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="collapsed",  # Can be "auto", "expanded", "collapsed"
	page_title='V and B App',  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
    )
    

# Set Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choisir une page", ["Page d'accueil", "Données", "Indicateurs", "Recommandateur"])
    if page != "Page d'accueil":
        sel_boisson = st.sidebar.selectbox('Choisir une boisson', ['Bière', 'Vin', 'Whisky', 'Rhum'])
        st.sidebar.title("Filtres")
        
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
        st.title('Présentation')
        st.write("Bienvenue sur ce site qui a été crée par et pour les amateurs de bons produits et de bonnes bouteilles. Ce site a pour objectif de pouvoir explorer et découvrir de bonnes bouteilles à travers des indicateurs et des modèles statistiques.")
            
        st.title("D'où proviennent les données")
        st.write("Ces données ont été scrapées sur le site www.vandb.com. La liste exhaustive des boissons proposées par l'enseigne V and B sur son site est recupérée dans un premier temps. Puis à partir de cette liste, les caractéristiques attribuées à ces boissons par la compagnie sont stockées. Ces caractéristiques sont des notions propres à chacun. Nous faisons entièrement confiance à V and B quand à la pertinence des notes choisies.")
            
        st.title('Utilisation du site')
        st.write("Il faut cliquer sur la petite flèche en haut à gauche pour afficher l'onglet de navigation. Vous pouvez donc choisir la page et la boisson que vous souhaitez. Puis filtrer les bouteilles en fonction de certaines variables.")
        st.subheader("Données :")
        st.write("Tableau brut des données permettant de découvrir les données disponibles, et d'explorer en détail les bouteilles sélectionnées.")
        st.subheader("Indicateurs :")
        st.write("Ensemble d'indicateurs de moyennes des notes gustatives (sur 10) des bouteilles. Ces indicateurs permettent de comparer les bouteilles filtrées en fonction de l'ensemble des bouteilles.")
        st.subheader("Recommandateur :")
        st.write("Choisissez une bouteille que vous aimez, l'algorithme va vous donner les bouteilles qui lui ressemble le plus. Ces recommandations sont basées sur les bouteilles que vous avez filtrez, et les variables explicatives que vous avez sélectionnées.")

# Données   
    if page == 'Données':
        st.title("Données des " + sel_boisson.lower() + "s présents chez V and B")
        st.write("Une ligne correspond à une bouteille et ses caractéristiques.")
        st.write("Vous pouvez mettre le tableau en plein écran afin de mieux visualiser le tableau.")
        st.markdown("---")
        st.dataframe(df_select)

# Indicateurs
    if page == 'Indicateurs':
        st.title("Indicateurs de moyenne")
        st.write("La gauge et la valeur au centre correspondent à la moyenne des bouteilles filtrées. Le trait rouge correspond à la moyenne pour l'ensemble des bouteilles. Le delta correspond à la différence entre la moyenne des bouteilles sélectionnées et la moyenne de l'ensemble des bouteilles.")
        #nb_bouteilles(df)
        #nb_bouteilles(df_select)

        if sel_boisson == 'Bière':
            col1, col2, col3= st.beta_columns(3)
            with col1:
                multi_gauges(df, df_select, ['Amertume', 'Acidité'])
            with col2:
                multi_gauges(df, df_select, ['Vivacité', 'Puissance'])
            with col3:
                multi_gauges(df, df_select, ['Rondeur', 'Fruité'])

        if sel_boisson == 'Vin':
            col1, col2= st.beta_columns(2)
            with col1:
                multi_gauges(df, df_select, ['Fruité', 'Vivacité'])
            with col2:
                multi_gauges(df, df_select, ['Puissance', 'Epicé/boisé'])

        if sel_boisson == 'Whisky':
            col1, col2, col3= st.beta_columns(3)
            with col1:
                multi_gauges(df, df_select, ['Epicé/Boisé', 'Tourbé/Fumé', 'Iodé'])
            with col2:
                multi_gauges(df, df_select, ['Fruité', 'Floral/Végétal', 'Malté'])
            with col3:
                multi_gauges(df, df_select, ['Organique', 'Vivacité', 'Pâtissier'])

        if sel_boisson == 'Rhum':
            col1, col2, col3= st.beta_columns(3)
            with col1:
                multi_gauges(df, df_select, ['Epicé/Boisé', 'Tourbé/Fumé', 'Fruité'])
            with col2:
                multi_gauges(df, df_select, ['Floral/Végétal', 'Vivacité'])
            with col3:
                multi_gauges(df, df_select, ['Sucrosité', 'Pâtissier'])

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

        if len(X_var) > 0:
            if n>=p:  
                with col2:
                    st.title('Recommandateur')
                    sel_simi = st.selectbox(sel_boisson + ' que tu aimes', sorted(df_acp.index))
                    nb_simi = st.number_input("Nombre de " + sel_boisson.lower() + "s les plus ressemblants", min_value=1, max_value=n-1, value=3)
                    df_near = get_indices_of_nearest_neighbours(df_acp, coord, nb_simi+1)
                    recos = same_reco(df_near, sel_simi, sel_boisson)

                data_reco_sel = df[df['Nom'] == sel_simi]
                data_reco = df[df['Nom'].isin(recos)].drop_duplicates(subset=['Nom'])
                data_reco_sort = data_reco.set_index('Nom').loc[recos].reset_index()
                df_reco_final = pd.concat([data_reco_sel, data_reco_sort]).reset_index().drop("index", axis = 1)

                st.write('\n')
                st.title("Données des bouteilles recommandées")
                st.write('\n')
                st.dataframe(df_reco_final)
            else:
                st.error("Nombre de " + sel_boisson.lower() + "s sélectionnées inférieur au nombre de variables explicatives.")
        else:
            st.error("Pas assez variables explicatives sélectionnées.")
            
# Bottom page
    st.write("\n") 
    st.write("\n")
    st.markdown("---")
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

def multi_gauges(data, data_select, gauges):
    for gaug in gauges:
        gauge(data, data_select, gaug)

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
        res.append(df[df[0] == ind].iloc[:,i].to_string()[5:].lstrip())
    res = res[1:]
    for i in range(len(res)):
        st.write(type_boisson + ' ressemblant n°' + str(i+1) +' : ' + res[i])
    return res

    
def data_reco(data, recos):
    for i in reco:
        st.write(i)


if __name__ == "__main__":
    main()
