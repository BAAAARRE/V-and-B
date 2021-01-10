import streamlit as st
import pandas as pd
import numpy as np

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.graph_objects as go
import plotly.express as px


def main():

# Set configs
    st.set_page_config(
	layout="centered",  # Can be "centered" or "wide". In the future also "dashboard", etc.
	initial_sidebar_state="collapsed",  # Can be "auto", "expanded", "collapsed"
	page_title='V and B App',  # String or None. Strings get appended with "• Streamlit". 
	page_icon=None,  # String, anything supported by st.image, or None.
    )
    

# Load Data
    df = pd.read_csv('clean_beer.csv')
    

# Set Sidebar
    st.sidebar.title('Navigation onglet')
    sel_type = st.sidebar.multiselect('Type de bière    ', sorted(df['Type de bière'].unique()))
    

# Configure generals filters
    df_type = multi_filter(df, sel_type, 'Type de bière')

    df_select = df[df.isin(df_type)].dropna()



# Page 1

    st.dataframe(df_select)
    st.title('Choix des variables')
    y_var = st.selectbox('Variable à expliquer', list(df_select.select_dtypes(include=np.object).iloc[:,:5].columns))
    X_var = st.multiselect('Variables expliquatives', list(df_select.select_dtypes(include=np.number).columns), default=list(df_select.select_dtypes(include=np.number).columns))


    df_acp, n, p, acp_, coord, eigval = acp(df = df_select, X = X_var, y = y_var)
    if n>=p & len(X_var) >= 2:

        st.title('Variance expliquée')
        choix_axe(acp_, p)      

        st.title('Choix dex axes à représenter')
        nb_axe_x = st.number_input('Numéro axe factoriel pour X max(' + str(p) + ')', 1, max_value=p)
        nb_axe_y = st.number_input('Numéro axe factoriel pour Y max(' + str(p) + ')', 2, max_value=p)

        st.title('Graphique des individus')
        x_inertie, y_inertie, plan_inertie = graph_ind(df_acp, coord, acp_, nb_axe_x, nb_axe_y)
        st.write(x_inertie)
        st.write(y_inertie)
        st.write(plan_inertie)

        corr = correlation(acp_, p, df_acp, nb_axe_x, nb_axe_y, eigval)
        st.title('Cercle des corrélations')
        cercle_corr(corr)
    else:
        st.error("Nombre d'individus inférieur au nombre de variables")


	
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


def choix_axe(acp, p):
    axe_var = np.round(acp.explained_variance_ratio_*100, 2)
    axe_var_sum = np.cumsum(axe_var)
    axe = np.arange(1,p+1)

    fig = px.bar(x = axe, y = axe_var, text = axe_var,
                labels=dict(x="Numéro de l'axe", y='Variance expliqué en %')
                )
    fig.update_traces(textposition='outside')
    fig.add_trace(go.Scatter(x=axe, y=axe_var, mode='lines'))
    st.write(fig)


def correlation(acp, p, df, nb_x_axe, nb_y_axe, eigval):
    sqrt_eigval = np.sqrt(eigval)
    corvar = np.zeros((p,p))
    for k in range(p):
        corvar[:,k] = acp.components_[k,:] * sqrt_eigval[k]

    corr = pd.DataFrame({'id':df.columns, 'Axe_' + str(nb_x_axe):corvar[:,nb_x_axe-1], 'Axe_' + str(nb_y_axe):corvar[:,nb_y_axe-1]})
    return corr


def cercle_corr(corr):
    fig = px.scatter(corr, x = corr.iloc[:,1], y = corr.iloc[:,2], text= 'id',
                     labels=dict(x=corr.columns[1], y=corr.columns[2])   
                    )

    fig.update_traces(textposition='top center')
    # Set axes properties
    fig.update_xaxes(range=[-1, 1], zeroline=False)
    fig.update_yaxes(range=[-1, 1])

    # Add circles
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=-1, y0=-1, x1=1, y1=1,
        line_color="LightSeaGreen",
    )

    # Add shapes
    for i in range(len(corr.iloc[:,1])):
        fig.add_shape(type="line",
            x0=0, y0=0, x1=corr.iloc[:,1][i], y1=corr.iloc[:,2][i],
            line=dict(color="RoyalBlue",width=2)
        )

    # Set figure size
    fig.update_layout(width=700, height=700)

    st.write(fig)


def graph_ind(df_acp, coord, acp, nb_x_axe, nb_y_axe):  
    x_inertie = str(round(acp.explained_variance_ratio_[nb_x_axe-1]*100, 2)) + " % des données expliquées sur l'axe n°" + str(nb_x_axe)
    y_inertie = str(round(acp.explained_variance_ratio_[nb_y_axe-1]*100, 2)) + " % des données expliquées sur l'axe n°" + str(nb_y_axe)
    plan_inertie = str(round(acp.explained_variance_ratio_[nb_x_axe-1]*100 + acp.explained_variance_ratio_[nb_y_axe-1]*100, 2)) + " % des données expliquées sur ce plan factoriel"
    fig = px.scatter(x = coord[:,nb_x_axe-1], y = coord[:,nb_y_axe-1], text=df_acp.index,
                 labels=dict(x='Axe n°' + str(nb_x_axe), y='Axe n°' + str(nb_y_axe))
                )

    fig.update_traces(textposition='top center')
    st.write(fig)
    return x_inertie, y_inertie, plan_inertie

if __name__ == "__main__":
    main()
