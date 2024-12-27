import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import requests

st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="Accueil")

# --- Initialisation de SessionState ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# --- Layout de la page d'accueil ---
# Tableau de bord Streamlit
st.title('Tableau de bord - Credit Scoring')
st.subheader('Application d\'aide à la décision de prêt')
st.write("""Cette application assiste l'agent de prêt dans sa décision d'accorder un prêt à un client.
     Pour ce faire, un algorithme d'apprentissage automatique est utilisé pour prédire les difficultés d'un client à rembourser le prêt.
     Pour plus de transparence, cette application fournit également des informations pour expliquer l'algorithme et les prédictions.""")

st.write(" ")
# Configuration des colonnes
col1, col2 = st.columns(2)

with col1:
    st.image("image/logo.png") 

st.write(" ")
# --- Description dans la deuxième colonne ---
with col2:
    st.write(" ")  # Espacement
    st.write(" ")
    st.subheader("Contenu de l'application :")
    st.markdown(
        """
        Cette Application est divisée en trois parties :
        1) **Information générale :** Données sur les clients et explication du modèle utilisé.
        2) **Analyse des clients :** Exploration des caractéristiques et comportements.
        3) **Prédiction :** Évaluation du risque de défaut de remboursement via une API.
        """
    )

# --- Sidebar pour informations supplémentaires ---
st.sidebar.title("À propos du tableau de bord")
st.write(" ")
st.sidebar.write(
    """
    **Auteur :** [Ch. Tidiane THIAM](https://www.linkedin.com/in/cheikh-tid-thiam)
    """
    )
st.write(" ")
st.sidebar.write(
        """
        Ce tableau de bord de scoring permet :
        - D'explorer les données des clients.
        - Visualiser les défauts de remboursement par tranche d'âge.
        - D'analyser les caractéristiques des clients.
        - De prédire le défaut de remboursement pour un nouveau client.

        Les données utilisées sont issues de la compétition Kaggle "Home Credit Default Risk".
        """
        )

# --- Chargement des données ---
# Chargement des données avec pickle
@st.cache_data
def load_data(file_path):
    """Chargement des données avec pickle."""
    return pd.read_pickle(file_path)

# Fonction d'analyse exploratoire
def perform_eda(df):
    '''
    Réalise une analyse exploratoire sur le DataFrame et génère une visualisation.

    Entrée :
        df : DataFrame contenant les données.
    Sortie :
        fig : Figure Plotly affichant les défauts de remboursement par tranche d'âge.
    '''
    # Copier le DataFrame pour éviter de modifier l'original
    df_copy = df.copy()

    # Informations sur l'âge dans un DataFrame séparé
    age_data = df_copy[['TARGET', 'DAYS_BIRTH']]

    # Ajouter une colonne contenant l'âge en années
    age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / -365

    # Diviser les données d'âge par intervalles de 5 ans
    age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

    # Regrouper les données par tranches d'âge et calculer les moyennes pour chaque tranche d'âge
    age_groups = age_data.groupby('YEARS_BINNED').mean()

   # On réinitialise l'index pour rendre YEARS_BINNED une colonne
    age_groups_reset = age_groups.reset_index()
    # Convertir les tranches d'âge en chaînes pour les utiliser comme catégories
    age_groups_reset['YEARS_BINNED'] = age_groups_reset['YEARS_BINNED'].astype(str)

    # Visualisation
    fig = px.bar(
        age_groups_reset,
        x='YEARS_BINNED',  # Tranches d'âge
        y='TARGET',        # Moyenne du taux de défaut
        text='TARGET',     # Afficher les valeurs sur les barres
        labels={'TARGET': "Défaut de remboursement (%)', 'YEARS_BINNED': 'Tranche d'âge (années)"},
        title="Défaut de remboursement par tranche d'âge"
    )

    # Modifier les valeurs pour afficher le pourcentage
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside', marker_color='lightskyblue')
    fig.update_yaxes(tickformat=".0%", title_text="Défaut de remboursement (%)")
    fig.update_layout(xaxis_title="Tranche d'âge (années)", title_x=0.5)

    return fig


# Chargement des données
file_path = 'train_red_format.pkl'
try:
    st.subheader("Chargement des données")
    df = load_data(file_path)
    st.success("Données chargées avec succès !") # Affichage d'un message de succès
    st.write("Aperçu des données :")
    st.write(df.head())
    st.text("Informations sur le DataFrame :")
    buffer = df.info(buf=None)  # Affichage des informations
    st.text(buffer)
except Exception as e:
    st.error(f"Erreur lors du chargement des données : {e}")

# Analyse exploratoire
if st.checkbox("Effectuer une analyse exploratoire (EDA)"):
    st.subheader("Analyse exploratoire des données")
    try:
        fig = perform_eda(df)
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Erreur lors de l'analyse exploratoire : {e}")


# --- Formulaire principal pour les entrées utilisateur ---
st.subheader("Prédiction du défaut de remboursement")
st.write("Veuillez entrer les caractéristiques du client.")

# Collecte des entrées utilisateur
EXT_SOURCE_2 = st.number_input('Valeur de EXT_SOURCE_2'
                               , min_value=0.0, value=0.5, step=0.01)

DAYS_BIRTH = st.number_input('Âge du client (en jours depuis la naissance, valeur négative)'
                             , min_value=-30000, max_value=0, value=-10000, step=100)

EXT_SOURCE_3 = st.number_input('Valeur de EXT_SOURCE_3'
                               , min_value=0.0, value=0.5, step=0.01)

bureau_DAYS_CREDIT_max = st.number_input('Durée maximale des crédits précédents (en jours)'
                                         , min_value=-5000.0, value=-1000.0, step=100.0)

bureau_DAYS_CREDIT_min = st.number_input('Durée minimale des crédits précédents (en jours)'
                                         , min_value=-5000.0, value=-1500.0, step=100.0)

bureau_DAYS_CREDIT_UPDATE_mean = st.number_input('Moyenne des mises à jour des crédits (en jours)', min_value=-5000.0, value=-500.0, step=100.0)
bureau_DAYS_CREDIT_mean = st.number_input('Durée moyenne des crédits précédents (en jours)', min_value=-5000.0, value=-1000.0, step=100.0)
bureau_CREDIT_ACTIVE_Closed_mean = st.number_input('Proportion moyenne des crédits fermés', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
CODE_GENDER_M = st.number_input('Code genre (1 pour homme, 0 pour femme)', min_value=0, max_value=1, value=0, step=1)
bureau_CREDIT_ACTIVE_Active_mean = st.number_input('Proportion moyenne des crédits actifs', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Envoi des données à l'API
if st.button('Prédire'):
    data = {
        "EXT_SOURCE_2": EXT_SOURCE_2,
        "DAYS_BIRTH": DAYS_BIRTH,
        "EXT_SOURCE_3": EXT_SOURCE_3,
        "bureau_DAYS_CREDIT_max": bureau_DAYS_CREDIT_max,
        "bureau_DAYS_CREDIT_min": bureau_DAYS_CREDIT_min,
        "bureau_DAYS_CREDIT_UPDATE_mean": bureau_DAYS_CREDIT_UPDATE_mean,
        "bureau_DAYS_CREDIT_mean": bureau_DAYS_CREDIT_mean,
        "bureau_CREDIT_ACTIVE_Closed_mean": bureau_CREDIT_ACTIVE_Closed_mean,
        "CODE_GENDER_M": CODE_GENDER_M,
        "bureau_CREDIT_ACTIVE_Active_mean": bureau_CREDIT_ACTIVE_Active_mean,
    }

    try:
        # Assurez-vous que l'URL correspond à l'endpoint `/predict`
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Résultat de la prédiction : {result['prediction']}")
        else:
            st.write(f"Erreur {response.status_code} : {response.text}")
    except requests.exceptions.RequestException as e:
        st.write(f"Erreur de connexion à l'API : {e}")

