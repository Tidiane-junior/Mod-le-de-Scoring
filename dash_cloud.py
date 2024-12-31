import streamlit as st
import requests

st.set_page_config(
    layout='wide',
    initial_sidebar_state='expanded',
    page_title="Accueil avec API Cloud")

# --- Initialisation de SessionState ---
if "load_state" not in st.session_state:
    st.session_state.load_state = False

# --- Layout de la page d'accueil ---
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
# auteur 
st.sidebar.write(
    """
    **Auteur :** [Ch. Tidiane THIAM](https://www.linkedin.com/in/cheikh-tid-thiam)
    """
    )

st.sidebar.write(
        """
        Ce tableau de bord de scoring permet :
        - D'explorer les données des clients.
        - Visualiser les défauts de remboursement par tranche d'âge.
        - D'analyser les caractéristiques des clients.
        - De prédire le défaut de remboursement pour un nouveau client.

        """
        )

# URL de l'API déployée sur le cloud
api_url = "https://fastapi-image-71943282713.europe-west9.run.app/predict/"

# --- # Interface utilisateur pour entrer les données ---
st.subheader("Prédiction du défaut de remboursement")
st.write("Veuillez entrer les caractéristiques du client.")

# Reseigner les données du client
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

# Appel de l'API lorsque l'agent clique sur le bouton "Prédire"
if st.button("Prédire"):
    # Préparation des données d'entrée pour l'API
    input_data = {
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

    # Envoie d'une requête POST à l'API
    response = requests.post(api_url, json=input_data)

    # Gérer la réponse de l'API
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prédiction : {result['prediction']}")
        st.info(f"Probabilité : {result['probability']:.2f}")
    else:
        st.error(f"Erreur {response.status_code} : {response.text}")
