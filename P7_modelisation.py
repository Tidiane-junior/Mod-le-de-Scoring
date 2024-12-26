# %%
# libriaries nécessaires pour la manipulation
import numpy as np
import pandas as pd
# Pour la visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer # Encodage des variables catégorielles
from sklearn.impute import SimpleImputer # Gérer les NaNs
from sklearn.model_selection import KFold  # Pour diviser le dataset en pli
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import RocCurveDisplay, roc_auc_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from sklearn.metrics import fbeta_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

import gc  # Pour libérer la mémoire
import os # Gestion du système de fichiers du drive

# Supprimer les avertissements
import warnings
warnings.filterwarnings('ignore')

import joblib

# %%
# Chargement des données
data = pd.read_pickle('train_red_format.pkl')
print('Dimension des données: ', data.shape)
data.head()

# %%
# Distribution de la variable cible
print(data['TARGET'].value_counts(normalize=True))

# Visualisation de la répartition
sns.countplot(data=data, x='TARGET')
plt.title('Distribution de la cible')
plt.show();

# %%
# Séparer les colonnes numériques et catégoriques
df = data.copy()
categorical_cols = df.select_dtypes(include = "object").columns
numeric_cols = df.drop(['TARGET', 'SK_ID_CURR'], axis = 1).select_dtypes(exclude = "object").columns

print(numeric_cols)
print(categorical_cols)

# %%
# Encodage des catégorielles avec One-Hot
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df.info()

# %%
df['NAME_CONTRACT_TYPE_Revolving loans'].unique()


# %% Normalisation des données numériques
# Instancier le scaler
scaler = MinMaxScaler()

# Normaliser uniquement les colonnes numériques
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# %% Séparation en jeux d’entraînement et de test

# Séparation X (features) et y (cible)
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])  # Exclure la cible et l'identifiant
y = df['TARGET']

# Division en jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # Stratify pour garder le déséquilibre
)

print(f"Taille du jeu d'entraînement : {X_train.shape}")
print(f"Taille du jeu de test : {X_test.shape}")


