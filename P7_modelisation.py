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

# %% Création du modèle

# %% Entraînement du modèle
# Instancier le modèle avec pondération des classes
model = lgb.LGBMClassifier(
    class_weight='balanced',  # Gestion automatique du déséquilibre
    random_state=42
)

# Entraîner le modèle
model.fit(X_train, y_train)

# %% Évaluation sur le jeu de test
# Prédictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probabilité pour la classe 1
y_pred_proba[:5]

# Affichage des performances
# %%
# 1. Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Prédictions')
plt.ylabel('Réel')
plt.title('Matrice de confusion')
plt.show();

# %%
# 2. Classification report
print("Classification Report :\n", classification_report(y_test, y_pred))

# %%
# 3. Courbe ROC et AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel("Taux de Faux Positifs (FPR)")
plt.ylabel("Taux de Vrais Positifs (TPR)")
plt.title("Courbe ROC")
plt.legend(loc="lower right")
plt.show();

# %%
# 4. Courbe Précision-Rappel
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, color='green')
plt.xlabel("Rappel")
plt.ylabel("Précision")
plt.title("Courbe Précision-Rappel")
plt.show();

# UN AUC-ROC de 0.74 montre que les classes sont bien séparées

# %% Évaluation avec les métriques F2 et F3

f2 = fbeta_score(y_test, y_pred, beta=2)  # F2-score
f3 = fbeta_score(y_test, y_pred, beta=3)  # F3-score

# Affichage des résultats
print("=== Scores supplémentaires ===")
print(f"F2-score : {f2:.4f}")
print(f"F3-score : {f3:.4f}")

# F2-score : 0.3984
# F3-score : 0.4945
# le score F2 est plus élevé que le F1, ce qui signifie que 
# le modèle est plus performant pour la classe minoritaire


# %% Optimisation des hyperparamètres
#  Récupérer les importances des features
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': model.feature_importances_}).sort_values(by='Importance', ascending=False)

# Affichage des 10 features les plus importantes
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:10][::-1], feature_importances['Importance'][:10][::-1], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 des features les plus importantes")
plt.show();

# Afficher tout le tableau si nécessaire
print("Feature Importances:\n", feature_importances)

# %% Réduction des features et réentraînement
# Liste des 10 features les plus importantes
top_features = feature_importances['Feature'][:10].tolist()

# Sous-ensemble des données d'entraînement et de test
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

# Réinitialisation du modèle avec les meilleurs paramètres trouvés
optimized_model = lgb.LGBMClassifier(
    learning_rate=0.1,
    max_depth=20,
    n_estimators=200,
    num_leaves=70,
    class_weight='balanced',  # Pour compenser le déséquilibre des classes
    random_state=42
)

# Réentraînement
optimized_model.fit(X_train_top, y_train)

# Prédictions
y_pred_top = optimized_model.predict(X_test_top)
y_pred_proba_top = optimized_model.predict_proba(X_test_top)[:, 1]


# %%
# Affichage des performances
f2_top = fbeta_score(y_test, y_pred_top, beta=2)
f3_top = fbeta_score(y_test, y_pred_top, beta=3)
print(f"F2-score (top features) : {f2_top:.4f}")
print(f"F3-score (top features) : {f3_top:.4f}")


# %% Voir la répartition des vraies et fausses prédictions par classe
## Classification report
report = classification_report(y_test, y_pred_top)
print("Classification Report:\n", report)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_top)
print("Confusion Matrix:\n", cm)


# %%
# Ajustement du seuil (exemple à 0.3 pour favoriser la classe minoritaire)
y_pred_adjusted = (y_pred_proba_top >= 0.3).astype(int)

# Calcul du F2-score avec le nouveau seuil
f2_adjusted = fbeta_score(y_test, y_pred_adjusted, beta=2)
f3_adjusted = fbeta_score(y_test, y_pred_adjusted, beta=3)
print(f"F2-score avec seuil ajusté : {f2_adjusted:.4f}")
print(f"F3-score avec seuil ajusté : {f3_adjusted:.4f}")


# %%
# Sauvegarde du modèle
joblib.dump(optimized_model, 'lgb_model.pkl')

# %%# Chargement du modèle
model_loaded = joblib.load('lgb_model.pkl')

# Vérification du chargement avec une prédiction
sample_data = np.array([[0.1, 30, 0.3, 100, 500, 250, 1000, 1, 0, 2]])  # Exemple de données
prediction = model_loaded.predict(sample_data)
print("Prédiction du modèle chargé :", prediction)
