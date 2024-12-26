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

