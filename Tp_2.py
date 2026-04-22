# ============================================================
# TP 2 — Régression Linéaire & Logistique
# Datasets : auto-mpg.csv  |  binary.csv
# ============================================================
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')
 
# ============================================================
# PARTIE 1 — RÉGRESSION LINÉAIRE  (auto-mpg.csv)
# ============================================================
print("=" * 60)
print("PARTIE 1 : RÉGRESSION LINÉAIRE — auto-mpg")
print("=" * 60)
 
# ----- 1.1 Chargement et exploration -----
df = pd.read_csv("auto-mpg.csv", na_values="?")
print("\n--- Aperçu ---")
print(df.head())
print("\n--- Infos ---")
print(df.info())
print("\n--- Statistiques descriptives ---")
print(df.describe())
print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())
 
# ----- 1.2 Prétraitement -----
# Suppression de la colonne 'car name' (non numérique)
df.drop(columns=["car name"], inplace=True, errors="ignore")
 
# Suppression des lignes avec valeurs manquantes
df.dropna(inplace=True)
 
# Vérification des types
df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)
 
print(f"\nDimensions après nettoyage : {df.shape}")
 
# ----- 1.3 Visualisations -----
plt.figure(figsize=(12, 5))
 
plt.subplot(1, 2, 1)
df["mpg"].hist(bins=20, color="steelblue", edgecolor="white")
plt.title("Distribution de mpg")
plt.xlabel("mpg")
plt.ylabel("Fréquence")
 
plt.subplot(1, 2, 2)
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.savefig("correlation_autoMpg.png", dpi=100)
plt.show()
print("→ Figure sauvegardée : correlation_autoMpg.png")
 
# ----- 1.4 Séparation features / cible -----
X = df.drop(columns=["mpg"])
y = df["mpg"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# Normalisation
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
 
# ----- 1.5 Entraînement -----
model_lr = LinearRegression()
model_lr.fit(X_train_sc, y_train)
 
# ----- 1.6 Évaluation -----
y_pred = model_lr.predict(X_test_sc)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
 
print("\n--- Résultats Régression Linéaire ---")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
 
# Coefficients
coeff_df = pd.DataFrame({
    "Feature"    : X.columns,
    "Coefficient": model_lr.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print("\n--- Coefficients ---")
print(coeff_df.to_string(index=False))
 
# ----- 1.7 Graphique prédictions vs réelles -----
plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color="steelblue")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Valeurs réelles (mpg)")
plt.ylabel("Valeurs prédites (mpg)")
plt.title(f"Régression Linéaire — R² = {r2:.3f}")
plt.tight_layout()
plt.savefig("pred_vs_real_mpg.png", dpi=100)
plt.show()
print("→ Figure sauvegardée : pred_vs_real_mpg.png")
 
 
# ============================================================
# PARTIE 2 — RÉGRESSION LOGISTIQUE  (binary.csv)
# ============================================================
print("\n" + "=" * 60)
print("PARTIE 2 : RÉGRESSION LOGISTIQUE — binary")
print("=" * 60)
 
# ----- 2.1 Chargement et exploration -----
df2 = pd.read_csv("binary.csv")
print("\n--- Aperçu ---")
print(df2.head())
print("\n--- Infos ---")
print(df2.info())
print("\n--- Distribution de la cible ---")
print(df2["admit"].value_counts())
print(df2["admit"].value_counts(normalize=True).round(3))
 
# ----- 2.2 Visualisation -----
plt.figure(figsize=(12, 4))
 
plt.subplot(1, 3, 1)
df2["gre"].hist(bins=20, color="coral", edgecolor="white")
plt.title("Distribution GRE")
 
plt.subplot(1, 3, 2)
df2["gpa"].hist(bins=20, color="mediumpurple", edgecolor="white")
plt.title("Distribution GPA")
 
plt.subplot(1, 3, 3)
df2["admit"].value_counts().plot(kind="bar", color=["#E24B4A", "#1D9E75"])
plt.title("Répartition admit (0/1)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("eda_binary.png", dpi=100)
plt.show()
print("→ Figure sauvegardée : eda_binary.png")
 
# ----- 2.3 Prétraitement -----
df2.dropna(inplace=True)
 
X2 = df2.drop(columns=["admit"])
y2 = df2["admit"]
 
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)
 
scaler2 = StandardScaler()
X2_train_sc = scaler2.fit_transform(X2_train)
X2_test_sc  = scaler2.transform(X2_test)
 
# ----- 2.4 Entraînement -----
model_log = LogisticRegression(random_state=42, max_iter=1000)
model_log.fit(X2_train_sc, y2_train)
 
# ----- 2.5 Évaluation -----
y2_pred      = model_log.predict(X2_test_sc)
y2_pred_prob = model_log.predict_proba(X2_test_sc)[:, 1]
 
acc = accuracy_score(y2_test, y2_pred)
cm  = confusion_matrix(y2_test, y2_pred)
 
print("\n--- Résultats Régression Logistique ---")
print(f"Accuracy : {acc:.4f}")
print("\n--- Rapport de classification ---")
print(classification_report(y2_test, y2_pred))
 
# Matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="Blues")
plt.title("Matrice de confusion — Régression Logistique")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.show()
print("→ Figure sauvegardée : confusion_matrix.png")
 
# Coefficients
coeff_log = pd.DataFrame({
    "Feature"    : X2.columns,
    "Coefficient": model_log.coef_[0]
}).sort_values("Coefficient", ascending=False)
print("\n--- Coefficients (log-odds) ---")
print(coeff_log.to_string(index=False))
 
print("\n✓ TP 2 terminé avec succès.")
 
