"""Exploration des données de churn.
Les graphiques sont enregistrés dans data/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # fonctionne aussi sur une VM sans écran
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

df = pd.read_csv(DATA / "raw" / "churn.csv")

print(df.head())

print("\nDimensions :")
print(df.shape)

print("\nTypes :")
print(df.dtypes)

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nStatistiques :")
print(df.describe())

print("\nRépartition du churn :")
print(df["churn"].value_counts())
print(df["churn"].value_counts(normalize=True).round(3))

plt.figure()
sns.countplot(data=df, x="churn")
plt.title("Répartition des clients selon le churn")
plt.savefig(DATA / "churn_distribution.png", bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Corrélations")
plt.savefig(DATA / "correlation.png", bbox_inches="tight")
plt.close()

print("\nGraphiques enregistrés : data/churn_distribution.png, data/correlation.png")
