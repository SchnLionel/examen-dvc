import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

params = yaml.safe_load(open("params.yaml"))
df = pd.read_csv('data/raw/raw.csv')

# --- LA CORRECTION EST ICI ---
# On supprime la colonne date car elle contient du texte
if 'date' in df.columns:
    df = df.drop(columns=['date'])
# -----------------------------

X = df.drop('silica_concentrate', axis=1)
y = df['silica_concentrate']

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=params['split']['test_size'], 
    random_state=params['split']['random_state']
)

os.makedirs('data/processed', exist_ok=True)
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Fini ! Fichiers créés sans la colonne date.")
