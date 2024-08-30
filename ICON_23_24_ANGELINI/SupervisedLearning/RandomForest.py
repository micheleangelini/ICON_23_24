import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from Utils import *

# Caricamento del dataset
listingsProcessedDF = pd.read_csv('../datasets/listingsProcessed.csv')

# Separazione delle variabili predittive e risposta in X e y
X = pd.DataFrame(listingsProcessedDF)
X.drop('price', axis=1, inplace=True)
y = pd.DataFrame(listingsProcessedDF["price"])

# Scaling dei dati
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

# Separazione dei dati in insieme di addestramento (train) ed in insieme di test (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Il codice per la ricerca degli iperparametri è commentato per evitare la ri-esecuzione che richiede tempo
'''
# Definizione del range degli iperparametri per la ricerca randomizzata
random_grid = {
    'n_estimators': [100, 200, 500, 1000, 1500, 2000, 2500],
    'max_features': ['sqrt', 'log2', 2, 5],
    'max_depth': [None, 10, 20, 40, 70, 100, 150],
    'min_samples_split': [2, 5, 10, 20, 40, 50],
    'min_samples_leaf': [1, 2, 4, 10, 20],
    'bootstrap': [True, False],
    'max_leaf_nodes': [None, 10, 20, 30, 50, 70, 100]
}

# Creazione del modello RandomForestRegressor
rf_tune = RandomForestRegressor()

rf_random = RandomizedSearchCV(
    estimator=rf_tune,
    param_distributions=random_grid,
    n_iter=200,
    cv=10,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    error_score='raise'
)

# Addestramento del modello con RandomizedSearchCV
rf_random.fit(X_train, y_train.values.ravel())

# Stampa dei migliori iperparametri trovati
print(f"Parametri migliori: {rf_random.best_params_}")
print(f"Stimatore migliore: {rf_random.best_estimator_}")
'''

best_params = {
    'n_estimators': 1000,
    'min_samples_split': 12,
    'min_samples_leaf': 4,
    'max_leaf_nodes': None,
    'max_depth': 90,
    'bootstrap': True}

RF = RandomForestRegressor(**best_params).fit(X_train, y_train.values.ravel())

trainPredictionRF = RF.predict(X_train)
testPredictionRF = RF.predict(X_test)

# Analisi importanza delle features
importancesRF = RF.feature_importances_
feat_imp = pd.DataFrame({'importance': importancesRF})
feat_imp['feature'] = X_train.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp.set_index('feature', drop=True, inplace=True)
print(feat_imp)

# Grafico delle predizioni vs i valori reali
graficoPredizioni(y_train, y_test, trainPredictionRF, testPredictionRF)

# Valutazione del modello
valutazioneModello(y_train, y_test, trainPredictionRF, testPredictionRF)

# K-Fold Cross Validation
mse, mae, r2 = kfoldCrossValidation(RF, X, y.values.ravel())

'''results_df = pd.DataFrame({'Modello': ['RandomForest'], 'R2': [r2], 'MSE': [mse], 'MAE': [mae]})
results_df.to_csv('results.csv', mode='a', header=False, index=False)'''