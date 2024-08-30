import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from catboost import CatBoostRegressor
from Utils import *

# Caricamento dei dati
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


catBoost = CatBoostRegressor(silent=True)

# Griglia di iperparametri per RandomizedSearchCV - commentata per evitare la ri-esecuzione che richiede tempo
'''
param_distributions = {
    'iterations': [950, 1000, 1050],
    'depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.005, 0.01, 0.015],
    'l2_leaf_reg': [4, 5, 6, 7, 8],
    'bagging_temperature': [0.1, 0.15, 0.2],
    'border_count': [19, 20, 21],
    'random_strength': [1, 2, 3, 4, 5],
    'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
}

# RandomizedSearchCV con 5-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=catBoost,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit del modello di ricerca RandomizedSearchCV
random_search.fit(X_train, y_train)

# Stampa dei migliori parametri trovati
print("\nMigliori parametri trovati con RandomizedSearchCV:\n", random_search.best_params_)
'''

params = {
    'random_strength': 3,
    'learning_rate': 0.01,
    'l2_leaf_reg': 5,
    'iterations': 1000,
    'grow_policy': 'SymmetricTree',
    'depth': 6,
    'border_count': 20,
    'bagging_temperature': 0.1
}

# Creazione e addestramento del modello con i migliori parametri
CatB = CatBoostRegressor(**params, silent=True)
CatB.fit(X_train, y_train, plot=True)

# Previsione sul set di addestramento e di test
trainPredictionCB = CatB.predict(X_train)
testPredictionCB = CatB.predict(X_test)

# Analisi importanza delle features
importancesCB = CatB.feature_importances_
feat_imp = pd.DataFrame({'importance': importancesCB})
feat_imp['feature'] = X_train.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp.set_index('feature', drop=True, inplace=True)
print(feat_imp)

# Grafico delle predizioni vs i valori reali
graficoPredizioni(y_train, y_test, trainPredictionCB, testPredictionCB)

# Valutazione del modello
valutazioneModello(y_train, y_test, trainPredictionCB, testPredictionCB)

# K-Fold Cross Validation
mse, mae, r2 = kfoldCrossValidation(CatB, X, y)

'''results_df = pd.DataFrame({'Modello': ['CatBoost'], 'R2': [r2], 'MSE': [mse], 'MAE': [mae]})
results_df.to_csv('results.csv', mode='a', header=False, index=False)'''