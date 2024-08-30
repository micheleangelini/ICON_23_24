import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
from Utils import *
import matplotlib.pyplot as plt

# Caricamento del dataset
listingsProcessedDF = pd.read_csv('../datasets/listingsProcessed.csv')

# Separazione delle variabili predittive (X) e risposta (y)
X = pd.DataFrame(listingsProcessedDF)
X.drop('price', axis=1, inplace=True)
y = pd.DataFrame(listingsProcessedDF["price"])

# Scaling dei dati
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

# Separazione dei dati in insieme di addestramento (train) ed in insieme di test (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Griglia dei parametri per RandomizedSearchCV
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0.1, 0.5, 1],
    'n_estimators': [500, 1000, 2000],
    'gamma': [0, 0.1, 0.2]
}
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# RandomizedSearchCV per la ricerca degli iperparametri migliori
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Esecuzione della ricerca - commentato per evitare la ri-esecuzione che richiede tempo
'''random_search.fit(X_train, y_train.values.ravel())
print("Best parameters:", random_search.best_params_)
print("Best estimator:", random_search.best_estimator_)'''


best_params = {
 'subsample': 0.8,
 'reg_lambda': 1,
 'reg_alpha': 1,
 'n_estimators': 1000,
 'max_depth': 4,
 'learning_rate': 0.01,
 'gamma': 0.1,
 'colsample_bytree': 0.5}

# Inizializzazione e addestramento del modello con i migliori iperparametri
xgb_reg = xgb.XGBRegressor(**best_params)
xgb_reg.fit(X_train, y_train.values.ravel())

# Predizioni sul training set e sul test set
trainPredictionXGB = xgb_reg.predict(X_train)
testPredictionXGB = xgb_reg.predict(X_test)

# Analisi importanza delle features
importancesXGB = xgb_reg.feature_importances_
feat_imp = pd.DataFrame({'importance': importancesXGB})
feat_imp['feature'] = X_train.columns
feat_imp.sort_values(by='importance', ascending=False, inplace=True)
feat_imp.set_index('feature', drop=True, inplace=True)
print(feat_imp)

# Grafico delle predizioni vs i valori reali
graficoPredizioni(y_train, y_test, trainPredictionXGB, testPredictionXGB)

# Valutazione del modello
valutazioneModello(y_train, y_test, trainPredictionXGB, testPredictionXGB)

# K-Fold Cross Validation
mse, mae, r2 = kfoldCrossValidation(xgb_reg, X, y.values.ravel())

'''results_df = pd.DataFrame({'Modello': ['XGBoost'], 'R2': [r2], 'MSE': [mse], 'MAE': [mae]})
results_df.to_csv('results.csv', mode='a', header=False, index=False)'''