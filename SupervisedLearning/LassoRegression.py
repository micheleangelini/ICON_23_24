'''import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from Utils import *

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

# Definizione del range di alpha su scala logaritmica
alphas = np.logspace(-4, 1, 100)

# LassoCV con validazione incrociata
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_cv.fit(X_train, y_train.values.ravel())

# Miglior valore di alpha trovato
best_alpha = lasso_cv.alpha_

# Uso del miglior modello per predizioni
trainPredictionLasso = lasso_cv.predict(X_train)
testPredictionLasso = lasso_cv.predict(X_test)

# Costruzione del DataFrame dei coefficienti
lassoDF = pd.DataFrame({'variable': X.columns, 'estimate': lasso_cv.coef_})

# Grafico che mostra l'andamento dei coefficienti beta per ogni predittore al variare di alpha
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(lassoDF.gym,'r',lassoDF.elevator,'g',lassoDF.hot_tub,'b', lassoDF.pool, 'c', lassoDF.internet,'y',lassoDF.bedrooms,'m')
ax.set_xlabel("Lambda")
ax.set_xticklabels(np.arange(-1, 100, 1))
ax.set_ylabel("Beta Estimate")
ax.set_title("Lasso Regression", fontsize=16)
ax.legend(labels=['gym','elevator','hot_tub', 'pool','internet','bedrooms'])
ax.grid(True)

# Grafico delle predizioni vs i valori reali
graficoPredizioni(y_train, y_test, trainPredictionLasso, testPredictionLasso)

# Valutazione del modello
valutazioneModello(y_train, y_test, trainPredictionLasso, testPredictionLasso)

# K-Fold Cross Validation
mse, mae, r2 = kfoldCrossValidation(lasso_cv, X, y.values.ravel())

results_df = pd.DataFrame({'Modello': ['LassoRegression'], 'R2': [r2], 'MSE': [mse], 'MAE': [mae]})
results_df.to_csv('results.csv', mode='a', header=False, index=False)'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
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

# Definizione del range di alpha su scala logaritmica
alphas = np.logspace(-4, 1, 100)

# LassoCV con validazione incrociata
lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=42)
lasso_cv.fit(X_train, y_train.values.ravel())

# Miglior valore di alpha trovato
best_alpha = lasso_cv.alpha_

# Uso del miglior modello per predizioni
trainPredictionLasso = lasso_cv.predict(X_train)
testPredictionLasso = lasso_cv.predict(X_test)

# Costruzione del DataFrame dei coefficienti
lassoDF = pd.DataFrame({'variable': X.columns, 'estimate': lasso_cv.coef_})

# Selezione delle variabili di interesse
variables_of_interest = ['gym', 'elevator', 'hot_tub', 'pool', 'internet', 'bedrooms']

# Controlla che tutte le variabili siano presenti nel DataFrame
variables_of_interest = [var for var in variables_of_interest if var in X.columns]

# Generazione del grafico che mostra l'andamento dei coefficienti per variabili specifiche al variare di alpha
coefs = {var: [] for var in variables_of_interest}

# Fit del modello Lasso per ogni alpha e memorizzazione dei coefficienti delle variabili di interesse
for alpha in alphas:
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train.values.ravel())
    for var in variables_of_interest:
        coefs[var].append(lasso.coef_[X.columns.get_loc(var)])

# Creazione del grafico che mostra l'andamento dei coefficienti beta per ogni predittore al variare di alpha
fig, ax = plt.subplots(figsize=(10, 10))
for var in variables_of_interest:
    ax.plot(alphas, coefs[var], label=var)
ax.axhline(y=0, color='black', linestyle='--')
ax.set_xscale('log')
ax.set_xlabel("Alpha")
ax.set_ylabel("Beta Estimate")
ax.set_title("Lasso Regression", fontsize=16)
ax.legend()
ax.grid(True)
plt.show()