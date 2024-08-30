import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
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

varArr = np.array(X.columns)

ridgeReg = Ridge(alpha=0).fit(X_train, y_train)

# Memorizzo le predizioni per ogni valore lambda/alpha con i valori di R2 e lambda
ridgeTrainPred = []
ridgeTestPred = []
ridgeR2score = []
lambdaVal = []

ridgeDF = pd.DataFrame({'variable': varArr, 'estimate': ridgeReg.coef_.ravel()})

# valori di lambda da 0 a 2000 con incrementi di 1
lambdas = np.arange(0, 2000, 1)

for alpha in lambdas:
    ridgeReg = Ridge(alpha=alpha)
    ridgeReg.fit(X_train, y_train)
    var_name = 'estimate' + str(alpha)
    ridgeDF[var_name] = ridgeReg.coef_.ravel()
    ridgeTrainPred.append(ridgeReg.predict(X_train))
    ridgeTestPred.append(ridgeReg.predict(X_test))
    ridgeR2score.append(ridgeReg.score(X_train, y_train))
    lambdaVal.append(alpha)

ridgeDF = ridgeDF.set_index('variable').T.rename_axis('estimate').rename_axis(1).reset_index()

bestRidgeR2 = max(ridgeR2score)

# Utilizzo il modello di regressione Ridge migliore per le predizioni
trainPredictionRidge = ridgeTrainPred[ridgeR2score.index(bestRidgeR2)]
testPredictionRidge = ridgeTestPred[ridgeR2score.index(bestRidgeR2)]

# Grafico che mostra l'andamento dei coefficienti beta per ogni predittore al variare di lambda
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(ridgeDF.gym, 'r', ridgeDF.elevator, 'g', ridgeDF.hot_tub, 'b', ridgeDF.pool, 'c', ridgeDF.internet, 'y',
        ridgeDF.bedrooms, 'm')
ax.axhline(y=0, color='black', linestyle='--')
ax.set_xlabel("Lambda")
ax.set_ylabel("Beta Estimate")
ax.set_title("Ridge Regression", fontsize=16)
ax.legend(labels=['gym', 'elevator', 'hot_tub', 'pool', 'internet', 'bedrooms'])
ax.grid(True)
plt.show()

# Grafico delle predizioni vs i valori reali
graficoPredizioni(y_train, y_test, trainPredictionRidge, testPredictionRidge)

# Valutazione del modello
valutazioneModello(y_train, y_test, trainPredictionRidge, testPredictionRidge)

# K-Fold Cross Validation
mse, mae, r2 = kfoldCrossValidation(ridgeReg, X, y)

'''results_df = pd.DataFrame({'Modello': ['RidgeRegression'], 'R2': [r2], 'MSE': [mse], 'MAE':[mae]})
results_df.to_csv('results.csv', mode='a', header=False, index=False)'''