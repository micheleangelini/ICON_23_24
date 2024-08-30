import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from Utils import *

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

listingsProcessedDF = pd.read_csv('../datasets/listingsProcessed.csv')

# Separazione delle variabili predittive e risposta in X e y
X = pd.DataFrame(listingsProcessedDF)
X.drop('price', axis=1, inplace=True)
y = pd.DataFrame(listingsProcessedDF['price'])

# Scaling dei dati
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))

# Separazione dei dati in insieme di addestramento (train) ed in insieme di test (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Coefficienti della regressione
print('Intercetta della retta di regressione: b = ', linreg.intercept_)

# Stampa dei coefficienti per ogni predittore
coef_df = pd.DataFrame({'Predittori': X_train.columns, 'Coefficienti': linreg.coef_[0]})
sorted_coef_df = coef_df.sort_values(by="Coefficienti", ascending=False, ignore_index=True)
print(sorted_coef_df)

# Predizione della risposta corrispondente ai predittori
trainPredictionLR = linreg.predict(X_train)
testPredictionLR = linreg.predict(X_test)

# Grafico valori predetti vs valori reali
graficoPredizioni(y_train, y_test, trainPredictionLR, testPredictionLR)

# Valutazione del modello
valutazioneModello(y_train, y_test, trainPredictionLR, testPredictionLR)

# K-Fold Cross Validation
mse, mae, r2 = kfoldCrossValidation(linreg, X, y)

'''results_df = pd.DataFrame({'Modello': ['LinearRegression'], 'R2': [r2], 'MSE': [mse], 'MAE': [mae]})
results_df.to_csv('results.csv', mode='a', header=False, index=False)'''