import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_validate, KFold

# funzione grafico predizione vs valori reali
def graficoPredizioni(y_train, y_test, trainPrediction, testPrediction):
    f, axes = plt.subplots(1, 2, figsize=(24, 12))
    axes[0].scatter(y_train, trainPrediction, color="blue")
    axes[0].plot(y_train, y_train, 'g-', linewidth=5)
    axes[0].set_xlabel("Valori reali della variabile di risposta (train)")
    axes[0].set_ylabel("Valori predetti della variabile di risposta (train)")
    axes[1].scatter(y_test, testPrediction, color="red")
    axes[1].plot(y_test, y_test, 'g-', linewidth=5)
    axes[1].set_xlabel("Valori reali della variabile di risposta (test)")
    axes[1].set_ylabel("Valori predetti della variabile di risposta (test)")
    plt.show()

# funzione valutazione modello
def valutazioneModello(y_train, y_test, trainPrediction, testPrediction):
    print()
    print("[TRAIN] MSE:", round(mean_squared_error(y_train, trainPrediction), 4))
    print("[TRAIN] MAE:", round(mean_absolute_error(y_train, trainPrediction), 4))
    print("[TRAIN] R^2:", round(r2_score(y_train, trainPrediction), 4))
    print()
    print("[TEST] MSE:", round(mean_squared_error(y_test, testPrediction), 4))
    print("[TEST] MAE:", round(mean_absolute_error(y_test, testPrediction), 4))
    print("[TEST] R^2:", round(r2_score(y_test, testPrediction), 4))
    print()


# funzione per k-fold cross validation
def kfoldCrossValidation(model, X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=None)
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    results_kfold = cross_validate(model, X, y, cv=kf, scoring=scoring)

    mean_r2 = round(results_kfold['test_r2'].mean(), 4)
    mean_mse = -round(results_kfold['test_neg_mean_squared_error'].mean(), 4)
    mean_mae = -round(results_kfold['test_neg_mean_absolute_error'].mean(), 4)

    print("[K-FOLD CV] MSE:", mean_mse)
    print("[K-FOLD CV] MAE:", mean_mae)
    print("[K-FOLD CV] R^2:", mean_r2)

    return mean_mse, mean_mae, mean_r2