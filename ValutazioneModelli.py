import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('SupervisedLearning/results.csv')

# Grafico R^2
df_sorted_r2 = df.sort_values(by='R^2', ascending=True)
plt.figure(figsize=(10, 5))
plt.barh(df_sorted_r2['Modello'], df_sorted_r2['R^2'], color='skyblue')
plt.title('Comparazione delle Prestazioni dei Modelli (R^2)')
plt.xlabel('R^2')
plt.ylabel('Modello')
plt.xlim(0, 1)
for index, value in enumerate(df_sorted_r2['R^2']):
    plt.text(value, index, f'{value:.2f}', va='center')
plt.show()

# Grafico MSE
df_sorted_mse = df.sort_values(by='MSE', ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(df_sorted_mse['Modello'], df_sorted_mse['MSE'], color='salmon')
plt.title('Comparazione delle Prestazioni dei Modelli (MSE)')
plt.xlabel('MSE')
plt.ylabel('Modello')
for index, value in enumerate(df_sorted_mse['MSE']):
    plt.text(value, index, f'{value:.0f}', va='center')
plt.show()

# Grafico MAE
df_sorted_mae = df.sort_values(by='MAE', ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(df_sorted_mae['Modello'], df_sorted_mae['MAE'], color='lightgreen')
plt.title('Comparazione delle Prestazioni dei Modelli (MAE)')
plt.xlabel('MAE')
plt.ylabel('Modello')
for index, value in enumerate(df_sorted_r2['MAE']):
    plt.text(value, index, f'{value:.0f}', va='center')
plt.show()