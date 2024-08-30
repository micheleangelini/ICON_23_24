import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

listingsProcessed = pd.read_csv('datasets/listingsProcessed.csv')

DFClustering = listingsProcessed[
    ['bedrooms', 'price', 'room_type_Entire_home_apt', 'room_type_Private_room',
     'room_type_Shared_room']
]

# Unione delle colonne room_type_Private_room e room_type_Shared_room in una sola colonna
DFClustering['room_type_Room'] = DFClustering['room_type_Private_room'] | DFClustering['room_type_Shared_room']
DFClustering.drop(['room_type_Private_room', 'room_type_Shared_room'], axis=1, inplace=True)

# Metodo del gomito per determinare il numero ottimale di cluster
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(DFClustering)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss)
plt.title('Metodo del gomito')
plt.xlabel('Numero di cluster')
plt.ylabel('WCSS - Within-Cluster Sum of Squares')
plt.show()

# K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(DFClustering)

# Aggiunta delle etichette dei cluster al dataset
DFClustering['cluster'] = kmeans.labels_

# Valutazione del modello usando il silhouette score
sil_score = silhouette_score(DFClustering, kmeans.labels_)
print(f'Silhouette Score: {round(sil_score, 3)}')

# Visualizzazione
pca = PCA(2) # Riduzione delle dimensioni a 2
X_pca = pca.fit_transform(DFClustering)
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
plt.title('K-means')
plt.xlabel('Prima componente principale')
plt.ylabel('Seconda componente principale')
plt.colorbar(label='Etichette dei cluster')
plt.show()

# Coordinate dei centroidi per ciascun cluster
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=DFClustering.columns[:-1])
print("Coordinate dei centrodi per ciascun cluster:")
print(centroids)

