# Obiettivo: individuare le caratteristiche / strutture / servizi di una proprietà che influenzano il prezzo

# DATA PREPROCESSING
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

sb.set()

listingsDF = pd.read_csv('datasets/listings.csv')
listingsDF.head()

print("Data type : ", type(listingsDF))
print("Data dims : ", listingsDF.shape)

# Selezione delle colonne rilevanti
listingDF = listingsDF[
    ['id', 'name', 'summary', 'longitude', 'latitude', 'space', 'description', 'instant_bookable',
     'neighborhood_overview', 'neighbourhood_cleansed', 'host_id', 'host_name', 'host_since',
     'host_response_time', 'street', 'zipcode', 'review_scores_rating', 'property_type', 'room_type', 'accommodates',
     'bathrooms', 'bedrooms', 'beds', 'reviews_per_month', 'amenities', 'cancellation_policy', 'number_of_reviews',
     'price']
]
listingDF.head()

# Sostituzione dei valori mancanti con 0
listingDF.fillna(0, inplace=True)

# Conversione dei prezzi da string a float
priceDF = listingDF['price']
prices = []

for p in priceDF:
    p = float(p[1:].replace(',', ''))
    prices.append(p)

listingDF['price'] = prices

# Rimozione degli annunci che hanno un numero di camere da letto, bagni, numero di persone che possono ospitare, prezzo,
# letti, nessuna valutazione e nessuna recensione in un mese
listingDF = listingDF[listingDF.bedrooms > 0]
listingDF = listingDF[listingDF.bathrooms > 0]
listingDF = listingDF[listingDF.accommodates > 0]
listingDF = listingDF[listingDF.price > 0]
listingDF = listingDF[listingDF.beds > 0]
listingDF = listingDF[listingDF.review_scores_rating > 0]
listingDF = listingDF[listingDF.reviews_per_month > 0]

# ANALISI DEGLI ANNUNCI IN BASE AL room_type
print("Number of room types :", len(listingDF["room_type"].unique()))
print(listingDF["room_type"].value_counts())

countplot_room_type = sb.catplot(
    x="room_type",
    hue="room_type",
    data=listingDF,
    kind="count",
    palette="Set2"
)
countplot_room_type.fig.suptitle("Numero degli annunci per tipo di stanza")
plt.show()

# ANALISI DEGLI ANNUNCI IN BASE AL property_type
print("Number of property types :", len(listingDF["property_type"].unique()))
print(listingDF["property_type"].value_counts())
countplot_property_type = sb.catplot(
    x="property_type",
    hue='property_type',
    data=listingDF,
    kind="count",
    palette="Set2",
    height=8,
    aspect=2
)
countplot_property_type.fig.suptitle("Numero degli annunci per tipo di proprietà")
plt.show()

# ANALISI DEI PREZZI PER I DIVERSI TIPI DI STANZE E PROPRIETÀ
roomProperty_DF = listingDF.groupby(['property_type', 'room_type']).price.mean()
roomProperty_DF = roomProperty_DF.reset_index()
roomProperty_DF = roomProperty_DF.sort_values('price', ascending=[0])
roomProperty_DF.head()

plt.figure(figsize=(10, 18))
heatmap_prezzi = sb.heatmap(
    listingDF.groupby(['property_type', 'room_type']).price.mean().unstack(),
    annot=True,
    fmt=".0f",
    cmap=sb.cm.rocket_r,
    cbar_kws={'label': 'mean_price'}
)
heatmap_prezzi.set_title("Heatmap della media dei prezzi per tipo di stanza e tipo di proprietà", loc='center')
plt.show()

# ANALISI DEGLI ANNUNCI IN BASE AL NUMERO DI CAMERE DA LETTO
plt.figure(figsize=(12, 12))
boxplot_bedrooms_price = sb.boxplot(
    x='bedrooms',
    y='price',
    data=listingDF[['bedrooms', 'price']],
    palette='coolwarm'
)
boxplot_bedrooms_price.set_title("Boxplot numero di camere da letto vs prezzo", loc='center')
plt.show()

# Per verificare il numero di annunci che contiene 7 camere da letto (eccezione nel boxplot sopra)
print("Number of bedrooms :", len(listingDF["bedrooms"].unique()))
print()
print("BedRms|Listings")
print(listingDF["bedrooms"].value_counts())
# Solo un annuncio con 7 camere da letto e che non segue il trend dei prezzi -> si può trascurare

# Swarmplot per visualizzare il numero di annunci per ciascun room_type e il numero di camere da letto
noRoomDF = listingDF[['property_type', 'bedrooms']]
plt.figure(figsize=(12, 12))
swarmplot_listings_vs_room_type_vs_bedrooms = sb.swarmplot(
    x='bedrooms',
    y='property_type',
    data=noRoomDF)
swarmplot_listings_vs_room_type_vs_bedrooms.set_title(
    "Swarmplot numero di annunci per tipo di stanza e numero di camere da letto",
    loc='center'
)
plt.show()

# Heatmap prezzi con numero di camere da letto per annuncio
plt.figure(figsize=(12, 12))
heatmap_prezzi_numero_camere = sb.heatmap(
    listingDF.groupby(['property_type', 'bedrooms']).price.mean().unstack(),
    annot=True, fmt=".0f",
    cmap=sb.cm.rocket_r,
    cbar_kws={'label': 'mean_price'}
)
heatmap_prezzi_numero_camere.set_title("Heatmap media dei prezzi per tipo di stanza e numero di camere", loc='center')
plt.show()

# ANALISI DEGLI ANNUNCI IN BASE ALLE amenities

import nltk
from nltk.corpus import stopwords
import re

amenitiesDF = listingDF[['amenities', 'price', 'id', ]]
amenitiesDFTopper = amenitiesDF.sort_values('price', ascending=[0])
amenitiesDFtop = amenitiesDFTopper.head(30)
allemenities = ''
for index, row in amenitiesDFtop.iterrows():
    p = re.sub('[^a-zA-Z]+', ' ', row['amenities'])
    allemenities += p

allemenities_data = nltk.word_tokenize(allemenities)
filtered_data = [word for word in allemenities_data if word not in stopwords.words('english')]
wnl = nltk.WordNetLemmatizer()
allemenities_data = [wnl.lemmatize(data) for data in filtered_data]
allemenities_words = ' '.join(allemenities_data)

from wordcloud import WordCloud

wordcloud = WordCloud(width=1000, height=700, background_color="white").generate(allemenities_words)
plt.figure(figsize=(15, 15))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()