import pandas as pd

listingDF = pd.read_csv('datasets/listings.csv')
# Selezione delle colonne rilevanti
newListingsDF = listingDF[['room_type', 'property_type', 'bedrooms', 'amenities', 'number_of_reviews', 'price']]

# Lista di tutti i possibili servizi
amenities_list = list(newListingsDF.amenities)
amenities_list_string = " ".join(amenities_list)
amenities_list_string = amenities_list_string.replace('{', '')
amenities_list_string = amenities_list_string.replace('}', ',')
amenities_list_string = amenities_list_string.replace('"', '')
amenities_set = [x.strip() for x in amenities_list_string.split(',')]
amenities_set = set(amenities_set)

# Creazione di colonne per ogni servizio e aggiunta nel DF originale
newListingsDF.loc[newListingsDF['amenities'].str.contains('Suitable for Events'), 'event_suitable'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('24-Hour Check-in'), 'check_in_24h'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Air Conditioning'),'air_conditioning'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Breakfast'), 'breakfast'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('TV'), 'tv'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Dishwasher|Dryer|Washer'), 'white_goods'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Elevator'), 'elevator'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Exercise equipment|Gym|gym'), 'gym'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Free Parking on Premises|parking'), 'parking'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Pool'), 'pool'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Hot Tub'), 'hot_tub'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Wireless Internet|Internet'), 'internet'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Cat(s)|Dog(s)|Other pet(s)|Pets Allowed'), 'pets_allowed'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Safe|Security system'), 'secure'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Step-free access|Wheelchair|Accessible'), 'accessible'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Heating'), 'heating'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Indoor Fireplace'), 'fireplace'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Smoking Allowed'), 'smoking_allowed'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Laptop Friendly Workspace'), 'workspace'] = 1
newListingsDF.loc[newListingsDF['amenities'].str.contains('Family/Kid Friendly|Children|children'), 'child_friendly'] = 1

# Rimozione della colonna amenities
newListingsDF.drop('amenities', axis=1, inplace=True)

# Sostituzione dei valori nulli con 0
cols_to_replace_nulls = newListingsDF.iloc[:, 5:].columns
newListingsDF[cols_to_replace_nulls] = newListingsDF[cols_to_replace_nulls].fillna(0)

# Raggruppamento delle tipologie di proprietà meno comuni in "Other" (inferiori a 30)
newListingsDF.loc[~newListingsDF.property_type.isin(
    ['House', 'Apartment', 'Townhouse', 'Condominium', 'Loft', 'Bed & Breakfast']), 'property_type'] = 'Other'

# Conversione dei prezzi da stringhe in interi
newListingsDF.price = newListingsDF.price.str[1:-3]
newListingsDF.price = newListingsDF.price.str.replace(",", "")
newListingsDF.price = newListingsDF.price.astype('int64')

# Verifica se ci sono valori nulli nelle colonne
newListingsDF.isnull().sum()

# Ci sono 6 valori nulli nella colonna bedrooms, si procede con la rimozione di queste righe
newListingsDF = newListingsDF.dropna(subset=['bedrooms'])

# One-hot encode
transformedDF = pd.get_dummies(newListingsDF, columns=['room_type', 'property_type'])

# Rinomino le categorie che contengono spazi oppure caratteri speciali
newTransformedDF = transformedDF.rename(columns={'room_type_Entire home/apt': 'room_type_Entire_home_apt'})
newTransformedDF = newTransformedDF.rename(columns={'room_type_Private room': 'room_type_Private_room'})
newTransformedDF = newTransformedDF.rename(columns={'room_type_Shared room': 'room_type_Shared_room'})
newTransformedDF = newTransformedDF.rename(columns={'property_type_Bed & Breakfast': 'property_type_Bed_and_Breakfast'})

newTransformedDF.to_csv('datasets/listingsProcessed.csv', index=False)