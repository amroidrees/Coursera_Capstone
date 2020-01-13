import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # 
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # 
import folium # map rendering library

#Lib for html handling
from lxml import html

print('Libraries imported.')

################################################
################################################
################################################

#Fetch wikipedia page as html
url = 'https://fi.wikipedia.org/wiki/Helsingin_alueellinen_jako'
pageContent=requests.get(url)
neighborhood_html = html.fromstring(pageContent.content)


################################################
################################################
################################################


# define the dataframe columns
column_names = ['Neighborhood', 'lat', 'lng'] 
df = pd.DataFrame(columns=column_names)
#Get table rows of Helsinki Neigborhoods with xpath
rows = neighborhood_html.xpath('//*[@id="mw-content-text"]/div/table[2]/tbody/tr')

for row in rows:
    #Append dataframe with variable values
    children = row[0].getchildren()
    
    for child in children:
        if child.tag == 'ul':
            for li in child[0]:
                #print('Suppea')
                df = df.append({'Neighborhood': li.text}, ignore_index=True)
        else:
            df = df.append({'Neighborhood': child.text}, ignore_index=True)
           
df.head()

################################################
################################################
################################################

# Dataframe contains all Helsinki neighborhoods or city areas which are in small capital city same thing.
df.head()

################################################
################################################
################################################

#Fetch housing price page as html
price_url = 'https://www.asuntojenhinnat.fi/myytyjen-asuntojen-tilastot/kunta/Helsinki/'
pricePageContent=requests.get(price_url)
price_html = html.fromstring(pricePageContent.content)


################################################
################################################
################################################


# define the dataframe columns
price_column_names = ['Neighborhood', 'Price'] 
df_price = pd.DataFrame(columns=price_column_names)

#Create lists for hood and price
hood = []
price = []

#Get table rows of Helsinki Neigborhoods with xpath
price_rows = price_html.xpath('//*[@id="main"]/section[4]/div/div/table/tbody/tr')

for row in price_rows:
    #Append dataframe with variable values
    hood.append(row[1].text_content())
    price.append(row[2].text)

df_price['Neighborhood'] = hood
df_price['Price'] = price
df_price.head()

################################################
################################################
################################################

#Merge pricing data to Dataframe
df = pd.merge(df, df_price, on='Neighborhood')
df.head()


# Let's get first coordinates of Helsinki area and see that data is correct
address = 'Helsinki, FI'

geolocator = Nominatim(user_agent="ny_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Helsinki City are {}, {}.'.format(latitude, longitude))

################################################
################################################
################################################

#Create lists for lat and long
lat = []
lng = []

#Loop through all neigborhoods in Helsinki
for adr in df['Neighborhood']:
    
    #Use geolocator to get coordinates of neigborhoods
    loc = geolocator.geocode(adr)
    #Append coordinates to lists
    lat.append(loc.latitude)
    lng.append(loc.longitude)

#Map coordinate lists to data frame 
df['lat'] = lat
df['lng'] = lng


################################################
################################################
################################################

#Neighborhoods with coordinates and m2 pricing
df.head()

################################################
################################################
################################################

## Forsquare API
CLIENT_ID = 'XDSVHHZ0OH2OITHZSB5MJSEHSUVR5J3CYY5EOHOQTV550IQ1' # your Foursquare ID
CLIENT_SECRET = '5MNXPTBQ4GVTH0KMT0H0UKX00KYCEDLTI0XLT5BKJGME4AO3' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)

################################################
################################################
################################################

#Get Kruunuhaka or City Center venues

LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
 # create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL


################################################
################################################
################################################

results = requests.get(url).json()

################################################
################################################
################################################

#Check that coordites work correctly
neighborhood_latitude = df.loc[0, 'lat'] # neighborhood latitude value
neighborhood_longitude = df.loc[0, 'lng'] # neighborhood longitude value

neighborhood_name = df.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, neighborhood_latitude, neighborhood_longitude))

# create map of Helsinki using latitude and longitude values
Helsinki_map = folium.Map(location=[latitude, longitude], zoom_start=11)

for index, row in df.iterrows():

    adr = row['Neighborhood']
    lat = row['lat']
    lng = row['lng']

    label = '{}'.format(adr)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(Helsinki_map)  

Helsinki_map

################################################
################################################
################################################

#Function to Get nearby venues
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

################################################
################################################
################################################


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


################################################
################################################
################################################

# Function for most common venue
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

################################################
################################################
################################################

venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()

################################################
################################################
################################################

all_helsinki_venues = getNearbyVenues(names=df['Neighborhood'],latitudes=df['lat'], longitudes=df['lng'])
all_helsinki_venues.head()

################################################
################################################
################################################

hood_venue = all_helsinki_venues[['Neighborhood', 'Venue']].copy()
hood_venues = hood_venue.groupby(['Neighborhood']).size().reset_index(name='Venues')
hood_venues.head()

################################################
################################################
################################################

#Merge number of venues to Dataframe
df = pd.merge(df, hood_venues, on='Neighborhood')
df.head()

hood_venues.sort_values(by=['Venues'])
hood_venues.plot.bar(x='Neighborhood', y='Venues', rot=90,figsize=(20,10))


################################################
################################################
################################################

# one hot encoding
helsinki_onehot = pd.get_dummies(all_helsinki_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
helsinki_onehot['Neighborhood'] = df['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [helsinki_onehot.columns[-1]] + list(helsinki_onehot.columns[:-1])
downtown_onehot = helsinki_onehot[fixed_columns]

helsinki_onehot.head()

################################################
################################################
################################################


helsinki_grouped = helsinki_onehot.groupby('Neighborhood').mean().reset_index()
helsinki_grouped.head()

################################################
################################################
################################################

#Get 10 top venues of neighborhood
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = helsinki_grouped['Neighborhood']

for ind in np.arange(helsinki_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(helsinki_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


#Check shape
helsinki_grouped.shape

all_helsinki_venues.shape

#Five top venues of Helsinki neighborhoods
num_top_venues = 5

for hood in df['Neighborhood']:
    print("----"+hood+"----")
    temp = helsinki_grouped[helsinki_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')

################################################
################################################
################################################


# set number of clusters
kclusters = 5

helsinki_grouped_clustering = helsinki_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(helsinki_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]

# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

helsinki_merged = df

# merge helsinki_grouped with helsinki_data to add latitude/longitude for each neighborhood
helsinki_merged = helsinki_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

helsinki_merged.head() # check the last columns!

most_common_in_cluster = helsinki_merged['Cluster Labels', '1st Most Common Venue'].copy()
most_common_in_cluster = most_common_in_cluster.groupby(['Cluster Labels', '1st Most Common Venue' ]).size().reset_index(name='Venues')
most_common_in_cluster

# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster, price in zip(helsinki_merged['lat'], helsinki_merged['lng'], helsinki_merged['Neighborhood'], helsinki_merged['Cluster Labels'],helsinki_merged['Price']):
    label = folium.Popup(str(poi) + ' Most of venue type: ' + str(cluster_name[cluster]) + '. Avg square price: ' + str(price), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters

helsinki_merged.set_index("Neighborhood", inplace=True)
helsinki_merged.head()

#Example of how real estate agent could fetch information by neighborhood when going to sales meeting
helsinki_merged.loc['Eira']

