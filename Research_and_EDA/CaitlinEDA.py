import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.style.use('ggplot')
import math
import seaborn as sns

# read data into DataFrame
train = pd.read_csv('../Assets/train.csv')

# look at data
train.head()

# check dtypes and null values
train.info()
# delete address fields because location has already been translated
# into latitude and longitude
del train['Address']
del train['Block']
del train['Street']
del train['AddressNumberAndStreet']
del train['AddressAccuracy']

# check whether all species carry WNV
train.groupby('Species').agg({'WnvPresent': np.sum})

# no observations among Culex Erraticus, Salinarius, Tarsalis or Territans, but
# CDC's website suggests these can all still carry the virus
# https://www.cdc.gov/westnile/resources/pdfs/mosquitospecies1999-2012.pdf

# aggregate observations that are only distinct because of
# hitting the 50 mosquito cap
grouped = train.groupby(['Date', 'Species', 'Trap', 'Latitude', 'Longitude'])
aggregated = pd.DataFrame(grouped.agg({'NumMosquitos': np.sum, 'WnvPresent': np.max})).reset_index()
aggregated.sort_values(by = 'NumMosquitos', ascending = False)
# Sort by date and change to datetime
aggregated.sort_values(by='Date', inplace=True)
aggregated['Date'] = pd.to_datetime(aggregated['Date'])
# set index to date
aggregated.set_index('Date', inplace=True)


# add month column
aggregated['Month'] = aggregated.index.month
aggregated['Year'] = aggregated.index.year

# check if mosquito vars should be categorical
aggregated.Species.value_counts()

# create dummy vars for mosquito types
aggregated = pd.get_dummies(aggregated, columns = ['Species'],drop_first=True)

# function to calculate distance between 2 lat/long points
# source: http://www.johndcook.com/blog/python_longitude_latitude/
def distance_on_unit_sphere(lat1, long1, lat2, long2):

    # Convert latitude and longitude to spherical coordinates in radians
    degrees_to_radians = math.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    # sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
    math.cos(phi1)*math.cos(phi2))
    arc = math.acos( cos )

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return arc

# weather stations 1 and 2 lat/longitude
station_1_lat = 41.995
station_1_lon = -87.933
station_2_lat = 41.786
station_2_lon = -87.752
# function to calculate whether station 1 or 2 (from weather.csv) is closer
def closest_station(lat, lon):
    if (distance_on_unit_sphere(lat, lon, station_1_lat, station_1_lon) <
        distance_on_unit_sphere(lat, lon, station_2_lat, station_2_lon)):
        return 1
    else: return 2

# add station to indicate whether station 1 or 2 is closer
aggregated['Station'] = [closest_station(a,b) for (a, b) in zip(aggregated.Latitude, aggregated.Longitude)]

aggregated.Station.value_counts()
# look at distribution of data
aggregated.describe()

# write to a csv

aggregated.to_csv('../Assets/train_cleaned.csv', encoding= 'utf-8')

## EDA ON TRAIN DATA
# look at dispersion of traps geographically
plt.scatter(aggregated.Longitude, aggregated.Latitude, c = aggregated.Station)
plt.scatter([station_1_lon, station_2_lon], [station_1_lat,station_2_lat], c = 'r')
plt.show()


# look at dispersion of virus incidence geographically
plt.scatter(aggregated.Longitude, aggregated.Latitude, c = aggregated.WnvPresent, alpha = .01)

# look at distribution of number of mosquitos
plt.hist(aggregated.NumMosquitos, 100)

# look at incidence of cases over time
aggregated[['Date', 'WnvPresent']].groupby('Date').sum().plot(kind='bar')

# look for patterns of seasonality
aggregated[['WnvPresent']].resample('M', how = 'sum')

# Cases peak in August, and run from July through October.
# How does this compare with seasonality of mosquito populations?
# (as approximated by number of mosquitos caught in traps)

aggregated[['NumMosquitos']].resample('M', how = 'sum')
