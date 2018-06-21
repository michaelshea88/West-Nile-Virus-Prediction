import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

train = pd.read_csv('../Assets/train_cleaned.csv')
weather = pd.read_csv('../Assets/weather_clean2.csv')

train.columns
weather.columns

merged = train.merge(weather, how = 'left', on = ['Station', 'Date'])

merged.shape

merged[['Station', 'Trap']].sort_values('Trap')

merged.Station.value_counts()

# Cleaning
# Date column to date_time
merged['Date'] = pd.to_datetime(merged.Date)

# Date column to index
merged.set_index('Date', inplace=True)

merged.to_csv('../Assets/merged.csv',encoding='utf-8')
