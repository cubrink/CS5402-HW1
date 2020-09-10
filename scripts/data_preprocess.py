# data_preprocessing
#
# Prepare data from census.csv for analysis

# Imports

import pandas as pd
from datetime import datetime
from feature_engine.discretisers import EqualWidthDiscretiser, EqualFrequencyDiscretiser

# Load data

DATA_SOURCE = r'../data/census.csv'
df = pd.read_csv(DATA_SOURCE)

# Sanitize date format

df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].apply(lambda x: x.replace(year=1994).date().strftime('%m/%d/%Y'))

# Discretisation

age_discretiser = EqualWidthDiscretiser(bins=10, variables=['age'])
hpw_discretiser = EqualFrequencyDiscretiser(q=5, variables=['hours-per-week'])

df = age_discretiser.fit_transform(df)
df = hpw_discretiser.fit_transform(df)

# Value replacement

# Workclass:       ? -> Other
# Occupation:      ? -> Other
# Native-country:  ? -> Unspecified

# Sex:             Starts with f -> female
#                  Starts with m -> male


df['workclass'].replace(to_replace='\?', value='Other', inplace=True, regex=True)
df['occupation'].replace(to_replace='\?', value='Other', inplace=True, regex=True)
df['native-country'].replace(to_replace='\?', value='Unspecified', inplace=True, regex=True)

df['sex'].replace(['^(\W)*[Ff].*', '^(\W)*[Mm].*'], ['Female', 'Male'], inplace=True, regex=True)

# Value normalization

# Normalize population-wgt
wgt = df['population-wgt']
df['population-wgt'] = (wgt-wgt.min())/(wgt.max() - wgt.min())

# Write sanitzed data to file

SAVE_LOCATION = r'../data/census_sanitized.csv'

df.to_csv(SAVE_LOCATION, index=False)
