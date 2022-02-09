import os
from pathlib import Path
import pandas as pd


# Read dataset from specified path
def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path,encoding='ISO-8859-1')
    return df

# Read the pre-processed olympics dataset, call pre-processing logic if not already done.
def read_olympics_dataset() -> pd.DataFrame:
    if not os.path.isfile("athlete_events_processed.csv"):
        save_cleaned_dataset()
    df = pd.read_csv("athlete_events_processed.csv")
    return df

# Preprocess the dataset, replace the NaN values with the mean.
def pre_processing_with_mean(df: pd.DataFrame):
    df["Age"].fillna((df["Age"].mean()), inplace=True)
    df["Weight"].fillna((df["Weight"].mean()), inplace=True)
    df["Height"].fillna((df["Height"].mean()), inplace=True)
    df['Medal'].fillna('No_Medal', inplace=True)
    return df

def get_dataset_in_desired_range(df: pd.DataFrame):
    df = df.loc[(df['Year'] >= 1960) & (df['Year'] <= 2016)]
    return df

def pre_process_nan_column(df: pd.DataFrame, columnname, value):
    df[columnname].fillna(0, inplace=True)

# Save the preprocessed dataset in the directory with a new name.
def save_cleaned_dataset():
    df = read_dataset('athlete_events.csv')
    pre_processed_df = pre_processing_with_mean(df)
    pre_processed_df.to_csv('athlete_events_processed.csv', index=False)

def preprocess_incomegroup():
    df = read_dataset('countrywise_income_group.csv')
    df.fillna("unknown", inplace=True)
    df = df.drop("Country Code", axis=1)
    return df

def preprocess_gdp():
    df = read_dataset('gdp.csv')
    df = pd.melt(df, 'Country Name', var_name='Year').sort_values(['Country Name', 'Year']).reset_index(
        drop=True)
    df = df.loc[(df['Year'] != 'Country Code') & (df['Year'] <= '2016')]

    df = df.rename({'value': 'GDP Value'}, axis=1)
    return df
    #print(df)

def preprocess_countriesLocation():
    df = read_dataset('countries_location.csv')
    df2 = read_dataset('noc_regions.csv')

    df = pd.merge(df2, df, how='left', left_on=['region'], right_on=['country'])

    df = df.drop(["country_code", "country", "notes", "NOC"], axis=1)
    df['latitude'].fillna(0, inplace=True)
    df['longitude'].fillna(0, inplace=True)
    df = df.rename({'region': 'Country Name'}, axis=1)

    return df

if __name__ == '__main__':
    preprocess_incomegroup()
    preprocess_gdp()
    preprocess_countriesLocation()
    print("ok")
