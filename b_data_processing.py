from typing import Dict

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from a_db_layer import read_dataset, read_olympics_dataset, get_dataset_in_desired_range, preprocess_gdp, \
    pre_processing_with_mean
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics

#TODO: preprocess the original dataset and send back to a_db_layer

#useless for now
def get_column_with_year(column: str) -> pd.DataFrame:
    df = read_dataset('athlete_events.csv')
    df_ret = df[[column, 'Year']]
    print(df_ret)
    return df_ret

def get_country_count_by_year_full(country: str, season: str):
    df = read_olympics_dataset()
    noc = read_dataset('noc_regions.csv')
    df = pd.merge(df, noc, how='left', left_on=['NOC'], right_on=['NOC'])
    #print(df)
    df_country = df.loc[(df['region'] == country) & (df['Medal'].isin(['Gold', 'Silver', 'Bronze'])) & (df['Season'] == season)]
    df_ret = df_country.groupby(['Year', 'Event']).size().to_frame('Count').reset_index()
    df_ret = df_ret.groupby(['Year']).size().to_frame('Count').reset_index()
    print('get_country_count_by_year - ', country, "\n", df_ret)
    return df_ret

def get_country_count_by_year(country: str, season: str):
    df = read_olympics_dataset()
    df = get_dataset_in_desired_range(df)
    noc = read_dataset('noc_regions.csv')
    df = pd.merge(df, noc, how='left', left_on=['NOC'], right_on=['NOC'])
    #print(df)
    df_country = df.loc[(df['region'] == country) & (df['Medal'].isin(['Gold', 'Silver', 'Bronze'])) & (df['Season'] == season)]
    df_ret = df_country.groupby(['Year', 'Event']).size().to_frame('Count').reset_index()
    df_ret = df_ret.groupby(['Year']).size().to_frame('Count').reset_index()
    #print('get_country_count_by_year - ', country, "\n", df_ret)

    return df_ret

def get_total_count_by_year(season: str):
    df = read_olympics_dataset()
    df_medals = df.loc[(df['Medal'].isin(['Gold', 'Silver', 'Bronze'])) & (df['Season'] == season)]
    df_ret = df_medals.groupby(['Year', 'Event']).size().to_frame('Count').reset_index()
    df_ret = df_ret.groupby(['Year']).size().to_frame('Count').reset_index()

    return df_ret

def get_top_players_by_age_sport(season: str , sport:str):
    df = read_olympics_dataset()
    df = pre_processing_with_mean(df)
    df_country = df.loc[(df['Medal'].isin(['Gold', 'Silver', 'Bronze'])) & (df['Season'] == season) & (df['Sport'] == sport)]
    df_country = df_country.sort_values(by=['Age'])

    #print(df_country[["Name","Medal","Age"]])
    #df.drop_duplicates(subset=['Name'])
    df_country = df_country.drop_duplicates(subset=['Name'])
    print(df_country[["Name", "Medal", "Age"]])

    return df_country

def get_avg_age_medallists_countrywise(season:str):
    df = read_olympics_dataset()
    noc = read_dataset('noc_regions.csv')
    df = df.loc[(df['Season'] == season)]
    df_ret = df.groupby('NOC')['Age'].mean().reset_index()
    df_ret = pd.merge(df_ret, noc, how='left', left_on=['NOC'], right_on=['NOC']).reset_index()
    df_ret = df_ret.sort_values(by=['Age'])
    df_ret = df_ret.drop_duplicates(subset=['region'])
    print(df_ret)

    return df_ret

def get_gdpmedal_count_per_year( season: str):
    df = read_olympics_dataset()
    locations = read_dataset('countries_location.csv')
    noc = read_dataset('noc_regions.csv')
    gdp = read_dataset('gdp.csv')
    gdp = preprocess_gdp()
    df_country = df.loc[(df['Medal'].isin(['Gold', 'Silver', 'Bronze'])) & (df['Season'] == season)]
    df_ret = df_country.groupby(['Year', 'Event','NOC']).size().to_frame('Medal Count').reset_index()
    df_ret = df_ret.groupby(['Year','NOC']).size().to_frame('Medal Count').reset_index()

    merged_df = pd.merge(df_ret, noc, on='NOC', how='left')
    merged_df = merged_df.rename(columns={'region': 'Country Name'})
    merged_df = (merged_df.drop("notes",axis=1))
    merged_df = get_dataset_in_desired_range(merged_df)

    merged_df['Year']=merged_df['Year'].astype(int)
    #print(gdp)
    gdp['Year'] = gdp['Year'].astype(int)
    new_df = pd.merge(merged_df, gdp,how='left', left_on=['Country Name', 'Year'], right_on=['Country Name', 'Year'])
    new_df = pd.merge(new_df, locations, how='left', left_on=['Country Name'], right_on=['country'])
    new_df = new_df.drop(['country','country_code'], axis=1)
    new_df = new_df.fillna(0.0)

    new_df.to_csv('final_gdp.csv', index=False)
    # print(new_df.head().to_string())

    #print(new_df)
    return new_df

def get_players_per_country(season: str):
    df = read_olympics_dataset()
    df = df[df['Season'] == season]
    df_ret = df.groupby(['NOC', 'Name']).size().to_frame('Entrants Count').reset_index()
    df_country_medal_count = df_ret.groupby('NOC')['Entrants Count'].sum().reset_index()

    print(df_country_medal_count)
    return df_country_medal_count

def get_medal_variety_won_both_seasons(season: str):
    df = read_olympics_dataset()
    df = df[df['Season'] == season]
    df = df.drop(["ID",'Name','Sex','Age','Height','Weight'], axis=1)
    df_medals = df.loc[(df['Medal'].isin(['Gold', 'Silver', 'Bronze']))]
    df_ret = df_medals.groupby(['NOC','Event','Medal']).size().to_frame('Count')
    df_ret = df_ret.groupby(['NOC' ,'Medal']).size().to_frame('Count').reset_index()
    return df_ret

def get_medal_won_both_seasons(season: str):
    df = read_olympics_dataset()
    df = df[df['Season'] == season]
    df = df.drop(["ID", 'Name', 'Sex', 'Age', 'Height', 'Weight'], axis=1)
    df_medals = df.loc[(df['Medal'].isin(['Gold', 'Silver', 'Bronze']))]
    df_ret = df_medals.groupby(['NOC', 'Medal']).size().to_frame('Medal Count').reset_index()
    df_country_medal_count = df_ret.groupby('NOC')['Medal Count'].sum().reset_index()
    return df_country_medal_count

def get_host_by_year():
    df = read_olympics_dataset()
    #print(df["Year"])
    #df.sort_values(by='Year')
    df = pd.DataFrame(df,columns=['Year','City','Season'])
    country = read_dataset('worldcities.csv')
    country = country.drop_duplicates()
    new_df = pd.merge(df, country, how='left', left_on=['City'], right_on=['city'])
    new_df = new_df.drop('city', axis=1)
    # print(new_df)
    return new_df

def avg_age_discipline( season: str ):
    df = read_dataset('athlete_events.csv')
    df = pre_processing_with_mean(df)
    df = df.loc[(df['Season'] == season)]
    df_ret = df.groupby('Sport')['Age'].mean().reset_index()

    print(df_ret)
    return df_ret

def avg_age_gender(season: str):
    df = read_dataset('athlete_events.csv')
    df = pre_processing_with_mean(df)
    df = df.loc[(df['Season'] == season)]
    df_ret = df.groupby('Sex')['Age'].mean().reset_index()
    print(df_ret)
    return df_ret

def get_genderwise_participation(season:str):
    df =read_olympics_dataset()
    df = df[df['Season']==season]
    df_ret = df.groupby(['Year','Sex']).size().to_frame('Count').reset_index()
    return df_ret

def predict_sports():
    #pick entries with medals
    df = read_olympics_dataset()
    sports = ["Beach Volleyball", "Weightlifting", "Basketball", "Swimming", "Rhythmic Gymnastics", "Football", "Synchronized Swimming", "Judo", "Handball"]

    df = df.loc[(df['Medal'].isin(['Gold', 'Silver', 'Bronze'])) & (df['Season']=='Summer')& (df['Sport'].isin(sports))]
    X = df[["Height","Weight"]]
    y = df[["Sport"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = DecisionTreeClassifier()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_confusion_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='micro')
    recall =recall_score(y_test, y_pred, average='micro')
    model_f1_score = f1_score(y_test, y_pred, average='micro')
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("Training accuracy: ", train_score)
    print("Testing accuracy: ", test_score)
    print("Accuracy",accuracy)
    print("Precision", precision)
    print("Confusion Marix", model_confusion_matrix)

    return dict(model=model, confusion_matrix=model_confusion_matrix, accuracy=accuracy, precision=precision,
                recall=recall, f1_score=model_f1_score)
if __name__ == '__main__':
    # df = get_country_count_by_year("IND", "Summer")
    #get_total_count_by_year("Summer")
    #get_gdpmedal_count_per_year("Summer")
    #get_players_per_country()
    #get_medal_won_both_seasons()
    #get_host_by_year()
    avg_age_discipline("Summer")
    avg_age_gender("Summer")
    print("ok")


