from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly import plot
from plotly.subplots import make_subplots
from scipy.ndimage import label

from a_db_layer import read_dataset, read_olympics_dataset, preprocess_countriesLocation, preprocess_incomegroup, \
    get_dataset_in_desired_range
from b_data_processing import get_column_with_year, get_country_count_by_year, \
    get_total_count_by_year, get_players_per_country, get_gdpmedal_count_per_year, \
    get_medal_variety_won_both_seasons, get_medal_won_both_seasons, get_host_by_year, avg_age_discipline, \
    avg_age_gender, get_avg_age_medallists_countrywise, get_top_players_by_age_sport, get_country_count_by_year_full, get_genderwise_participation, \
    predict_sports


def country_line_medal_by_year(country: str, season:str):
    # fig, ax = plt.subplots()
    df_country = get_country_count_by_year_full(country, season)
    # ax.plot(df_country["Year"], df_country["Count"], label=country)
    # ax.set(xlabel='year', ylabel='Medal Count', title='Medal count of '+country+' by year')
    # ax.grid()
    fig = px.line(df_country, x="Year", y="Count", title='Medal count of '+country+' by year')

    return fig

def top_countries_line_medal_by_year(season: str):
    fig, ax = plt.subplots()

    df_total = get_total_count_by_year(season)
    fig = px.bar(df_total, x="Year", y="Count")

    df_country = get_country_count_by_year_full("United States", season)
    fig.add_trace(go.Scatter(x=df_country['Year'], y=df_country['Count'], name="USA"))

    df_country = get_country_count_by_year_full("Germany", season)
    fig.add_trace(go.Scatter(x=df_country['Year'], y=df_country['Count'], name="Germany"))

    df_country = get_country_count_by_year_full("UK", season)
    fig.add_trace(go.Scatter(x=df_country['Year'], y=df_country['Count'], name="UK"))

    df_country = get_country_count_by_year_full("France", season)
    fig.add_trace(go.Scatter(x=df_country['Year'], y=df_country['Count'], name="France"))

    df_country = get_country_count_by_year_full("Russia", season)
    fig.add_trace(go.Scatter(x=df_country['Year'], y=df_country['Count'], name="Russia"))

    ax.set(xlabel='year', ylabel='Medal Count', title='Specific countries medal count year-wise')
    fig.update_layout(
        title='Competitor countries medal count year-wise',
        xaxis_title='Year',
        yaxis_title='Medal Count'
    )
    return fig

def height_weight_scatter(season: str):
    df = read_olympics_dataset()
    #only height and weight
    # fig = px.scatter(df, x="Weight", y="Height")
    #height and weight colored by Sport
    df = df[df['Season'] == season]
    fig = px.scatter(df, x = "Weight", y = "Height", color = "Sport",
                     title="Scatterplot of height and weight of participants across sports")
    return fig

def avg_age_discipline_bar(season:str):
    df = avg_age_discipline(season)
    fig = px.bar(df, x="Sport", y="Age", color="Sport", title="Avg age split by Discipline (Sport) for "+ season + " Olympics")
    fig.update_traces(width=2)
    #fig.show()
    return fig

def avg_age_gender_bar(season:str):
    df = avg_age_gender(season)
    fig = px.bar(df, x="Sex", y="Age", color="Sex", title= "Avg age split by Gender for "+ season + " Olympics")
    #fig.update_traces(width=2)
    return fig

def avg_age_medallists_countrywise_bar(season:str):
    df = get_avg_age_medallists_countrywise(season)
    trace1 = go.Bar(x=df["region"].head(11), y=df["Age"].head(11), xaxis='x2', yaxis='y2',
                    marker=dict(color='#0099ff'),
                    name='Top Youngest Medallists')
    trace2 = go.Bar(x=df["region"].tail(10), y=df["Age"].tail(10), xaxis='x2', yaxis='y2',
                    marker=dict(color='#404040'),
                    name='Top Oldest Medallists')
    data = [trace1, trace2]
    fig = go.Figure(data=data)
    fig.layout.margin.update({'t': 75, 'l': 50})
    fig.layout.update({'title': 'Average age of medallists per country for ' + season + " Olympics"})
    #fig.show()
    return fig

def top_players_by_age_bar(season:str, sport:str):
    df = get_top_players_by_age_sport(season, sport)
    trace1 = go.Bar(x=df["Name"].head(11), y=df["Age"].head(11), xaxis='x2', yaxis='y2',
                    marker=dict(color='#0099ff'),
                    name='Top Youngest Medallists')
    trace2 = go.Bar(x=df["Name"].tail(10), y=df["Age"].tail(10), xaxis='x2', yaxis='y2',
                    marker=dict(color='#404040'),
                    name='Top Oldest Medallists')
    data = [trace1,trace2]
    fig = go.Figure(data=data)
    fig.layout.margin.update({'t': 75, 'l': 50})
    fig.layout.update({'title': 'Top 10 youngest/oldest medallists for '+season+ " Olympics in "+ sport})
    ##fig.show()
    # fig, (ax1, ax2) = plt.subplots(2)
    # fig.suptitle('Top 10 youngest/oldest medallists for '+season+ " Olympics in "+ sport)
    # print(df["Age"].tail(10))
    # print(df["Age"].head(10))
    # ax2.barh(df["Name"].tail(10), df["Age"].tail(10))
    # ax2.set_title('Top Oldest Medallists')
    # ax2.set_xlabel('Age')
    # ax1.barh(df["Name"].head(10), df["Age"].head(10))
    # ax1.set_title('Top Youngest Medallists')
    # ax1.set_xlabel('Age')
    # plt.show()
    #
    # fig.show()
    return fig

def top_countries_by_medal(season: str, ncount:int):
    df = read_olympics_dataset()
    df = df[df['Season'] == season]
    noc = read_dataset('noc_regions.csv')
    merged_df = pd.merge(df, noc, on='NOC', how='left')
    top_countries = merged_df.groupby('region')['Medal'].count().nlargest(ncount).reset_index()
    print(top_countries)
    fig = px.bar(top_countries, x="region", y="Medal")
    fig.layout.update({'title': 'Top countries in olympics by medal count.'})
    return fig

def entrants_per_medal(season: str):
    df_entrants = get_players_per_country(season)
    df_medals = get_medal_won_both_seasons(season)
    df_noc = read_dataset('noc_regions.csv')
    new_df = pd.merge(df_entrants, df_medals, how='left', left_on=['NOC'],
                      right_on=['NOC'])
    new_df["Medal Count"].fillna(0, inplace=True)
    df_host = get_host_by_year()
    #print("uu",df_host)
    #print(df_noc)
    df_host = pd.merge(df_host,df_noc , how='left', left_on=['country'],
                      right_on=['region'])
    df_host = df_host.drop(['region','notes'],axis=1)
    hostcountry = list(df_host["NOC"])
   # $print(df_host)#
    new_df["Host City"] = new_df["NOC"].isin(hostcountry)
    print(new_df.head(50))

    print(new_df)
    #print(df_host)
    fig = px.bar(new_df, y="NOC", x="Entrants Count" ,orientation='h',color='Host City',height=1000,text=new_df["Entrants Count"]/new_df["Medal Count"])
    fig.update_traces(width=1.3)
    fig.add_scatter(x=new_df["Medal Count"], y = new_df["NOC"], mode="markers",
                    marker=dict(size=2,  color="Yellow", line=dict(color='Black', width=1)),
                    name="Medals Won")
    # fig.show()
    return fig


def medal_variety_won(season: str):
    df = get_medal_variety_won_both_seasons(season)
    fig = px.scatter(df, y="NOC", x="Count", color="Medal", symbol="Medal")
    fig.update_traces(marker_size=10)
    return fig

def game_segration():
    df = read_olympics_dataset()
    fig = px.sunburst(df, path=['Season', 'Sport', 'Event'], title="Segregation of Events between Winter and Summer Olympics",
                      width=900, height=1000)
    # fig.show()
    return fig

def gdp_per_year_per_country(season: str):
    df = get_gdpmedal_count_per_year(season)
    df = df.drop("NOC", axis = 1)
    df_income = read_dataset('countrywise_income_group.csv')
    df_income = preprocess_incomegroup()
    df_country_medal_count = df.groupby('Country Name')['Medal Count','GDP Value'].sum()
    locations = read_dataset('countries_location.csv')
    locations = preprocess_countriesLocation()
    new_df = pd.merge(df_country_medal_count, df_income, how='left', left_on=['Country Name'], right_on=['Country Name'])
    new_df = new_df.drop('Region', axis=1)

    new_df = pd.merge(new_df, locations, how='left', left_on=['Country Name'], right_on=['Country Name'])
    new_df = new_df.dropna()
    new_df = new_df.drop_duplicates()
    fig, ax = plt.subplots()
    fig = px.scatter_mapbox(new_df, lat=new_df["latitude"].astype(float), lon=new_df["longitude"].astype(float), size_max=40, zoom=1,
                                  color=new_df['IncomeGroup'].astype(str),size=new_df['GDP Value'], hover_name= new_df['Country Name'],hover_data = [new_df["Medal Count"]],
                            width=1200, height=800)
    fig.update_layout(mapbox_style="light",
                     mapbox_accesstoken="pk.eyJ1IjoiZGVla3NoYXNhcmVlbiIsImEiOiJja3dlYm93bGUwMzVpMndwOWJ1M2M0dnlpIn0.aCVlRBubTH65ExTcmOuNPg")
    return fig
    
def athletes_per_sport(season: str):
    df = read_olympics_dataset()
    athlete_count = df.groupby(['Season'])['Sport'].value_counts()
    print("athlete_count\n", athlete_count)
    df = pd.DataFrame(data={'Athlete Count': athlete_count.values}, index=athlete_count.index).reset_index()
    print("df\n", df)
    df1 = df[df['Season'] == season]
    fig = px.pie(df1, values='Athlete Count', names='Sport', title='Number of participating athletes per sport')
    # fig.show()
    return fig

def hypothesis_testing_medals(country:str, season: str):
    df = get_country_count_by_year(country, season)
    host = get_host_by_year()
    host.drop("City", axis=1, inplace=True)
    host = host[host["Season"] == "Summer"]
    host.dropna(inplace=True)
    host = get_dataset_in_desired_range(host)
    host.drop_duplicates(keep="first", inplace=True)
    host = host.reset_index()
    host.drop(["index", "Season"], axis=1, inplace=True)
    host = host[host["country"] == country]
    host = pd.merge(host, df, how='left', left_on=['Year'], right_on=['Year'])
    df = df.rename(columns={"Count":"Medal Tally"})
    fig = px.line(df, x="Year", y="Medal Tally", markers=True, title="Medal count per year for " + country)
    fig.add_scatter(x=host['Year'], y=host['Count'], mode="markers",
                    marker=dict(size=20, color="greenyellow", line=dict(color='Black', width=2)), name="Olympic Host")

    # fig.show()
    return fig

def hypothesis_testing_gdp(country: str, season: str):
    df = get_gdpmedal_count_per_year(season)
    df = df.drop("NOC", axis = 1)
    host = get_host_by_year()
    host.drop("City",axis=1,inplace=True)
    host = host[host["Season"] == "Summer"]

    host.dropna(inplace=True)
    host = get_dataset_in_desired_range(host)
    # print(host)
    host.drop_duplicates(keep="first", inplace=True)
    host = host.reset_index()
    host.drop(["index","Season"],axis=1,inplace=True)
   # print(host)
    host = host[host["country"]==country]
    host = pd.merge(host, df, how='left', left_on=['country', 'Year'], right_on=['Country Name','Year'])
    #
    # print(host)
    # print(df)
    fig = px.line(df[df["Country Name"] == country],x="Year",y="GDP Value",markers=True,title="GDP values per year for "+country)
    #fig.add_bar(x=host["Year"],y=host['GDP Value'], marker_color='Blue',width=1)
    fig.add_scatter(x = host['Year'],y=host['GDP Value'], mode="markers",
                    marker=dict(size=20, color="Yellow",line=dict(color='Black',width=2)),name="Olympic Host")

    #fig.show()
    return fig
	
def gdp_correlation_with_medal_tally():
    df = read_dataset('final_gdp.csv')
    medal_tally_gdp = df.loc[:, ['Year', 'Country Name', 'GDP Value' , 'Medal Count']].drop_duplicates()
    correlation = medal_tally_gdp['GDP Value'].corr(medal_tally_gdp['Medal Count'])
    print("Correlation", correlation)
    fig = px.scatter(medal_tally_gdp, x="GDP Value", y="Medal Count",  color="Country Name" ,trendline="ols", trendline_scope="overall",title="GDP versus Medal Tally")
    fig.update_traces(marker_size=10)
    #fig.show()
    return fig

def genderwise_participation_line(season:str):
    df = get_genderwise_participation(season)
    fig = px.line(df,y="Count",x="Year",color="Sex",labels={
                     "Count": "Team players",}, markers=True, title="Number of male and female participants in the games")
    return fig

def prediction_heat(height:int, weight:int):
    model = predict_sports()
    sports = ["Beach Volleyball", "Weightlifting", "Basketball", "Swimming", "Rhythmic Gymnastics", "Football",
              "Synchronized Swimming", "Judo", "Handball"]

    fig = px.imshow(model['confusion_matrix'],title="Heatmap of correlation matrix",labels=dict(x="Sports", y="Sports"),x=sports,y=sports,
                    width=1000, height=600)
    # fig.show()
    model = model['model']
    df = pd.DataFrame(columns=['Height', 'Weight'])
    df.loc[0] = [height, weight]
    #Basketball H 212 W115
    #Judo H 115 W212
    #Basketball (Tall and Light) - Judo (Heavy and Short)
    print("MERA DDFFFFF\n", df)
    sport = model.predict(df)
    print(sport)
    return sport




if __name__ == '__main__':
    # country_line_medal_by_year("USA")
    # top_countries_line_medal_by_year()
    #height_weight_scatter()
    #top_countries_by_medal()
    # players_per_country()
    #gdp_per_year_per_country()
    # medal_variety_won()
    # entrants_per_medal()
    # avg_age_discipline_bar("Summer")
    # avg_age_gender_bar("Summer")
    # country_line_medal_by_year("USA")
    # top_players_by_age_bar("Summer", "Swimming")
    # athletes_per_sport()
    # country_line_medal_by_year("United States", "Summer")
    # country_line_medal_by_year("United States", "Winter")
    # print(prediction_heat()[1])
    prediction_heat()
    print("ok")
