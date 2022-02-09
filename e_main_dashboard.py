from typing import Tuple
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from a_db_layer import read_dataset, read_olympics_dataset, preprocess_countriesLocation
from b_data_processing import get_country_count_by_year, get_total_count_by_year
import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
# import dash_core_components as dcc
from dash import dcc
# import dash_html_components as html
from dash import html
import dash_bootstrap_components as dbc
import json

#Main dashboard
from c_visualization_layer import country_line_medal_by_year, avg_age_discipline_bar, top_countries_line_medal_by_year, \
    height_weight_scatter, avg_age_gender_bar, avg_age_medallists_countrywise_bar, top_players_by_age_bar, \
    top_countries_by_medal, entrants_per_medal, medal_variety_won, game_segration, gdp_per_year_per_country, \
    athletes_per_sport, hypothesis_testing_medals, hypothesis_testing_gdp, gdp_correlation_with_medal_tally, \
    genderwise_participation_line, prediction_heat


def dash_home():
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Populating country list for dropdown
    df_countries = preprocess_countriesLocation()
    countryarr = df_countries['Country Name'].unique()
    countryarr = np.append(countryarr[0:159], countryarr[160:206],axis = 0)
    countrylist = countryarr.tolist()

    # Populating sports list for dropdown
    sportdf = read_olympics_dataset()
    sportarr = sportdf['Sport'].unique()
    sportlist = sportarr.tolist()

    # Just the left navigation pane code
    # https://dash-bootstrap-components.opensource.faculty.ai/examples/simple-sidebar/
    leftNavBar = html.Div(
        [
            html.H2("Sumdi Analytica", style={"color": "#1c4c74"}),
            html.Hr(),
            html.P(
                "Please select one of the options to view visualizations.", className="lead"
            ),
            dbc.Nav(
                [
                    # Page Links
                    dbc.NavLink("Home", href="/", active="exact"),
                    dbc.NavLink("Game Segregation", href="/page-1", active="exact"),
                    dbc.NavLink("Country Medal Count", href="/page-2", active="exact"),
                    dbc.NavLink("Top countries by Medal Count", href="/page-3", active="exact"),
                    dbc.NavLink("Competitor Medal Count", href="/page-4", active="exact"),
                    dbc.NavLink("More participation leads to more medals?", href="/page-5", active="exact"),
                    dbc.NavLink("Medal Variety", href="/page-6", active="exact"),
                    dbc.NavLink("Genderwise Participation", href="/page-7", active="exact"),
                    dbc.NavLink("Hypothesis - Medals", href="/page-8", active="exact"),
                    dbc.NavLink("Athletes per sport", href="/page-9", active="exact"),
                    dbc.NavLink("Age Analysis", href="/page-10", active="exact"),
                    dbc.NavLink("Hypothesis - GDP", href="/page-11", active="exact"),
                    dbc.NavLink("The GDP Connection", href="/page-12", active="exact"),
                    dbc.NavLink("GDP Correlation with medals", href="/page-13", active="exact"),
                    dbc.NavLink("Height-Weight Scatter", href="/page-14", active="exact"),
                    dbc.NavLink("Which sport you can be good at?", href="/page-15", active="exact"),
                ],
                vertical=True,
                pills=True,
            ),
        ]
    )

    app.layout = dbc.Row([
            # Left Navigation pane
            dbc.Col(children=[
                dcc.Location(id="url"), leftNavBar
                ],
                md=3, style={"padding": "2rem 2rem", "border-right": "1px solid #6c757d"},
            ),
            # Main dashboard area
            dbc.Col(children=[
                dbc.Row([
                    html.H3(children='Visual Analytics of Olympic History', style={"color": "1C4E80"}),
                ]),
                dbc.Row(id="seasonrow", children=[
                    dbc.RadioItems(
                        id="p1_radio",
                        options=[
                            {'label': 'Summer', 'value': 'Summer'},
                            {'label': 'Winter', 'value': 'Winter'}
                        ],
                        value='Summer',
                        labelStyle={'display': 'inline-block'},
                        inputStyle={"margin-left": "10px", "margin-right": "5px"}
                    ),
                    html.Div(id='seasondiv', style={'display':'none'})
                ], style={'display': 'none'}),
                dbc.Row(children=[
                    dcc.Dropdown(
                        id='country-dropdown', value='United States',
                        options=[{'label': country, 'value': country} for country in countrylist],
                        style={}
                    ),
                    html.Div(id='countrydiv', style={'display': 'none'})
                ], style={'width': '20%', 'display': 'inline-block'}),
                dbc.Row(id="sliderrow", children=[
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label(id='slider1-value'),
                            dcc.Slider(id="slider1", min=1, max=100, step=1, value=5),
                        ]),
                    ], md=6)

                ], style={'width': '100%', 'display': 'inline-block', 'marginTop':'5px'}),

                dbc.Row(children=[
                    html.Div(id="maindash", children=[], style= {'width': '80%'}),
                    ]),

                dbc.Row(children=[
                    dcc.Dropdown(
                        id='sport-dropdown', value='Swimming',
                        options=[{'label': sport, 'value': sport} for sport in sportlist],
                        style={}
                    ),
                ], style={'width': '20%', 'display': 'inline-block'}),
                dbc.Row(id="predictionrow", children=[
                    dbc.Col([
                        dcc.Input(
                            id="height".format("number"),
                            value=110,
                            type="number",
                            placeholder="Height (cms)".format("number"),
                        ),
                        dcc.Input(
                            id="weight".format("number"),
                            value=110,
                            type="number",
                            placeholder="Weight (kgs)".format("number"),
                        ),
                    ], md=6)

                ], style={'width': '100%', 'display': 'inline-block', 'marginTop': '5px'}),
                ],
                md=9, style={"padding": "2rem 2rem"},
            ),
        ])

    # Callback function conditionally displaying elements based on the page selected.
    @app.callback([Output(component_id="maindash", component_property="children"),
                   Output(component_id="country-dropdown", component_property="style"),
                   Output(component_id="sliderrow", component_property="style"),
                   Output(component_id="sport-dropdown", component_property="style"),
                   Output(component_id="predictionrow", component_property="style"),
                   Output(component_id="seasonrow", component_property="style")
                   ],
                  [Input("url", "pathname"), Input("p1_radio", "value"), Input("country-dropdown", "value"),
                   Input("slider1", "value"),
                   Input("sport-dropdown", "value"),
                   Input("height".format("number"), "value"),
                   Input("weight".format("number"), "value")
                   ])
    def render_page_content(pathname, radiovalue, p1_country, slidervalue, sportvalue, height, weight):
        print("render_page_content")
        p1_season = radiovalue
        if pathname == "/":
            return [html.P("Analysing and Visualizing 120 years of Olympic History on a dashboard!"),
                    html.P("This project developed as part of a final project in CSCI 6612 Visual Analytics.")], \
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'}
        elif pathname == "/page-1":
            return html.Div(children=[
                html.P("Game segregation - Summer and Winter combination."),
                dbc.Col(dcc.Graph(id='example-graph', figure=game_segration()), ),
            ]),\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'}
        elif pathname == "/page-2":
            return html.Div(children=[
                html.P("This visualization shows the medal count of a specific country year-wise."),
                dbc.Col(dcc.Graph(id='example-graph', figure=country_line_medal_by_year(p1_country, p1_season)), ),
            ]), \
                   {'display': 'block'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-3":
            return html.Div(children=[
                html.P("This visualization ranks top performing countries in the olympics."),
                dbc.Col(dcc.Graph(id='example-graph', figure=top_countries_by_medal(p1_season, slidervalue)), ),
            ]),\
                   {'display': 'none'},\
                   {'width': '100%', 'display': 'inline-block', 'marginTop':'5px'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-4":
            return html.Div(children=[
                html.P("Competitive comparison of Medal Count between top countries year wise."),
                dbc.Col(dcc.Graph(id='example-graph', figure=top_countries_line_medal_by_year(p1_season)), ),
            ]),\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-5":
            return html.Div(children=[
                html.P("This graph visualizes relation between sending more participants and winning more medals. Metric used here is entrants per medal."),
                dbc.Col(dcc.Graph(id='example-graph', figure=entrants_per_medal(p1_season)), ),
            ]),\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-6":
            return html.Div(children=[
                html.P("This visualization shows the variety of medals won by a country."),
                dbc.Col(dcc.Graph(id='example-graph', figure=medal_variety_won(p1_season)), ),
            ]),\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-7":
            return html.Div(children=[
                html.P("This visualization shows an analysis on the changes of male and female participation over the years"),
                dbc.Col(dcc.Graph(id='example-graph', figure=genderwise_participation_line(p1_season)), ),
            ]), \
                   {'display': 'none'}, \
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-8":
            return html.Div(children=[
                html.P("This visualization shows the relation between the medal tally and the country being a host country."),
                dbc.Col(dcc.Graph(id='example-graph', figure=hypothesis_testing_medals(p1_country, p1_season)), ),
            ]),\
                   {'display': 'block'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-9":
            return html.Div(children=[
                html.P(""),
                dbc.Col(dcc.Graph(id='example-graph', figure=athletes_per_sport(p1_season)), ),
            ]),\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-10":
            return html.Div(children=[
                html.P("Age analysis based on the type of sport."),
                dbc.Col(dcc.Graph(id='example-graph', figure=avg_age_discipline_bar(p1_season)), ),
                html.Hr(),
                html.P("Age analysis based on the gender."),
                dbc.Col(dcc.Graph(id='example-graph', figure=avg_age_gender_bar(p1_season)), ),
                html.Hr(),
                html.P("Average age of medalists by country."),
                dbc.Col(dcc.Graph(id='example-graph', figure=avg_age_medallists_countrywise_bar(p1_season)), ),
                html.Hr(),
                html.P("Top 10 youngest and oldest medalists."),
                dbc.Col(dcc.Graph(id='example-graph', figure=top_players_by_age_bar(p1_season, sportvalue)), ),
            ]),\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'block'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-11":
            return html.Div(children=[
                html.P("This visualization shows the relation between the GDP value and the country being a host country."),
                dbc.Col(dcc.Graph(id='example-graph', figure=hypothesis_testing_gdp(p1_country, p1_season)), ),
            ]),\
                   {'display': 'block'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-12":
            return html.Div(children=[
                html.P("This visualization shows a comparative analysis between the income levels of a country and the medals won."),
                dbc.Col(dcc.Graph(id='example-graph', figure=gdp_per_year_per_country(p1_season)), ),
            ]),\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-13":
            return html.Div(children=[
                html.P("This visualization shows a correlation between the medal count and the GDP value of different countries."),
                html.P("Correlation: 0.5009461901052723."),
                dbc.Col(dcc.Graph(id='example-graph', figure=gdp_correlation_with_medal_tally()), ),
            ]),\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'}
        elif pathname == "/page-14":
            return html.Div(children=[
                html.P(""),
                dbc.Col(dcc.Graph(id='example-graph', figure=height_weight_scatter(p1_season)), ),
            ]),\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'display': ''}
        elif pathname == "/page-15":
            return html.Div(children=[
                html.Hr(),
                html.P("Participant sports prediction using height and weight."),
                html.P("Enter height in CMS, and weight in KGS"),
                html.H4(children='Sport this person can be good at:', style={"color": "#1C4E80"}),
                html.H3(id="predictionresult", children=prediction_heat(height, weight), style={"color": "#1C4E80"}),
            ]), \
                   {'display': 'none'}, \
                   {'display': 'none'},\
                   {'display': 'none'},\
                   {'width': '100%', 'display': 'inline-block', 'marginTop': '5px'}, \
                   {'display': 'none'}

    @app.callback(Output('slider1-value', 'children'), [Input('slider1', 'value')])
    def update_slider_value(slider1):
        return f'Number of top countries: {slider1}'

    return app

if __name__ == "__main__":
    app_olympics = dash_home()
    app_olympics.run_server(debug=True)