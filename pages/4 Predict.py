import streamlit as st
import pygame
import numpy as np
from PIL import Image
import os
import pandas as pd
import base64
import io
from io import BytesIO
from streamlit import components
import joblib
import requests
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import PlayerCareerStats
os.environ["SDL_VIDEODRIVER"] = "dummy"
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('./images/nba.png')
# initialize Pygame
pygame.init()





st.write("<h1 style='font-size: 30px; text-align: center;'> Let\'s pick a line up! </h1>", unsafe_allow_html=True)


# define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
BROWN = (139, 69, 19)
ORANGE = (255, 140, 0)

# set the size of the window
WINDOW_SIZE = (2000, 1000)
screen = pygame.Surface(WINDOW_SIZE)

# set the display mode
screen = pygame.display.set_mode(WINDOW_SIZE)
# set the title of the window
pygame.display.set_caption("Basketball court")

# load the image of the basketball court
court_image = Image.open('./images/basketball_court.jpg')
court_width, court_height = court_image.size

# resize the image to fit the window size
court_image = court_image.resize(WINDOW_SIZE, resample=Image.BOX)

# convert the image to a Pygame surface
court_surface = pygame.image.fromstring(court_image.tobytes(), court_image.size, court_image.mode)

# draw the court surface onto the screen
screen.blit(court_surface, (0, 0))
# define the dimensions of the court
court_width = 2000
court_height = 1000
court_margin = 200


# define the positions of the starting players
home_players = {
    'PG': (145, 250),
    'SG': (305,500),
    'SF': (545,150),
    'PF': (545,600),
    'C': (695,350)
}

away_players = {
    'PG': (1600, 250),
    'SG': (1500,500),
    'SF': (1200,150),
    'PF': (1200,600),
    'C': (1050,350)
}


from nba_api.stats.static import players

# get all NBA players
nba_players = players.get_players()

# create a dictionary to store the player names and their ID numbers
player_ids = {player['full_name']: player['id'] for player in nba_players}


def p_stat_calculater(dfs):
    """
    This with take in a dictionary of frames for  players of a team and calculate the per game stats regardless 
    of season. #Later add the playoff to the stats
    """

    #the lists below will contine stats of  individual players
    FG_A=[]
    FG_M=[]
    FT_A=[]
    FT_M=[]
    FG3_A=[]
    FG3_M=[]
    AS=[]
    RE=[]
    TO=[]
    for key in dfs.keys():
            #drop nan row
        dataframe=pd.DataFrame()
        dataframe = dfs[key]
        dataframe = dataframe.dropna(how='any')
        dfs[key] = dataframe
        AS.append(dfs[key]['AST'].sum()/dfs[key]['GP'].sum())
        RE.append(dfs[key]['REB'].sum()/dfs[key]['GP'].sum())
        TO.append(dfs[key]['TOV'].sum()/dfs[key]['GP'].sum())
        #calculate per game attempt and made of everey player in a list
        FG_A.append(dfs[key]['FGA'].sum()/dfs[key]['GP'].sum())
        FG_M.append(dfs[key]['FGM'].sum()/dfs[key]['GP'].sum())   
        FT_A.append(dfs[key]['FTA'].sum()/dfs[key]['GP'].sum())
        FT_M.append(dfs[key]['FTM'].sum()/dfs[key]['GP'].sum()) 
        FG3_A.append(dfs[key]['FG3A'].sum()/dfs[key]['GP'].sum())
        FG3_M.append(dfs[key]['FG3M'].sum()/dfs[key]['GP'].sum())               

    FG_PCT=sum(FG_M)/sum(FG_A)
    FT_PCT=sum(FT_M)/sum(FT_A)
    FG3_PCT=sum(FG3_M)/sum(FG3_A)
    AST=np.sum(AS)
    REB=np.sum(RE)
    TOV=np.sum(TO)
    if len(dfs)==1:
        AST=AST*5
        REB=REB*5
        TOV=TOV*5 
    stats_dict = {'FG_PCT': FG_PCT, 'FT_PCT': FT_PCT, 'FG3_PCT': FG3_PCT, 'AST': AST, 'REB': REB, 'TOV': TOV}
    stats_df = pd.DataFrame(stats_dict, index=[0])
    return stats_df

dfs1={}
dfs2={}
# create a dropdown menu to let the user choose the home team player
home_column, away_column = st.columns(2)

home_team_players = []
for pos, pos_pos in home_players.items():
    with home_column:
        player_name = st.selectbox(f'***Select home team {pos}***', ['None'] + list(player_ids.keys()))
        if player_name != 'None':
            player_id = player_ids[player_name]
            response = requests.get(f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png")
            if response.status_code == 200:
                try:
                    player_image = pygame.image.load(BytesIO(response.content))
                except pygame.error:
                    # If loading the image fails, use the fallback image
                    fallback_image_path = "./images/fallback.png"
                    player_image = pygame.image.load(fallback_image_path)
            else:
                 # If the response status code is not 200, use the fallback image
                fallback_image_path = "./images/fallback.png"
                player_image = pygame.image.load(fallback_image_path)
            #player_image = pygame.image.load(BytesIO(response.content))
            home_team_players.append((pos, player_image, pos_pos))
            player_stats = PlayerCareerStats(player_id=player_id)
            dfs1[pos] = player_stats.get_data_frames()[0]

            #st.write(df)

import time
# create a dropdown menu to let the user choose the away team player
away_team_players = []
for pos, pos_pos in away_players.items():
    with away_column:
        player_name = st.selectbox(f'***Select away team {pos}***', ['None'] + list(player_ids.keys()))
        if player_name != 'None':
            player_id = player_ids[player_name]
            time.sleep(.3)
            response = requests.get(f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png")
            if response.status_code == 200:
                try:
                    player_image = pygame.image.load(BytesIO(response.content))
                except pygame.error:
                    # If loading the image fails, use the fallback image
                    fallback_image_path = "./images/fallback.png"
                    player_image = pygame.image.load(fallback_image_path)
            else:
                 # If the response status code is not 200, use the fallback image
                fallback_image_path = "./images/fallback.png"
                player_image = pygame.image.load(fallback_image_path) 
            #player_image = pygame.image.load(BytesIO(response.content))
            away_team_players.append((pos, player_image, pos_pos))
            time.sleep(1)
            player_stats = PlayerCareerStats(player_id=player_id)
            dfs2[pos]= player_stats.get_data_frames()[0]
        else:
            # Remove the key-value pair from dfs2 dictionary
            if pos in dfs2:
                del dfs2[pos]
                        #st.write(df)



    
def draw_player(player_image, position):
    screen.blit(player_image, position)
    
for player in home_team_players:
    draw_player(player[1], player[2])
    
for player in away_team_players:
    draw_player(player[1], player[2])


image_array = np.array(pygame.surfarray.array3d(screen)).swapaxes(0, 1)
st.image(image_array)

submitted = st.button("Predict")
  
if submitted:
    #user_data_home = home_team_players
    #user_data_away = away_team_players
    #Models names and errors
    models_name = [ "LinearRegression.pkl", "Ridge.pkl", "Lasso.pkl", "ElasticNet.pkl"]
    rmses=[6.944043233035345,6.941490841645733,9.664759120252853,9.665046975303072]
    # models_name = [
    # "RandomForestRegressor.pkl", "LinearRegression.pkl", "Ridge.pkl", "Lasso.pkl", "ElasticNet.pkl","SVR.pkl",
    # "DecisionTreeRegressor.pkl", "KNeighborsRegressor.pkl", "GradientBoostingRegressor.pkl",
    # "AdaBoostRegressor.pkl", "XGBRegressor.pkl"]
    # rmses=[7.1709254870222665,6.944043233035345,6.941490841645733,9.664759120252853,9.665046975303072,
    # 9.63608964815864,10.396960145512226,10.345836464048702,6.975835114681261,
    # 7.585558024483646,6.9671111027529635]

    df_home=p_stat_calculater(dfs1)
    df_home.columns = ['fg_pct_home', 'ft_pct_home', 'fg3_pct_home', 'ast_home', 'reb_home', 'to_home']
    dfs1={}
    player_1="Home Team"
    
    #st.write(df_home.head())
    df_away=p_stat_calculater(dfs2)
    df_away.columns = ['fg_pct_away', 'ft_pct_away', 'fg3_pct_away', 'ast_away', 'reb_away', 'to_away']
    dfs2={}
    player_2='Away Team'
  
    #st.write(df_away.head())
    df = pd.concat([df_home, df_away], axis=1)
    # Predict
    y_pred=[]
    valid=[]        
    x=''
    for model in (models_name):
        my_trained=joblib.load(f"./{model}")
        y_pred.append(np.floor(my_trained.predict(df)[0]))
  

        # create a table to compare the results
    table = pd.DataFrame({
        "Model": [modeln for modeln in models_name],
        "RMSE": rmses,
        f"{player_1} vs {player_2}":y_pred
    })
    del df
    #display the table
    #st.write(table)
    import plotly.graph_objs as go

    # Create a trace for the bar chart with error bars
    data = go.Bar(
        x=table['Model'],
        y=table[f"{player_1} vs {player_2}"],
        error_y=dict(
            type='data',
            array=table['RMSE'],
            visible=True
        )
    )

    # Create the layout for the bar chart
    layout = go.Layout(
        title=f'{player_1} vs {player_2} Performance Comparison',
        xaxis=dict(title='Model'),
        yaxis=dict(title=f'{player_1} vs {player_2} '),
        template='plotly_white'
    )

    # Create the figure object and add the trace and layout
    fig = go.Figure(data=[data], layout=layout)

    # Display the figure
    st.plotly_chart(fig)
#st.markdown("*The sanity test check weather the model predict draw for identical teams.")
        
# # Create a container for the logo
# # Load the logo image
# logo_1 = Image.open("./images/spiced.jpg")
# logo_1=logo_1.resize((140, 140))
# logo_2=Image.open("./images/logo_2.png")  
# logo_2=logo_2.resize((140, 120))
# #st.image(logo_3)


# # Load the logo image
# logo = Image.open("./images/spiced.jpg")

# # Create a container for the logo
# logo_container = st.sidebar.container()

# # Display the logo in the container
# with logo_container:
#     st.image(logo_1, use_column_width=False)

# logo_container1 = st.sidebar.container()
# with logo_container1:
#     st.image(logo_2, use_column_width=False)

# quit Pygame
pygame.quit()