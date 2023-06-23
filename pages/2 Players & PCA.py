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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# initialize Pygame
pygame.init()

# # First page
# Load logo images

co1, co2=st.columns([12,1])
# logo_nba = Image.open("./images/nba.jpg")
# with co2:
#     st.image(logo_nba, width=200)  # Add the NBA logo image
with co1:
  st.title('Principal Componenet Analysis')
st.write(f"<h1 style='font-size: 20px;'>PCA here allows us to visually classify players in comparison to each other. </h1>", unsafe_allow_html=True)

# Function to load the dataframe from a CSV file
def load_dataframe():
    return pd.read_csv('./data/players_pca.csv', index_col=0)

##### Ask for a playerf from the user, calculater its stats using the function

selected_players=[]
final_df=load_dataframe()
st.write(' ')
st.markdown("<h1 style='font-size: 20px;'>To start with: A Dataframe constituted of 150 randomly chosen NBA players.</h1>", unsafe_allow_html=True)
st.write(' ')

styled_df = final_df.head(5).style.set_table_styles([
    {
        'selector': 'th.col_heading',
        'props': [
            ('background-color', 'darkblue'),
            ('color', 'white')
        ]
    }
])
st.table(styled_df)
#st.write(f"<h1 style='font-size: 20px;'>Data is being obtained live with: Official NBA_API </h1>", unsafe_allow_html=True)
st.write(' ')
st.write(f"<h1 style='font-size: 20px;'>Data is being obtained live via: Official NBA_API </h1>", unsafe_allow_html=True)
def pca_calculator(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=3)
    pca.fit(df_scaled)
    #st.write("Explained Variance Ratios:")
    #st.write(pca.explained_variance_ratio_)

    pca_scores = pca.transform(df_scaled)

    pc1 = pca_scores[:, 0] 
    pc2 = pca_scores[:, 1]

    plt.figure(figsize=(6, 4))
    plt.scatter(pc1, pc2)

    for i, player in enumerate(df.index):
        if player in ['LeBron James', 'Michael Jordan'] or player in selected_players:
            plt.annotate(player, (pc1[i], pc2[i]))

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Plot of Players')
    st.pyplot()
       ############ 
    pc1 = pca_scores[:, 0] 
    pc2 = pca_scores[:, 2]

    plt.figure(figsize=(6, 4))
    plt.scatter(pc1, pc2)

    for i, player in enumerate(df.index):
        if player in ['LeBron James', 'Michael Jordan'] or player in selected_players:
            plt.annotate(player, (pc1[i], pc2[i]))

    plt.xlabel('PC1')
    plt.ylabel('PC3')
    plt.title('PCA Plot of Players')

    st.pyplot()
           ############ 
    # pc1 = pca_scores[:, 1]
    # pc2 = pca_scores[:, 2]

    # plt.figure(figsize=(6, 4))
    # plt.scatter(pc1, pc2)

    # for i, player in enumerate(df.index):
    #     if player in ['LeBron James', 'Michael Jordan'] or player in selected_players:
    #         plt.annotate(player, (pc1[i], pc2[i]))

    # plt.xlabel('PC2')
    # plt.ylabel('PC3')
    # plt.title('PCA Plot of Players')

    # st.pyplot()
    
    st.write("***PCA Components:***")
    pc_df={}
    pc_df=pd.DataFrame(data=(pca.components_), index=['PC1', 'PC2', 'PC3'], columns=final_df.columns)
    pc_style=pc_df.style.set_table_styles([
    {
        'selector': 'th.col_heading',
        'props': [
            ('background-color', 'darkblue'),
            ('color', 'white')
        ]
    }
])
    st.table(pc_style)
    return None

st.set_option('deprecation.showPyplotGlobalUse', False)

def p_stat_calculater(df):
    """
    This function takes in the dataframe for a player and calculates the per game stats regardless 
    of season. It drops rows with null values before calculating the stats. 
    If a division by zero occurs, it sets the result to zero.
    """
    # Drop rows with null values
    df = df.dropna()
    
    total_games = df['GP'].sum()
    if total_games<50.0:
        return None
    
    try:
        FG_PCT = df['FGM'].sum() / df['FGA'].sum()
    except ZeroDivisionError:
        FG_PCT = 0.0
    
    try:
        FT_PCT = df['FTM'].sum() / df['FTA'].sum()
    except ZeroDivisionError:
        FT_PCT = 0.0
    
    try:
        FG3_PCT = df['FG3M'].sum() / df['FG3A'].sum()
    except ZeroDivisionError:
        FG3_PCT = 0.0
    
    AST = df['AST'].sum() / total_games
    REB = df['REB'].sum() / total_games
    TOV = df['TOV'].sum() / total_games
    BLK = df['BLK'].sum() / total_games
    STL = df['STL'].sum() / total_games
    PTS = df['PTS'].sum() / total_games
    
    stats_dict = {'PTS': PTS,'FG_PCT': FG_PCT, 'FT_PCT': FT_PCT, 'FG3_PCT': FG3_PCT, 'AST': AST, 'REB': REB,  'STL': STL,'BLK': BLK, 'TOV': TOV}
    
    stats_df = pd.DataFrame(stats_dict, index=[0])
    
    return stats_df


    

# Function to save the dataframe to a CSV file
def save_dataframe(df):
    df.to_csv('./data/players_pca.csv')



from nba_api.stats.static import players
import requests
# get all NBA players
nba_players = players.get_players()
# create a dictionary to store the player names and their ID numbers
player_ids = {player['full_name']: player['id'] for player in nba_players}

#let the user choose the player:
player_name = st.selectbox(f'**Select your player**', ['None'] + list(player_ids.keys()))
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
            selected_players.append(player_name)

col1,col2, col3=st.columns([1,1,4])

with col1:
    submitted = st.button("Submit",help="Send player for PCA analysis")
with col2:
    OKAY = st.button("Update",help="Show player on the graph")
#df = 
# Display the initial PCA graph for final_df
pca_calculator(final_df)

# Empty placeholder for the PCA graph
pca_placeholder = st.empty()
#st.dataframe(df.head())
import time
if submitted:
    # Clear the output area
    pca_placeholder.empty()
    # Iterate over each player
    for name in selected_players:
        time.sleep(3)
        player_id=player_ids[name]  
        try:
            # Fetch the player's data using PlayerCareerStats(player_id=player_id)
            player_stats = PlayerCareerStats(player_id=player_id).get_data_frames()[0]
            
            # Calculate the player's stats using p_stat_calculator function
            stats_df = p_stat_calculater(player_stats)
            if stats_df is None:
                continue
            # Add player's name as an index to the dataframe
            stats_df['Player Name'] = name
            stats_df.set_index('Player Name', inplace=True)
            
            # Merge the player's stats into the main dataframe
            final_df = pd.concat([final_df, stats_df])
            save_dataframe(final_df)
            # Sleep for 1 second before processing the next player
            time.sleep(1)
            pca_placeholder.pyplot(pca_calculator(final_df))   
            #pca_explained() 
        except requests.exceptions.ReadTimeout:
            # Handle ReadTimeout error
            #print(f"Timeout error occurred for player: {name}. Skipping...")
           player_stats = None
        

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
# st.markdown('Players data: Obtained using NBA_API and post_processed')
# st.write('Here is the head of the DataFrame:')