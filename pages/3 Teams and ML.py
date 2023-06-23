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
# Create a container for the logo


st.markdown("<h1 style='font-size: 25px;text-align: center; color: red;'>The following content does not promote gambling!</h1>", unsafe_allow_html=True)
st.write(' ')
st.write(' ')
st.write(' ')

#def load_dataframe():
    #return pd.read_csv('./data/games_details.csv', index_col=0)

#game_details=load_dataframe()
#game_details.index=game_details.index.astype(str)
st.markdown("<h1 style='font-size: 20px;'>Original data contains all NBA players' performances in every single match:</h1>", unsafe_allow_html=True)

#"<h1 style='font-size: 20px;text-align: center; color: red;'>....</h1>", unsafe_allow_html=True)

st.write(f"<h1 style='font-size: 20px; text-align: center;'>DataFrame dimension: {28} x {668628}</h1>", unsafe_allow_html=True)
st.write(" ")
#st.write("<h1 style='font-size: 20px;text-align: center; color: Blue;'>After some painful pandas operations we arriv at target DataFrame:</h1>", unsafe_allow_html=True)
s1,s2,s3=st.columns([2,3,2])
hard=Image.open('./images/hardwork.png')
with s2:
    st.image(hard,use_column_width=True)

games=pd.read_csv('./data/target.csv',index_col=0)

games.index=games.index.astype(str)
# Create a custom CSS style for the title row
title_style = [
    {'selector': 'th', 'props': [('background-color', 'blue'), ('color', 'white')]}
]

# Apply the custom style to the DataFrame
styled_games = games.head(5).style.set_table_styles(title_style)

# Display the styled DataFrame using st.table()
st.table(styled_games)
# games_home=pd.read_csv('./data/target_home.csv', index_col=0)
# games_home.index=games_home.index.astype(str)
# games_away=pd.read_csv('./data/target_away.csv', index_col=0)
# games_away.index=games_away.index.astype(str)
# games_result=pd.read_csv('./data/target_result.csv', index_col=0)
# games_result.index = games_result.index.astype(str)
#st.markdown("<h1 style='font-size: 25px;'>  ... </h1>", unsafe_allow_html=True)



import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
@st.cache_data()
def plotmydata():
    cols = ['fg_pct_home', 'ft_pct_home', 'fg3_pct_home', 'ast_home', 'reb_home', 'to_home']
    num_cols = len(cols)
    num_rows = (num_cols + 1) // 2  # Calculate the number of rows needed
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
    axs = axs.flatten()  # Flatten the axes array

    for i, col in enumerate(cols):
        ax = axs[i]

        sns.scatterplot(x=col, y='result', data=games, ax=ax)

        # Fit the line using numpy.polyfit
        slope, intercept, r_value, p_value, std_err = stats.linregress(games[col], games['result'])
        line = slope * games[col] + intercept

        # Plot the fitted line with thicker line width
        ax.plot(games[col], line, color='red', linewidth=2)

        ax.set_xlabel(col,fontweight='bold', fontsize=14)
        ax.set_ylabel('result', fontweight='bold', fontsize=14)  # Bold and bigger y-axis label

        # Customize tick size and style
        ax.tick_params(axis='both', which='both', labelsize=12, width=1, length=5, pad=5)

        # Set bold and bigger tick labels
        ax.xaxis.set_tick_params(width=2, size=7)
        ax.yaxis.set_tick_params(width=2, size=7)
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontweight('bold')

        # Set bigger title and axis label sizes
        # ax.set_title(col, fontsize=16, fontweight='bold')
        # ax.xaxis.label.set_size(14)

    plt.tight_layout()
    st.pyplot(fig)

st.write(f"<h1 style='font-size: 20px;text-align: center;'>DataFrame dimension: {games.columns.size} x {games.index.size}</h1>", unsafe_allow_html=True)
st.write(' ')
st.markdown("<h1 style='font-size: 25px;'>  Let's explore how do those features relate to the result:</h1>", unsafe_allow_html=True)


plotmydata()


st.markdown("<h1 style='font-size: 20px;'>  All features have a linear effect on the spread result: Some feature engineering.</h1>", unsafe_allow_html=True)
st.write(' ')

st.markdown("<h1 style='font-size: 25px;'> <span style='color: red;'>Attention:</span> Real sport prediction is radically different than this.</h1>", unsafe_allow_html=True)
st.write(' ')
st.write(' ')
st.write(' ')
st.markdown("<h1 style='font-size: 25px;text-align: center;'>Time to let the Machine finally Learn.</h1>", unsafe_allow_html=True)
st.write(' ')
st.write(' ')


models_name = [
"RandomForestRegressor.pkl", "LinearRegression.pkl", "Ridge.pkl", "Lasso.pkl", "ElasticNet.pkl","SVR.pkl",
"DecisionTreeRegressor.pkl", "KNeighborsRegressor.pkl", "GradientBoostingRegressor.pkl",
"AdaBoostRegressor.pkl", "XGBRegressor.pkl"]
failed_models=["DecisionTreeRegressor.pkl","KNeighborsRegressor.pkl", "GradientBoostingRegressor.pkl",
"AdaBoostRegressor.pkl", "XGBRegressor.pkl"]
almost=["RandomForestRegressor.pkl","SVR.pkl"]
rmses_test=[7.1709254870222665,6.944043233035345,6.941490841645733,9.664759120252853,9.665046975303072,
9.63608964815864,10.396960145512226,10.345836464048702,6.975835114681261,
7.585558024483646,6.9671111027529635]
rmses_train=[2.711735, 6.973573, 6.974939, 9.755895, 9.752933, 9.722809, 
             0.000000, 8.564980, 6.736251, 7.627707, 5.037337]
valid=[]        
x=''
for i, model in enumerate(models_name):
    my_trained=joblib.load(f"./{model}")
    #st.write(i, y[0])
    if  model in failed_models:
        x='    Failed'
    elif model in almost:
        x='     Meh'
    else:
        x='     Passed'
    valid.append(x)

    # create a table to compare the results
table = pd.DataFrame({
    "Model": [modeln for modeln in models_name],
    "RMSE_TRAIN": rmses_train,    
    "RMSE_TEST": rmses_test,
    "Sanity test": valid,


})
def color_passed_green(row):
    if row['Sanity test'] == '     Passed':
        return ['background-color: green'] * len(row)
    return [''] * len(row)

# Apply the conditional formatting to the DataFrame
styled_table = table.style.apply(color_passed_green, axis=1).set_table_styles([
    {'selector': 'th', 'props': [('background-color', 'blue'), ('color', 'white')]}
])
#display the table
st.table(styled_table)
st.markdown("💡 The sanity test checks whether a model predicts the draw for identical teams or not.")


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

pygame.quit()