import streamlit as st
from PIL import Image
import pandas as pd


import streamlit as st
st.markdown("<h1 style='text-align: center; font-size: 25px;'>  Who is the GOAT?</h1>", unsafe_allow_html=True)
# Define the table data
table_data = [
    ['30.1', 'Points', '27.2'],
    ['6.2', 'Rebounds', '7.5'],
    ['5.3', 'Assists', '7.3'],
    ['2.3', 'Steals', '1.5'],
    ['0.8', 'Blocks', '0.8'],
    ['2.7', 'Turnovers', '3.5'],
    ['38.3', 'Minutes', '38.1'],
    ['49.7', 'Field goal %', '50.5'],
    ['32.7', '3-point %', '34.5'],
    ['83.5', 'Free throw %', '73.5']
]


# Create a DataFrame from the data
#df = pd.DataFrame(data)
df = pd.DataFrame(table_data[1:], columns=table_data[0])
# Set up the page layout
col1, col2, col3 = st.columns([1.5, 2, 1.5])

# Add the images of LeBron James and Michael Jordan to either side of the table
with col1:
    st.image('./images/michael.png', use_column_width=True)
with col3:
    st.image('./images/lebron.png', use_column_width=True)

# Add the table to the center column
with col2:
    st.write('')
    st.write('')
    st.write('')
    # Create an HTML table
    html_table = "<table align='center'>"
    for row in table_data:
        html_table += "<tr>"
        for cell in row:
            # Check if the first cell is greater than the third cell
            if cell == row[0] and float(cell) > float(row[2]):
                cell = f'<td style="background-color: green;">{cell}</td>'
            # Check if the first cell is smaller than the third cell
            elif cell == row[2] and float(cell) > float(row[0]):
                cell = f'<td style="background-color: green;">{cell}</td>'
            else:
                cell = f'<td>{cell}</td>'
            html_table += cell
        html_table += "</tr>"
    html_table += "</table>"

    # Display the HTML table in Streamlit
    st.markdown(html_table, unsafe_allow_html=True)

st.write(" ")
st.write(" ")
st.write(" ")
# Video from YouTube

st.markdown("<h1 style='font-size: 25px;'>  Watch this to see the gravity of the situation :)</h1>", unsafe_allow_html=True)
st.write(" ")
st.write(" ")
video_id = "mavBH9eFaM0"
video_url = f"https://www.youtube.com/watch?v={video_id}"
st.video(f"https://www.youtube.com/watch?v={video_id}")


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