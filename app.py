import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.subplots as sp
import folium
from streamlit_folium import folium_static
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")


# ##########################################################################################################################################


@st.cache_data
def load_data1():
    df = pd.read_csv('data/count.csv', low_memory=False)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df


def load_data2():
    dft = pd.read_csv('data/dft.csv', low_memory=False)
    dft.drop('Unnamed: 0', axis=1, inplace=True)
    return dft


def round_dataframe_to_integers(dataframe):
    rounded_data = dataframe.copy()
    for col in dataframe.columns[1:]:
        rounded_data[col] = dataframe[col].round(0).astype(int)
    return rounded_data

# ##########################################################################################################################################


def lineplot_subplots(dataframe):
    x_column = dataframe.columns[0]  # Use the first column as x-values
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### {y_column} - Line Chart")

        # Create the line chart using Plotly Express
        fig = px.line(dataframe, x=x_column, y=y_column,
                      title=f"{y_column} - Line Chart", width=1100)
        fig.update_xaxes(tickangle=45)

        # Display the plot using Streamlit
        st.plotly_chart(fig)


def lineplot_top_subplots(dataframe):
    x_column = dataframe.columns[0]  # Use the first column as x-values
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### Top 10 {y_column} - Line Chart")

        # Sort the column in descending order and take the top 10 values
        sorted_data = dataframe.sort_values(
            by=y_column, ascending=False).head(10)

        # Create the line chart using Plotly Express
        fig = px.line(sorted_data, x=x_column, y=y_column,
                      title=f"Top 10 {y_column} - Line Chart")
        fig.update_xaxes(tickangle=45)

        # Display the plot using Streamlit
        st.plotly_chart(fig)


def lineplot_bottom_subplots(dataframe):
    x_column = dataframe.columns[0]  # Use the first column as x-values
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### Bottom 10 {y_column} - Line Chart")

        # Sort the column in descending order and take the bottom 10 values
        sorted_data = dataframe.sort_values(
            by=y_column, ascending=True).head(10)

        # Create the line chart using Plotly Express
        fig = px.line(sorted_data, x=x_column, y=y_column,
                      title=f"Bottom 10 {y_column} - Line Chart")
        fig.update_xaxes(tickangle=45)

        # Display the plot using Streamlit
        st.plotly_chart(fig)

# ##########################################################################################################################################


def pieplot_subplots(dataframe):
    x_column = dataframe.columns[0]  # Use the first column as category names
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### {y_column} - Pie Chart")

        # Create the pie chart using Plotly Express
        fig = px.pie(dataframe, names=x_column, values=y_column,
                     title=f"{y_column} - Pie Chart", width=800)

        # Display the plot using Streamlit
        st.plotly_chart(fig)


def pieplot_top_subplots(dataframe, num_values=5):
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### Top {num_values} {y_column} - Pie Chart")

        # Sort the column in descending order and take the top values
        sorted_data = dataframe.sort_values(
            by=y_column, ascending=False).head(10)

        # Create the pie chart using Plotly Express
        fig = px.pie(sorted_data, names=dataframe.columns[0], values=y_column,
                     title=f"Top {num_values} {y_column} - Pie Chart", width=800)

        # Display the plot using Streamlit
        st.plotly_chart(fig)


def pieplot_bottom_subplots(dataframe, num_values=5):
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### Bottom {num_values} {y_column} - Pie Chart")

        # Sort the column in ascending order and take the bottom values
        sorted_data = dataframe.sort_values(
            by=y_column, ascending=True).head(10)

        # Create the pie chart using Plotly Express
        fig = px.pie(sorted_data, names=dataframe.columns[0], values=y_column,
                     title=f"Bottom {num_values} {y_column} - Pie Chart", width=800)

        # Display the plot using Streamlit
        st.plotly_chart(fig)

# ##########################################################################################################################################


def boxplot_subplots(dataframe):
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### {y_column} - Box Plot")

        # Create the box plot using Plotly Express
        fig = px.box(
            dataframe, x=dataframe.columns[0], y=y_column, title=f"{y_column} - Box Plot", width=800)

        # Display the plot using Streamlit
        st.plotly_chart(fig)


def boxplot_top_subplots(dataframe, num_values=5):
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### Top {num_values} {y_column} - Box Plot")

        # Sort the column in descending order and take the top values
        sorted_data = dataframe.sort_values(
            by=y_column, ascending=False).head(10)

        # Create the box plot using Plotly Express
        fig = px.box(sorted_data, x=dataframe.columns[0], y=y_column,
                     title=f"Top {num_values} {y_column} - Box Plot", width=800)

        # Display the plot using Streamlit
        st.plotly_chart(fig)


def boxplot_bottom_subplots(dataframe, num_values=5):
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### Bottom {num_values} {y_column} - Box Plot")

        # Sort the column in ascending order and take the bottom values
        sorted_data = dataframe.sort_values(
            by=y_column, ascending=True).head(10)

        # Create the box plot using Plotly Express
        fig = px.box(sorted_data, x=dataframe.columns[0], y=y_column,
                     title=f"Bottom {num_values} {y_column} - Box Plot", width=800)

        # Display the plot using Streamlit
        st.plotly_chart(fig)


# ##########################################################################################################################################

def scatterplot_subplots(dataframe):
    x_column = dataframe.columns[0]  # Use the first column as x-values
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### {y_column} - Scatter Plot")

        # Create the scatter plot using Plotly Express
        fig = px.scatter(dataframe, x=x_column, y=y_column,
                         title=f"{y_column} - Scatter Plot", width=800)

        # Display the plot using Streamlit
        st.plotly_chart(fig)


def scatterplot_top_subplots(dataframe, num_values=5):
    x_column = dataframe.columns[0]  # Use the first column as x-values
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### Top {num_values} {y_column} - Scatter Plot")

        # Sort the column in descending order and take the top values
        sorted_data = dataframe.sort_values(
            by=y_column, ascending=False).head(10)

        # Create the scatter plot using Plotly Express
        fig = px.scatter(sorted_data, x=x_column, y=y_column,
                         title=f"Top {num_values} {y_column} - Scatter Plot", width=800)

        # Display the plot using Streamlit
        st.plotly_chart(fig)


def scatterplot_bottom_subplots(dataframe, num_values=5):
    x_column = dataframe.columns[0]  # Use the first column as x-values
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### Bottom {num_values} {y_column} - Scatter Plot")

        # Sort the column in ascending order and take the bottom values
        sorted_data = dataframe.sort_values(
            by=y_column, ascending=True).head(10)

        # Create the scatter plot using Plotly Express
        fig = px.scatter(sorted_data, x=x_column, y=y_column,
                         title=f"Bottom {num_values} {y_column} - Scatter Plot", width=800)

        # Display the plot using Streamlit
        st.plotly_chart(fig)

# ##########################################################################################################################################


def heatmap_subplots(dataframe):
    x_column = dataframe.columns[0]  # Use the first column as x-values
    for i in range(1, len(dataframe.columns)):
        y_column = dataframe.columns[i]

        st.write(f"### Heatmap for {y_column}")

        # Create the heatmap using Plotly Express
        fig = px.imshow(dataframe.pivot(x_column, y_column, y_column),
                        x=x_column, y=y_column, title=f"Heatmap for {y_column}")

        # Display the plot using Streamlit
        st.plotly_chart(fig)


# ##########################################################################################################################################

def custom_lineplot(dataframe):
    st.title('Custom Line Plot')
    
    # Create a multiselect widget to allow the user to choose columns for the y-axis
    selected_columns = st.multiselect("Select columns for the y-axis", dataframe.columns[1:])

    if not selected_columns:
        st.warning("Please select at least one column.")
        return

    # Create the line plot using Plotly Express
    fig = px.line(dataframe, x='country', y=selected_columns, title="Custom Line Plot")

    # Customize the plot layout if needed
    fig.update_layout(
        xaxis_title='Country',
        yaxis_title='Value',
    )

    # Display the plot using Streamlit
    st.plotly_chart(fig)


def custom_pieplot(dataframe):
    st.title('Custom Pie Chart')
    
    # Create a multiselect widget to allow the user to choose columns for the pie chart
    selected_column = st.selectbox("Select a column for the pie chart", dataframe.columns[1:])

    if not selected_column:
        st.warning("Please select a column.")
        return

    # Create the pie chart using Plotly Express
    fig = px.pie(dataframe, names='country', values=selected_column, title="Custom Pie Chart")

    # Customize the chart layout if needed
    fig.update_traces(textinfo='percent+label')

    # Display the pie chart using Streamlit
    st.plotly_chart(fig)


def custom_boxplot(dataframe):
    st.title('Custom Box Plot')
    
    # Create a selectbox widget to allow the user to choose a column for the box plot
    selected_column = st.selectbox("Select a column for the box plot", dataframe.columns[1:])

    if not selected_column:
        st.warning("Please select a column.")
        return

    # Create the box plot using Plotly Express
    fig = px.box(dataframe, x='country', y=selected_column, title="Custom Box Plot")

    # Customize the chart layout if needed
    fig.update_traces(boxpoints='outliers')

    # Display the box plot using Streamlit
    st.plotly_chart(fig)



def custom_scatterplot(dataframe):
    st.title('Custom Scatter Plot')
    
    # Create a multiselect widget to allow the user to choose columns for the y-axis
    selected_columns = st.multiselect("Select columns for the y-axis", dataframe.columns[1:])

    if not selected_columns:
        st.warning("Please select at least one column.")
        return

    # Create the scatter plot using Plotly Express
    fig = px.scatter(dataframe, x='country', y=selected_columns, title="Custom Scatter Plot")

    # Customize the chart layout if needed

    # Display the scatter plot using Streamlit
    st.plotly_chart(fig)


# ##########################################################################################################################################

def country_lineplot(df):
    st.title("Country Line Plots")

    # Check if there are at least two columns in the DataFrame
    if len(df.columns) < 2:
        st.write("The DataFrame should have at least two columns.")
        return

    # Use the first column as y-values
    y_column = df.columns[0]

    # Use the remaining columns as x-values
    x_columns = df.columns[1:]

    # Create line charts for each x value
    for x_column in x_columns:
        st.subheader(f"Line Plot for {x_column}")
        fig = px.line(df, x=x_column, y=y_column,
                      title=f"{y_column} vs. {x_column}")
        st.plotly_chart(fig)


def country_boxplot(df):
    st.title("Country Box Plots")

    # Check if there are at least two columns in the DataFrame
    if len(df.columns) < 2:
        st.write("The DataFrame should have at least two columns.")
        return

    # Use the first column as y-values
    y_column = df.columns[0]

    # Use the remaining columns as x-values
    x_columns = df.columns[1:]

    # Create box plots for each x value
    for x_column in x_columns:
        st.subheader(f"Box Plot for {x_column}")
        fig = px.box(df, x=x_column, y=y_column,
                     title=f"{y_column} vs. {x_column}")
        st.plotly_chart(fig)


def country_scatterplot(df):
    st.title("Country Scatter Plots")

    # Check if there are at least two columns in the DataFrame
    if len(df.columns) < 2:
        st.write("The DataFrame should have at least two columns.")
        return

    # Use the first column as y-values
    y_column = df.columns[0]

    # Use the remaining columns as x-values
    x_columns = df.columns[1:]

    # Create scatter plots for each x value
    for x_column in x_columns:
        st.subheader(f"Scatter Plot for {x_column}")
        fig = px.scatter(df, x=x_column, y=y_column,
                         title=f"{y_column} vs. {x_column}")
        st.plotly_chart(fig)


# ##########################################################################################################################################

def decade_lineplot(df):
    st.title("Line Charts by Decade")

    # Create checkboxes for each decade
    decades = sorted(df_decades['Decade'].unique())
    selected_decades = st.multiselect(
        "Select Decades", decades, default=decades)

    # Filter the DataFrame based on selected decades
    filtered_df = df_decades[df_decades['Decade'].isin(selected_decades)]

    # Create line charts for the selected decades
    if not filtered_df.empty:
        fig = px.line(filtered_df, x='Year', y='GDP Per Capita',
                      color='country', title='GDP Per Capita by Country')
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected decades.")


def decade_pieplot(df):
    # Create a Streamlit app
    st.title("Pie Charts by Decade")

    # Create checkboxes for each decade
    decades = sorted(df_decades['Decade'].unique())
    selected_decades = st.multiselect(
        "Select Decades", decades, default=decades)

    # Filter the DataFrame based on selected decades
    filtered_df = df_decades[df_decades['Decade'].isin(selected_decades)]

    # Create pie charts for the selected decades
    if not filtered_df.empty:
        fig = px.pie(filtered_df, names='country',
                     values='GDP Per Capita', title='GDP Per Capita by Country')
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected decades.")


def decade_boxplot(df):

    # Create a Streamlit app
    st.title("Box Plots by Decade")

    # Create checkboxes for each decade
    decades = sorted(df_decades['Decade'].unique())
    selected_decades = st.multiselect(
        "Select Decades", decades, default=decades)

    # Filter the DataFrame based on selected decades
    filtered_df = df_decades[df_decades['Decade'].isin(selected_decades)]

    # Create box plots for the selected decades
    if not filtered_df.empty:
        fig = px.box(filtered_df, x='Decade', y='GDP Per Capita',
                     points="all", title='GDP Per Capita by Decade')
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected decades.")


def decade_scatterplot(df):

    # Create a Streamlit app
    st.title("Scatter Plots by Decade")

    # Create checkboxes for each decade
    decades = sorted(df_decades['Decade'].unique())
    selected_decades = st.multiselect(
        "Select Decades", decades, default=decades)

    # Filter the DataFrame based on selected decades
    filtered_df = df_decades[df_decades['Decade'].isin(selected_decades)]

    # Create scatter plots for the selected decades
    if not filtered_df.empty:
        fig = px.scatter(filtered_df, x='Year', y='GDP Per Capita',
                         color='country', title='GDP Per Capita by Country')
        st.plotly_chart(fig)
    else:
        st.write("No data available for the selected decades.")


# ########################################################################################################################################
d1 = load_data1()
d2 = load_data2()

df1 = round_dataframe_to_integers(d1)
df2 = round_dataframe_to_integers(d2)
dft1 = df1.iloc[:, [0] + list(range(1, len(df1.columns)))]
dft1.sort_index(axis=1)
# st.write(dft1)
# st.write(df2.sort_values(by='country'))
# st.write(df2)

df_decades = df2.copy()
df_decades.columns = df_decades.columns.astype(str)
df_decades = df_decades.melt(
    id_vars=['country'], var_name='Year', value_name='GDP Per Capita')
df_decades['Decade'] = (df_decades['Year'].astype(int) // 10) * 10
# st.write(df_decades)
# st.write(df1.keys())
# lineplot_subplots(df2)

# ##########################################################################################################################################

st.sidebar.title("EDA Options Currently")
options = [
    'Introduction',
    'About',
    'View Data',
    'Analysis - Line Chart By Years',
    'Analysis - Pie Chart By Years',
    'Analysis - Box Chart By Years',
    'Analysis - Scatter Chart By Years',
    'Analysis - Heatmap Chart By Years',
    'Analysis - Custom Charts By Years',
    'Analysis - Charts By Country',
    'Analysis - Charts By Decades',
    'Conclusion'
]
menu = st.sidebar.radio("Select an Option", options)

# ##########################################################################################################################################

if menu == options[0]:
    st.header("Introduction")

# ##########################################################################################################################################
if menu == options[1]:
    st.header("About")

# ##########################################################################################################################################
if menu == options[2]:
    st.header("View Data")
    cols = ['By Year', 'By Country', 'By Decade', 'Country Names']
    choice = st.selectbox("select a columns", cols)
    if choice == cols[0]:
        st.subheader("Data By Year")
        st.write(dft1)
    if choice == cols[1]:
        st.subheader("Data by Country")
        st.write(df2)
    if choice == cols[2]:
        st.subheader("Data by Decade")
        st.write(df_decades)
    if choice == cols[3]:
        st.subheader("Country Names")
        st.write(dft1.keys())

# ##########################################################################################################################################
if menu == options[3]:
    st.header("Analysis - Line Chart By Years")
    cols = ['Top 10', 'Bottom 10', 'Country']
    choice = st.selectbox("select a columns", cols)

    if choice == cols[0]:
        st.subheader("Top 10")
        st.write(lineplot_top_subplots(df2))
    if choice == cols[1]:
        st.subheader("Bottom 10")
        st.write(lineplot_bottom_subplots(df2))
    if choice == cols[2]:
        st.subheader("Country")
        st.write(lineplot_subplots(df2))
# ##########################################################################################################################################
if menu == options[4]:
    st.header("Analysis - Pie Chart By Years")
    cols = ['Top 10', 'Bottom 10', 'Country']
    choice = st.selectbox("select a columns", cols)

    if choice == cols[0]:
        st.subheader("Top 10")
        st.write(pieplot_top_subplots(df2))
    if choice == cols[1]:
        st.subheader("Bottom 10")
        st.write(pieplot_bottom_subplots(df2))
    if choice == cols[2]:
        st.subheader("Country")
        st.write(pieplot_subplots(df2))
# ##########################################################################################################################################
if menu == options[5]:
    st.header("Analysis - Box Chart By Years")
    cols = ['Top 10', 'Bottom 10', 'Country']
    choice = st.selectbox("select a columns", cols)

    if choice == cols[0]:
        st.subheader("Top 10")
        st.write(boxplot_top_subplots(df2))
    if choice == cols[1]:
        st.subheader("Bottom 10")
        st.write(boxplot_bottom_subplots(df2))
    if choice == cols[2]:
        st.subheader("Country")
        st.write(boxplot_subplots(df2))
# ##########################################################################################################################################
if menu == options[6]:
    st.header("Analysis - Scatter Chart By Years")
    cols = ['Top 10', 'Bottom 10', 'Country']
    choice = st.selectbox("select a columns", cols)

    if choice == cols[0]:
        st.subheader("Top 10")
        st.write(scatterplot_top_subplots(df2))
    if choice == cols[1]:
        st.subheader("Bottom 10")
        st.write(scatterplot_bottom_subplots(df2))
    if choice == cols[2]:
        st.subheader("Country")
        st.write(scatterplot_subplots(df2))
# ##########################################################################################################################################
if menu == options[7]:
    st.header("Analysis - Heatmap Chart By Years")
    st.subheader("WIP")
    # st.write(df)
    df = df2
    df.set_index('country', inplace=True)
    # Create the heatmap using Seaborn
    plt.figure(figsize=(100, 75))
    sns.heatmap(df, cmap="YlGnBu", annot=True, fmt=".1f", linewidths=0.5)
    plt.title('Heatmap of Data by Year and Country')

    # Display the heatmap using Streamlit
    st.pyplot(plt)
# ##########################################################################################################################################
if menu == options[8]:
    st.header("Analysis - Custom Charts By Years")
    cols = ['Line', 'Pie', 'Box', 'Scatter']
    choice = st.selectbox("select a columns", cols)

    if choice == cols[0]:
        st.subheader("Line Plot")
        st.write(custom_lineplot(df2))
    if choice == cols[1]:
        st.subheader("Pie Plot")
        st.write(custom_pieplot(df2))
    if choice == cols[2]:
        st.subheader("Box Plot")
        st.write(custom_boxplot(df2))
    if choice == cols[3]:
        st.subheader("Scatter Plot")
        st.write(custom_scatterplot(df2))
# ##########################################################################################################################################
if menu == options[9]:
    st.header("Analysis - Charts By Country")
    cols = ['Line', 'Box', 'Scatter']
    choice = st.selectbox("select a columns", cols)

    if choice == cols[0]:
        st.subheader("Line Plot")
        st.write(country_lineplot(dft1))
    if choice == cols[1]:
        st.subheader("Box Plot")
        st.write(country_boxplot(dft1))
    if choice == cols[2]:
        st.subheader("Scatter Plot")
        st.write(country_scatterplot(dft1))
# ##########################################################################################################################################
if menu == options[10]:
    st.header("Analysis - Charts By Decades")
    cols = ['Line', 'Pie', 'Box', 'Scatter']
    choice = st.selectbox("select a columns", cols)

    if choice == cols[0]:
        st.subheader("Line Plot")
        st.write(decade_lineplot(df_decades))
    if choice == cols[1]:
        st.subheader("Pie Plot")
        st.write(decade_pieplot(df_decades))
    if choice == cols[2]:
        st.subheader("Box Plot")
        st.write(decade_boxplot(df_decades))
    if choice == cols[3]:
        st.subheader("Scatter Plot")
        st.write(decade_scatterplot(df_decades))
# ##########################################################################################################################################
if menu == options[11]:
    st.header("Conclusion")
