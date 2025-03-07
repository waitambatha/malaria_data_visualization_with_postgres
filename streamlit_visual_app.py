import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Load data
def load_data():
    file_path = "malaria_hf_dqa.csv"  # Update if needed
    df = pd.read_csv(file_path)
    return df.fillna(0)  # Replace NaN values with 0


df = load_data()

# Sidebar Filters
st.sidebar.header("Filter Data")
columns = st.sidebar.multiselect("Select Columns to Display", df.columns.tolist(), default=df.columns[:5])

# Detailed Summary Section
st.title("Malaria Data Analysis Dashboard")
st.write("### Dataset Overview")

selected_column = st.selectbox("Select a Column for Summary", df.columns)

if selected_column:
    st.write(f"### Summary of {selected_column}")
    st.write(f"- **Unique Values:** {df[selected_column].nunique()}")
    st.write(f"- **Most Common Value:** {df[selected_column].mode()[0]}")
    st.write(f"- **Missing Values:** {df[selected_column].isna().sum()}")
    st.write(f"- **Data Type:** {df[selected_column].dtype}")

    if df[selected_column].dtype == 'object':
        st.write("- **Frequent Values:**")
        st.write(df[selected_column].value_counts())
    else:
        st.write(f"- **Mean:** {df[selected_column].mean():.2f}")
        st.write(f"- **Median:** {df[selected_column].median():.2f}")
        st.write(f"- **Standard Deviation:** {df[selected_column].std():.2f}")

# Data Overview
st.write("### Dataset Preview")
st.dataframe(df[columns])

# Summary Statistics
st.write("### Data Summary Statistics")
st.dataframe(df.describe().transpose())

# Interactive Charts
st.write("### Data Visualization")

# Bar Chart
st.write("#### Bar Chart")
x_axis_bar = st.selectbox("Select X-axis for Bar Chart", df.columns, key="bar_x")
y_axis_bar = st.selectbox("Select Y-axis for Bar Chart", df.columns, key="bar_y")
fig_bar = px.bar(df, x=x_axis_bar, y=y_axis_bar, title=f"Bar Chart: {x_axis_bar} vs {y_axis_bar}")
st.plotly_chart(fig_bar)

# Line Chart
st.write("#### Line Chart")
x_axis_line = st.selectbox("Select X-axis for Line Chart", df.columns, key="line_x")
y_axis_line = st.selectbox("Select Y-axis for Line Chart", df.columns, key="line_y")
fig_line = px.line(df, x=x_axis_line, y=y_axis_line, title=f"Line Chart: {x_axis_line} vs {y_axis_line}")
st.plotly_chart(fig_line)

# Donut Chart
st.write("#### Donut Chart")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

if categorical_cols and numerical_cols:
    x_axis_donut = st.selectbox("Select Category for Donut Chart", categorical_cols, key="donut_x")
    y_axis_donut = st.selectbox("Select Values for Donut Chart", numerical_cols, key="donut_y")
    donut_data = df[[x_axis_donut, y_axis_donut]].copy()
    donut_data[y_axis_donut] = pd.to_numeric(donut_data[y_axis_donut], errors='coerce').fillna(0)
    fig_donut = px.pie(donut_data, names=x_axis_donut, values=y_axis_donut, hole=0.4,
                       title=f"Donut Chart: {x_axis_donut}")
    st.plotly_chart(fig_donut)
else:
    st.warning("No categorical or numerical columns available for donut chart.")

# Heatmap
st.write("#### Heatmap")
selected_columns = st.multiselect("Select Columns for Heatmap", numerical_cols, default=numerical_cols[:5])
if len(selected_columns) > 1:
    fig_heatmap = plt.figure(figsize=(10, 6))
    sns.heatmap(df[selected_columns].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig_heatmap)
else:
    st.warning("Select at least two numerical columns for heatmap.")

# Download Data
st.write("### Download Filtered Data")
st.download_button("Download CSV", df.to_csv(index=False), "filtered_data.csv", "text/csv")
