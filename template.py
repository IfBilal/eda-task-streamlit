# lab_eda_gui.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration

st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Exploratory Data Analysis Interface")


# 2. Sidebar: Dataset Ingestion

# set header for sidebar
st.sidebar.header("Dataset Controls")

# create a file uploader in the sidebar for CSV files
uploaded_file = st.sidebar.file_uploader("Upload CSV File for Analysis", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded file as a valid CSV: {e}")
        st.stop()

    # 3. Dataset Overview

    # set subheader for dataset overview
    st.subheader("Dataset Preview & Metadata")
    st.write("**First 5 Rows:**")
    # display the first 5 rows of the dataset
    st.dataframe(df.head())

    # display the shape of the dataset
    st.write("**Shape:**", df.shape)

    st.write("**Column Data Types:**")
    # display the data types of each column in the dataset
    st.dataframe(df.dtypes.astype(str).rename("Data Type"))

    # Missing value summary
    st.write("**Missing Values per Column:**")
    # display the count and percentage of missing values for each column in the dataset
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_pct
    })
    st.dataframe(missing_df)

    # Basic statistics for numerical columns
    st.write("**Basic Numerical Statistics:**")
    # display the basic statistics
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe().loc[["mean", "50%", "min", "max"]].rename(index={"50%": "median"}))
    else:
        st.info("No numerical columns found in this dataset.")

    # 4. Attribute Selection

    # set header for attribute selection in the sidebar
    st.sidebar.header("Attribute Selection")
    # create a selectbox in the sidebar to choose an attribute for visualization
    selected_column = st.sidebar.selectbox("Select Attribute for Visualization", df.columns)

    # Detect column type
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        column_type = "Numerical"
    else:
        column_type = "Categorical"

    # 5. Visualization Rendering

    st.subheader("Visualization")

    if column_type == "Numerical":
        # Histogram with seaborn
        fig, ax = plt.subplots()
        sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
        ax.set_title(f"Histogram of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        # Bar chart for categorical
        counts = df[selected_column].value_counts()
        percentages = (counts / counts.sum() * 100).round(2)

        fig, ax = plt.subplots()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
        ax.set_title(f"Frequency Counts of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        st.write("**Percentage Breakdown:**")
        st.dataframe(pd.DataFrame({"Count": counts, "Percentage (%)": percentages}))

else:
    st.info("Please upload a CSV file to start EDA.")
