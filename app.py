import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from joblib import load
import numpy as np

# Page configuration set as the first command
st.set_page_config(page_title='Engine Health Analysis', layout='wide')


# Load the dataset
@st.cache_data
def load_data(filename):
    data = pd.read_csv(filename)
    return data

df = load_data('engine_data.csv')  # Ensure 'engine_data.csv' is in the same directory as your app.py file

# Title and introduction
st.title('Engine Health Analysis Dashboard')
st.write("""
The goal of this visualization project is to delve into the intricate relationship between various engine performance metrics and engine health. This exploration is pivotal for enhancing our understanding of engine performance and facilitating predictive maintenance.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", ["About", "Data Visualization", "Correlation Heatmap", "Categorical Analysis", "Predict Engine Health"])

st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
# Now display the logo and developer credit at what is effectively the bottom of the sidebar
logo = Image.open('logo.png')
# Resize the logo as desired
logo_small = logo.resize((int(logo.width / 5), int(logo.height / 5)))

# Display the resized logo under the navigation menu
st.sidebar.image(logo_small, use_column_width=False)  # Set use_column_width to False to keep the resized dimensions
st.sidebar.write("Developed by M. Habib Agrebi")

# About section
if selection == "About":
    st.header("About This Project")
    st.write("""
    In this project, we aim to uncover the intricate relationships between various engine performance metrics and their impact on engine health. By examining metrics such as engine RPM, lubricating oil pressure, and coolant temperature, among others, we seek to answer pertinent questions regarding engine health.
    """)
    # Load the engine image
    engine_image = Image.open('engine.jpg')  # Replace 'engine_image.jpg' with your engine image file name
    
    # Resize the image to a specific width while maintaining aspect ratio
    base_width = 800  # Adjust this value as needed for the desired width
    w_percent = (base_width / float(engine_image.size[0]))
    h_size = int((float(engine_image.size[1]) * float(w_percent)))
    engine_image = engine_image.resize((base_width, h_size), Image.ANTIALIAS)

    # Display the resized engine image
    st.image(engine_image, caption='Engine Visualization', use_column_width=False)


# Data Visualization section
if selection == "Data Visualization":
    st.header("Data Visualization")

    # Sub-options for Data Visualization
    viz_selection = st.selectbox(
        "Select a visualization",
        ["Engine RPM vs. Lub Oil Pressure", "Pairplot of Engine Metrics", "RPM Distribution", "Metric Distributions"]
    )

    # Visualization 1: Engine RPM vs. Lub Oil Pressure
    if viz_selection == "Engine RPM vs. Lub Oil Pressure":
        st.subheader("Engine RPM vs. Lub Oil Pressure by Engine Condition")
        fig1 = px.scatter(df, x="Engine rpm", y="Lub oil pressure", color="Engine Condition",
                          title="Engine RPM vs. Lub Oil Pressure by Engine Condition")
        st.plotly_chart(fig1)

    # Visualization 2: Pairplot of Engine Metrics by Engine Condition
    elif viz_selection == "Pairplot of Engine Metrics":
        st.subheader("Pairplot of Engine Metrics by Engine Condition")
        features = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "Engine Condition"]
        fig2 = px.scatter_matrix(df[features], color="Engine Condition",
                                 title="Pairplot of Engine Metrics by Engine Condition")
        st.plotly_chart(fig2)

    # Visualization 5: Distribution of Engine RPM by Condition
    elif viz_selection == "RPM Distribution":
        st.subheader("Distribution of Engine RPM by Condition")
        fig5 = px.histogram(df, x='Engine rpm', color='Engine Condition', barmode='overlay',
                            title='Distribution of Engine RPM by Condition',
                            labels={'Engine rpm': 'Engine RPM'})
        st.plotly_chart(fig5)
 # Visualization 6: Pairwise Relationships of Engine Metrics by Condition
    elif viz_selection == "Pairwise Relationships":
        st.subheader("Pairwise Relationships of Engine Metrics by Condition")
        pairplot_fig = sns.pairplot(df, hue='Engine Condition', diag_kind='kde', markers=["o", "s"], palette="Set1")
        plt.suptitle('Pairwise Relationships of Engine Metrics by Condition', y=1.02)
        st.pyplot(plt)

    # Visualization 7: Engine Metrics Distribution by Condition
    elif viz_selection == "Engine Metrics Distribution":
        st.subheader("Engine Metrics Distribution by Condition")
        metrics = df.columns[:-1]  # Assuming 'Engine Condition' is the last column
        fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 20))
        for i, metric in enumerate(metrics):
            sns.boxplot(x='Engine Condition', y=metric, data=df, ax=axs[i], palette="Set3")
            axs[i].set_title(metric)
            axs[i].set_xlabel('Engine Condition')
            axs[i].set_ylabel(metric)
        plt.tight_layout()
        st.pyplot(plt)

    # Interactive Visualization: Metric Distributions by Engine Condition
    elif viz_selection == "Metric Distributions":
        st.subheader("Metric Distributions by Engine Condition")
        # Dropdown for user to select metric
        metric = st.selectbox("Select metric to view distribution", options=df.columns[:-1], index=df.columns.tolist().index('Engine rpm'))
        
        # Create and display the histogram based on selected metric
        fig, ax = plt.subplots()
        sns.histplot(df, x=metric, hue="Engine Condition", kde=True, element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {metric} by Engine Condition')
        plt.xlabel(metric)
        plt.ylabel('Density')
        st.pyplot(fig)



# Correlation Heatmap
elif selection == "Correlation Heatmap":
    st.header("Correlation Heatmap of Engine Metrics")
    correlation = df.corr()  # Compute the correlation matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
    st.pyplot(plt)

# Categorical Analysis
elif selection == "Categorical Analysis":
    st.header("Analysis of Engine Conditions")
    condition_counts = df['Engine Condition'].value_counts()
    fig3 = px.bar(condition_counts, 
                  title='Distribution of Engine Conditions',
                  labels={'index': 'Condition', 'value': 'Count'},
                  text_auto=True)  # Automatically add text on bars
    st.plotly_chart(fig3)

    # Optionally, add more analysis or interactive features
    # Example: Comparing conditions across different types of engines
    if 'Engine Type' in df.columns:
        selected_engine_type = st.selectbox('Select Engine Type', df['Engine Type'].unique())
        filtered_data = df[df['Engine Type'] == selected_engine_type]
        condition_counts_filtered = filtered_data['Engine Condition'].value_counts()
        fig4 = px.bar(condition_counts_filtered,
                      title=f'Distribution of Engine Conditions for {selected_engine_type}',
                      labels={'index': 'Condition', 'value': 'Count'},
                      text_auto=True)
        st.plotly_chart(fig4)

# Load the model from the same directory
model = load('best_random_forest_classifier.joblib')

# Prediction section
if selection == "Predict Engine Health":
    st.header("Predict Engine Health")
    
    # Input fields for all engine metrics used in the model
    engine_rpm = st.number_input("Engine RPM", min_value=0.0, format="%.2f")
    lub_oil_pressure = st.number_input("Lubricating Oil Pressure", min_value=0.0, format="%.2f")
    fuel_pressure = st.number_input("Fuel Pressure", min_value=0.0, format="%.2f")
    coolant_pressure = st.number_input("Coolant Pressure", min_value=0.0, format="%.2f")
    lub_oil_temp = st.number_input("Lubricating Oil Temperature", min_value=0.0, format="%.2f")
    coolant_temp = st.number_input("Coolant Temperature", min_value=0.0, format="%.2f")
    # Ensure that you have inputs for all the features the model was trained on
    
    # Button to make prediction
    if st.button('Predict Condition'):
        # Create a numpy array from input data with all features
        new_data = np.array([[engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp]])
        
        # Make prediction
        prediction = model.predict(new_data)
        
        # Display the prediction
        condition = 'Healthy' if prediction[0] == 0 else 'Unhealthy'
        st.write(f"The predicted engine condition is: **{condition}**")