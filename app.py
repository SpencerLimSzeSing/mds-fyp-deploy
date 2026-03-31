import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import folium
import math
import pandas as pd
import plotly.express as px
from sklearn.model_selection import KFold

st.set_page_config(layout="wide")

@st.cache_resource
def load_knn_model():
    return joblib.load('best_knn_model.joblib')

@st.cache_resource
def load_rf_model():
    return joblib.load('best_rf_model.joblib')

@st.cache_resource
def load_xgb_model():
    return joblib.load('best_xgb_model.joblib')

@st.cache_resource
def load_meta_ann_model():
    return tf.keras.models.load_model('BestTuned_meta_ann_model.keras', compile=False)
    
import base64
def get_base64_image(file_path):
    """Convert image file to Base64."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Model loading with caching
@st.cache_resource
def load_knn_model():
    return joblib.load('models/best_knn_model.joblib')

@st.cache_resource
def load_rf_model():
    return joblib.load('models/best_rf_model.joblib')

@st.cache_resource
def load_xgb_model():
    return joblib.load('models/best_xgb_model.joblib')

@st.cache_resource
def load_meta_ann_model():
    return tf.keras.models.load_model('models/Tuned_meta_ann_model.keras', compile=False)
    
# Pre-load models (will only load once due to caching)
knn_model = load_knn_model()
rf_model = load_rf_model()
xgb_model = load_xgb_model()
meta_ann = load_meta_ann_model()
base_models = [knn_model, rf_model, xgb_model]
# Category mapping
category_mapping = {
    0: 'Rainfall_Category_No Rain',
    1: 'Rainfall_Category_Moderate Rain',
    2: 'Rainfall_Category_Heavy Rain',
    3: 'Rainfall_Category_Very Heavy Rain'
}

# Function to add a background image globally
def add_bg_from_local(image_file):
    """
    Adds a background image to a Streamlit app.

    Parameters:
    - image_file: Path to the local image file
    """
    with open(image_file, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()
    
    # Inject CSS to set the background image globally
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function once, globally
add_bg_from_local("assets/pexels.jpg")

# Load dataset
def load_data():
    file_path = "tab3/datasetdeploy.csv"
    return pd.read_csv(file_path)

data = load_data()

# Tabs for Introduction and Prediction
tab1, tab2, tab3 = st.tabs(["Introduction", "Rainfall Prediction","Dashboard"])

# Tab 1: Introduction
with tab1:
    st.title("Welcome to the Rainfall Prediction App")
    st.write("""
    This application predicts rainfall categories based on Meteorological factors. 
    You can input parameters such as temperature, wind speed, humidity, and atmospheric pressure to get predictions 
    using an advanced ensemble of machine learning models.
    - Predict rainfall categories: **No Rain (<5mm), Moderate Rain (5-20mm), Heavy Rain (20-50mm), Very Heavy Rain (>50mm)**
    """)
    st.markdown("### MDS Project Tittle:")
    st.write("""
    Meteorological Variable-based Using Stacking Ensemble Learning and ANN.
    """)
    st.markdown("### Author:") 
    st.write("""
    Lim Sze Sing (22109557)
    """)
    st.markdown("### Introduction:")
    st.write("""
    Accurate daily rainfall prediction is critical for effective water resource management,
    disaster preparedness, and short-term decision-making. This research focuses on
    enhancing the accuracy of daily rainfall category classification using a combination of
    stacking ensemble learning and Artificial Neural Networks (ANN). Traditional machine
    learning models, while useful, but often struggle with the complexity in meteorological
    data. To address these challenges, this study proposes the development of a stacking
    ensemble model that integrates various base learners, including K-Nearest Neighbors
    (KNN), Random Forest (RF), Extreme Gradient Boosting (XGB) with ANN as the Meta Learner (Meta-ANN).
    The dataset utilized in this research comprises 10 years of daily weather observations from various 
    Australian weather stations. The performance of the models were evaluated using metrics such as accuracy, recall,
    precision and F1-score, and compared against individual base models and a standalone
    ANN model. This study has shown that the Meta-ANN significantly outperformed the
    standalone ANN and base models, achieving higher accuracy and balanced performance
    across rainfall categories. Adjustments such as class weighting and hyperparameter
    tuning further enhanced the model's ability to address class imbalances. These findings
    highlight the effectiveness of ensemble methods in capturing the complex relationships
    between meteorological factors, leading to a reliable approach to daily rainfall prediction.
    """)
    st.markdown("### Data source:")
    st.write("""
    The dataset are from Bureau of Meteorology of Australia, comprises 10 years of daily weather 
    observations with a comprehensive range of meteorological variables. 
    The observations were gathered from a multitude of weather stations as image below:
    """)
    st.image("tab3/map.png", caption="Map of Weather Stations in Australia", width=500)



# Tab 2: Rainfall Prediction
with tab2:
    st.title("Rainfall Prediction App")
    st.write("Predict rainfall category based on climatic factors.")
    st.header("Input Features")

    # Function to add a background image
    def add_bg_from_local(image_file):
        """
        Adds a background image to a Streamlit app.

        Parameters:
        - image_file: Path to the local image file
        """
        with open(image_file, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        
        # Inject CSS to set the background image
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_image}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

        # Add the background image (call the function)
        add_bg_from_local("assets/pexels.jpg")  


    # Helper function for styling sections
    def styled_section(header, color):
        st.markdown(
            f"""
            <div style="background-color: {color}; padding: 2px 10px; border-radius: 5px; margin-bottom: 10px;">
                <h4 style="margin: 0; color: black;">{header}</h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Date Inputs
    styled_section("Date", "#f0f8ff")  # Light blue background
    col1, col2 = st.columns([1, 1]) 
    with col1:
        month = st.number_input("Month (1-12):", min_value=1, max_value=12, step=1)
    with col2:
        day = st.number_input("Day (1-31):", min_value=1, max_value=31, step=1)

    # Create a row with three sections
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    # Weather Features in the first column
    with col1:
        styled_section("Weather Features", "#f0f8ff")
        sunshine = st.number_input("Sunshine (hours):", min_value=0.0, max_value=15.0, step=0.1)
        st.caption("Hours of bright sunshine.")
        evaporation = st.number_input("Evaporation (mm):", min_value=0.0, max_value=150.0, step=0.1)
        st.caption("Evaporation in millimeters (mm).")
    # Atmospheric Pressure Features in the second column
    with col2:
        styled_section("Atmospheric Pressure Features", "#f0f8ff")
        pressure_9am = st.number_input("Pressure at 9am (hPa):", min_value=900.0, max_value=1100.0, step=0.1)
        st.caption("Atmospheric pressure at 9am in hPa.")
        pressure_3pm = st.number_input("Pressure at 3pm (hPa):", min_value=900.0, max_value=1100.0, step=0.1)
        st.caption("Atmospheric pressure at 3pm in hPa.")
    # Temperature Features in the third column
    with col3:
        styled_section("Temperature Features", "#f0f8ff")
        min_temp = st.number_input("Min Temperature (°C):", min_value=-10.0, max_value=50.0, step=0.1)
        max_temp = st.number_input("Max Temperature (°C):", min_value=-10.0, max_value=50.0, step=0.1)
        temp_9am = st.number_input("Temperature at 9am (°C):", min_value=-10.0, max_value=50.0, step=0.1)
        temp_3pm = st.number_input("Temperature at 3pm (°C):", min_value=-10.0, max_value=50.0, step=0.1)
    # Wind Features in the first column
    with col4:
        styled_section("Wind Features", "#f0f8ff")  
        wind_gust_speed = st.number_input("Wind Gust Speed (km/h):", min_value=0.0, max_value=200.0, step=1.0)
        st.caption("The speed (km/h) of the strongest wind gust in the 24 hours to midnight.")
        wind_speed_9am = st.number_input("Wind Speed at 9am (km/h):", min_value=0.0, max_value=150.0, step=1.0)
        st.caption("Wind speed (km/hr) averaged over 10 minutes prior to 9am.")
        wind_speed_3pm = st.number_input("Wind Speed at 3pm (km/h):", min_value=0.0, max_value=150.0, step=1.0)
        st.caption("Wind speed (km/hr) averaged over 10 minutes prior to 3pm.")
    # Humidity Features in the second column
    with col5:
        styled_section("Humidity Features", "#f0f8ff")  
        humidity_9am = st.number_input("Humidity at 9am (%):", min_value=0.0, max_value=100.0, step=1.0)
        st.caption("Humidity percentage at 9am.")
        humidity_3pm = st.number_input("Humidity at 3pm (%):", min_value=0.0, max_value=100.0, step=1.0)
        st.caption("Humidity percentage at 3pm.")
    
    # Target-encoded mapping for locations
    location_target_encoded = {
        'Adelaide': 1.5663539307667422, 'Albany': 2.2638594164456234, 'Albury': 1.9141149119893721,
        'AliceSprings': 0.8828496042216359, 'BadgerysCreek': 2.1931010928961747, 'Ballarat': 1.7400264200792603,
        'Bendigo': 1.6193803559657218, 'Brisbane': 3.144890857323632, 'Cairns': 5.742034805890228,
        'Canberra': 1.7417203042715037, 'Cobar': 1.1273092369477913, 'CoffsHarbour': 5.061496782932611,
        'Dartmoor': 2.1465669612508496, 'Darwin': 5.092452239273411, 'GoldCoast': 3.7693959731543623,
        'Hobart': 1.601819322459222, 'Katherine': 3.2010897435897436, 'Launceston': 2.011988110964333,
        'Melbourne': 1.8700616016427105, 'MelbourneAirport': 1.4519774011299436, 'Mildura': 0.9450615231127371,
        'Moree': 1.6302032235459005, 'MountGambier': 2.087561860772022, 'MountGinini': 3.292260061919505,
        'Newcastle': 3.183891708967851, 'Nhil': 0.9348629700446144, 'NorahHead': 3.387299419597132,
        'NorfolkIsland': 3.127665317139001, 'Nuriootpa': 1.3903429903429902, 'PearceRAAF': 1.66908037653874,
        'Penrith': 2.1753036437246966, 'Perth': 1.906295020357031, 'PerthAirport': 1.761648388168827,
        'Portland': 2.5303738317757007, 'Richmond': 2.1384615384615384, 'Sale': 1.5101666666666667,
        'SalmonGums': 1.0343824027072759, 'Sydney': 3.324543002697033, 'SydneyAirport': 3.009916805324459,
        'Townsville': 3.485591823277283, 'Tuggeranong': 2.1640426951300866, 'Uluru': 0.7843626806833114,
        'WaggaWagga': 1.7099462365591398, 'Walpole': 2.9068463994324225, 'Watsonia': 1.860820273424475,
        'Williamtown': 3.591108499804152, 'Witchcliffe': 2.8956639566395665, 'Wollongong': 3.594902749832327,
        'Woomera': 0.49040454697425606
    }

    # Add dropdown for location selection
    styled_section("Select Location", "#f0f8ff")
    all_locations = [
        'Adelaide', 'Albany', 'Albury',
        'AliceSprings', 'BadgerysCreek', 'Ballarat',
        'Bendigo', 'Brisbane', 'Cairns',
        'Canberra', 'Cobar', 'CoffsHarbour',
        'Dartmoor', 'Darwin', 'GoldCoast',
        'Hobart', 'Katherine', 'Launceston',
        'Melbourne', 'MelbourneAirport', 'Mildura',
        'Moree', 'MountGambier', 'MountGinini',
        'Newcastle', 'Nhil', 'NorahHead',
        'NorfolkIsland', 'Nuriootpa', 'PearceRAAF',
        'Penrith', 'Perth', 'PerthAirport',
        'Portland', 'Richmond', 'Sale',
        'SalmonGums', 'Sydney', 'SydneyAirport',
        'Townsville', 'Tuggeranong', 'Uluru',
        'WaggaWagga', 'Walpole', 'Watsonia',
        'Williamtown', 'Witchcliffe', 'Wollongong',
        'Woomera'
    ]
    selected_location = st.selectbox("Select Location:", list(location_target_encoded.keys()))
    # Get the encoded value for the selected location
    location_encoded = location_target_encoded[selected_location]

    st.write(f"**Selected Location:** {selected_location}")

 

    # Derived features
    dew_point_estimate = min_temp * (humidity_9am / 100)
    avg_humidity = (humidity_9am + humidity_3pm) / 2
    avg_temperature = (min_temp + max_temp) / 2
    temp_range = max_temp - min_temp
    temp_sunshine_interaction = max_temp * sunshine
    pressure_difference = pressure_3pm - pressure_9am
    temp_difference = temp_3pm - temp_9am



    # Ensure feature names match those used during model training
    feature_names = [
        "MinTemp", "MaxTemp", "Evaporation", "WindGustSpeed", "WindSpeed9am",
        "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
        "Temp9am", "Temp3pm", "Month", "Day", "Location_Target_Encoded", "DewPointEstimate",
        "AvgHumidity", "AvgTemperature", "TempRange", "TempSunshineInteraction",
        "PressureDifference", "TempDifference"
    ]

    # Combine all raw features into a single array for base model predictions
    input_features = np.array([[min_temp, max_temp, evaporation, wind_gust_speed, wind_speed_9am,
                                wind_speed_3pm, humidity_9am, humidity_3pm, pressure_9am, pressure_3pm,
                                temp_9am, temp_3pm, month, day, location_encoded, dew_point_estimate,
                                avg_humidity, avg_temperature, temp_range, temp_sunshine_interaction,
                                pressure_difference, temp_difference]])

    # Convert input_features to a DataFrame with proper column names
    input_features_df = pd.DataFrame(input_features, columns=feature_names)



    display_mapping = {
        'Rainfall_Category_No Rain': 'No Rain (0 mm to 5 mm)',
        'Rainfall_Category_Moderate Rain': 'Moderate Rain (5 mm to 20 mm)',
        'Rainfall_Category_Heavy Rain': 'Heavy Rain (20 mm to 50 mm)',
        'Rainfall_Category_Very Heavy Rain': 'Very Heavy Rain (50 mm and higher)'
    }

    if st.button("Predict"):
        st.write("### Predicting Rainfall Category...")

            # Step 1: Generate Meta-Features for Prediction
        meta_features = np.zeros((1, len(base_models)))  # Initialize meta-feature array for a single sample

        for i, model in enumerate(base_models):
                # Directly predict using the pre-trained base models
            meta_features[0, i] = model.predict(input_features_df)[0]  # Store the prediction

            # Step 2: Use Meta-Learner for Final Prediction
        meta_pred = meta_ann.predict(meta_features)
        meta_pred_class = np.argmax(meta_pred, axis=1)[0]  # Get the class with the highest probability
        meta_pred_label = category_mapping[meta_pred_class]
        meta_pred_display = display_mapping[meta_pred_label]
            # Step 3: Display Results

            # Load Base64 icon dynamically
        icon_mapping = {
                'Rainfall_Category_No Rain': get_base64_image('assets/rain1.png'),
                'Rainfall_Category_Moderate Rain': get_base64_image('assets/rain2.png'),
                'Rainfall_Category_Heavy Rain': get_base64_image('assets/rain3.png'),
                'Rainfall_Category_Very Heavy Rain': get_base64_image('assets/rain4.png')
            }
        icon_file = icon_mapping[meta_pred_label]  # Get corresponding icon file

            # Additional information mapping for categories
        category_info = {
                'Rainfall_Category_No Rain': """
                - **Conditions**: Dry weather, no precipitation.
                - **Impact**: Ideal for outdoor activities.
                """,
                'Rainfall_Category_Light Rain': """
                - **Conditions**: A gentle drizzle rainfall.
                - **Impact**: Minimal disruption to daily activities, but outdoor activities may not be convenient.
                """,
                'Rainfall_Category_Moderate Rain': """
                - **Conditions**: A consistent moderate rainfall.
                - **Impact**: May require umbrellas or raincoats for outdoor activities. Some potential for minor disruptions to transportation and outdoor events.
                """,
                'Rainfall_Category_Heavy Rain': """
                - **Conditions**: Intense rainfall, potentially with strong winds and thunderstorms.
                - **Impact**: It is recommended to stay indoors. Potentially leading to transportation delays, road closures, and potential flooding in low-lying areas.
                """,
                'Rainfall_Category_Very Heavy Rain': """
                - **Conditions**: Unusually heavy downpour, often comes with strong winds and thunderstorms.
                - **Impact**: It is recommended to stay indoors. Potentially leading to flash flooding, stay alert if it happens on consecutive days.
                """
            }

            # Fetch the description dynamically based on the predicted label
        predicted_category_description = category_info.get(meta_pred_label, "No description available.")

            # Display the result with the correct description
        st.markdown(
                f"""
                <div style="background-color: #ffffff; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                    <h2 style="color: #000;">Predicted Rainfall Category:</h2>
                    <div style="display: flex; align-items: center; justify-content: flex-start; margin-bottom: 20px;">
                        <h3 style="margin: 0; color: #000;">{meta_pred_display}</h3>
                        <img src="data:image/png;base64,{icon_file}" width="80" style="margin-left: 10px;"/>
                    </div>
                    <ul style="color: #000; list-style-type: disc; padding-left: 20px;">
                        {predicted_category_description}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Additional considerations section
        st.markdown(
                """
                <div style="background-color: #ffffff; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
                    <h2 style="color: #000;">Additional Considerations</h2>
                    <ul style="color: #000; list-style-type: disc; padding-left: 20px;">
                        <li>Rainfall patterns and intensity vary across Australia's diverse climate zones.</li>
                        <li>For the most accurate and up-to-date information on rainfall conditions in Australia, please refer to the <a href="https://bom.gov.au/" target="_blank">Bureau of Meteorology website</a>.</li>
                        <li>Heavy rain can be dangerous. Always prioritize safety and follow any evacuation orders or warnings from authorities.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )


# Tab 3: 
with tab3:
    st.title("Dashboard:")
    #st.write("This tab provides insights into the dataset with interactive visualizations.")
    # Filters Section
    st.subheader("")
    col1, col2 = st.columns(2)

    # Add date range slider
    with col1:
        styled_section("Filter by Date Range", "#f0f8ff")
        data['Date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
        min_date = data['Date'].min().date()  # Extract the earliest date
        max_date = data['Date'].max().date()  # Extract the latest date
        date_range = st.slider(
        "Select Date Range:",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
        filtered_data = data[
        (data['Date'] >= pd.to_datetime(date_range[0])) &
        (data['Date'] <= pd.to_datetime(date_range[1]))
    ]


    # Ensure Location_Target_Encoded column exists
    from sklearn.preprocessing import LabelEncoder

    if 'Location_Target_Encoded' not in data.columns:
        le = LabelEncoder()
        data['Location_Target_Encoded'] = le.fit_transform(data['Location'])

    # Create a mapping of location names to their encoded values
    location_target_encoded = dict(zip(data['Location'], data['Location_Target_Encoded']))
    
    # Location Dropdown in the second column
    with col2:
        # Add location filter using dropdown
        styled_section("Filter by Location", "#f0f8ff")
        # List of all locations
        all_locations = ['All'] + [
            'Adelaide', 'Albany', 'Albury',
            'AliceSprings', 'BadgerysCreek', 'Ballarat',
            'Bendigo', 'Brisbane', 'Cairns',
            'Canberra', 'Cobar', 'CoffsHarbour',
            'Dartmoor', 'Darwin', 'GoldCoast',
            'Hobart', 'Katherine', 'Launceston',
            'Melbourne', 'MelbourneAirport', 'Mildura',
            'Moree', 'MountGambier', 'MountGinini',
            'Newcastle', 'Nhil', 'NorahHead',
            'NorfolkIsland', 'Nuriootpa', 'PearceRAAF',
            'Penrith', 'Perth', 'PerthAirport',
            'Portland', 'Richmond', 'Sale',
            'SalmonGums', 'Sydney', 'SydneyAirport',
            'Townsville', 'Tuggeranong', 'Uluru',
            'WaggaWagga', 'Walpole', 'Watsonia',
            'Williamtown', 'Witchcliffe', 'Wollongong',
            'Woomera'
        ]

        # Dropdown to select location
        selected_location = st.selectbox("Select Location:", options=all_locations, key="location_selectbox")

        # Filter data by selected location
        if selected_location == 'All':
            st.markdown("### Showing data for all locations.")
        else:
            st.markdown(f"### Showing data for: **{selected_location}**")
            filtered_data = filtered_data[filtered_data['Location'] == selected_location]

        
# Scorecard Section
    styled_section("Metrics", "#f0f8ff")
    if not filtered_data.empty:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Min Temperature (°C)", filtered_data["MinTemp"].min())
            col2.metric("Max Temperature (°C)", filtered_data["MaxTemp"].max())
            col3.metric("Average Temperature (°C)", f"{(filtered_data['MinTemp'].mean() + filtered_data['MaxTemp'].mean()) / 2:.1f}")
            col4.metric("Total Rainfall (mm)",f"{filtered_data['Rainfall'].sum():.1f}")
            col5.metric("Max Rainfall (mm)", filtered_data["Rainfall"].max())
            col6.metric(
                "Raining Days",
                len(filtered_data[filtered_data["Rainfall_Category"].isin(["Moderate Rain", "Heavy Rain", "Very Heavy Rain"])])
            )
    else:
        st.write("No data available for the selected filters.")

    # Line Graphs Section
    
    if not filtered_data.empty:
            # Arrange graphs in a grid
            col1, col2 = st.columns(2)

            # Rainfall Graph
            with col1:
                fig1 = px.line(filtered_data, x="Date", y="Rainfall", title="Rainfall Over Time")
                st.plotly_chart(fig1, use_container_width=True)

            # MinTemp & MaxTemp Graph
            with col2:
                fig2 = px.line(filtered_data, x="Date", y=["MinTemp", "MaxTemp"], title="Min & Max Temperatures Over Time")
                st.plotly_chart(fig2, use_container_width=True)

            # WindGustSpeed Graph
            col1, col2 = st.columns(2)
            with col1:
                fig3 = px.line(filtered_data, x="Date", y="WindGustSpeed", title="Wind Gust Speed Over Time")
                st.plotly_chart(fig3, use_container_width=True)

            # Humidity Graph
            with col2:
                fig4 = px.line(filtered_data, x="Date", y=["Humidity9am", "Humidity3pm"], title="Humidity at 9am & 3pm Over Time")
                st.plotly_chart(fig4, use_container_width=True)

            # Pressure Graph
            col1, col2 = st.columns(2)
            with col1:
                fig5 = px.line(filtered_data, x="Date", y=["Pressure9am", "Pressure3pm"], title="Pressure at 9am & 3pm Over Time")
                st.plotly_chart(fig5, use_container_width=True)

            # Filtered Data Table
            with col2:
                st.subheader("Filtered Data Table")
                st.dataframe(filtered_data)
    else:
            st.write("No data available to display graphs.")
