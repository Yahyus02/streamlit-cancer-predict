import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def get_clean_data():
    data = pd.read_csv(r"C:/Users/user/Downloads/streamlit/data/data.csv") 
    
    data = data.drop(["id", "Unnamed: 32"], axis=1,  errors='ignore')

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

def add_sidebar():
    st.sidebar.markdown("""
    <div style="background-color:#e8f5e9; padding:15px; border-radius:10px;">
        <h3 style='color:#2e7d32;'>Adjust Cell Nuclei Measurements</h3>
    </div>
""", unsafe_allow_html=True)

    data = get_clean_data()

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    return input_dict 

def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):

    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness', 
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[v for k, v in input_data.items() if k.endswith('_mean')],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[v for k, v in input_data.items() if k.endswith('_se')],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[v for k, v in input_data.items() if k.endswith('_worst')],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
    showlegend=True
    )

    return fig 

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_array = pd.DataFrame([input_data])

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction == 0:
        st.write("Benign")
    else:
        st.write("Malignant")
    
    st.write("Probability of being benign:",  model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant:",  model.predict_proba(input_array_scaled)[0][1])

    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")

def main():
    st.set_page_config(
        page_title="Breast Cancer Diagnosis",
        page_icon=".female-doctor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
        <style>
            /* Change whole app background to black and text to white */
            .stApp {
                background-color: #000000 !important;
                color: white !important;
            }

            /* Customize the sidebar background */
            .css-1d391kg {  
                background-color: #121212 !important;
            }

            /* Style the title and headers */
            h1 {
                color: #4CAF50;
                font-size: 36px;
            }

            h2, h3 {
                color: #388E3C;
            }

            /* Style Streamlit buttons */
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                border-radius: 10px;
                border: none;
                font-size: 16px;
            }

            /* Style containers or boxes */
            .block-container {
                padding: 2rem 1rem;
                background-color: #121212 !important;
                color: white !important;
            }
        </style>
    """, unsafe_allow_html=True)

    input_data = add_sidebar()
    st.write(input_data)
    
    with st.container():
        st.markdown("<h1 style='color:#4CAF50;'>🧬 Breast Cancer Diagnosis</h1>", unsafe_allow_html=True)
        st.write("Please connect this lab to your cytology lab to help diagnose breast cancer from your tissue samples. This app predicts using a machines learning model whether a breast mass is benign or malignant based on the measurements its receives from your cytosis lab. You can also update the measurement by hand using the sliders in the sidebar.")

    col1, col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()
