import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 1. Page & Aesthetic Configuration
# ==========================================
st.set_page_config(
    page_title="Cuisine Intelligence Pro",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ==========================================
# 1. Page & Aesthetic Configuration (REFINED)
# ==========================================

# Custom CSS for a Luxury Tech Theme - Professional Balanced Version


# Material Icons Library Load + Custom CSS
# Material Icons Library Load + Full Professional CSS
st.markdown("""
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<style>
    /* 1. Global App Styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* 2. Sidebar Main Container */
    section[data-testid="stSidebar"] {
        background-color: #002b2b !important;
        color: white !important;
    }
    
    /* 3. Sidebar Compact Spacing Logic */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.7rem !important; 
        padding-top: 1rem !important;
    }

    /* 4. Labels Styling (No Overlap) */
    section[data-testid="stSidebar"] label {
        color: #e0f2f2 !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        margin-bottom: 2px !important; 
        display: block !important;
    }

    /* 5. Input & Slider Container Styling */
    .stNumberInput, .stSlider {
        background: rgba(255,255,255,0.08);
        padding: 8px 12px !important;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 5px !important;
    }

    /* 6. Icon Styling */
    .micons {
        vertical-align: middle;
        margin-right: 8px;
        color: #00b3b3;
        font-size: 20px;
    }

    /* 7. Divider Styling */
    [data-testid="stSidebar"] hr {
        margin: 0.6rem 0 !important;
        border: none;
        border-top: 1px solid rgba(255,255,255,0.2);
    }

    /* 8. Headers & UI Elements */
    h1 { color: #004d4d !important; font-weight: 800 !important; }
    
    .prediction-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-left: 8px solid #008080;
        margin-bottom: 20px;
    }

    .cuisine-tag {
        background: linear-gradient(45deg, #008080, #00b3b3);
        color: white;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 600;
        margin: 6px;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(0,128,128,0.2);
    }

    div.stButton > button:first-child {
        background: linear-gradient(to right, #008080, #00b3b3);
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        width: 100%;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Data Loading (Cached)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    try:
        df = pd.read_csv("dataset.csv", encoding="utf-8-sig")
        required_cols = ["Average Cost for two", "Price range", "Aggregate rating", "Votes", "Cuisines"]
        if not all(col in df.columns for col in required_cols):
            st.error("Missing columns in dataset!")
            st.stop()
            
        df["Cuisines_Cleaned"] = df["Cuisines"].fillna("").apply(
            lambda x: [c.strip().title() for c in x.split(",") if c.strip() != ""]
        )
        return df
    except FileNotFoundError:
        st.error("Error: 'dataset.csv' not found!")
        st.stop()

df = load_and_preprocess_data()

# ==========================================
# 3. Sidebar - Profile Inputs
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3443/3443338.png", width=80)
    st.title("Settings")
    st.markdown("Configure the restaurant parameters below.")
    st.divider()

    # Financial Tier with Icon
    st.markdown('#### <i class="material-icons micons">payments</i> Financial Tier', unsafe_allow_html=True)
    max_dataset_cost = int(df["Average Cost for two"].max())
    input_cost = st.number_input("Avg Cost for Two", 0, max_dataset_cost, 1000, 50)
    input_price_range = st.slider("Price Range Tier", 1, 4, 2)

    st.divider()

    # Performance with Icon
    st.markdown('#### <i class="material-icons micons">analytics</i> Performance', unsafe_allow_html=True)
    input_rating = st.slider("Target Rating", 0.0, 5.0, 3.8, 0.1)
    max_dataset_votes = int(df["Votes"].max())
    input_votes = st.number_input("Total Votes", 0, max_dataset_votes, 200, 10)

    st.divider()
    predict_btn = st.button("Generate Prediction 🚀")
    
# ==========================================
# 4. Main Panel
# ==========================================
st.title("🍽️ Smart Cuisine Classifier")
st.markdown("##### AI-driven statistical analysis for restaurant categorization")
st.divider()

if predict_btn:
    with st.spinner('Scanning dataset for architectural matches...'):
        # Tolerance logic
        cost_tol, rate_tol = 0.25, 0.4
        
        similar_restaurants = df[
            (df["Price range"] == input_price_range) &
            (df["Average Cost for two"].between(input_cost*(1-cost_tol), input_cost*(1+cost_tol))) &
            (df["Aggregate rating"].between(input_rating-rate_tol, input_rating+rate_tol))
        ]
        
        # Fallback if too specific
        if len(similar_restaurants) < 5:
            similar_restaurants = df[(df["Price range"] == input_price_range)].head(50)

        all_predicted = [c for sub in similar_restaurants["Cuisines_Cleaned"] for c in sub]
        
        if all_predicted:
            prediction_series = pd.Series(all_predicted).value_counts().head(5)
            
            # Prediction Results Section
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.subheader("🎯 Primary Cuisine Predictions")
            
            # Displaying Tags
            tag_html = ""
            for cuisine in prediction_series.index:
                tag_html += f'<span class="cuisine-tag">{cuisine}</span>'
            st.markdown(tag_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Columns for Charts and Insights
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Probability Analysis")
                # Visual Chart
                chart_data = pd.DataFrame({
                    'Cuisine': prediction_series.index,
                    'Confidence': prediction_series.values / len(similar_restaurants)
                }).set_index('Cuisine')
                st.bar_chart(chart_data, color="#008080")

            with col2:
                st.subheader("💡 Insights")
                st.info(f"""
                - **Sample Size:** {len(similar_restaurants)} similar restaurants found.
                - **Top Match:** {prediction_series.index[0]}
                - **Rating Match:** Consistent with {input_rating} stars.
                """)
        else:
            st.warning("Could not find enough data to make a confident prediction.")

else:
    # Welcome View
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Welcome to the Cuisine Lab")
        st.write("This tool uses statistical patterns from your dataset to predict which cuisines a restaurant is likely to serve based on its cost and rating profile.")
        st.image("https://images.unsplash.com/photo-1504674900247-0877df9cc836?auto=format&fit=crop&q=80&w=1000", use_container_width=True)
    with col_b:
        st.success("⬅️ Start by adjusting the parameters in the sidebar and clicking Predict!")
        with st.expander("Why this matters?"):
            st.write("Understanding the correlation between pricing and cuisine helps in market positioning and competitor analysis.")

# ==========================================
# 5. Footer
# ==========================================
st.divider()

st.caption("Cuisine Intelligence Pro | Developed by Umashankar Sahu | AI/ML Student")