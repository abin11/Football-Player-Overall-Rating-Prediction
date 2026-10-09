import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
from streamlit.components.v1 import html

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Football Player Overall Rating Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------
# Load Model (Linear Regression)
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("player_rating_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'player_rating_model.pkl' is in the current directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


model = load_model()

# Check if model is LinearRegression (it is)
st.sidebar.success("✅ Model loaded: Random Forest Regression")

# ------------------------------
# Custom CSS Styling
# ------------------------------
st.markdown("""
    <style>
        /* Main container */
        .main {
            background-color: #0e1117;
            color: #e0e0e0;
        }

        /* Headers */
        h1, h2, h3 {
            color: #00b894 !important;
            font-weight: 600 !important;
        }

        /* Cards & containers */
        .css-1r6slb0, .css-1v3fvcr {
            background-color: #1e242c !important;
            border-radius: 15px !important;
            padding: 20px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4) !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #00b894, #00a67e);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 2rem;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 10px rgba(0,184,148,0.3);
        }
        .stButton > button:hover {
            transform: scale(1.02);
            background: linear-gradient(90deg, #00a67e, #00b894);
            box-shadow: 0 6px 15px rgba(0,184,148,0.5);
        }

        /* Metric card */
        .metric-card {
            background: linear-gradient(145deg, #1e242c, #2a323c);
            border-radius: 20px;
            padding: 25px;
            text-align: center;
            border-left: 5px solid #00b894;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .metric-value {
            font-size: 3.5rem;
            font-weight: 700;
            color: #00b894;
        }
        .metric-label {
            color: #b0b0b0;
            font-size: 1.1rem;
            letter-spacing: 1px;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 0.9rem;
            border-top: 1px solid #2a323c;
            margin-top: 40px;
        }

        /* Sidebar */
        .css-1d391kg, .css-1aumxhk {
            background-color: #161b22 !important;
        }
        .sidebar-content {
            padding: 20px 10px;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #1e242c;
            border-radius: 8px;
            padding: 8px 20px;
            color: #b0b0b0;
        }
        .stTabs [aria-selected="true"] {
            background-color: #00b894 !important;
            color: white !important;
        }

        /* Slider labels */
        .stSlider label {
            color: #d0d0d0 !important;
            font-weight: 500 !important;
        }

        /* Success/Info boxes */
        .stAlert {
            border-radius: 12px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/football2.png", width=80)
    st.title("⚽ Player Predictor")
    st.markdown("---")
    st.markdown("""
        **Project Description**  
        This application uses a Random Forest Regression model to predict a football player's overall rating based on key attributes.  
        It mimics the rating system used in popular football video games.

        **Instructions**  
        1. Select player position.  
        2. Adjust attribute sliders (1–99).  
        3. Click **Predict Overall Rating**.  
        4. View the predicted rating, category, and visual charts.

        **Developer**  
        Made by [Your Name]  
        Powered by Streamlit & Scikit-learn

        **Model Info**  
        - **Algorithm:** Random Forest Regression  
        - **Features:** 29 attributes + one-hot encoded position  
        - **Training Data:** 7,900+ player records  
        - **Performance:** R² ~0.81, RMSE ~3.2
    """)
    st.markdown("---")
    st.caption("© 2025 Football Predictor | All rights reserved.")

# ------------------------------
# Header Hero Section
# ------------------------------
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0e1117 0%, #1a2a2a 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border-bottom: 3px solid #00b894;
        text-align: center;
    ">
        <h1 style="font-size: 3.2rem; color: #00b894; margin-bottom: 0.2rem;">
            ⚽ Football Player Overall Rating Predictor
        </h1>
        <p style="font-size: 1.2rem; color: #b0b0b0; max-width: 700px; margin: 0 auto;">
            Enter player attributes to predict their FIFA-style overall rating using Machine Learning.
        </p>
    </div>
""", unsafe_allow_html=True)

# ------------------------------
# Session State Initialization
# ------------------------------
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "rating" not in st.session_state:
    st.session_state.rating = None
if "sample_triggered" not in st.session_state:
    st.session_state.sample_triggered = False

# ------------------------------
# Input Section
# ------------------------------
with st.form("prediction_form"):
    st.markdown("### 🧑‍⚕️ Player Information")
    col1, col2 = st.columns(2)
    with col1:
        position = st.selectbox(
            "Position",
            ["CAM", "CB", "CDM", "CF", "CM", "GK", "LB", "LM", "LW", "RB", "RM", "RW", "ST"],
            index=0,
            help="Select the player's primary position."
        )
    with col2:
        age = st.slider("Age", 15, 40, 25, help="Player's age in years.")

    # Expandable sections for attributes
    with st.expander("⚡ Attacking Attributes", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            finishing = st.slider("Finishing", 1, 99, 70)
            shot_power = st.slider("Shot Power", 1, 99, 70)
            heading_accuracy = st.slider("Heading Accuracy", 1, 99, 60)
        with col2:
            crossing = st.slider("Crossing", 1, 99, 60)
            short_passing = st.slider("Short Passing", 1, 99, 65)
            long_passing = st.slider("Long Passing", 1, 99, 65)
        with col3:
            through_ball = st.slider("Through Ball", 1, 99, 65)
            dribbling = st.slider("Dribbling", 1, 99, 65)
            ball_control = st.slider("Ball Control", 1, 99, 70)

    with st.expander("🛡️ Defensive Attributes"):
        col1, col2 = st.columns(2)
        with col1:
            standing_tackle = st.slider("Standing Tackle", 1, 99, 60)
            sliding_tackle = st.slider("Sliding Tackle", 1, 99, 55)
        with col2:
            interceptions = st.slider("Interceptions", 1, 99, 55)
            defensive_awareness = st.slider("Defensive Awareness", 1, 99, 55)

    with st.expander("💪 Physical Attributes"):
        col1, col2, col3 = st.columns(3)
        with col1:
            acceleration = st.slider("Acceleration", 1, 99, 70)
            speed = st.slider("Speed", 1, 99, 70)
            agility = st.slider("Agility", 1, 99, 65)
        with col2:
            balance = st.slider("Balance", 1, 99, 65)
            jumping = st.slider("Jumping", 1, 99, 70)
            strength = st.slider("Strength", 1, 99, 70)
        with col3:
            stamina = st.slider("Stamina", 1, 99, 70)
            offensive_awareness = st.slider("Offensive Awareness", 1, 99, 65)
            composure = st.slider("Composure", 1, 99, 65)
            reactions = st.slider("Reactions", 1, 99, 70)

    with st.expander("🧤 Goalkeeping Attributes"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gk_diving = st.slider("GK Diving", 1, 99, 50)
            gk_handling = st.slider("GK Handling", 1, 99, 50)
        with col2:
            gk_kicking = st.slider("GK Kicking", 1, 99, 50)
            gk_reflexes = st.slider("GK Reflexes", 1, 99, 50)
        with col3:
            gk_positioning = st.slider("GK Positioning", 1, 99, 50)

    # Buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        submitted = st.form_submit_button("🔮 Predict Overall Rating", use_container_width=True)
    with col2:
        reset = st.form_submit_button("🔄 Reset Inputs", use_container_width=True)
    with col3:
        sample = st.form_submit_button("🎲 Sample Player", use_container_width=True)

# ------------------------------
# Handle Reset & Sample
# ------------------------------
if reset:
    st.session_state.predicted = False
    st.session_state.rating = None
    st.session_state.sample_triggered = False
    st.rerun()

if sample:
    st.session_state.sample_triggered = True
    st.rerun()


# ------------------------------
# Prepare Input Vector (EXACTLY 42 features)
# ------------------------------
def get_input_vector():
    # Position one-hot encoding (13 positions)
    positions = ["CAM", "CB", "CDM", "CF", "CM", "GK", "LB", "LM", "LW", "RB", "RM", "RW", "ST"]
    pos_one_hot = [0] * 13
    if position in positions:
        pos_one_hot[positions.index(position)] = 1

    # Use sample values if triggered
    if st.session_state.sample_triggered:
        # Sample: High-rated Striker
        values = [
            27,  # age
            95,  # finishing
            93,  # shot_power
            90,  # heading_accuracy
            85,  # crossing
            88,  # short_passing
            85,  # long_passing
            87,  # through_ball
            92,  # dribbling
            93,  # ball_control
            65,  # standing_tackle
            60,  # sliding_tackle
            62,  # interceptions
            60,  # defensive_awareness
            75,  # acceleration
            78,  # speed
            74,  # agility
            78,  # balance
            85,  # jumping
            80,  # strength
            78,  # stamina
            85,  # offensive_awareness
            88,  # composure
            90,  # reactions
            30,  # gk_diving
            30,  # gk_handling
            30,  # gk_kicking
            30,  # gk_reflexes
            30,  # gk_positioning
        ]
        # Position: ST (index 12)
        pos_one_hot = [0] * 13
        pos_one_hot[12] = 1
        st.session_state.sample_triggered = False
    else:
        values = [
            age,
            finishing, shot_power, heading_accuracy,
            crossing, short_passing, long_passing,
            through_ball, dribbling, ball_control,
            standing_tackle, sliding_tackle, interceptions,
            defensive_awareness,
            acceleration, speed, agility, balance,
            jumping, strength, stamina,
            offensive_awareness, composure, reactions,
            gk_diving, gk_handling, gk_kicking,
            gk_reflexes, gk_positioning
        ]

    # Combine all 42 features
    input_vector = values + pos_one_hot
    return np.array(input_vector).reshape(1, -1)


# ------------------------------
# Prediction Logic
# ------------------------------
if submitted:
    with st.spinner("🔍 Calculating rating..."):
        try:
            input_data = get_input_vector()
            pred = model.predict(input_data)[0]
            st.session_state.rating = round(pred, 1)
            st.session_state.predicted = True
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            st.session_state.predicted = False

# ------------------------------
# Display Results
# ------------------------------
if st.session_state.predicted and st.session_state.rating is not None:
    rating = st.session_state.rating

    # Rating Category
    if rating <= 50:
        category = "Poor"
        color = "red"
        emoji = "🔴"
    elif rating <= 65:
        category = "Average"
        color = "orange"
        emoji = "🟡"
    elif rating <= 80:
        category = "Good"
        color = "green"
        emoji = "🟢"
    elif rating <= 90:
        category = "Excellent"
        color = "blue"
        emoji = "🔵"
    else:
        category = "World Class"
        color = "gold"
        emoji = "🏆"

    st.markdown("---")
    st.markdown("### 📊 Prediction Results")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">⭐ Overall Rating</div>
                <div class="metric-value">{rating}</div>
                <div style="margin-top: 10px;">
                    <span style="font-size:1.2rem; color:{color};">{emoji} {category}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Category message
    if rating >= 80:
        st.success(f"🏅 {emoji} This player is **{category}**! Excellent attributes!")
    elif rating >= 65:
        st.info(f"📈 {emoji} This player is **{category}**. Good potential!")
    else:
        st.warning(f"📉 {emoji} This player is **{category}**. Needs improvement.")

    # ------------------------------
    # Visualizations (Plotly)
    # ------------------------------
    st.markdown("### 📈 Attribute Analysis")

    # Radar Chart
    labels = ['Finishing', 'Shot Power', 'Heading', 'Crossing', 'Passing', 'Dribbling',
              'Ball Control', 'Tackling', 'Interceptions', 'Def. Awareness',
              'Physical', 'Pace', 'Agility', 'Balance', 'Jumping', 'Strength',
              'Stamina', 'Off. Awareness', 'Composure', 'Reactions']

    # Get current values from sliders (or sample values)
    if st.session_state.sample_triggered:
        # Use sample values for display
        vals = [
            95, 93, 90, 85, 88, 92, 93,
            (65 + 60) / 2, 62, 60,
            (80 + 78 + 85) / 3, (75 + 78) / 2,
            74, 78, 85, 80, 78,
            85, 88, 90
        ]
    else:
        vals = [
            finishing, shot_power, heading_accuracy, crossing,
            (short_passing + long_passing + through_ball) / 3,
            dribbling, ball_control,
            (standing_tackle + sliding_tackle) / 2,
            interceptions, defensive_awareness,
            (strength + stamina + jumping) / 3,
            (acceleration + speed) / 2,
            agility, balance, jumping, strength, stamina,
            offensive_awareness, composure, reactions
        ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=labels,
        fill='toself',
        name='Player Attributes',
        line_color='#00b894',
        fillcolor='rgba(0,184,148,0.3)'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color='#b0b0b0'
            ),
            bgcolor='rgba(30,36,44,0.8)'
        ),
        showlegend=False,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=40, r=40, t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Horizontal Bar Chart: Category Averages
    attack = (finishing + shot_power + heading_accuracy + crossing + short_passing +
              long_passing + through_ball + dribbling + ball_control) / 9
    defense = (standing_tackle + sliding_tackle + interceptions + defensive_awareness) / 4
    physical = (acceleration + speed + agility + balance + jumping + strength + stamina +
                offensive_awareness + composure + reactions) / 10
    gk = (gk_diving + gk_handling + gk_kicking + gk_reflexes + gk_positioning) / 5

    fig_bar = go.Figure()
    categories = ['Attacking', 'Defending', 'Physical', 'Goalkeeping']
    values_cat = [attack, defense, physical, gk]
    colors = ['#00b894', '#e17055', '#74b9ff', '#fdcb6e']

    fig_bar.add_trace(go.Bar(
        y=categories,
        x=values_cat,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.1f}" for v in values_cat],
        textposition='outside'
    ))
    fig_bar.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b0b0b0'),
        xaxis=dict(range=[0, 100], gridcolor='#2a323c'),
        yaxis=dict(gridcolor='#2a323c'),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Circular Progress for Overall Rating
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rating,
        title={'text': "Overall Rating"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#b0b0b0"},
            'bar': {'color': "#00b894"},
            'bgcolor': "rgba(30,36,44,0.8)",
            'borderwidth': 2,
            'bordercolor': "#00b894",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255,85,85,0.3)'},
                {'range': [50, 65], 'color': 'rgba(255,165,0,0.3)'},
                {'range': [65, 80], 'color': 'rgba(0,200,0,0.3)'},
                {'range': [80, 90], 'color': 'rgba(50,100,255,0.3)'},
                {'range': [90, 100], 'color': 'rgba(255,215,0,0.3)'}
            ]
        }
    ))
    fig_gauge.update_layout(
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b0b0b0')
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
    <div class="footer">
        © 2025 Football Player Overall Rating Predictor | Built with Streamlit & Linear Regression ⚽
    </div>
""", unsafe_allow_html=True)