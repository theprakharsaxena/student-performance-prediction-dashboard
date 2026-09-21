import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time

# Page Configuration
st.set_page_config(
    page_title="Student Academic Performance Prediction | ANN Deep Learning",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load Trained Model, Weights, and Scaler
@st.cache_resource
def load_artifacts():
    model = None
    try:
        from tensorflow.keras.models import load_model
        model = load_model("student_performance_model.keras")
    except Exception:
        model = None

    weights = joblib.load("model_weights.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, weights, scaler

@st.cache_data
def load_dataset():
    return pd.read_csv("student_performance.csv")

def predict_ann(scaled_input, model, weights):
    if model is not None:
        try:
            raw = model.predict(scaled_input, verbose=0)
            return float(raw[0][0])
        except Exception:
            pass
            
    x = np.array(scaled_input)
    # Layer 1: Dense(32, ReLU)
    a1 = np.maximum(0, np.dot(x, weights[0][0]) + weights[0][1])
    # Layer 2: Dense(16, ReLU)
    a2 = np.maximum(0, np.dot(a1, weights[1][0]) + weights[1][1])
    # Layer 3: Dense(8, ReLU)
    a3 = np.maximum(0, np.dot(a2, weights[2][0]) + weights[2][1])
    # Layer 4: Dense(1, Linear)
    out = np.dot(a3, weights[3][0]) + weights[3][1]
    return float(out[0][0])

try:
    model, weights, scaler = load_artifacts()
    df = load_dataset()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    st.error(f"Error loading model artifacts: {e}")

# Sidebar Header Badge
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 15px 0;">
    <div style="background: linear-gradient(135deg, #1E3A8A, #3B82F6); border-radius: 50%; width: 85px; height: 85px; margin: 0 auto; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(0,0,0,0.2);">
        <span style="font-size: 45px;">🎓</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.title("Navigation & Info")
st.sidebar.info("This web application uses an **Artificial Neural Network (ANN)** to predict a student's final exam score based on study habits and academic metrics.")

navigation_choice = st.sidebar.radio(
    "Select Section:",
    ["🔮 Predict Exam Score", "📊 Dataset & Analytics", "🧠 ANN Theory & Architecture"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Quick Settings")
show_raw_inputs = st.sidebar.checkbox("Show Raw Normalized Inputs", value=False)
theme_select = st.sidebar.selectbox("Dashboard Accent Color", ["Blue", "Purple", "Emerald"])

# App Header
st.markdown('<div class="main-title">🎓 Student Performance Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by Deep Learning Artificial Neural Network (ANN Regression)</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict Score", "📊 Dataset & EDA", "🧠 ANN Architecture & Theory"])

# ---------------------------------------------------------
# TAB 1: PREDICTION INTERFACE
# ---------------------------------------------------------
with tab1:
    st.header("🔮 Final Exam Score Estimator")
    st.text("Adjust the student's study habits and parameters below to compute predicted final exam marks.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📚 Study Metrics")
        study_hours = st.slider("Study Hours / Week", min_value=1.0, max_value=50.0, value=25.0, step=0.5)
        prev_score = st.number_input("Previous Semester Exam Score (%)", min_value=30.0, max_value=100.0, value=75.0, step=0.5)
        
    with col2:
        st.subheader("🏫 Attendance & Rest")
        attendance = st.slider("Class Attendance Percentage (%)", min_value=40.0, max_value=100.0, value=85.0, step=0.5)
        sleep_hours = st.number_input("Sleep Hours / Night", min_value=4.0, max_value=10.0, value=7.0, step=0.5)
        
    with col3:
        st.subheader("📝 Practice & Support")
        practice_tests = st.number_input("Mock / Practice Tests Taken", min_value=0, max_value=15, value=5, step=1)
        tutoring_sessions = st.slider("Tutoring Sessions Attended", min_value=0, max_value=10, value=3, step=1)

    st.markdown("---")
    predict_btn = st.button("🚀 Predict Final Exam Score", width="stretch")
    
    if predict_btn:
        with st.spinner("Calculating neural network activations..."):
            # Simulate progress bar for UI demo
            progress_bar = st.progress(0)
            for p in range(100):
                time.sleep(0.005)
                progress_bar.progress(p + 1)
                
            input_features = pd.DataFrame([[
                study_hours,
                attendance,
                prev_score,
                sleep_hours,
                practice_tests,
                tutoring_sessions
            ]], columns=['Study_Hours_Per_Week', 'Attendance_Percentage', 'Previous_Exam_Score', 'Sleep_Hours_Per_Night', 'Practice_Tests_Taken', 'Tutoring_Sessions'])
            
            scaled_features = scaler.transform(input_features)
            raw_prediction = predict_ann(scaled_features, model, weights)
            predicted_score = float(np.clip(raw_prediction, 0.0, 100.0))
            
            st.success("✅ Prediction Completed!")
            
            # Display Metric KPIs
            m_col1, m_col2, m_col3 = st.columns(3)
            score_diff = predicted_score - prev_score
            delta_str = f"{score_diff:+.1f}% vs Previous Score"
            
            with m_col1:
                st.metric("Predicted Final Exam Score", f"{predicted_score:.1f} / 100", delta=delta_str)
            with m_col2:
                grade_label = "Grade A 🌟" if predicted_score >= 85 else ("Grade B 👍" if predicted_score >= 70 else "Grade C ⚠️")
                st.metric("Expected Performance Grade", grade_label)
            with m_col3:
                st.metric("Target Exam Score Benchmark", "75.0 / 100")
                
            # Dynamic Feedback Banner
            if predicted_score >= 85.0:
                st.info("🌟 **Outstanding Performance!** The student is on track for top honors.")
            elif predicted_score >= 70.0:
                st.success("👍 **Solid Performance!** Consistent effort will yield high results.")
            elif predicted_score >= 50.0:
                st.warning("⚠️ **Average Performance.** Increasing study hours or tutoring could boost marks.")
            else:
                st.error("❌ **Critical Warning!** High risk of low marks. Immediate intervention recommended.")
                
            if show_raw_inputs:
                st.subheader("Raw Normalized Feature Vector")
                st.code(f"Scaled Inputs (X_scaled): {scaled_features.tolist()}")
                
            # Feature Radar Chart
            st.markdown("---")
            st.subheader("📊 Student Feature Profile Visualization")
            categories = ['Study Hours', 'Attendance %', 'Previous Score', 'Sleep Hours', 'Practice Tests', 'Tutoring']
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[study_hours/50*100, attendance, prev_score, sleep_hours/10*100, practice_tests/15*100, tutoring_sessions/10*100],
                theta=categories,
                fill='toself',
                name='Student Profile'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Normalized Student Preparation Metrics (0-100% scale)"
            )
            st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------
# TAB 2: DATASET & EDA
# ---------------------------------------------------------
with tab2:
    st.header("📊 Dataset Overview & Exploratory Data Analysis")
    st.write("Explore the dataset used to train the Artificial Neural Network model.")
    
    # KPI Summaries
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Records", len(df))
    kpi2.metric("Total Input Features", len(df.columns) - 1)
    kpi3.metric("Average Final Score", f"{df['Final_Exam_Score'].mean():.1f}")
    kpi4.metric("Max Exam Score", f"{df['Final_Exam_Score'].max():.1f}")
    
    st.subheader("📋 Dataset Sample Table")
    selected_cols = st.multiselect("Filter Columns to View", options=list(df.columns), default=list(df.columns))
    st.dataframe(df[selected_cols].head(10), width="stretch")
    
    # Download Button
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Full Dataset (CSV)",
        data=csv_data,
        file_name="student_performance.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    st.subheader("📈 Statistical Summary")
    st.table(df.describe().T[['mean', 'std', 'min', '50%', 'max']])
    
    # Interactive Plots
    st.subheader("🔥 Feature Correlation Analysis")
    fig_corr = px.imshow(
        df.corr(),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Viridis",
        title="Pearson Correlation Heatmap"
    )
    st.plotly_chart(fig_corr, width="stretch")
    
    st.subheader("🎯 Study Hours vs Final Exam Score Scatter Plot")
    fig_scatter = px.scatter(
        df,
        x="Study_Hours_Per_Week",
        y="Final_Exam_Score",
        color="Attendance_Percentage",
        size="Previous_Exam_Score",
        hover_data=["Practice_Tests_Taken", "Tutoring_Sessions"],
        title="Study Hours vs Final Score (Color: Attendance %, Size: Prev Score)"
    )
    st.plotly_chart(fig_scatter, width="stretch")

# ---------------------------------------------------------
# TAB 3: ANN ARCHITECTURE & THEORY
# ---------------------------------------------------------
with tab3:
    st.header("🧠 Artificial Neural Network Theory & Architecture")
    st.markdown("""
    An **Artificial Neural Network (ANN)** is a computing system inspired by biological neural networks in human brains.
    """)
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        st.subheader("Biological vs Artificial Neuron")
        st.markdown("""
        | Biological Neural Network | Artificial Neural Network (ANN) |
        |---|---|
        | Dendrites | Input Signals ($X$) |
        | Synaptic Weights | Weights ($W$) |
        | Cell Body (Soma) | Summation & Bias ($Z = W \\cdot X + B$) |
        | Axon Output | Activation Output ($A = f(Z)$) |
        """)
        
    with col_t2:
        st.subheader("Mathematical Neuron Formula")
        st.latex(r"Z = \sum_{i=1}^{n} (W_i \cdot X_i) + B")
        st.latex(r"A = \text{ReLU}(Z) = \max(0, Z)")
        st.markdown("Where **$W$** represents connection weights, **$X$** represents student input metrics, and **$B$** is the bias term.")
        
    st.markdown("---")
    st.subheader("🏗️ Keras Model Architecture Definition")
    st.code("""
# Keras Sequential Model Architecture
model = Sequential([
    Dense(32, activation='relu', input_shape=(6,)),   # Layer 1: Input (6) -> Dense (32)
    Dense(16, activation='relu'),                     # Layer 2: Hidden (16)
    Dense(8, activation='relu'),                      # Layer 3: Hidden (8)
    Dense(1, activation='linear')                     # Layer 4: Output (1 continuous prediction)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
""", language="python")

    with st.expander("🔍 Click to read about Activation Functions & Loss Optimization"):
        st.markdown("""
        - **ReLU (Rectified Linear Unit)**: $f(x) = \\max(0, x)$. Provides non-linear transformation while preventing vanishing gradient problem.
        - **Linear Activation**: $f(x) = x$. Used in the output layer for continuous numerical target prediction.
        - **Adam Optimizer**: An adaptive learning rate optimization algorithm that computes individual learning rates for different parameters.
        - **Mean Squared Error (MSE)**: The loss function minimized during training: $\text{MSE} = \\frac{1}{N} \\sum (y_{actual} - y_{predicted})^2$.
        """)
