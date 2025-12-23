import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow import keras
from PIL import Image
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="ChestMNIST AI Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark professional styling
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    /* Main container */
    .main {
        background: transparent;
        padding: 1rem;
    }
    
    /* Header styling */
    h1 {
        color: #e0e7ff !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 800;
        text-align: center;
        padding: 1.5rem 0;
        text-shadow: 0 0 30px rgba(99, 102, 241, 0.5);
        letter-spacing: -0.5px;
    }
    
    h2, h3 {
        color: #cbd5e1 !important;
        font-weight: 600;
    }
    
    /* Paragraph text */
    p, label, .stMarkdown {
        color: #cbd5e1 !important;
    }
    
    /* Cards and containers */
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] strong {
        color: #cbd5e1 !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #818cf8 !important;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        font-weight: 700;
        padding: 0.875rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        font-size: 1.1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(99, 102, 241, 0.6);
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(99, 102, 241, 0.5);
        border-radius: 15px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"] label {
        color: #e0e7ff !important;
        font-weight: 600;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricValue"] {
        color: #e0e7ff !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricDelta"] {
        color: #a5b4fc !important;
    }
    
    /* Info/Success/Warning boxes */
    .stAlert {
        background: rgba(30, 41, 59, 0.8) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border-left: 4px solid #6366f1;
        color: #cbd5e1 !important;
    }
    
    .stSuccess {
        border-left-color: #10b981 !important;
    }
    
    .stInfo {
        border-left-color: #3b82f6 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        color: #cbd5e1 !important;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-top: none;
        color: #cbd5e1 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 41, 59, 0.4);
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(99, 102, 241, 0.5);
        color: #cbd5e1;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton>button:hover {
        background: rgba(99, 102, 241, 0.3);
        border-color: #6366f1;
        transform: translateY(-2px);
    }
    
    /* JSON viewer */
    .stJson {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #6366f1 !important;
    }
    
    /* Custom cards */
    .custom-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #6366f1;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #8b5cf6;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "frontend/model/saved_model/v2"

# ChestMNIST class labels (14 pathologies)
CLASS_LABELS = [
    "Atelectasis",
    "Cardiomegaly", 
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

@st.cache_resource
def load_model():
    """Load the TensorFlow SavedModel"""
    try:
        return keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess(img):
    """Preprocess image for model input"""
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    return img

def predict(img, model):
    """Generate predictions from preprocessed image"""
    img = np.expand_dims(img, axis=0)
    out_dict = model(img)
    out_tensor = list(out_dict.values())[0]
    preds = out_tensor.numpy()[0]
    return preds

def create_prediction_chart(predictions, labels):
    """Create an interactive bar chart for predictions"""
    # Sort predictions and labels together
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_preds = predictions[sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    # Color coding based on probability
    colors = ['#ef4444' if p > 0.7 else '#f59e0b' if p > 0.4 else '#10b981' 
              for p in sorted_preds]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_preds * 100,
            y=sorted_labels,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f'{p*100:.1f}%' for p in sorted_preds],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Pathology Detection Probabilities',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#e0e7ff', 'family': 'Arial Black'}
        },
        xaxis_title='Probability (%)',
        yaxis_title='Pathology',
        height=600,
        template='plotly_dark',
        font=dict(size=12, color='#cbd5e1'),
        xaxis=dict(range=[0, 100], gridcolor='rgba(99, 102, 241, 0.2)'),
        yaxis=dict(gridcolor='rgba(99, 102, 241, 0.2)'),
        margin=dict(l=150, r=50, t=80, b=50),
        plot_bgcolor='rgba(15, 23, 42, 0.8)',
        paper_bgcolor='rgba(30, 41, 59, 0.6)',
    )
    
    return fig

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability > 0.7:
        return "HIGH", "#ef4444"
    elif probability > 0.4:
        return "MODERATE", "#f59e0b"
    else:
        return "LOW", "#10b981"

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913133.png", width=100)
    st.markdown("### ChestMNIST AI Classifier")
    st.markdown("---")
    
    st.markdown("""
    **Advanced Medical Imaging Analysis**
    
    This AI-powered system analyzes chest X-ray images to detect 14 different pathologies using deep learning.
    
    **Supported Conditions:**
    - Atelectasis
    - Cardiomegaly
    - Effusion
    - Infiltration
    - Mass & Nodule
    - Pneumonia & Pneumothorax
    - And 7 more conditions
    
    **Instructions:**
    1. Upload a chest X-ray image
    2. Click 'Analyze Image'
    3. Review detailed results
    
    **Disclaimer:** This tool is for research and educational purposes only. Always consult qualified healthcare professionals for medical diagnosis.
    """)
    
    st.markdown("---")
    st.markdown(f"**Session:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Main content
st.title("ChestMNIST AI Diagnostic Platform")
st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; font-weight: 500; margin-top: -1rem;'>Advanced Deep Learning for Medical Image Analysis</p>", unsafe_allow_html=True)

# Load model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check the model path.")
    st.stop()

# Create two columns for layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Upload Chest X-ray")
    uploaded = st.file_uploader(
        "Choose an X-ray image file",
        type=["png", "jpg", "jpeg"],
        help="Upload a chest X-ray image in PNG, JPG, or JPEG format"
    )
    
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded X-ray Image", use_column_width=True)
        
        # Image info
        st.markdown("**Image Information:**")
        st.info(f"**Size:** {img.size[0]} x {img.size[1]} pixels\n**Format:** {img.format}\n**Mode:** {img.mode}")

with col2:
    st.markdown("### Analysis Results")
    
    if uploaded:
        if st.button("Analyze Image", use_container_width=True):

            with st.spinner("Processing image and running AI analysis..."):
                # Preprocess and predict
                pre = preprocess(img)
                preds = predict(pre, model)
                
                # Store in session state
                st.session_state['predictions'] = preds
                st.session_state['analyzed'] = True
        
        # Display results if available
        if 'analyzed' in st.session_state and st.session_state['analyzed']:
            preds = st.session_state['predictions']
            
            st.success("Analysis Complete!")
            
            # Top findings
            st.markdown("#### Top Findings")
            top_indices = np.argsort(preds)[-3:][::-1]
            
            for idx in top_indices:
                risk_label, risk_color = get_risk_level(preds[idx])
                st.markdown(f"""
                <div class='custom-card' style='border-left: 4px solid {risk_color};'>
                    <strong style='font-size: 1.2rem; color: #e0e7ff;'>{CLASS_LABELS[idx]}</strong><br>
                    <span style='color: {risk_color}; font-weight: bold; font-size: 1rem;'>{risk_label}</span> 
                    <span style='color: #cbd5e1;'>- {preds[idx]*100:.1f}% probability</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Upload an X-ray image to begin analysis")

# Full results section
if 'analyzed' in st.session_state and st.session_state['analyzed']:
    st.markdown("---")
    st.markdown("### Comprehensive Pathology Report")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Visual Analysis", "Detailed Results", "Export Data"])
    
    with tab1:
        preds = st.session_state['predictions']
        fig = create_prediction_chart(preds, CLASS_LABELS)
        st.plotly_chart(fig, width='stretch')
        
        # Summary statistics
        st.markdown("#### Summary Statistics")
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            st.metric("Highest Probability", f"{np.max(preds)*100:.1f}%", 
                     CLASS_LABELS[np.argmax(preds)])
        with col_b:
            high_risk = np.sum(preds > 0.7)
            st.metric("High Risk Findings", high_risk, 
                     "Alert" if high_risk > 0 else "Clear")
        with col_c:
            moderate_risk = np.sum((preds > 0.4) & (preds <= 0.7))
            st.metric("Moderate Risk", moderate_risk,
                     "Monitor" if moderate_risk > 0 else "Clear")
        with col_d:
            avg_prob = np.mean(preds)
            st.metric("Average Probability", f"{avg_prob*100:.1f}%")
    
    with tab2:
        st.markdown("#### Complete Pathology Breakdown")
        
        # Create detailed table
        for i, (label, prob) in enumerate(zip(CLASS_LABELS, preds)):
            risk_label, risk_color = get_risk_level(prob)
            
            with st.expander(f"{i+1}. {label} - {prob*100:.2f}%"):
                col_x, col_y = st.columns([3, 1])
                with col_x:
                    st.progress(float(prob))
                    st.markdown(f"**Probability:** {prob*100:.2f}%")
                    st.markdown(f"**Risk Level:** {risk_label}")
                with col_y:
                    st.markdown(f"<h1 style='text-align: center; color: {risk_color};'>{prob*100:.0f}%</h1>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### Export Analysis Results")
        
        # Prepare export data
        results_dict = {label: f"{prob*100:.2f}%" for label, prob in zip(CLASS_LABELS, preds)}
        
        import json
        results_json = json.dumps(results_dict, indent=2)
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            st.download_button(
                label="Download JSON Report",
                data=results_json,
                file_name=f"chest_xray_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col_export2:
            csv_data = "Pathology,Probability\n" + "\n".join([f"{label},{prob*100:.2f}" for label, prob in zip(CLASS_LABELS, preds)])
            st.download_button(
                label="Download CSV Report",
                data=csv_data,
                file_name=f"chest_xray_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.markdown("**Preview:**")
        st.json(results_dict)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; padding: 2rem 0;'>
    <p style='color: #cbd5e1;'><strong>ChestMNIST AI Classifier v2.0</strong></p>
    <p style='color: #94a3b8;'>Powered by TensorFlow & Deep Learning | For Research & Educational Use Only</p>
    <p style='color: #f87171;'>WARNING: Not approved for clinical diagnosis - Consult healthcare professionals</p>
</div>
""", unsafe_allow_html=True)