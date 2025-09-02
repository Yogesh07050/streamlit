import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import io
import requests
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
import pandas as pd
from datetime import datetime
import base64
 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
# Configure page
st.set_page_config(
    page_title="CLIP Classification System",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
    # Custom CSS for professional styling compatible with dark theme
def load_custom_css():
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
   
    .main-header h1 {
        text-align: center;
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
    }
   
    .section-header {
        padding: 1rem;
        border-left: 4px solid #3b82f6;
        margin: 1.5rem 0;
        border-radius: 8px;
    }
   
    .section-header h3 {
        margin: 0;
        color: #3b82f6;
        font-weight: 600;
    }
   
    .metric-card {
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
    }
   
    .success-box {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #22c55e;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
   
    .warning-box {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        color: #fbbf24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
   
    .error-box {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
   
    .result-positive {
        border-left: 4px solid #22c55e;
        background: rgba(34, 197, 94, 0.05);
    }
   
    .result-negative {
        border-left: 4px solid #ef4444;
        background: rgba(239, 68, 68, 0.05);
    }
   
    .stTextArea textarea {
        border-radius: 8px !important;
    }
   
    .stSelectbox > div > div {
        border-radius: 8px !important;
    }
   
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)
 
@st.cache_resource
def load_clip_model(model_name: str = "ViT-B/32"):
    """
    Load CLIP model and preprocessing function with error handling
   
    Args:
        model_name: Name of the CLIP model to load
       
    Returns:
        Tuple of (model, preprocess_function, device) or (None, None, None) on error
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP model '{model_name}' on device: {device}")
       
        model, preprocess = clip.load(model_name, device=device)
        model.eval()  # Set to evaluation mode
       
        logger.info("CLIP model loaded successfully")
        return model, preprocess, device
   
    except Exception as e:
        logger.error(f"Error loading CLIP model: {e}")
        return None, None, None
 
def validate_prompts(positive_prompts: List[str], negative_prompts: List[str]) -> Tuple[bool, str]:
    """
    Validate prompt lists
   
    Args:
        positive_prompts: List of positive classification prompts
        negative_prompts: List of negative classification prompts
       
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not positive_prompts:
        return False, "At least one positive prompt is required"
   
    if not negative_prompts:
        return False, "At least one negative prompt is required"
   
    if len(positive_prompts) + len(negative_prompts) > 50:
        return False, "Total number of prompts should not exceed 50 for optimal performance"
   
    # Check for empty prompts
    if any(not prompt.strip() for prompt in positive_prompts + negative_prompts):
        return False, "Empty prompts are not allowed"
   
    return True, ""
 
def preprocess_image(input_data, input_source: str) -> Optional[Image.Image]:
    """
    Preprocess image from various sources
   
    Args:
        input_data: Image data (file upload or URL string)
        input_source: Source type ("upload" or "url")
       
    Returns:
        PIL Image or None on error
    """
    try:
        if input_source == "url":
            response = requests.get(input_data, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
        else:
            image = Image.open(input_data)
       
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
           
        # Basic size validation
        if max(image.size) > 4096:
            logger.warning("Image size is very large, this may affect performance")
           
        return image
   
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None
 
def classify_input(
    model,
    preprocess,
    device,
    input_data,
    positive_prompts: List[str],
    negative_prompts: List[str],
    input_type: str = "image"
) -> Optional[Dict[str, Any]]:
    """
    Classify input using CLIP model with comprehensive error handling
   
    Args:
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: Computing device
        input_data: Input data to classify
        positive_prompts: List of positive prompts
        negative_prompts: List of negative prompts
        input_type: Type of input ("image" or "text")
       
    Returns:
        Classification results dictionary or None on error
    """
    try:
        # Validate inputs
        is_valid, error_msg = validate_prompts(positive_prompts, negative_prompts)
        if not is_valid:
            raise ValueError(error_msg)
       
        # Prepare text prompts
        all_prompts = positive_prompts + negative_prompts
       
        # Handle long prompts by truncating if necessary
        truncated_prompts = []
        for prompt in all_prompts:
            if len(prompt) > 77:  # CLIP's max token length
                truncated_prompts.append(prompt[:77])
                logger.warning(f"Prompt truncated: {prompt[:50]}...")
            else:
                truncated_prompts.append(prompt)
       
        text_inputs = clip.tokenize(truncated_prompts).to(device)
       
        if input_type == "image":
            # Process image
            if isinstance(input_data, str):
                image = preprocess_image(input_data, "url")
            else:
                image = preprocess_image(input_data, "upload")
           
            if image is None:
                raise ValueError("Failed to process image")
           
            image_input = preprocess(image).unsqueeze(0).to(device)
           
            # Get features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
               
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
               
                # Calculate similarities
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                similarities = similarities[0].cpu().numpy()
       
        elif input_type == "text":
            # Process text input
            if len(input_data) > 77:
                logger.warning("Input text is too long and will be truncated")
                input_data = input_data[:77]
           
            input_text = clip.tokenize([input_data]).to(device)
           
            with torch.no_grad():
                input_features = model.encode_text(input_text)
                text_features = model.encode_text(text_inputs)
               
                # Normalize features
                input_features = input_features / input_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
               
                # Calculate similarities
                similarities = (100.0 * input_features @ text_features.T).softmax(dim=-1)
                similarities = similarities[0].cpu().numpy()
       
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
       
        # Calculate scores for positive and negative categories
        positive_scores = similarities[:len(positive_prompts)]
        negative_scores = similarities[len(positive_prompts):]
       
        positive_total = np.sum(positive_scores)
        negative_total = np.sum(negative_scores)
       
        # Determine classification with confidence calculation
        is_positive = positive_total > negative_total
        confidence = max(positive_total, negative_total)
        margin = abs(positive_total - negative_total)
       
        return {
            'classification': 'Positive' if is_positive else 'Negative',
            'confidence': float(confidence),
            'margin': float(margin),
            'positive_score': float(positive_total),
            'negative_score': float(negative_total),
            'timestamp': datetime.now().isoformat(),
            'input_type': input_type,
            'detailed_scores': {
                'positive_prompts': [(prompt, float(score)) for prompt, score in zip(positive_prompts, positive_scores)],
                'negative_prompts': [(prompt, float(score)) for prompt, score in zip(negative_prompts, negative_scores)]
            }
        }
   
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        st.error(f"Classification error: {str(e)}")
        return None
 
def export_results(results: Dict[str, Any], input_type: str) -> str:
    """
    Export classification results to JSON format
   
    Args:
        results: Classification results dictionary
        input_type: Type of input classified
       
    Returns:
        JSON string of results
    """
    export_data = {
        'classification_results': results,
        'export_timestamp': datetime.now().isoformat(),
        'input_type': input_type
    }
    return json.dumps(export_data, indent=2)
 
def display_results_dashboard(result: Dict[str, Any]):
    """
    Display classification results in a professional dashboard format
   
    Args:
        result: Classification results dictionary
    """
    classification = result['classification']
    confidence = result['confidence']
    margin = result['margin']
   
    # Main classification result with styling
    result_class = "result-positive" if classification == "Positive" else "result-negative"
   
    st.markdown(f"""
    <div class="metric-card {result_class}">
        <h3>Classification Result: {classification}</h3>
        <p><strong>Confidence Score:</strong> {confidence:.4f}</p>
        <p><strong>Decision Margin:</strong> {margin:.4f}</p>
    </div>
    """, unsafe_allow_html=True)
   
    # Metrics in columns
    col1, col2, col3 = st.columns(3)
   
    with col1:
        st.metric(
            "Confidence Score",
            f"{confidence:.4f}",
            delta=None
        )
   
    with col2:
        st.metric(
            "Positive Score",
            f"{result['positive_score']:.4f}",
            delta=None
        )
   
    with col3:
        st.metric(
            "Negative Score",
            f"{result['negative_score']:.4f}",
            delta=None
        )
   
    # Detailed scores
    st.markdown('<div class="section-header"><h4>Detailed Score Analysis</h4></div>', unsafe_allow_html=True)
   
    # Create dataframes for better visualization
    positive_df = pd.DataFrame(
        result['detailed_scores']['positive_prompts'],
        columns=['Prompt', 'Score']
    )
    negative_df = pd.DataFrame(
        result['detailed_scores']['negative_prompts'],
        columns=['Prompt', 'Score']
    )
   
    col_pos, col_neg = st.columns(2)
   
    with col_pos:
        st.subheader("Positive Prompts Scores")
        st.dataframe(positive_df, use_container_width=True)
       
        # Progress bars for positive scores
        for _, row in positive_df.iterrows():
            st.progress(float(row['Score']), text=f"{row['Prompt']}: {row['Score']:.4f}")
   
    with col_neg:
        st.subheader("Negative Prompts Scores")
        st.dataframe(negative_df, use_container_width=True)
       
        # Progress bars for negative scores
        for _, row in negative_df.iterrows():
            st.progress(float(row['Score']), text=f"{row['Prompt']}: {row['Score']:.4f}")
 
def main():
    """Main application function"""
    load_custom_css()
   
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>CLIP Classification System</h1>
    </div>
    """, unsafe_allow_html=True)
   
    st.markdown("**Professional AI-powered classification system using OpenAI's CLIP model**")
   
    # Load model
    with st.spinner("Initializing CLIP model..."):
        model, preprocess, device = load_clip_model()
   
    if model is None:
        st.markdown("""
        <div class="error-box">
            <strong>Model Loading Error:</strong> Failed to load CLIP model. Please ensure all dependencies are installed correctly.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
   
    st.markdown(f"""
    <div class="success-box">
        <strong>System Status:</strong> CLIP model loaded successfully on {device.upper()}
    </div>
    """, unsafe_allow_html=True)
   
    # Sidebar configuration
    with st.sidebar:
        st.header("System Configuration")
       
        # Model selection
        available_models = ["ViT-B/32", "ViT-B/16", "RN50", "RN101", "RN50x4"]
        selected_model = st.selectbox("CLIP Model", available_models, index=0)
       
        if selected_model != "ViT-B/32":
            st.info("Note: Changing model requires reloading. This may take a moment.")
       
        # Input type selection
        input_type = st.radio("Input Type", ["Image", "Text"])
       
        st.markdown("---")
       
        # Prompt configuration
        st.subheader("Classification Prompts")
       
        # Positive prompts
        with st.expander("Positive Class Definition", expanded=True):
            positive_prompts_text = st.text_area(
                "Positive Prompts (one per line)",
                value="professional appearance\nformal attire\nbusiness setting\npositive expression",
                height=120,
                help="Define characteristics that should be classified as positive"
            )
       
        # Negative prompts
        with st.expander("Negative Class Definition", expanded=True):
            negative_prompts_text = st.text_area(
                "Negative Prompts (one per line)",
                value="casual appearance\ninformal attire\nrelaxed setting\nnegative expression",
                height=120,
                help="Define characteristics that should be classified as negative"
            )
       
        # Template presets for quick setup
        st.markdown("---")
        st.subheader("Quick Setup Templates")
       
        template_options = {
            "Sentiment Analysis": {
                "positive": ["positive sentiment\nhappy emotion\njoyful expression\noptimistic tone"],
                "negative": ["negative sentiment\nsad emotion\nangry expression\npessimistic tone"]
            },
            "Content Moderation": {
                "positive": ["appropriate content\nsafe for work\nfamily friendly\nprofessional"],
                "negative": ["inappropriate content\noffensive material\ninappropriate language\nunprofessional"]
            },
            "Product Quality": {
                "positive": ["high quality\npremium product\nexcellent condition\nprofessional grade"],
                "negative": ["low quality\ncheap product\npoor condition\namateurish"]
            }
        }
       
        selected_template = st.selectbox("Apply Template", ["Custom"] + list(template_options.keys()))
       
        if selected_template != "Custom" and st.button("Apply Template", use_container_width=True):
            st.rerun()
       
        # Process prompts with template handling
        if selected_template != "Custom" and 'template_applied' not in st.session_state:
            if selected_template in template_options:
                positive_prompts_text = template_options[selected_template]["positive"][0]
                negative_prompts_text = template_options[selected_template]["negative"][0]
                st.session_state.template_applied = selected_template
       
        positive_prompts = [p.strip() for p in positive_prompts_text.split('\n') if p.strip()]
        negative_prompts = [p.strip() for p in negative_prompts_text.split('\n') if p.strip()]
       
        # Prompt validation
        is_valid, validation_error = validate_prompts(positive_prompts, negative_prompts)
       
        if is_valid:
            st.success(f"Configuration Valid: {len(positive_prompts)} positive, {len(negative_prompts)} negative prompts")
        else:
            st.error(f"Configuration Error: {validation_error}")
       
        # Advanced settings
        with st.expander("Advanced Settings"):
            enable_export = st.checkbox("Enable Results Export", value=True)
            show_raw_scores = st.checkbox("Show Raw Similarity Scores", value=False)
   
    # Main content area
    col1, col2 = st.columns([1, 1])
   
    with col1:
        st.markdown('<div class="section-header"><h3>Input Data</h3></div>', unsafe_allow_html=True)
       
        input_data = None
       
        if input_type == "Image":
            # Image input interface
            image_source = st.radio("Image Source", ["File Upload", "URL"], horizontal=True)
           
            if image_source == "File Upload":
                uploaded_file = st.file_uploader(
                    "Select Image File",
                    type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'],
                    help="Supported formats: PNG, JPG, JPEG, GIF, BMP, WebP"
                )
               
                if uploaded_file:
                    input_data = uploaded_file
                   
                    # Display image with metadata
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Selected: {uploaded_file.name}", use_column_width=True)
                   
                    # Enhanced image metadata
                    file_size = len(uploaded_file.getvalue()) / 1024  # KB
                    st.markdown(f"""
                    **Image Information:**
                    - **Filename:** {uploaded_file.name}
                    - **Dimensions:** {image.size[0]} × {image.size[1]} pixels
                    - **Format:** {image.format}
                    - **File Size:** {file_size:.1f} KB
                    - **Color Mode:** {image.mode}
                    """)
           
            else:  # URL input
                image_url = st.text_input(
                    "Image URL",
                    placeholder="https://example.com/image.jpg",
                    help="Enter a direct link to an image file"
                )
               
                if image_url:
                    try:
                        with st.spinner("Loading image from URL..."):
                            response = requests.get(image_url, timeout=10)
                            response.raise_for_status()
                            image = Image.open(io.BytesIO(response.content))
                           
                        input_data = image_url
                        st.image(image, caption="Image from URL", use_column_width=True)
                        st.info(f"Image Size: {image.size[0]} x {image.size[1]} pixels")
                       
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
       
        else:  # Text input
            text_input = st.text_area(
                "Text Input",
                height=200,
                placeholder="Enter the text you want to classify...",
                help="Maximum length: 77 characters (CLIP limitation)"
            )
           
            if text_input.strip():
                input_data = text_input.strip()
               
                # Text preview
                st.markdown("**Text Preview:**")
                st.info(f"Length: {len(text_input)} characters")
               
                if len(text_input) > 77:
                    st.warning("Text will be truncated to 77 characters for CLIP processing")
               
                with st.expander("Full Text"):
                    st.text(text_input)
   
    with col2:
        st.markdown('<div class="section-header"><h3>Classification Results</h3></div>', unsafe_allow_html=True)
       
        # Classification button
        can_classify = input_data is not None and is_valid
       
        if st.button(
            "Run Classification",
            type="primary",
            use_container_width=True,
            disabled=not can_classify
        ):
            if not can_classify:
                if not input_data:
                    st.warning("Please provide input data for classification")
                if not is_valid:
                    st.error(f"Configuration error: {validation_error}")
            else:
                with st.spinner("Processing classification..."):
                    result = classify_input(
                        model, preprocess, device, input_data,
                        positive_prompts, negative_prompts,
                        input_type.lower()
                    )
               
                if result:
                    # Store result in session state
                    st.session_state.last_result = result
                   
                    # Display results
                    display_results_dashboard(result)
                   
                    # Export functionality
                    if enable_export:
                        st.markdown("---")
                        st.subheader("Export Results")
                       
                        export_json = export_results(result, input_type.lower())
                       
                        st.download_button(
                            label="Download Results (JSON)",
                            data=export_json,
                            file_name=f"clip_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                   
                    # Raw scores display
                    if show_raw_scores:
                        st.markdown("---")
                        st.subheader("Raw Similarity Scores")
                        st.json(result['detailed_scores'])
       
        else:
            if not input_data:
                st.info("Please provide input data to begin classification")
            elif not is_valid:
                st.error("Please fix configuration errors before proceeding")
   
    # System information and documentation
    with st.expander("System Information & Usage Guide"):
        st.markdown("""
        ### CLIP Classification System
       
        **System Capabilities:**
        - Multi-modal classification (images and text)
        - Customizable positive/negative class definitions
        - Professional results dashboard
        - Exportable classification reports
       
        **Usage Instructions:**
        1. **Configure Prompts**: Define positive and negative class characteristics
        2. **Select Input Type**: Choose between image or text classification
        3. **Provide Input**: Upload image/enter URL or input text
        4. **Run Classification**: Execute classification and review results
       
        **Technical Details:**
        - Model: OpenAI CLIP (Contrastive Language-Image Pre-training)
        - Supported Image Formats: PNG, JPG, JPEG, GIF, BMP, WebP
        - Text Length Limit: 77 characters (CLIP tokenization limit)
        - Processing: GPU-accelerated when available
       
        **Performance Notes:**
        - Larger images may require more processing time
        - More prompts increase computation but may improve accuracy
        - Results include confidence scores and decision margins for reliability assessment
        """)
 
if __name__ == "__main__":
    main()
