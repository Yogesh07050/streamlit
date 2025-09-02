import streamlit as st
import torch
from PIL import Image
import numpy as np
import io
import requests
from typing import List, Tuple
from transformers import CLIPProcessor, CLIPModel

# Configure page
st.set_page_config(
    page_title="CLIP Custom Classifier",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4ecdc4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_clip_model():
    """Load CLIP model using Hugging Face Transformers"""
    try:
        st.info("Loading CLIP model via Hugging Face Transformers...")
        
        # Create a temporary directory for cache
        import tempfile
        import os
        temp_cache_dir = tempfile.mkdtemp()
        
        # Set cache directory environment variable
        os.environ['HF_HOME'] = temp_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = temp_cache_dir
        os.environ['HF_DATASETS_CACHE'] = temp_cache_dir
        
        # Load model and processor with custom cache
        model_name = "openai/clip-vit-base-patch32"
        
        from transformers import CLIPModel, CLIPProcessor
        
        model = CLIPModel.from_pretrained(model_name, cache_dir=temp_cache_dir)
        processor = CLIPProcessor.from_pretrained(model_name, cache_dir=temp_cache_dir)
        
        device = "cpu"
        model.to(device)
        
        return model, processor, device
        
    except Exception as e:
        st.error(f"Error loading CLIP model: {e}")
        
        # Fallback: Try loading without custom cache
        try:
            st.info("Trying fallback loading method...")
            from transformers import CLIPModel, CLIPProcessor
            
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            device = "cpu"
            model.to(device)
            
            return model, processor, device
            
        except Exception as e2:
            st.error(f"Fallback also failed: {e2}")
            st.info("This might be a temporary issue. Please try refreshing the page in a few minutes.")
            return None, None, None

def classify_input(model, processor, device, input_data, positive_prompts, negative_prompts, input_type="image"):
    """
    Classify input based on positive and negative prompts using CLIP
    """
    try:
        # Prepare text prompts
        all_prompts = positive_prompts + negative_prompts
        
        if input_type == "image":
            # Process image
            if isinstance(input_data, str):  # URL
                response = requests.get(input_data, timeout=10)
                image = Image.open(io.BytesIO(response.content))
            else:  # Uploaded file
                image = Image.open(input_data)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process inputs
            inputs = processor(text=all_prompts, images=image, return_tensors="pt", padding=True)
            
            # Get outputs
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        elif input_type == "text":
            # For text-to-text comparison, we'll use text features
            # Process the input text and all prompts
            all_texts = [input_data] + all_prompts
            
            inputs = processor(text=all_texts, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
                
                # Calculate similarities between input text and prompts
                input_features = text_features[0:1]  # First text is input
                prompt_features = text_features[1:]  # Rest are prompts
                
                # Compute cosine similarity
                similarities = torch.cosine_similarity(input_features, prompt_features, dim=1)
                probs = torch.softmax(similarities * 100, dim=0).cpu().numpy()
        
        # Calculate scores for positive and negative categories
        positive_scores = probs[:len(positive_prompts)]
        negative_scores = probs[len(positive_prompts):]
        
        positive_total = np.sum(positive_scores)
        negative_total = np.sum(negative_scores)
        
        # Determine classification
        is_positive = positive_total > negative_total
        confidence = max(positive_total, negative_total)
        
        return {
            'classification': 'Positive' if is_positive else 'Negative',
            'confidence': float(confidence),
            'positive_score': float(positive_total),
            'negative_score': float(negative_total),
            'detailed_scores': {
                'positive_prompts': [(prompt, float(score)) for prompt, score in zip(positive_prompts, positive_scores)],
                'negative_prompts': [(prompt, float(score)) for prompt, score in zip(negative_prompts, negative_scores)]
            }
        }
    
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">CLIP Custom Classifier</h1>', unsafe_allow_html=True)
    st.markdown("### Define your own positive and negative prompts to classify images or text!")
    
    # Add info box
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> This app uses OpenAI's CLIP model to classify images or text based on your custom prompts. 
        Define what you consider "positive" and "negative" examples, and let AI do the classification!
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, processor, device = load_clip_model()
    
    if model is None:
        st.error("Failed to load CLIP model. Please refresh the page and try again.")
        st.stop()
    
    st.success(f"CLIP model loaded successfully on {device.upper()}")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Input type selection
        input_type = st.radio("Select input type:", ["Image", "Text"], help="Choose what type of content you want to classify")
        
        st.header("Define Classification Prompts")
        
        # Positive prompts
        st.subheader("Positive Category")
        positive_prompts_text = st.text_area(
            "Enter positive prompts (one per line):",
            value="happy face\nsmiling person\njoyful expression\npositive emotion",
            height=120,
            help="These prompts define what should be classified as 'Positive'"
        )
        
        # Negative prompts
        st.subheader("Negative Category")
        negative_prompts_text = st.text_area(
            "Enter negative prompts (one per line):",
            value="sad face\nangry person\nfrowning expression\nnegative emotion",
            height=120,
            help="These prompts define what should be classified as 'Negative'"
        )
        
        # Process prompts
        positive_prompts = [p.strip() for p in positive_prompts_text.split('\n') if p.strip()]
        negative_prompts = [p.strip() for p in negative_prompts_text.split('\n') if p.strip()]
        
        # Display prompt counts
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Positive", len(positive_prompts))
        with col2:
            st.metric("Negative", len(negative_prompts))
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        input_data = None
        
        if input_type == "Image":
            # Image input options
            image_option = st.radio("Choose image source:", ["Upload", "URL"], horizontal=True)
            
            if image_option == "Upload":
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                    help="Supported formats: PNG, JPG, JPEG, GIF, BMP"
                )
                if uploaded_file:
                    input_data = uploaded_file
                    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            else:  # URL
                image_url = st.text_input(
                    "Enter image URL:",
                    placeholder="https://example.com/image.jpg",
                    help="Paste a direct link to an image"
                )
                if image_url:
                    try:
                        with st.spinner("Loading image..."):
                            response = requests.get(image_url, timeout=10)
                            image = Image.open(io.BytesIO(response.content))
                            input_data = image_url
                            st.image(image, caption="Image from URL", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image from URL: {e}")
        
        else:  # Text input
            text_input = st.text_area(
                "Enter text to classify:",
                height=200,
                placeholder="Type your text here...",
                help="Enter any text you want to classify"
            )
            if text_input.strip():
                input_data = text_input.strip()
                st.text_area("Text to classify:", value=text_input, height=100, disabled=True)
    
    with col2:
        st.header("Classification Results")
        
        if input_data and positive_prompts and negative_prompts:
            if st.button("Classify Now", type="primary", use_container_width=True):
                with st.spinner("AI is analyzing..."):
                    result = classify_input(
                        model, processor, device, input_data, 
                        positive_prompts, negative_prompts,
                        input_type.lower()
                    )
                
                if result:
                    # Main classification result
                    classification = result['classification']
                    confidence = result['confidence']
                    
                    # Display result with color coding
                    if classification == "Positive":
                        st.markdown("### Classification: <span style='color: #28a745; font-weight: bold;'>POSITIVE</span>", 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown("### Classification: <span style='color: #dc3545; font-weight: bold;'>NEGATIVE</span>", 
                                  unsafe_allow_html=True)
                    
                    # Confidence and scores in columns
                    col_conf, col_pos, col_neg = st.columns(3)
                    
                    with col_conf:
                        st.metric("Confidence", f"{confidence:.1%}")
                    with col_pos:
                        st.metric("Positive Score", f"{result['positive_score']:.3f}")
                    with col_neg:
                        st.metric("Negative Score", f"{result['negative_score']:.3f}")
                    
                    # Detailed breakdown
                    st.subheader("Detailed Breakdown")
                    
                    # Create tabs for better organization
                    tab1, tab2 = st.tabs(["Positive Scores", "Negative Scores"])
                    
                    with tab1:
                        st.write("**Individual prompt scores:**")
                        for prompt, score in result['detailed_scores']['positive_prompts']:
                            st.progress(float(score), text=f"{prompt}: {score:.3f}")
                    
                    with tab2:
                        st.write("**Individual prompt scores:**")
                        for prompt, score in result['detailed_scores']['negative_prompts']:
                            st.progress(float(score), text=f"{prompt}: {score:.3f}")
        
        elif not positive_prompts or not negative_prompts:
            st.warning("Please define both positive and negative prompts in the sidebar.")
        
        elif not input_data:
            st.info("Please provide input data to classify in the left panel.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Made with love using [OpenAI CLIP](https://openai.com/research/clip) and [Streamlit](https://streamlit.io) | "
        "Hosted on [Hugging Face Spaces](https://huggingface.co/spaces)"
    )

if __name__ == "__main__":
    main()
