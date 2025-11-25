import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Image Text Translator",
    page_icon="🍌",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .upload-text {
        text-align: center;
        padding: 20px;
        border: 2px dashed #444;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

def init_gemini(api_key):
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def get_image_text(image, prompt=None):
    """
    Uses Gemini Pro Vision (or similar) to extract text.
    """
    if prompt is None:
        prompt = """
        Analyze this image and transcribe EVERY single piece of text visible, exactly as it appears. 
        Pay special attention to:
        1. All headers and subheaders.
        2. All form labels, input values, and placeholders.
        3. All button text and navigation items.
        4. Small print, footnotes, and any text that appears faint or unhighlighted.
        5. Text inside icons or logos if legible.
        
        Do not summarize. Provide a complete transcription of all textual content found in the image.
        """
    try:
        # Using gemini-2.5-pro for robust text recognition (available in list)
        model = genai.GenerativeModel('gemini-2.5-pro') 
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        st.error(f"Error in text recognition: {str(e)}")
        return None

def generate_translated_image(original_image, extracted_text, target_language, style_prompt=""):
    """
    Uses Nano Banana Pro (nano-banana-pro-preview) to generate the image.
    """
    try:
        # Using the explicit preview model found in the list
        model_name = "nano-banana-pro-preview" 
        
        # Construct the prompt for image generation/editing
        prompt = f"""
        Generate an image that looks exactly like the original image provided, 
        but replace the text "{extracted_text}" with its translation in {target_language}.
        Maintain the same font style, color, and background.
        {style_prompt}
        """
        
        model = genai.GenerativeModel(model_name)
        
        # Call the model with prompt and original image
        response = model.generate_content([prompt, original_image])
        
        # Check for different response formats
        if hasattr(response, 'parts') and response.parts:
            for part in response.parts:
                # Direct image attribute (some SDK versions)
                if hasattr(part, 'image') and part.image:
                    return part.image
                
                # Inline data (standard for binary responses)
                if hasattr(part, 'inline_data') and part.inline_data:
                    import io
                    return Image.open(io.BytesIO(part.inline_data.data))
        
        # Fallback for other structures
        if hasattr(response, 'images') and response.images:
            return response.images[0]
            
        # Only check text if no binary parts were found/processed to avoid errors
        try:
            if hasattr(response, 'text') and response.text:
                st.warning(f"Model returned text instead of image: {response.text}")
        except Exception:
            # Ignore errors when accessing .text on binary responses
            pass
            
        return None
    except Exception as e:
        st.error(f"Error in image generation ({model_name}): {str(e)}")
        st.info(f"Note: Ensure your API key has access to the '{model_name}' model.")
        return None

def main():
    st.title("🍌 Nano Banana Pro Image Translator")
    st.markdown("### Select an image, recognize text, and reproduce it in a target language.")

    # Load API Key from Streamlit secrets (for cloud) or environment (for local)
    api_key = None
    
    # Try Streamlit secrets first (for Streamlit Cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'GOOGLE_API_KEY' in st.secrets:
            api_key = st.secrets['GOOGLE_API_KEY']
    except Exception:
        pass
    
    # Fall back to environment variable (for local development)
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("GOOGLE_API_KEY not found. Please configure it in Streamlit secrets (for cloud) or .env file (for local).")
        st.info("""
        **For Streamlit Cloud:**
        1. Go to your app settings
        2. Click on 'Secrets' in the left sidebar
        3. Add: `GOOGLE_API_KEY = "your-api-key-here"`
        
        **For Local Development:**
        Create a `.env` file with: `GOOGLE_API_KEY=your-api-key-here`
        """)
        return

    init_gemini(api_key)

    # Main content
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Layout: 2 Columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input & Settings")
                
                # Language Selection (Default: Italian)
                languages = ["Spanish", "French", "German", "Italian", "Japanese", "Korean", "Chinese", "Hindi"]
                default_index = languages.index("Italian") if "Italian" in languages else 0
                target_language = st.selectbox("Target Language", languages, index=default_index)
                
                # Show Original Image
                st.image(image, caption="Original Image", width="stretch")
                
                process_btn = st.button("Process Image")

            # Placeholders for results
            recognized_text_placeholder = col1.empty()
            generated_image_placeholder = col2.empty()

            if process_btn:
                with st.spinner("Recognizing text with Gemini 2.5 Pro..."):
                    extracted_text = get_image_text(image)
                
                if extracted_text:
                    # Show recognized text in Left Panel (Col 1)
                    with col1:
                        st.success("Text Recognized!")
                        st.text_area("Recognized Text", extracted_text, height=150)
                    
                    with st.spinner(f"Generating new image in {target_language} with Nano Banana Pro..."):
                        generated_image = generate_translated_image(image, extracted_text, target_language)
                    
                    # Show generated image in Right Panel (Col 2)
                    with col2:
                        st.subheader(f"Generated Image ({target_language})")
                        if generated_image:
                            st.image(generated_image, caption=f"Translated to {target_language}", width="stretch")
                        else:
                            st.error("Failed to generate image. Please check model availability.")
                            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
