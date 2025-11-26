import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure page settings
st.set_page_config(
    page_title="Image to Image Translator - Multipurpose",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS - using Streamlit's built-in theme system
st.markdown("""
    <style>
    /* Description box - minimal styling, text uses Streamlit theme */
    .description-box {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem auto 1.5rem auto;
        max-width: 800px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .description-box p {
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Button styling - keep visual improvements */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        border: none;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader - minimal styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(102, 126, 234, 0.5) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        background: rgba(102, 126, 234, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(102, 126, 234, 0.8) !important;
        background: rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Image styling */
    .stImage img {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        object-fit: contain;
    }
    
    /* Ensure images align properly */
    [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    /* Fix column alignment */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
    # Header section with improved styling
    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    st.title("🖼️ Image to Image Translator - Multipurpose")
    
    # App description
    st.markdown("""
    <div class="description-box">
        <p>
            Transform images with multilingual text translation capabilities. Upload any image containing text, 
            and our AI-powered system will recognize the text and generate a new image with the text translated 
            into your chosen language while preserving the original design, layout, and visual style.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

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
        st.error("⚠️ GOOGLE_API_KEY not found. Please configure it in Streamlit secrets (for cloud) or .env file (for local).")
        with st.expander("📋 Setup Instructions", expanded=True):
            st.markdown("""
            **For Streamlit Cloud:**
            1. Go to your app settings
            2. Click on 'Secrets' in the left sidebar
            3. Add: `GOOGLE_API_KEY = "your-api-key-here"`
            
            **For Local Development:**
            Create a `.env` file in your project root with: `GOOGLE_API_KEY=your-api-key-here`
            """)
        return

    init_gemini(api_key)

    # Main content area
    st.markdown("---")
    
    # File uploader with better styling
    uploaded_file = st.file_uploader(
        "📤 Choose an image to translate", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image containing text that you want to translate"
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Initialize session state
            if 'recognized_text' not in st.session_state:
                st.session_state.recognized_text = None
            if 'generated_images' not in st.session_state:
                st.session_state.generated_images = {}
            if 'original_image' not in st.session_state:
                st.session_state.original_image = None
            if 'current_file_id' not in st.session_state:
                st.session_state.current_file_id = None
            if 'view_language' not in st.session_state:
                st.session_state.view_language = None
            
            # Reset state if a new file is uploaded
            current_file_id = uploaded_file.name + str(uploaded_file.size)
            if st.session_state.current_file_id != current_file_id:
                st.session_state.recognized_text = None
                st.session_state.generated_images = {}
                st.session_state.view_language = None
                st.session_state.current_file_id = current_file_id
            
            # Store original image in session state
            st.session_state.original_image = image
            
            # Available languages
            languages = ["Spanish", "French", "German", "Italian", "Japanese", "Korean", "Chinese", "Hindi"]
            
            # Language selection FIRST
            st.markdown("### 🌍 Select Target Language")
            target_language = st.selectbox(
                "Choose the language for translation:",
                languages,
                index=2 if "German" in languages else 0,
                help="Select the language you want to translate the text to"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Layout: 2 Columns for side-by-side comparison
            col1, col2 = st.columns(2, gap="large")
            
            with col1:
                st.markdown("### 📷 Original Image")
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col2:
                st.markdown("### ✨ Generated Image")
                
                # Show language selector if we have generated images
                if st.session_state.generated_images:
                    available_languages = list(st.session_state.generated_images.keys())
                    
                    # Set default view language if not set or if current selection is not available
                    if not st.session_state.view_language or st.session_state.view_language not in available_languages:
                        st.session_state.view_language = available_languages[0]
                    
                    # Language selector dropdown
                    selected_view_lang = st.selectbox(
                        "📋 View Translation:",
                        available_languages,
                        index=available_languages.index(st.session_state.view_language) if st.session_state.view_language in available_languages else 0,
                        help="Select a generated language to view",
                        key="view_language_selector"
                    )
                    
                    # Update session state
                    st.session_state.view_language = selected_view_lang
                    
                    # Display the selected generated image
                    generated_img = st.session_state.generated_images[selected_view_lang]
                    st.image(generated_img, caption=f"Translated to {selected_view_lang}", use_container_width=True)
                    st.success(f"✅ Image translated to {selected_view_lang}!")
                    
                    # Show count of generated languages
                    if len(available_languages) > 1:
                        st.caption(f"📚 {len(available_languages)} translation(s) available")
                else:
                    # No generated images yet
                    st.info(f"👈 Click 'Process Image' to generate translation in {target_language}")
            
            # Process button - does text recognition AND image generation
            process_btn = st.button("🚀 Process Image", use_container_width=True, type="primary")
            
            # Show recognized text if available (collapsible)
            if st.session_state.recognized_text:
                with st.expander("📝 View Extracted Text", expanded=False):
                    st.text_area(
                        "Recognized Text", 
                        st.session_state.recognized_text, 
                        height=150,
                        label_visibility="collapsed",
                        key="display_recognized_text",
                        disabled=True
                    )
            
            # Show all generated languages summary
            if st.session_state.generated_images:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📚 Generated Translations")
                available_languages = list(st.session_state.generated_images.keys())
                cols = st.columns(min(len(available_languages), 4))
                for idx, lang in enumerate(available_languages):
                    with cols[idx % len(cols)]:
                        if lang == st.session_state.view_language:
                            st.markdown(f"**✓ {lang}** (viewing)")
                        else:
                            st.markdown(f"{lang} ✓")
            
            # Process when button is clicked
            if process_btn:
                # Step 1: Recognize text (only if not already done)
                if not st.session_state.recognized_text:
                    with st.spinner("🔍 Recognizing text with Gemini 2.5 Pro..."):
                        extracted_text = get_image_text(image)
                    
                    if extracted_text:
                        st.session_state.recognized_text = extracted_text
                    else:
                        st.error("❌ Failed to recognize text. Please try again.")
                        return
                else:
                    extracted_text = st.session_state.recognized_text
                
                # Step 2: Generate image for selected language (only if not already generated)
                if target_language not in st.session_state.generated_images:
                    with st.spinner(f"🎨 Generating image in {target_language} with Nano Banana Pro..."):
                        generated_image = generate_translated_image(
                            st.session_state.original_image, 
                            extracted_text, 
                            target_language
                        )
                    
                    if generated_image:
                        st.session_state.generated_images[target_language] = generated_image
                        st.session_state.view_language = target_language  # Set view to newly generated language
                        st.success(f"✅ Image successfully translated to {target_language}!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to generate image. Please check model availability.")
                else:
                    st.info(f"✅ Translation for {target_language} already exists! Select it in the dropdown above to view.")
                    # Update view language to the newly selected target language
                    if target_language in st.session_state.generated_images:
                        st.session_state.view_language = target_language
                        st.rerun()
                            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.exception(e)
    else:
        # Show helpful information when no file is uploaded
        st.markdown("""
        <div style='text-align: center; padding: 3rem;'>
            <p style='font-size: 1.1rem; margin: 0;'>👆 Upload an image above to get started</p>
            <p style='font-size: 0.9rem; margin-top: 1rem; margin-bottom: 0;'>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
