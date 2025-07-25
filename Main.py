import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import requests
from io import BytesIO
import base64
import uuid
import pandas as pd
import time
import random

# Custom CSS for improved UI and enhanced progress bar
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stExpander {
        background-color: #ffffff;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .image-container {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .centered-progress {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 200px;
        width: 100%;
    }
    .stProgress > div > div > div {
        background-color: #4CAF50 !important;
        transition: width 0.3s ease-in-out;
        border-radius: 5px;
    }
    .loading-text {
        font-size: 1.2em;
        color: #2c3e50;
        margin-bottom: 10px;
        text-align: center;
    }
    .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #4CAF50;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin-bottom: 10px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ================== PRE-TRAINED MODELS SETUP ==================
class ImageCaptioningSystem:
    """Image Captioning using pre-trained BLIP model"""
    
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def generate_caption(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
            
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
        return caption
    
    def generate_conditional_caption(self, image, text_prompt="a picture of"):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
            
        inputs = self.processor(image, text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=50, num_beams=5)
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
        return caption

class ImageSegmentationSystem:
    """Image Segmentation using pre-trained DeepLab model"""
    
    def __init__(self):
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'
        ]
        
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
        
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def segment_image(self, image):
        original_size = None
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            original_size = image.size
        elif isinstance(image, np.ndarray):
            original_size = (image.shape[1], image.shape[0])
            image = Image.fromarray(image).convert('RGB')
        else:
            original_size = image.size
            
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            
        predictions = torch.argmax(output, dim=0).cpu().numpy()
        
        predictions_resized = cv2.resize(
            predictions.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        return predictions_resized
    
    def create_segmentation_overlay(self, image, segmentation_mask, alpha=0.6):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        colored_mask = np.zeros_like(image)
        unique_classes = np.unique(segmentation_mask)
        
        for class_id in unique_classes:
            if class_id == 0:
                continue
            mask = segmentation_mask == class_id
            colored_mask[mask] = self.colors[class_id % len(self.colors)]
        
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        return overlay, unique_classes
    
    def get_detected_objects(self, segmentation_mask):
        unique_classes = np.unique(segmentation_mask)
        detected_objects = []
        
        for class_id in unique_classes:
            if class_id == 0:
                continue
            if class_id < len(self.classes):
                object_name = self.classes[class_id]
                pixel_count = np.sum(segmentation_mask == class_id)
                detected_objects.append({
                    'class': object_name,
                    'class_id': int(class_id),
                    'pixel_count': int(pixel_count)
                })
        
        return detected_objects

# ================== UTILITY FUNCTIONS ==================
def load_sample_images():
    sample_urls = [
        "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e?w=400",
        "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400",
        "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=400",
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",
    ]
    
    images = []
    for i, url in enumerate(sample_urls):
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            images.append((f"Sample Image {i+1}", image))
        except:
            continue
    
    return images

def process_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            return np.array(image)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    return None

def image_to_base64(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ================== EVALUATION METRICS ==================
class ModelEvaluator:
    @staticmethod
    def calculate_bleu_score(reference, hypothesis):
        from collections import Counter
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        common_words = Counter(ref_words) & Counter(hyp_words)
        precision = sum(common_words.values()) / len(hyp_words) if hyp_words else 0
        return precision
    
    @staticmethod
    def calculate_iou(mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        return iou

# ================== STREAMLIT WEB APPLICATION ==================
def main():
    st.set_page_config(
        page_title="Image Analysis System",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize models with enhanced progress bar
    if 'captioning_model' not in st.session_state:
        st.markdown("<div class='centered-progress'>", unsafe_allow_html=True)
        loading_messages = [
            "Loading BLIP model for captioning...",
            "Initializing DeepLab for segmentation...",
            "Preparing neural networks...",
            "Optimizing for your device..."
        ]
        message_placeholder = st.empty()
        progress_bar = st.progress(0)
        spinner = st.markdown("<div class='spinner'></div>", unsafe_allow_html=True)
        
        try:
            # Simulate loading progress with dynamic messages
            for i in range(100):
                current_message = loading_messages[i // 25]  # Change message every 25%
                message_placeholder.markdown(
                    f"<div class='loading-text'>{current_message} ({i+1}%)</div>",
                    unsafe_allow_html=True
                )
                progress_bar.progress(i + 1)
                time.sleep(0.05)  # Adjust timing for smooth animation
                if i == 50:  # Load captioning model at 50%
                    st.session_state.captioning_model = ImageCaptioningSystem()
            
            # Load segmentation model after captioning
            st.session_state.segmentation_model = ImageSegmentationSystem()
            progress_bar.progress(100)
            message_placeholder.markdown(
                "<div class='loading-text'>Models loaded successfully! (100%)</div>",
                unsafe_allow_html=True
            )
            time.sleep(0.5)  # Brief pause to show completion
            progress_bar.empty()
            message_placeholder.empty()
            spinner.empty()
        except Exception as e:
            progress_bar.empty()
            message_placeholder.empty()
            spinner.empty()
            st.error(f"Failed to load models: {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🖼️ Image Analysis System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>Advanced Image Captioning & Segmentation</p>", unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        with st.expander("Task Selection", expanded=True):
            task = st.selectbox(
                "Analysis Mode",
                ["Both Tasks", "Image Captioning Only", "Image Segmentation Only"],
                help="Choose which analysis tasks to perform"
            )
        
        with st.expander("Image Input", expanded=True):
            input_method = st.radio(
                "Input Method",
                ["Upload Image", "Use Sample Images", "Camera Input"],
                help="Select how to provide the input image"
            )
            
            image = None
            image_name = "Uploaded Image"
            
            if input_method == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Upload an image",
                    type=['jpg', 'jpeg', 'png', 'bmp'],
                    help="Supported formats: JPG, JPEG, PNG, BMP"
                )
                if uploaded_file:
                    image = process_uploaded_image(uploaded_file)
                    image_name = uploaded_file.name
            
            elif input_method == "Use Sample Images":
                sample_images = load_sample_images()
                if sample_images:
                    sample_options = [name for name, _ in sample_images]
                    selected_image = st.selectbox(
                        "Select sample image",
                        options=sample_options,
                        help="Choose a pre-loaded sample image"
                    )
                    image = np.array(next(img for name, img in sample_images if name == selected_image))
                    image_name = selected_image
            
            elif input_method == "Camera Input":
                camera_image = st.camera_input("Take a picture")
                if camera_image:
                    image = process_uploaded_image(camera_image)
                    image_name = "Camera Capture"
    
    # Main content
    if image is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.markdown(f"### Original Image: {image_name}")
            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        results = {}
        
        if task in ["Both Tasks", "Image Captioning Only"]:
            with st.expander("📝 Captioning Results", expanded=True):
                with st.spinner("Generating captions..."):
                    try:
                        caption = st.session_state.captioning_model.generate_caption(image)
                        results['caption'] = caption
                        st.success(f"**Generated Caption:** {caption}")
                        
                        if st.checkbox("Enable conditional captioning", key="conditional"):
                            prompt = st.text_input(
                                "Enter prompt",
                                value="a picture of",
                                help="Enter a text prompt to guide the caption generation"
                            )
                            if prompt:
                                conditional_caption = st.session_state.captioning_model.generate_conditional_caption(
                                    image, prompt
                                )
                                results['conditional_caption'] = conditional_caption
                                st.info(f"**Conditional Caption:** {conditional_caption}")
                    except Exception as e:
                        st.error(f"Error generating captions: {str(e)}")
        
        if task in ["Both Tasks", "Image Segmentation Only"]:
            with st.expander("🎯 Segmentation Results", expanded=True):
                with st.spinner("Performing segmentation..."):
                    try:
                        segmentation_mask = st.session_state.segmentation_model.segment_image(image)
                        overlay, unique_classes = st.session_state.segmentation_model.create_segmentation_overlay(
                            image, segmentation_mask
                        )
                        detected_objects = st.session_state.segmentation_model.get_detected_objects(segmentation_mask)
                        results['detected_objects'] = detected_objects
                        
                        with col2:
                            st.markdown("### Segmentation Overlay")
                            st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                            st.image(overlay, use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        
                        st.markdown("#### Detected Objects")
                        if detected_objects:
                            df = pd.DataFrame(detected_objects)
                            df = df[['class', 'pixel_count']].rename(columns={
                                'class': 'Object',
                                'pixel_count': 'Pixel Count'
                            })
                            st.dataframe(df.style.format({'Pixel Count': '{:,}'}), use_container_width=True)
                        else:
                            st.warning("No objects detected")
                        
                        if st.checkbox("Show raw segmentation mask", key="show_mask"):
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.imshow(segmentation_mask, cmap='tab20')
                            ax.set_title("Segmentation Mask")
                            ax.axis('off')
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error performing segmentation: {str(e)}")
        
        # Download results
        if results:
            st.markdown("---")
            with st.expander("💾 Download Results"):
                results_text = f"Image Analysis Results\n\n"
                if 'caption' in results:
                    results_text += f"Generated Caption: {results['caption']}\n"
                if 'conditional_caption' in results:
                    results_text += f"Conditional Caption: {results['conditional_caption']}\n"
                if 'detected_objects' in results and results['detected_objects']:
                    results_text += "\nDetected Objects:\n"
                    for obj in results['detected_objects']:
                        results_text += f"- {obj['class']}: {obj['pixel_count']:,} pixels\n"
                
                st.download_button(
                    label="Download Results",
                    data=results_text,
                    file_name=f"image_analysis_{uuid.uuid4()}.txt",
                    mime="text/plain"
                )
                
                if 'detected_objects' in results and results['detected_objects']:
                    df = pd.DataFrame(results['detected_objects'])
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Objects CSV",
                        data=csv,
                        file_name=f"detected_objects_{uuid.uuid4()}.csv",
                        mime="text/csv"
                    )
                
                if task in ["Both Tasks", "Image Segmentation Only"]:
                    overlay_base64 = image_to_base64(overlay)
                    st.download_button(
                        label="Download Segmentation Overlay",
                        data=base64.b64decode(overlay_base64),
                        file_name=f"segmentation_overlay_{uuid.uuid4()}.png",
                        mime="image/png"
                    )
    
    else:
        st.info("👆 Please select an image using the sidebar to begin analysis")
    
    # About section
    st.markdown("---")
    with st.expander("ℹ️ About This System", expanded=False):
        st.markdown("""
        ### System Overview
        This application provides advanced image analysis capabilities using state-of-the-art pre-trained models.
        
        #### Image Captioning
        - Powered by Salesforce BLIP model
        - Generates descriptive captions
        - Supports conditional captioning
        - No training required
        
        #### Image Segmentation
        - Uses DeepLabV3 with ResNet-101
        - Detects 21 object classes
        - Provides visualization overlays
        - Includes object statistics
        
        #### Technologies
        - PyTorch & Torchvision
        - Transformers
        - OpenCV
        - Streamlit
        - PIL
        """)
    st.markdown("Made with ❤️ by Bhaumik Senwal")

# ================== JUPYTER NOTEBOOK FUNCTIONS ==================
def run_captioning_demo(image_path):
    captioning_system = ImageCaptioningSystem()
    image = Image.open(image_path)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    caption = captioning_system.generate_caption(image)
    plt.subplot(1, 2, 2)
    plt.text(0.5, 0.5, f"Caption:\n{caption}", 
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12, wrap=True)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return caption

def run_segmentation_demo(image_path):
    segmentation_system = ImageSegmentationSystem()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    segmentation_mask = segmentation_system.segment_image(image)
    overlay, unique_classes = segmentation_system.create_segmentation_overlay(image, segmentation_mask)
    detected_objects = segmentation_system.get_detected_objects(segmentation_mask)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_mask, cmap='tab20')
    plt.title("Segmentation Mask")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("Detected Objects:")
    for obj in detected_objects:
        print(f"- {obj['class']}: {obj['pixel_count']:,} pixels")
    
    return segmentation_mask, detected_objects

def run_combined_demo(image_path):
    print("Running Image Captioning and Segmentation Demo")
    print("=" * 50)
    print("\n📝 CAPTIONING RESULTS:")
    caption = run_captioning_demo(image_path)
    print(f"Generated Caption: {caption}")
    print("\n🎯 SEGMENTATION RESULTS:")
    mask, objects = run_segmentation_demo(image_path)
    return caption, mask, objects

# ================== COMMAND LINE INTERFACE ==================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "streamlit":
            main()
        elif sys.argv[1] == "demo" and len(sys.argv) > 2:
            image_path = sys.argv[2]
            run_combined_demo(image_path)
        else:
            print("Usage:")
            print("python script.py streamlit  # Run web interface")
            print("python script.py demo <image_path>  # Run demo")
    else:
        main()
