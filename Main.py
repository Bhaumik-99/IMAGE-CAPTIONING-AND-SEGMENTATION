# Image Captioning and Segmentation System
# Using Pre-trained Models - No Custom Training Required
# Requirements: pip install torch torchvision transformers opencv-python pillow numpy matplotlib streamlit sentence-transformers

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModel
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import requests
from io import BytesIO
import base64

# ================== PRE-TRAINED MODELS SETUP ==================

class ImageCaptioningSystem:
    """Image Captioning using pre-trained BLIP model"""
    
    def __init__(self):
        print("Loading BLIP model for image captioning...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def generate_caption(self, image):
        """Generate caption for input image"""
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
        """Generate caption with text prompt"""
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
        print("Loading DeepLab model for image segmentation...")
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # COCO classes for segmentation
        self.classes = [
            'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow'
        ]
        
        # Color map for visualization
        self.colors = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)
        
        # Preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def segment_image(self, image):
        """Perform semantic segmentation on input image"""
        original_size = None
        
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            original_size = image.size
        elif isinstance(image, np.ndarray):
            original_size = (image.shape[1], image.shape[0])
            image = Image.fromarray(image).convert('RGB')
        else:
            original_size = image.size
            
        # Preprocess image
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
            
        # Get predictions
        predictions = torch.argmax(output, dim=0).cpu().numpy()
        
        # Resize to original image size
        predictions_resized = cv2.resize(
            predictions.astype(np.uint8), 
            original_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        return predictions_resized
    
    def create_segmentation_overlay(self, image, segmentation_mask, alpha=0.6):
        """Create colored overlay of segmentation mask on original image"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            
        # Create colored mask
        colored_mask = np.zeros_like(image)
        unique_classes = np.unique(segmentation_mask)
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
                continue
            mask = segmentation_mask == class_id
            colored_mask[mask] = self.colors[class_id % len(self.colors)]
        
        # Blend with original image
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        return overlay, unique_classes
    
    def get_detected_objects(self, segmentation_mask):
        """Get list of detected objects from segmentation mask"""
        unique_classes = np.unique(segmentation_mask)
        detected_objects = []
        
        for class_id in unique_classes:
            if class_id == 0:  # Skip background
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
    """Load sample images for demonstration"""
    sample_urls = [
        "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e?w=400",  # Dog
        "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400",  # Food
        "https://images.unsplash.com/photo-1568605114967-8130f3a36994?w=400",  # House
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # Mountains
    ]
    
    images = []
    for url in sample_urls:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            images.append(image)
        except:
            continue
    
    return images

def process_uploaded_image(uploaded_file):
    """Process uploaded image file"""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return np.array(image)
    return None

# ================== EVALUATION METRICS ==================

class ModelEvaluator:
    """Evaluation metrics for captioning and segmentation"""
    
    @staticmethod
    def calculate_bleu_score(reference, hypothesis):
        """Simple BLEU score calculation"""
        from collections import Counter
        
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Calculate precision
        common_words = Counter(ref_words) & Counter(hyp_words)
        precision = sum(common_words.values()) / len(hyp_words) if hyp_words else 0
        
        return precision
    
    @staticmethod
    def calculate_iou(mask1, mask2):
        """Calculate Intersection over Union for segmentation masks"""
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
        return iou

# ================== STREAMLIT WEB APPLICATION ==================

def main():
    st.set_page_config(
        page_title="Image Captioning & Segmentation System",
        page_icon="🖼️",
        layout="wide"
    )
    
    st.title("🖼️ Image Captioning & Segmentation System")
    st.markdown("---")
    
    # Initialize models
    if 'captioning_model' not in st.session_state:
        with st.spinner("Loading models... This may take a few minutes."):
            st.session_state.captioning_model = ImageCaptioningSystem()
            st.session_state.segmentation_model = ImageSegmentationSystem()
    
    # Sidebar for options
    st.sidebar.header("Options")
    task = st.sidebar.selectbox(
        "Select Task",
        ["Both Tasks", "Image Captioning Only", "Image Segmentation Only"]
    )
    
    # Image input options
    st.sidebar.header("Image Input")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Upload Image", "Use Sample Images", "Camera Input"]
    )
    
    image = None
    
    if input_method == "Upload Image":
        uploaded_file = st.sidebar.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        if uploaded_file:
            image = process_uploaded_image(uploaded_file)
    
    elif input_method == "Use Sample Images":
        sample_images = load_sample_images()
        if sample_images:
            selected_idx = st.sidebar.selectbox(
                "Select sample image:",
                range(len(sample_images)),
                format_func=lambda x: f"Sample Image {x+1}"
            )
            image = np.array(sample_images[selected_idx])
    
    elif input_method == "Camera Input":
        camera_image = st.sidebar.camera_input("Take a picture")
        if camera_image:
            image = process_uploaded_image(camera_image)
    
    # Main content area
    if image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Process image based on selected task
        if task in ["Both Tasks", "Image Captioning Only"]:
            st.subheader("📝 Image Captioning Results")
            
            with st.spinner("Generating caption..."):
                caption = st.session_state.captioning_model.generate_caption(image)
            
            st.success(f"**Generated Caption:** {caption}")
            
            # Conditional captioning
            if st.checkbox("Use conditional captioning"):
                prompt = st.text_input("Enter prompt:", value="a picture of")
                if prompt:
                    conditional_caption = st.session_state.captioning_model.generate_conditional_caption(
                        image, prompt
                    )
                    st.info(f"**Conditional Caption:** {conditional_caption}")
        
        if task in ["Both Tasks", "Image Segmentation Only"]:
            st.subheader("🎯 Image Segmentation Results")
            
            with st.spinner("Performing segmentation..."):
                segmentation_mask = st.session_state.segmentation_model.segment_image(image)
                overlay, unique_classes = st.session_state.segmentation_model.create_segmentation_overlay(
                    image, segmentation_mask
                )
                detected_objects = st.session_state.segmentation_model.get_detected_objects(segmentation_mask)
            
            with col2:
                st.subheader("Segmentation Overlay")
                st.image(overlay, use_container_width=True)
            
            # Display detected objects
            st.subheader("Detected Objects")
            if detected_objects:
                for obj in detected_objects:
                    st.write(f"- **{obj['class']}** (pixels: {obj['pixel_count']:,})")
            else:
                st.write("No objects detected")
            
            # Show segmentation mask
            if st.checkbox("Show raw segmentation mask"):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(segmentation_mask, cmap='tab20')
                ax.set_title("Segmentation Mask")
                ax.axis('off')
                st.pyplot(fig)
    
    else:
        st.info("👆 Please select an image using the sidebar options to get started!")
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About This System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Image Captioning:**
        - Uses pre-trained BLIP model from Salesforce
        - Generates descriptive captions for images
        - Supports conditional captioning with prompts
        - No custom training required
        """)
    
    with col2:
        st.markdown("""
        **Image Segmentation:**
        - Uses pre-trained DeepLabV3 with ResNet-101 backbone
        - Performs semantic segmentation on 21 object classes
        - Creates colored overlay visualization
        - Provides object detection statistics
        """)
    
    st.markdown("""
    **Technologies Used:**
    - PyTorch & Torchvision for deep learning
    - Transformers library for pre-trained models
    - OpenCV for image processing
    - Streamlit for web interface
    - PIL for image handling
    """)

# ================== JUPYTER NOTEBOOK FUNCTIONS ==================

def run_captioning_demo(image_path):
    """Demo function for Jupyter notebook - Image Captioning"""
    captioning_system = ImageCaptioningSystem()
    
    # Load and display image
    image = Image.open(image_path)
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Generate caption
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
    """Demo function for Jupyter notebook - Image Segmentation"""
    segmentation_system = ImageSegmentationSystem()
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform segmentation
    segmentation_mask = segmentation_system.segment_image(image)
    overlay, unique_classes = segmentation_system.create_segmentation_overlay(image, segmentation_mask)
    detected_objects = segmentation_system.get_detected_objects(segmentation_mask)
    
    # Display results
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
    """Demo function for Jupyter notebook - Combined tasks"""
    print("Running Image Captioning and Segmentation Demo")
    print("=" * 50)
    
    # Run captioning
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
            # Run Streamlit app
            main()
        elif sys.argv[1] == "demo" and len(sys.argv) > 2:
            # Run demo with image path
            image_path = sys.argv[2]
            run_combined_demo(image_path)
        else:
            print("Usage:")
            print("python script.py streamlit  # Run web interface")
            print("python script.py demo <image_path>  # Run demo")
    else:
        # Default: run Streamlit app
        main()