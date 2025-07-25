# 🖼️ Image Captioning and Segmentation System

A complete web-based system for generating **image captions** and performing **semantic segmentation** using **pre-trained models** from PyTorch and Hugging Face Transformers. Built using **Streamlit**, it requires **no custom model training**.

---

## 🌐 Live Demo

👉 [Try it Live on Streamlit](https://image-caption-bhaumik.streamlit.app/)

---

## 🚀 Features

- ✨ **Image Captioning**:
  - Uses **BLIP (Bootstrapped Language Image Pretraining)** model from Salesforce.
  - Generates descriptive captions from images.
  - Supports **conditional captioning** with user-defined prompts.

- 🎯 **Image Segmentation**:
  - Uses **DeepLabV3 + ResNet101** pretrained on COCO.
  - Provides colored overlays of segmented regions.
  - Identifies and counts multiple object categories.

- 📷 **Input Options**:
  - Upload image
  - Use sample images
  - Capture via camera (OpenCV or Streamlit webcam)

- 📊 **Evaluation Metrics**:
  - BLEU score for caption quality
  - IoU (Intersection over Union) for segmentation

---

## 🛠 Requirements

Install all dependencies using:

```bash
pip install torch torchvision transformers opencv-python pillow numpy matplotlib streamlit sentence-transformers
```
## 📁 Directory Structure

```bash
image-captioning-and-segmentation/
│
├── Main.py                    # Main app file for Streamlit
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── packages.txt               # System-level dependencies for Streamlit Cloud
└── (Other helper files/images if needed)
```

## 💻 Local Usage

Run the Streamlit app locally with:

```bash
streamlit run Main.py
```
## 📦 Deployment on Streamlit Cloud

If deploying to Streamlit Cloud, ensure the following files are included:

### `requirements.txt`
```txt
torch
torchvision
transformers
opencv-python
numpy
Pillow
matplotlib
streamlit
requests
sentence-transformers
```

### `packages.txt`
```txt
libgl1-mesa-glx
```

## 🧠 Models Used

- 🤖 `Salesforce/blip-image-captioning-base` (HuggingFace Transformers)  
- 🧠 `deeplabv3_resnet101` (Torchvision)

## 📷 Sample Image Sources

All sample images are fetched from [Unsplash](https://unsplash.com).

## 📜 License

MIT License. Feel free to use, share, and modify this project.

## 👨‍💻 Author

Built with ❤️ by **Bhaumik Senwal**

