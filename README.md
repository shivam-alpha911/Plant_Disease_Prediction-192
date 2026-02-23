# 🌿 Plant Disease Prediction Model

This project uses a Convolutional Neural Network (CNN) to detect and classify diseases in plant leaf images. The model is trained on a large dataset of plant leaf images and can predict the disease type from an uploaded image. It helps farmers and researchers quickly identify plant diseases using AI.

## 📁 Dataset

The dataset for training and evaluation is hosted on Google Drive.  
👉 **Download / access the dataset here:**  
https://drive.google.com/drive/folders/1wGNFhdjxl7J1rljxicbArto9Z9AJT8WA?usp=sharing

> Make sure the folder is set to *“Anyone with the link can view”* so that users can download the data without restrictions.:contentReference[oaicite:0]{index=0}

## 🧠 Model Details

- Built using Python and TensorFlow / Keras  
- Uses CNN for feature extraction and classification  
- Preprocessing includes resizing, normalization & augmentation  
- Achieves high accuracy on validation images

## 🚀 How to Run

1. Clone the repository:  
   `git clone https://github.com/<your-username>/<your-repo>.git`
2. Install dependencies:  
   `pip install -r requirements.txt`
3. Place downloaded dataset in the `data/` folder  
4. Train the model:  
   `python train.py`  
5. Run prediction:  
   `python predict.py --image <path_to_leaf_image>`

## 📊 Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | ~95%+ |
| Validation Accuracy | ~90%+ |

## 🧩 Deployment

You can deploy this as a web or mobile app using Flask, Streamlit, or FastAPI for users to upload images and get predictions.

---

Feel free to customize the text further to match your project and add images, results, and model graphs!:contentReference[oaicite:1]{index=1}
::contentReference[oaicite:2]{index=2}
