# 🌿 Bean Leaf Disease Detector

A Streamlit web app built for a **Computer Vision Lab Assignment**. Upload a bean leaf image, apply different image enhancement techniques, and classify the leaf disease using a deep learning model (MobileNetV2).

## ✨ Features

- 📤 Upload a bean leaf image (JPG, JPEG, PNG, BMP, WEBP)
- 🛠️ 17 image processing options with adjustable parameters (sidebar sliders)
- 🖼️ Side-by-side view of the original and processed image
- 📊 RGB and grayscale histograms (before vs after)
- ⬇️ Download the processed image as PNG
- 🔍 Disease prediction with confidence scores, run on either the original or the processed image

## 🧪 Image Processing Techniques

| Category | Techniques |
|---|---|
| Geometric | Zoom, Resize |
| Smoothing | Average Blur, Gaussian Blur, Median Blur |
| Contrast Enhancement | Histogram Equalization, CLAHE (Adaptive Histogram) |
| Edge Detection | Canny, Sobel, Laplacian |
| Intensity | Grayscale, Brightness / Contrast, Negative, Binary Threshold |
| Detail | Sharpen |
| Morphology | Erosion, Dilation, Opening, Closing |

## 🦠 Disease Classes

The model predicts one of three classes:

1. **Angular Leaf Spot**
2. **Bean Rust**
3. **Healthy**

## 🧰 Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/) – web interface
- [TensorFlow / Keras](https://www.tensorflow.org/) – MobileNetV2 model
- [OpenCV](https://opencv.org/) – image processing
- [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [Pillow](https://python-pillow.org/), [scikit-learn](https://scikit-learn.org/)

## 📁 Project Structure

```
cvlab-mini-app/
├── app.py              # Streamlit application
├── beans_model.keras   # Trained classification model
├── requirements.txt    # Python dependencies
└── README.md
```

## 🚀 Installation & Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

5. Open the link shown in the terminal (usually `http://localhost:8501`).

> ⚠️ Make sure `beans_model.keras` is in the same folder as `app.py`.

## 🧭 How to Use

1. Upload a bean leaf image.
2. Choose a technique from the **Image Processing** sidebar and adjust its parameters.
3. Compare the original and processed images, and check the histograms.
4. Select whether to predict on the **Original** or **Processed** image.
5. Click **Predict** to see the disease class and confidence for each class.

## 🔬 How It Works

- Images are resized to **224 × 224** before being passed to the model.
- Pixel scaling is handled inside the model, so raw 0–255 RGB values are fed in.
- The model outputs probabilities for the three classes, and the highest one is shown as the prediction.

## 📌 Notes

- Heavy processing (e.g., binary threshold, edge detection) may change the image a lot and affect prediction accuracy. Try comparing predictions on the original and processed versions.
- This project is made for educational purposes and is not a replacement for expert agricultural diagnosis.

## 👩‍💻 Author

**<Bushra irshad>**

## 📄 License

This project is licensed under the MIT License. Feel free to use and modify it for learning purposes.
