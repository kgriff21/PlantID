# 🪴 Plant Genera Classification using Deep Learning

## Overview
This project focuses on classifying five different genera of plants using deep learning. The model currently identifies species from the following genera:
- **Alocasia**
- **Monstera**
- **Pothos**
- **Anthurium**
- **Calathea**

The model can be confirmed (albeit locally) on the Streamlit app with user image input for correct identification.

Initially, the project was intended to classify specific species within the *Monstera* genus, but this was deferred due to complexity. The first part of the script, which includes scraping images for *Monstera* species using the Google API, has been commented out for future extension.

### Future Extension
- Implement a more detailed classification system to differentiate between specific *Monstera* species.
- Enhance the dataset with additional labeled images.
- Improve the model’s performance using advanced techniques such as fine-tuning pre-trained models or using transformers for image recognition.

## Data Source
The images for the five genera are sourced from a Kaggle repository: [Plant Species Classification - ResNet50 EDA](https://www.kaggle.com/code/macaronimutton/plant-species-classification-resnet50-eda/input). Unlike the original plan to use Google’s API for *Monstera* species, all current training images come from this dataset.

## Project Structure
```
.
├── models
│   └── saved_model_freeze.keras
├── test_images
│   ├── alocasia_polly.jpeg
│   ├── anthurium_ihor.jpg
│   ├── calathea.jpg
│   ├── mon_adan.jpeg
│   ├── mon_var.jpeg
│   ├── monstera_minimia_petra.JPG
│   └── pothos_.jpeg
├── README.md
├── main.ipynb
└── streamlit.py

```

## Steps Involved
### 1. Image Preprocessing
- Images are loaded, resized to 256x256, and converted to RGB format if necessary.
- Duplicate and corrupted images are removed.
- Images are augmented using **ImageDataGenerator** to enhance model generalization.

### 2. Train-Test Split
- The dataset is split into training (80%) and validation (20%) while ensuring a balanced distribution of plant genera.

### 3. Model Architecture
- **Base Model:** EfficientNetV2L (pre-trained on ImageNet, used for feature extraction).
- **New Layers:** A fully connected classifier head is added on top to distinguish between the five plant genera.
- **Fine-Tuning:** Initially, only the classifier head is trained. Later, the last 100 layers of the base model are unfrozen for fine-tuning.

### 4. Training
- Model is trained using **categorical crossentropy** and **Adam optimizer**.
- Learning rate reduction and early stopping are applied to prevent overfitting.
- Training history, accuracy, and loss curves are visualized.

### 5. Model Evaluation
- The model is evaluated using:
  - **Validation Accuracy & Loss Metrics**
  - **Confusion Matrix**
  - **Classification Report**

## Usage
### Running the Training Script
To train the model, run:
```bash
python train_model.py
```

### Inference
To classify new plant images, use:
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("models/saved_model.keras")

def classify_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize
    prediction = model.predict(image)
    class_label = np.argmax(prediction)
    return class_label

print(classify_image("path_to_image.jpg"))
```

## Dependencies
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- Pandas
- Matplotlib
- Seaborn
- PIL (Pillow)

Install required dependencies using:
```bash
pip install tensorflow numpy pandas matplotlib seaborn opencv-python pillow
```

## Next Steps
- Expand to include fine-grained classification of *Monstera* species.
- Train on a larger dataset to improve generalization.
- Deploy the model via an API for real-time classification.

## Acknowledgments
Special thanks to the Kaggle community for the dataset and TensorFlow for pre-trained models.

---
For any issues or contributions, feel free to open a pull request or contact the author.

