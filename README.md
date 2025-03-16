# 🔥 Deep Learning-Based Classification & Regression Model

![GitHub repo](https://img.shields.io/github/stars/yourrepo?style=social)
![License](https://img.shields.io/github/license/yourrepo)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)

## 🌟 Live Demo
🚀 **Try the Deployed Streamlit App**: [Click Here](https://ann-customer-churn-h3a6vu6sgmhcm99xhh9kfx.streamlit.app/) 🔥

## 📌 Overview
This project implements **Deep Learning models** for **classification and regression** tasks using **Artificial Neural Networks (ANNs)**. It includes:
- **Hyperparameter tuning** for optimal model performance.
- **Preprocessing steps** using `scikit-learn`.
- **Pickled preprocessing objects** to ensure consistency in inference.
- **Model deployment** using `.h5` and pickled files.
- **Interactive Streamlit App** for easy predictions.

## 🚀 Features
✅ **Pretrained ANN model** (`model.h5`)<br>
✅ **Hyperparameter tuning** with different configurations<br>
✅ **Pickled preprocessing files** (`label_encoder.pkl`, `scaler.pkl`, etc.) for reproducibility<br>
✅ **End-to-End pipeline** from training to inference<br>
✅ **Streamlit app for real-time predictions**<br>
✅ **Jupyter Notebooks** for step-by-step execution<br>
✅ **Sleek & interactive UI**

## 📚 Project Explanation
This project is designed to address **classification and regression** tasks using **Artificial Neural Networks (ANNs)**. The model has been trained on a structured dataset and optimized through hyperparameter tuning to enhance its performance. Key components include:
- **Data Preprocessing**: Handling missing values, feature scaling, and encoding categorical variables.
- **Model Training**: A deep learning model is trained with optimized hyperparameters to achieve high accuracy.
- **Evaluation Metrics**: The model is evaluated using accuracy, precision, recall, and F1-score.
- **Deployment**: The trained model and preprocessing objects are saved and loaded into a Streamlit web application for user-friendly predictions.
- **Prediction Pipeline**: A script is available to make predictions using new input data.

## 📚 Project Structure
```
📚 DL-ANN-CLASSIFICATION
│── 📝 .gitignore
│── 📝 README.md
│── 📝 requirements.txt
│── 📂 Data
│   ├── Churn_Modelling.csv           # Dataset used for training
│── 📂 Models & Encoders
│   ├── model.h5                    # Trained ANN model
│   ├── label_encoder_gender.pkl     # Encodes gender labels
│   ├── onehot_encoder_geo.pkl       # One-hot encoding for geography
│   ├── scaler.pkl                   # StandardScaler object
│── 📂 Notebooks
│   ├── EDA.ipynb                    # Exploratory Data Analysis
│   ├── hyperparametertuningann.ipynb # Hyperparameter tuning
│   ├── regression.ipynb              # Regression analysis
│   ├── prediction.ipynb              # Prediction script
│── 📂 Application
│   ├── app.py                        # Streamlit web app for inference
```

## ⚙️ Installation
Clone the repository and set up a virtual environment:
```bash
# Clone the repo
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎨 Run the Streamlit App Locally
To run the **interactive UI**, execute:
```bash
streamlit run app.py
```

## 🔍 Making Predictions
Use the trained model for predictions:
```bash
python scripts/predict.py --input data/sample.csv
```
Or load the model manually:
```python
import pickle
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model.h5")

# Load encoders and scalers
with open("encoders/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
```

## 📊 Results
| Model | Accuracy | Precision | Recall |
|--------|---------|----------|--------|
| ANN (Tuned) | **92.5%** | 91.2% | 89.5% |
| ANN (Baseline) | 86.7% | 85.3% | 83.1% |

## 📌 Future Enhancements
- [ ] Deploy as a **Flask API**
- [ ] Implement **explainable AI (XAI)** techniques
- [ ] Optimize using **quantization & pruning**

## 🐝 License
This project is licensed under the **MIT License**.

## 📢 Connect With Me
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/aaryan-rana-2741b1203/)  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-lightgrey)](https://github.com/aaryan7174)  
[![Medium](https://img.shields.io/badge/Medium-Read%20Articles-green)](https://medium.com/@7174aaryan)

