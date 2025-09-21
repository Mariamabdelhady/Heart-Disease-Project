#  Heart Disease Prediction Project

This project aims to **predict heart disease** using medical data from the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).  
We implemented a full pipeline including **data preprocessing**, **feature selection**, **model training**, and finally **an interactive UI** built with Streamlit.

---

##  Project Structure

```
Heart_Disease_Project/
├── data/                 # Raw data + cleaned data
│   ├── heart_disease.csv
│   └── heart_disease_clean.csv
├── models/               # Final trained model
│   └── final_model.pkl
├── notebooks/            # Jupyter Notebooks organized step-by-step
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_feature_selection.ipynb
│   ├── 03_supervised_learning.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   └── 05_results_analysis.ipynb
├── results/              # Model results & evaluation files
│   ├── evaluation_metrics.txt
│   ├── predictions_test.csv
│   ├── roc_curve.png
│   └── confusion_matrix.png
├── ui/                   # Streamlit web app
│   └── app.py
└── requirements.txt      # Project dependencies
```

---

##  How to Run Locally

1. Make sure you have **Python 3.10+** installed.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app:
```bash
streamlit run ui/app.py
```
4. Open the URL shown in the terminal:
```
http://localhost:8501
```

---

##  Final Results

###  Best Model Parameters
- **Model:** Logistic Regression  
- **Parameters:** `{'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}`

###  Test Set Performance:
- **Accuracy:** 0.8689  
- **Precision:** 0.8125  
- **Recall:** 0.9286  
- **F1-score:** 0.8667  
- **ROC-AUC:** 0.9307  

###  Visualizations
| ROC Curve | Confusion Matrix |
|----------|----------------|
| ![ROC Curve](results/roc_curve (1).png) | ![Confusion Matrix](results/confusion_matrix (1).png) |

---

##  Web App
The Streamlit app allows users to input the following features:
- **thalach** (maximum heart rate)
- **thal** (thalassemia status)
- **cp** (chest pain type)
- **ca** (number of major vessels)
- **oldpeak** (ST depression)

Output:
-  **No heart disease detected** or  
-  **Heart disease detected** with probability score.

---

##  Requirements
- Python >= 3.10  
- Libraries:  
  `pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit`

---

##  Deployment
You can deploy the project online using [Streamlit Community Cloud](https://streamlit.io/cloud) or test it quickly using `ngrok` on Google Colab.

---

##  Contributions
- Data cleaning and missing value imputation using SimpleImputer.
- Feature selection using RFE and Chi2.
- Logistic Regression training with GridSearchCV.
- Interactive UI with Streamlit.
- Model evaluation with ROC Curve & Confusion Matrix.
---
## Author

Developed by Mariam abdelhady
-  Email: [ mariamabdelhady20125@gmail.com](mailto:mariamabdelhady20125@gmail.com)  
-  LinkedIn: [https://www.linkedin.com/in/mariam-abd-elhady-236a2236b](https://www.linkedin.com/in/https://www.linkedin.com/in/mariam-abd-elhady-236a2236b)

---

##  Data Source
Dataset from [UCI Machine Learning Repository – Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease).
---
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heart-disease-project-mddyipbub83thaj8ed4thh.streamlit.app/)
