# 🩺 Diabetes Risk Assessment Web App

This is an interactive Streamlit web application that predicts the risk of diabetes using clinical features. The underlying machine learning model is a Random Forest classifier trained on the well-known Pima Indians Diabetes dataset. The app includes intuitive input sliders, real-time predictions, explainability, and visual data analysis.

---

## 🔍 Features

* **User-friendly interface** for entering patient data
* **Risk prediction** based on a trained Random Forest classifier
* **Interactive visualizations** (Glucose, Age, BMI analysis)
* **Feature engineering** for Insulin & Glucose status
* **Robust data preprocessing** with missing value handling and robust scaling
* **Clinical recommendations** based on prediction results
* **Streamlit deployment ready**

---

## 📊 Input Features

| Feature                    | Description                                    |
| -------------------------- | ---------------------------------------------- |
| Pregnancies                | Number of pregnancies                          |
| Glucose                    | Plasma glucose concentration (mg/dL)           |
| Blood Pressure             | Diastolic blood pressure (mm Hg)               |
| Skin Thickness             | Triceps skin fold thickness (mm)               |
| Insulin                    | 2-Hour serum insulin (μU/mL)                   |
| BMI                        | Body mass index (weight in kg / height²)       |
| Diabetes Pedigree Function | Likelihood of diabetes based on family history |
| Age                        | Patient age in years                           |

---

## 🚀 How to Run

### 🔧 Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---


## 📈 Model Details

* **Model**: Random Forest Classifier
* **Hyperparameters**:

  * n\_estimators = 200
  * max\_depth = 12
  * min\_samples\_split = 5
  * class\_weight = "balanced"
* **Scaling**: RobustScaler for outlier-resistant normalization
* **Feature Engineering**: Categorical bins for Glucose & Insulin

---

## 📊 Visualizations Included

* **Glucose vs Diabetes Outcome** (Box Plot)
* **Age Distribution by Outcome** (Histogram)
* **BMI vs Glucose Scatter** (Color-coded by outcome)

---

## 🧠 Dataset Used

* **Name**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Source**: UCI Machine Learning Repository
* **Shape**: 768 rows × 9 columns

---

## ⚠️ Disclaimer

This application is intended **for educational and informational purposes only**. It does **not constitute medical advice**. Please consult a healthcare professional for personalized medical guidance.

---

## 👨‍💻 Author

Developed by **[Himel Sarder](https://www.linkedin.com/in/himel-sarder/)**

---

## 📢 License

This project is open-source and free to use under the [MIT License](LICENSE).
