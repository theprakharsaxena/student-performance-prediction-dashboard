# Student Academic Performance Prediction using Artificial Neural Networks (ANN)

## 📌 Project Overview
This project applies an **Artificial Neural Network (ANN Regression)** model to predict a student's final exam score based on study habits, attendance percentage, previous test history, sleep hours, practice tests, and tutoring sessions.

Included is a comprehensive Jupyter Notebook with deep learning theory, data visualization, training routines, and an interactive Streamlit UI web application.

---

## 📊 Features & Target Dataset

| Feature Name | Type | Description / Range |
|--------------|------|---------------------|
| `Study_Hours_Per_Week` | Float | Hours spent studying per week (1.0 - 50.0) |
| `Attendance_Percentage` | Float | Class attendance rate (40.0% - 100.0%) |
| `Previous_Exam_Score` | Float | Score in previous semester exam (30.0 - 100.0) |
| `Sleep_Hours_Per_Night` | Float | Average sleep duration (4.0 - 10.0 hours) |
| `Practice_Tests_Taken` | Integer | Number of mock practice exams taken (0 - 15) |
| `Tutoring_Sessions` | Integer | Number of tutoring sessions attended (0 - 10) |
| **`Final_Exam_Score`** | **Float** | **Target Output: Final Exam Mark (0.0 - 100.0)** |

---

## 🏗️ Neural Network Architecture

The project uses a Sequential Artificial Neural Network implemented using TensorFlow/Keras:

```text
Input Layer (6 features) ──> Dense(32, ReLU) ──> Dense(16, ReLU) ──> Dense(8, ReLU) ──> Dense(1, Linear Output)
```

- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Evaluation Metrics**: Mean Absolute Error (MAE), $R^2$ Score

---

## 🛠️ Project File Structure

- `app.py`: Interactive Streamlit Dashboard application.
- `ANN_Student_Performance.ipynb`: Complete Jupyter Notebook containing theory & training code.
- `student_performance.csv`: Clean tabular dataset.
- `student_performance_model.keras`: Trained Keras ANN Model.
- `scaler.pkl`: Fitted StandardScaler object for feature normalization.
- `requirements.txt`: Python package dependencies.
- `README.md`: Project documentation.

---

## 🚀 How to Run locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Model / Generate Notebook**:
   ```bash
   python train_model.py
   ```

3. **Run Streamlit Web Application**:
   ```bash
   streamlit run app.py
   ```
