import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1200

study_hours = np.round(np.random.uniform(5.0, 45.0, n_samples), 1)
attendance = np.round(np.random.uniform(50.0, 100.0, n_samples), 1)
prev_score = np.round(np.random.uniform(40.0, 98.0, n_samples), 1)
sleep_hours = np.round(np.random.uniform(4.5, 9.5, n_samples), 1)
practice_tests = np.random.randint(0, 16, n_samples)
tutoring_sessions = np.random.randint(0, 11, n_samples)

# Synthetic target score calculation
raw_score = (
    0.35 * prev_score +
    0.70 * study_hours +
    0.20 * attendance +
    0.80 * sleep_hours +
    1.10 * practice_tests +
    1.30 * tutoring_sessions +
    np.random.normal(0, 3.5, n_samples)
)

final_score = np.clip(np.round(raw_score, 1), 0.0, 100.0)

df = pd.DataFrame({
    'Study_Hours_Per_Week': study_hours,
    'Attendance_Percentage': attendance,
    'Previous_Exam_Score': prev_score,
    'Sleep_Hours_Per_Night': sleep_hours,
    'Practice_Tests_Taken': practice_tests,
    'Tutoring_Sessions': tutoring_sessions,
    'Final_Exam_Score': final_score
})

df.to_csv('/Users/prakharsaxena/Downloads/untitled folder 23/student_performance.csv', index=False)
print("student_performance.csv created successfully with shape:", df.shape)
