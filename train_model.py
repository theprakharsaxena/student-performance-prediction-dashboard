import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
df = pd.read_csv('/Users/prakharsaxena/Downloads/untitled folder 23/student_performance.csv')
print("Dataset shape:", df.shape)

# Feature and Target Split
X = df.drop(columns=['Final_Exam_Score'])
y = df['Final_Exam_Score']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, '/Users/prakharsaxena/Downloads/untitled folder 23/scaler.pkl')
print("Scaler saved as scaler.pkl")

# Build ANN Architecture
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

print(model.summary())

# Train Model
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    verbose=1
)

# Save Model
model.save('/Users/prakharsaxena/Downloads/untitled folder 23/student_performance_model.keras')
print("Model saved as student_performance_model.keras")

# Evaluate Model
predictions = model.predict(X_test_scaled).flatten()
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"--- Model Evaluation ---")
print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R2 Score: {r2:.4f}")
