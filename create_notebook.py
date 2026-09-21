import json

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Artificial Neural Network (ANN) - Student Performance Prediction\n",
    "\n",
    "## What is ANN?\n",
    "An **Artificial Neural Network (ANN)** is a Machine Learning & Deep Learning model inspired by the structure and function of biological neural networks in the human brain. ANNs process complex non-linear data through interconnected nodes (neurons), learning representations by iteratively updating connection weights and biases during training.\n",
    "\n",
    "---\n",
    "\n",
    "## Human Brain vs ANN\n",
    "| Human Brain Component | ANN Equivalent |\n",
    "|-----------------------|----------------|\n",
    "| Biological Neuron | Artificial Neuron (Node) |\n",
    "| Synapse | Weight ($W$) |\n",
    "| Electrical Signal | Input Feature ($X$) |\n",
    "| Brain Output / Action | Network Prediction ($Y$) |\n",
    "\n",
    "---\n",
    "\n",
    "## ANN Architecture\n",
    "```\n",
    "Input Layer (X1, X2, ..., Xn)\n",
    "      │\n",
    "      ▼\n",
    "Hidden Layer 1 (Dense + ReLU)\n",
    "      │\n",
    "      ▼\n",
    "Hidden Layer 2 (Dense + ReLU)\n",
    "      │\n",
    "      ▼\n",
    "Output Layer (1 Unit - Linear Regression Output)\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "## Artificial Neuron Equation\n",
    "Each neuron computes a weighted sum of its inputs plus a bias term:\n",
    "\n",
    "$$\n",
    "Z = \\sum_{i=1}^{n} (W_i \\cdot X_i) + B\n",
    "$$\n",
    "\n",
    "- $X_i$: Input feature values\n",
    "- $W_i$: Connection weights\n",
    "- $B$: Bias\n",
    "- $Z$: Weighted sum\n",
    "\n",
    "---\n",
    "\n",
    "## Activation Functions\n",
    "An activation function introduces non-linearity into the network, enabling it to learn complex patterns:\n",
    "\n",
    "$$\n",
    "A = f(Z)\n",
    "$$\n",
    "\n",
    "1. **Rectified Linear Unit (ReLU)**:\n",
    "   $$\n",
    "   f(z) = \\max(0, z)\n",
    "   $$\n",
    "2. **Sigmoid** (Used for binary classification probabilities):\n",
    "   $$\n",
    "   \\sigma(z) = \\frac{1}{1 + e^{-z}}\n",
    "   $$\n",
    "3. **Linear** (Used for continuous target regression):\n",
    "   $$\n",
    "   f(z) = z\n",
    "   $$\n",
    "\n",
    "---\n",
    "\n",
    "## Forward and Backward Propagation\n",
    "- **Forward Propagation**: Data flows from the input layer through hidden layers to produce a final prediction $\\hat{y}$.\n",
    "- **Loss Computation**: The error between predicted output $\\hat{y}$ and true ground truth $y$ is computed using Mean Squared Error (MSE).\n",
    "- **Backward Propagation (Backprop)**: Gradient Descent calculates the gradients of loss with respect to each weight using the Chain Rule and updates weights using an optimizer (e.g., Adam).\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 1: Import Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense\n",
    "\n",
    "print('Libraries imported successfully!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 2: Load & Explore Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('student_performance.csv')\n",
    "print('Dataset Shape:', df.shape)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.info()\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 3: Data Visualization (Exploratory Data Analysis)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(10, 6))\n",
    "sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')\n",
    "plt.title('Feature Correlation Matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 4: Data Preprocessing & Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(columns=['Final_Exam_Score'])\n",
    "y = df['Final_Exam_Score']\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "joblib.dump(scaler, 'scaler.pkl')\n",
    "print('StandardScaler saved to scaler.pkl')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 5: Build Neural Network Architecture"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential([\n",
    "    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),\n",
    "    Dense(16, activation='relu'),\n",
    "    Dense(8, activation='relu'),\n",
    "    Dense(1, activation='linear')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 6: Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "history = model.fit(\n",
    "    X_train_scaled, y_train,\n",
    "    epochs=100,\n",
    "    batch_size=32,\n",
    "    validation_split=0.15,\n",
    "    verbose=1\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 7: Plot Training & Validation Loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(8, 5))\n",
    "plt.plot(history.history['loss'], label='Training Loss (MSE)')\n",
    "plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')\n",
    "plt.title('ANN Training & Validation Loss Curve')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Mean Squared Error')\n",
    "plt.legend()\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Step 8: Model Evaluation & Save Artifacts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = model.predict(X_test_scaled).flatten()\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "mae = mean_absolute_error(y_test, y_pred)\n",
    "r2 = r2_score(y_test, y_pred)\n",
    "\n",
    "print(f'Test Mean Squared Error (MSE): {mse:.4f}')\n",
    "print(f'Test Mean Absolute Error (MAE): {mae:.4f}')\n",
    "print(f'Test R2 Score: {r2:.4f}')\n",
    "\n",
    "model.save('student_performance_model.keras')\n",
    "print('Trained Keras model saved successfully as student_performance_model.keras')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open('/Users/prakharsaxena/Downloads/untitled folder 23/ANN_Student_Performance.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)

print('ANN_Student_Performance.ipynb generated successfully!')
