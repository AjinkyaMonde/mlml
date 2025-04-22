# Step 1: Import required libraries
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Ionosphere dataset
ionosphere = fetch_openml(name='ionosphere', version=1, as_frame=True)
df = ionosphere.frame

# Step 3: Prepare features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target: 'g' -> 1, 'b' -> 0
le = LabelEncoder()
y = le.fit_transform(y)

# Step 4: Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Logistic Regression model
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

# Step 7: Predict and Evaluate
y_pred = log_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy using Logistic Regression: {accuracy:.4f}")

# Step 8: Detailed Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['bad (0)', 'good (1)']))

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap='Greens', fmt='d', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title("Confusion Matrix - Logistic Regression (Ionosphere)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
