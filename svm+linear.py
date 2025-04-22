# Step 1: Import libraries
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Ionosphere dataset
ionosphere = fetch_openml(name='ionosphere', version=1, as_frame=True)
df = ionosphere.frame

# Step 3: Prepare X and y
X = df.drop(columns='class')
y = df['class']

# Encode target: 'g' -> 1, 'b' -> 0
le = LabelEncoder()
y = le.fit_transform(y)

# Step 4: Split into train and test (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train SVM with linear kernel
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

# Step 7: Predict and Evaluate
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy using Linear SVM: {accuracy:.4f}")

# Step 8: Confusion Matrix and Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Bad (0)', 'Good (1)']))

# Step 9: Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title("Confusion Matrix - Linear SVM (Ionosphere)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
