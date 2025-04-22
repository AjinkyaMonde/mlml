# Step 1: Import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: One-hot encode the target
y_encoded = to_categorical(y)

# Step 4: Train-Test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Step 5: Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Build the model
model = Sequential()
model.add(Dense(64, input_shape=(4,), activation='relu'))   # Hidden Layer 1
model.add(Dense(32, activation='relu'))                     # Hidden Layer 2
model.add(Dense(16, activation='relu'))                     # Hidden Layer 3
model.add(Dense(3, activation='softmax'))                   # Output Layer

# Step 7: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=0)

# Step 9: Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {accuracy:.4f}")

# Step 10: Classification Report
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=iris.target_names))

# Step 11: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap='YlOrBr', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix - Neural Network (Iris)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
