import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the dataset (downloads automatically)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df = pd.read_csv(url)

print("✓ Data loaded!")
print("Shape:", df.shape)
print("Class counts:\n", df['status'].value_counts())
# Separate features and target
X = df.drop(['name', 'status'], axis=1)
y = df['status']

# Check for missing values
print("Missing values:", df.isnull().sum().sum())

# Split into train and test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit+transform on train only
X_test_scaled  = scaler.transform(X_test)        # transform only on test

print("✓ Preprocessing done!")
# Train the SVM model
from sklearn.svm import SVC

model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
model.fit(X_train_scaled, y_train)
print("✓ Model trained!")

# Save model and scaler
joblib.dump(model,  'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✓ Model and scaler saved!")
# Evaluate the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = model.predict(X_test_scaled)

print("✓ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 1), "%")
print("\nFull Report:")
print(classification_report(y_test, y_pred, target_names=['Healthy', "Parkinson's"]))

# Confusion matrix chart
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
    xticklabels=['Predicted Healthy', "Predicted Parkinson's"],
    yticklabels=['Actual Healthy', "Actual Parkinson's"])
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("✓ Confusion matrix saved!")
# Inference module — predict for a new patient
def predict_parkinsons(voice_features):
    loaded_model  = joblib.load('model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    
    sample = np.array(voice_features).reshape(1, -1)
    sample_scaled = loaded_scaler.transform(sample)  # transform only!
    
    prediction = loaded_model.predict(sample_scaled)[0]
    confidence = loaded_model.predict_proba(sample_scaled)[0]
    
    return {
        "result"    : "Parkinson's Detected" if prediction == 1 else "Healthy",
        "confidence": f"{max(confidence)*100:.1f}%"
    }

# Test it on a real sample from test set
real_sample = X_test.iloc[0].tolist()
actual      = y_test.iloc[0]

result = predict_parkinsons(real_sample)

print("="*40)
print(f"Actual    : {'Parkinson s' if actual==1 else 'Healthy'}")
print(f"Predicted : {result['result']}")
print(f"Confidence: {result['confidence']}")
print("="*40)
print("✓ Pipeline complete!")