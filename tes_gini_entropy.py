from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, learning_curve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv(r"D:\skripsi\Smart Agriculture Technology for Reliable Intelligent Automation\CAPSTONE PROJECT\Datasets\dataset 1000.csv")

X = df[["suhu","kelembaban","kelembaban_tanah","intensitas_cahaya"]]  # fitur
y = df["label"]  # target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ================================
# 1. Model dengan Gini Index
# ================================
model_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
model_gini.fit(X_train, y_train)
y_pred_gini = model_gini.predict(X_test)

print("=== HASIL MENGGUNAKAN GINI INDEX ===")
print("Akurasi:", accuracy_score(y_test, y_pred_gini))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gini))
print("Classification Report:\n", classification_report(y_test, y_pred_gini))

# Visualisasi pohon (Gini)
plt.figure(figsize=(20,10))
plot_tree(model_gini, feature_names=X.columns, class_names=[str(c) for c in y.unique()],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (Gini Index)")
plt.show()

# ================================
# 2. Model dengan Entropy
# ================================
model_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
model_entropy.fit(X_train, y_train)
y_pred_entropy = model_entropy.predict(X_test)

print("\n=== HASIL MENGGUNAKAN ENTROPY ===")
print("Akurasi:", accuracy_score(y_test, y_pred_entropy))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_entropy))
print("Classification Report:\n", classification_report(y_test, y_pred_entropy))

# Visualisasi pohon (Entropy)
plt.figure(figsize=(20,10))
plot_tree(model_entropy, feature_names=X.columns, class_names=[str(c) for c in y.unique()],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (Entropy)")
plt.show()

# ================================
# 3. Confusion Matrix Heatmap
# ================================
fig, ax = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(confusion_matrix(y_test, y_pred_gini), annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Confusion Matrix - Gini")
sns.heatmap(confusion_matrix(y_test, y_pred_entropy), annot=True, fmt="d", cmap="Greens", ax=ax[1])
ax[1].set_title("Confusion Matrix - Entropy")
plt.show()

# ================================
# 4. Perbandingan Akurasi
# ================================
akurasi = [accuracy_score(y_test, y_pred_gini), accuracy_score(y_test, y_pred_entropy)]
metode = ["Gini", "Entropy"]

plt.bar(metode, akurasi, color=["blue","green"])
plt.ylim(0,1)
plt.ylabel("Akurasi")
plt.title("Perbandingan Akurasi Gini vs Entropy")
plt.show()

# ================================
# 5. Feature Importance
# ================================
plt.figure(figsize=(6,4))
sns.barplot(x=model_gini.feature_importances_, y=X.columns, color="blue")
plt.title("Feature Importance - Gini")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x=model_entropy.feature_importances_, y=X.columns, color="green")
plt.title("Feature Importance - Entropy")
plt.show()

# ================================
# 6. Learning Curve (contoh untuk Gini)
# ================================
train_sizes, train_scores, test_scores = learning_curve(model_gini, X, y, cv=5, scoring="accuracy")

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
plt.plot(train_sizes, test_mean, label="Cross-validation Score", color="orange")
plt.title("Learning Curve - Decision Tree (Gini)")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
