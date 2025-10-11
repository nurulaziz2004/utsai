from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_selection import RFE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================================
# Load dataset
# ================================
df = pd.read_csv(r"D:\Semester 7\AI\PAK ANAN\dataset_selada_no_age.csv")

X = df[["suhu","kelembaban","kelembaban_tanah","intensitas_cahaya"]]  # fitur
y = df["label"]  # target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ================================
# Feature Selection (RFE)
# ================================
base_tree = DecisionTreeClassifier(random_state=42)
rfe = RFE(base_tree, n_features_to_select=4)  # pilih 3 fitur terbaik
rfe.fit(X_train, y_train)

print("Fitur terpilih (RFE):", list(X.columns[rfe.support_]))

X_train_sel = rfe.transform(X_train)
X_test_sel = rfe.transform(X_test)

# ================================
# Model dengan Gini Index + Pruning
# ================================
model_gini = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)  # pruning dengan max_depth
model_gini.fit(X_train_sel, y_train)
y_pred_gini = model_gini.predict(X_test_sel)

print("\n=== HASIL GINI (dengan Feature Selection + Pruning) ===")
print("Akurasi:", accuracy_score(y_test, y_pred_gini))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gini))
print("Classification Report:\n", classification_report(y_test, y_pred_gini))

# Visualisasi pohon Gini
plt.figure(figsize=(15,8))
plot_tree(model_gini, feature_names=X.columns[rfe.support_], class_names=[str(c) for c in y.unique()],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (Gini + Pruning)")
plt.show()

# ================================
# Model dengan Entropy + Pruning
# ================================
model_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)  # pruning
model_entropy.fit(X_train_sel, y_train)
y_pred_entropy = model_entropy.predict(X_test_sel)

print("\n=== HASIL ENTROPY (dengan Feature Selection + Pruning) ===")
print("Akurasi:", accuracy_score(y_test, y_pred_entropy))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_entropy))
print("Classification Report:\n", classification_report(y_test, y_pred_entropy))

# Visualisasi pohon Entropy
plt.figure(figsize=(15,8))
plot_tree(model_entropy, feature_names=X.columns[rfe.support_], class_names=[str(c) for c in y.unique()],
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (Entropy + Pruning)")
plt.show()

# ================================
# Feature Importance Comparison
# ================================
plt.figure(figsize=(6,4))
sns.barplot(x=model_gini.feature_importances_, y=X.columns[rfe.support_], color="blue")
plt.title("Feature Importance - Gini (Pruned)")
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(x=model_entropy.feature_importances_, y=X.columns[rfe.support_], color="green")
plt.title("Feature Importance - Entropy (Pruned)")
plt.show()

# ================================
# Learning Curve (contoh Gini)
# ================================
train_sizes, train_scores, test_scores = learning_curve(model_gini, X_train_sel, y_train, cv=5, scoring="accuracy")

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
plt.plot(train_sizes, test_mean, label="Validation Score", color="orange")
plt.title("Learning Curve - Decision Tree (Gini, Pruned)")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
