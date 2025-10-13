# ===============================================
# Import libraries
# ===============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# ===============================================
# Load dataset
# ===============================================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)

# ===============================================
# Pisahkan fitur dan target
# ===============================================
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# ===============================================
# RFE untuk seleksi fitur
# ===============================================
base_model = DecisionTreeClassifier(random_state=42)
rfe = RFE(estimator=base_model, n_features_to_select=5)
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]

print("\n=== Fitur Terpilih oleh RFE ===")
print(selected_features)

# Gunakan fitur terpilih
X_selected = X[selected_features]

# ===============================================
# Split data
# ===============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.3, random_state=42
)

# ===============================================
# Decision Tree dengan Pre-Pruning (max_depth)
# ===============================================
model = DecisionTreeClassifier(
    random_state=42,
    max_depth=4,            # batas kedalaman pohon
    min_samples_split=2,    # cegah cabang terlalu kecil
    min_samples_leaf=10    # minimal sampel di daun
)
model.fit(X_train, y_train)

# ===============================================
# Evaluasi Model
# ===============================================
y_pred = model.predict(X_test)
print("\n=== HASIL EVALUASI MODEL ===")
print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ===============================================
# Visualisasi Pohon Keputusan
# ===============================================
plt.figure(figsize=(75, 50))
plot_tree(model, filled=True, feature_names=selected_features, class_names=['No Diabetes', 'Diabetes'])
plt.title("Decision Tree dengan RFE dan Max Depth Pruning")
plt.show()

# ===============================================
# Visualisasi Feature Importance
# ===============================================
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 10))
plt.bar(range(len(selected_features)), importances[indices])
plt.xticks(range(len(selected_features)), selected_features[indices], rotation=45)
plt.title("Feature Importance Setelah RFE dan Max Depth Pruning")
plt.xlabel("Fitur")
plt.ylabel("Kepentingan")
plt.tight_layout()
plt.show()

# Cetak ranking fitur
print("\n=== Ranking Feature Importance ===")
for i in range(len(selected_features)):
    print(f"{i+1}. {selected_features[indices[i]]}: {importances[indices[i]]:.4f}")
