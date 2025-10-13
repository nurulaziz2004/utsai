from collections import Counter
import numpy as np
import pandas as pd

# ==============================================
# Fungsi Gini Impurity
# ==============================================
def gini_impurity(y):
    """Hitung impurity Gini dari label y"""
    hist = np.bincount(y)
    ps = hist / len(y)
    return 1 - np.sum(ps ** 2)


# ==============================================
# Struktur Node dan Decision Tree
# ==============================================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None
        self.feature_importances_ = None  # menyimpan importance fitur

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)
        # Hitung importance sederhana berdasarkan frekuensi fitur digunakan
        self.feature_importances_ = self._compute_feature_importance(X)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Kriteria berhenti
        if (depth >= self.max_depth) or (n_labels == 1) or (n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Pilih split terbaik
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_loss = gini_impurity(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = gini_impurity(y[left_idxs]), gini_impurity(y[right_idxs])
        child_loss = (n_l / n) * g_l + (n_r / n) * g_r

        return parent_loss - child_loss

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _compute_feature_importance(self, X):
        """Hitung importance fitur berdasarkan frekuensi fitur digunakan"""
        importance = np.zeros(X.shape[1])
        self._traverse_and_count(self.root, importance)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        return importance

    def _traverse_and_count(self, node, importance):
        if node is None or node.is_leaf_node():
            return
        importance[node.feature] += 1
        self._traverse_and_count(node.left, importance)
        self._traverse_and_count(node.right, importance)


# ==============================================
# FUNGSI RFE (Recursive Feature Elimination)
# ==============================================
def recursive_feature_elimination(X, y, n_features_to_keep, feature_names):
    """Hapus fitur paling tidak penting sampai tersisa n_features_to_keep"""
    features = np.arange(X.shape[1])
    feature_names = np.array(feature_names)

    while len(features) > n_features_to_keep:
        clf = DecisionTree(max_depth=5)
        clf.fit(X[:, features], y)
        importances = clf.feature_importances_

        if importances.sum() == 0:
            print("⚠️ Semua fitur sama pentingnya, proses dihentikan.")
            break

        least_important = np.argmin(importances)
        print(f"🔥 Menghapus fitur '{feature_names[least_important]}' (indeks {features[least_important]}), sisa {len(features)-1} fitur.")
        features = np.delete(features, least_important)
        feature_names = np.delete(feature_names, least_important)

    return features, feature_names


# ==============================================
# MAIN PROGRAM
# ==============================================
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Load dataset
    df = pd.read_csv(r"D:\Semester 7\AI\PAK ANAN\datadt.csv")
    print("Data sample:")
    print(df.head())

    # Pisahkan fitur dan target (asumsi kolom terakhir = target)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)
    feature_names = df.columns[:-1]  # nama kolom fitur

    # Jalankan RFE untuk memilih 5 fitur terbaik
    selected_features, selected_names = recursive_feature_elimination(X, y, n_features_to_keep=5, feature_names=feature_names)
    print("\n✅ Fitur terpilih (berdasarkan nama):", list(selected_names))

    # Split data menggunakan fitur terpilih
    X_train, X_test, y_train, y_test = train_test_split(
        X[:, selected_features], y, test_size=0.3, random_state=42
    )

    # Buat model Decision Tree
    clf = DecisionTree(max_depth=5)
    clf.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = clf.predict(X_test)

    print("\n=== HASIL EVALUASI ===")
    print("Akurasi:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
