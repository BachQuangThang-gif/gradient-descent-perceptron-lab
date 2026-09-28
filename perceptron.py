import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ==============================================================
# BÀI 3.29: XÂY DỰNG LỚP PERCEPTRON TỪ ĐẦU (FROM SCRATCH)
# ==============================================================
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        X: Mảng numpy shape (n_samples, n_features)
        y: Mảng numpy chứa nhãn {-1, 1}
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            errors = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = np.where(linear_output >= 0, 1, -1)
                
                # Điều kiện phân lớp sai
                if y[idx] * y_pred <= 0:
                    update = self.lr * y[idx]
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            if errors == 0:
                break

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)


# ==============================================================
# BÀI 3.30: THỰC NGHIỆM TRÊN BỘ DỮ LIỆU PHÂN LỚP NHỊ PHÂN
# ==============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("THỰC NGHIỆM MÔ HÌNH PERCEPTRON (BÀI 3.29 & 3.30)")
    print("=" * 60)

    # 1. Tải tập dữ liệu nhị phân (Chẩn đoán Y khoa: Lành tính vs Ác tính)
    data = load_breast_cancer()
    X, y_raw = data.data, data.target

    # Đưa nhãn 0, 1 về chuẩn Perceptron: -1 và 1
    y = np.where(y_raw == 0, -1, 1)

    # 2. Chia tập Train (80%) và Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Chuẩn hóa dữ liệu (Standardization) để Perceptron hội tụ tốt nhất
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Khởi tạo và huấn luyện mô hình Perceptron
    model = Perceptron(learning_rate=0.01, n_iters=1000)
    model.fit(X_train, y_train)

    # 5. Dự báo trên tập Test
    y_pred = model.predict(X_test)

    # Đổi về nhãn 0 và 1 để tính toán các chỉ số đánh giá
    y_test_eval = np.where(y_test == 1, 1, 0)
    y_pred_eval = np.where(y_pred == 1, 1, 0)

    # 6. Đánh giá chất lượng mô hình theo các độ đo
    acc = accuracy_score(y_test_eval, y_pred_eval)
    prec = precision_score(y_test_eval, y_pred_eval)
    rec = recall_score(y_test_eval, y_pred_eval)
    f1 = f1_score(y_test_eval, y_pred_eval)

    print("\n--- KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ---")
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("=" * 60)