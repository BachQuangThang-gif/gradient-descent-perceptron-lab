# Bài Tập Machine Learning: Gradient Descent & Perceptron

Bài nộp bài tập gồm phần giải lý thuyết các bài toán 3.26 - 3.28 và mã nguồn thực nghiệm cho bài 3.29 - 3.30.

## 1. Lời giải lý thuyết

- **Bài 3.26:** Gradient Descent cho \(f(x)=x^2-4x+5\), \(\eta=0.2, x^{(0)}=5\). Sau 4 bước: \(x^{(4)} = 2.3888\), \(f(x^{(4)}) = 1.1512\) (hội tụ về cực tiểu \(x=2, f(x)=1\)).
- **Bài 3.27:** \(w^Tx = 1 > 0 \implies \hat{y} = +1\). Với \(y=-1\), mẫu bị phân lớp sai.
- **Bài 3.28:** \(w^Tx = -1 \implies \hat{y} = -1 \neq y=1\). Cập nhật: \(w*{new} = w + y \cdot x = [0, 4, 1]^T\). Giá trị mới: \(w*{new}^T x = 13 > 0\).

## 2. Phần thực hành (Bài 3.29 & 3.30)

- Xây dựng lớp `Perceptron` từ đầu hỗ trợ `fit` và `predict`.
- Thực nghiệm phân loại nhị phân trên tập dữ liệu Breast Cancer sau khi chuẩn hóa `StandardScaler`.
- Đánh giá chất lượng qua các chỉ số: Accuracy, Precision, Recall, F1-Score.

## Cách chạy

```bash
python perceptron.py
```
