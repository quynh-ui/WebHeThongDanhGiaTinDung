
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Đọc dữ liệu (dùng đường dẫn tuyệt đối theo vị trí file train_model.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "LoanPrediction", "dataset", "train.csv")

if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "train.csv")

print(f"Đang đọc dataset từ: {DATASET_PATH}")
train = pd.read_csv(DATASET_PATH)

# 2. Tách X và y
X = train.drop(columns=["Loan_Status", "Loan_ID"])
y = train["Loan_Status"]

# 3. Chia tập train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Định nghĩa pipeline tiền xử lý
categorical_features = [
    "Gender", "Married", "Dependents", 
    "Education", "Self_Employed", "Property_Area"
]

numerical_features = [
    "ApplicantIncome", "CoapplicantIncome", 
    "LoanAmount", "Loan_Amount_Term", "Credit_History"
]

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_features)
])

# 5. Huấn luyện 3 mô hình
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Y")
    rec = recall_score(y_test, y_pred, pos_label="Y")
    f1 = f1_score(y_test, y_pred, pos_label="Y")
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "Instance": model
    })

# 6. Đánh giá và chọn model tốt nhất theo F1-score
results_df = pd.DataFrame(results)
print("=== KẾT QUẢ ĐÁNH GIÁ CÁC MÔ HÌNH ===")
print(results_df[["Model", "Accuracy", "Precision", "Recall", "F1-score"]].round(4).to_string(index=False))

best_idx = results_df["F1-score"].idxmax()
best_model_name = results_df.loc[best_idx, "Model"]
best_model = results_df.loc[best_idx, "Instance"]

print(f"\n=> Chọn model tốt nhất: {best_model_name} (F1-score: {results_df.loc[best_idx, 'F1-score']:.4f})")

# 7. Lưu file model ra thư mục gốc để app.py sử dụng
MODEL_FILE = os.path.join(BASE_DIR, "loan_prediction_model.pkl")
joblib.dump(best_model, MODEL_FILE)
print(f"✅ Đã lưu model thành công tại: {MODEL_FILE}")

# ============================================================
# 8. TỰ ĐỘNG GHI NHẬN PHIÊN BẢN VÀ HIỆU NĂNG THỰC TẾ VÀO JSON
# ============================================================
import json
from datetime import datetime

history_path = os.path.join(BASE_DIR, 'models_history.json')

# Lấy Accuracy thực tế của mô hình tốt nhất vừa huấn luyện
actual_accuracy = round(float(results_df.loc[best_idx, 'Accuracy']) * 100, 1)
current_time_str = datetime.now().strftime("%H:%M %d/%m/%Y")

# 1. Đọc lịch sử các phiên bản cũ
history = []
if os.path.exists(history_path):
    with open(history_path, 'r', encoding='utf-8') as f:
        try:
            history = json.load(f)
        except Exception:
            history = []

# Đổi trạng thái tất cả các mô hình cũ thành 'Lưu trữ'
for item in history:
    item['status'] = 'Lưu trữ'

# 2. Tạo phiên bản mới tiếp theo (v4, v5,...) lấy trực tiếp thông số vừa train
next_ver_num = len(history) + 1
new_record = {
    "version": f"v{next_ver_num}",
    "name": f"ML v{next_ver_num} ({best_model_name})",
    "accuracy": actual_accuracy,
    "algorithm": str(best_model_name),
    "train_size": f"{len(X_train)} mẫu (80%)",
    "trained_at": current_time_str,
    "status": "Đang sử dụng"
}
# ============================================================
# LƯU FILE MODEL & GHI NHẬN LỊCH SỬ HUẤN LUYỆN
# ============================================================
MODEL_FILE = os.path.join(BASE_DIR, "loan_prediction_model.pkl")
joblib.dump(best_model, MODEL_FILE)
print(f"✅ Đã lưu model thành công tại: {MODEL_FILE}")

import json
from datetime import datetime

history_path = os.path.join(BASE_DIR, 'models_history.json')

# 1. Lấy chỉ số thực tế vừa huấn luyện
actual_accuracy = round(float(results_df.loc[best_idx, 'Accuracy']) * 100, 2)
actual_f1 = round(float(results_df.loc[best_idx, 'F1-score']) * 100, 2)

# 2. Đọc lịch sử cũ
history = []
if os.path.exists(history_path):
    with open(history_path, 'r', encoding='utf-8') as f:
        try:
            history = json.load(f)
        except Exception:
            history = []

# 👉 ĐOẠN Ở ẢNH BƯỚC 4 BẮT ĐẦU TỪ ĐÂY 👈
# Lấy bản ghi cuối cùng trước đó để tính toán độ tăng trưởng (Delta)
prev_acc = history[-1]['accuracy'] if history else 0
prev_f1 = history[-1]['f1_score'] if history else 0

diff_acc = round(actual_accuracy - prev_acc, 2)
diff_f1 = round(actual_f1 - prev_f1, 2)

# Chuyển trạng thái các mô hình cũ thành 'Lưu trữ'
for item in history:
    item['status'] = 'Lưu trữ'

# 3. Tạo bản ghi mới phiên bản tiếp theo
next_ver_num = len(history) + 1
new_record = {
    "version": f"ML v{next_ver_num}",
    "dataset_name": "Loan Prediction Dataset",
    "dataset_version": "Dataset v1",
    "records_count": len(train),
    "model_name": str(best_model_name),
    "train_ratio": "80%",
    "test_ratio": "20%",
    "random_state": 42,
    "accuracy": actual_accuracy,
    "precision": round(float(results_df.loc[best_idx, 'Precision']) * 100, 2),
    "recall": round(float(results_df.loc[best_idx, 'Recall']) * 100, 2),
    "f1_score": actual_f1,
    "trained_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
    "trainer": "Admin",
    "status": "active",
    "model_file": "loan_prediction_model.pkl",
    "diff_accuracy": diff_acc,
    "diff_f1": diff_f1,
    "preprocess": {
        "imputation": True,
        "encoding": True,
        "drop_id": True,
        "scaling": True
    }
}

history.append(new_record)

# 4. Ghi lại vào file JSON
with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print(f"✅ Đã ghi nhận [ML v{next_ver_num}] vào models_history.json (Acc: {actual_accuracy}%, ΔAcc: {diff_acc}%)")

history.append(new_record)

# 3. Ghi lại vào file JSON (bảo toàn v1, v2, v3 cũ)
with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print(f"✅ Đã cập nhật thành công: ML v{next_ver_num} ({best_model_name}) đạt {actual_accuracy}% vào models_history.json")