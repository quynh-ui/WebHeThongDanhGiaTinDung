# ============================================================
# ĐỀ TÀI: DỰ ĐOÁN KHẢ NĂNG ĐƯỢC DUYỆT KHOẢN VAY
# Dataset: Loan Prediction Problem Dataset
# ============================================================


# ============================================================
# 1. IMPORT THƯ VIỆN
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 2. ĐỌC DATASET
# ============================================================

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print("=" * 60)
print("      LOAN PREDICTION MACHINE LEARNING PROJECT")
print("=" * 60)


# ============================================================
# B1. KÍCH THƯỚC DATASET
# ============================================================

print("\n===== B1. KÍCH THƯỚC DATASET =====")

print("Kích thước train:", train.shape)
print("Kích thước test :", test.shape)


# ============================================================
# B2. DANH SÁCH CÁC CỘT
# ============================================================

print("\n===== B2. CÁC CỘT =====")

print(train.columns.tolist())


# ============================================================
# B3. KIỂU DỮ LIỆU
# ============================================================

print("\n===== B3. KIỂU DỮ LIỆU =====")

print(train.dtypes)


# ============================================================
# B4. THÔNG TIN DATASET
# ============================================================

print("\n===== B4. INFO DATASET =====")

train.info()


# ============================================================
# B5. 5 DÒNG ĐẦU
# ============================================================

print("\n===== B5. 5 DÒNG ĐẦU =====")

print(train.head())


# ============================================================
# B6. THỐNG KÊ DỮ LIỆU
# ============================================================

print("\n===== B6. THỐNG KÊ DỮ LIỆU =====")

print(train.describe())


# ============================================================
# B7. KIỂM TRA DỮ LIỆU THIẾU
# ============================================================

print("\n===== B7. KIỂM TRA DỮ LIỆU THIẾU =====")

missing = train.isnull().sum()

print(missing)

print("\nTổng số giá trị thiếu:", missing.sum())


# ============================================================
# B8. KIỂM TRA DỮ LIỆU TRÙNG
# ============================================================

print("\n===== B8. KIỂM TRA DỮ LIỆU TRÙNG =====")

duplicate_count = train.duplicated().sum()

print("Số dòng bị trùng:", duplicate_count)


# ============================================================
# B9. KIỂM TRA LOAN_STATUS
# ============================================================

print("\n===== B9. LOAN STATUS =====")

print(train["Loan_Status"].value_counts())


# ============================================================
# B10. TỶ LỆ LOAN_STATUS
# ============================================================

print("\n===== B10. TỶ LỆ LOAN STATUS =====")

loan_status_percent = (
    train["Loan_Status"]
    .value_counts(normalize=True)
    * 100
)

print(loan_status_percent.round(2))


# ============================================================
# B11. KIỂM TRA GIÁ TRỊ CÁC CỘT PHÂN LOẠI
# ============================================================

print("\n===== B11. GIÁ TRỊ CÁC CỘT PHÂN LOẠI =====")

print("\n--- Gender ---")
print(train["Gender"].value_counts(dropna=False))

print("\n--- Married ---")
print(train["Married"].value_counts(dropna=False))

print("\n--- Dependents ---")
print(train["Dependents"].value_counts(dropna=False))

print("\n--- Education ---")
print(train["Education"].value_counts(dropna=False))

print("\n--- Self_Employed ---")
print(train["Self_Employed"].value_counts(dropna=False))

print("\n--- Property_Area ---")
print(train["Property_Area"].value_counts(dropna=False))


# ============================================================
# B12. THỐNG KÊ CÁC BIẾN SỐ
# ============================================================

print("\n===== B12. THỐNG KÊ BIẾN SỐ =====")

print(train.describe())


# ============================================================
# B13. KIỂM TRA CÁC BIẾN QUAN TRỌNG
# ============================================================

print("\n===== B13. CÁC BIẾN SỐ QUAN TRỌNG =====")

print("\n--- ApplicantIncome ---")
print(train["ApplicantIncome"].describe())

print("\n--- CoapplicantIncome ---")
print(train["CoapplicantIncome"].describe())

print("\n--- LoanAmount ---")
print(train["LoanAmount"].describe())

print("\n--- Loan_Amount_Term ---")
print(train["Loan_Amount_Term"].describe())

print("\n--- Credit_History ---")
print(train["Credit_History"].value_counts(dropna=False))


# ============================================================
# B14. BIỂU ĐỒ LOAN STATUS
# ============================================================

print("\n===== B14. VẼ BIỂU ĐỒ LOAN STATUS =====")

plt.figure(figsize=(7, 5))

sns.countplot(
    data=train,
    x="Loan_Status"
)

plt.title("Phân bố kết quả khoản vay")
plt.xlabel("Loan Status")
plt.ylabel("Số lượng")

plt.tight_layout()


# ============================================================
# B15. CREDIT HISTORY VÀ LOAN STATUS
# ============================================================

print("\n===== B15. CREDIT HISTORY VÀ LOAN STATUS =====")

plt.figure(figsize=(7, 5))

sns.countplot(
    data=train,
    x="Credit_History",
    hue="Loan_Status"
)

plt.title(
    "Mối quan hệ giữa lịch sử tín dụng và kết quả khoản vay"
)

plt.xlabel("Credit History")
plt.ylabel("Số lượng")

plt.tight_layout()


# ============================================================
# B16. EDUCATION VÀ LOAN STATUS
# ============================================================

print("\n===== B16. EDUCATION VÀ LOAN STATUS =====")

plt.figure(figsize=(7, 5))

sns.countplot(
    data=train,
    x="Education",
    hue="Loan_Status"
)

plt.title(
    "Mối quan hệ giữa trình độ học vấn và kết quả khoản vay"
)

plt.xlabel("Education")
plt.ylabel("Số lượng")

plt.tight_layout()


# ============================================================
# B17. PROPERTY AREA VÀ LOAN STATUS
# ============================================================

print("\n===== B17. PROPERTY AREA VÀ LOAN STATUS =====")

plt.figure(figsize=(7, 5))

sns.countplot(
    data=train,
    x="Property_Area",
    hue="Loan_Status"
)

plt.title(
    "Mối quan hệ giữa khu vực tài sản và kết quả khoản vay"
)

plt.xlabel("Property Area")
plt.ylabel("Số lượng")

plt.tight_layout()


# ============================================================
# B18. CHUẨN BỊ X VÀ y
# ============================================================

print("\n===== B18. CHUẨN BỊ DỮ LIỆU MACHINE LEARNING =====")

# X = các biến đầu vào
# y = biến mục tiêu

X = train.drop(
    columns=["Loan_Status", "Loan_ID"]
)

y = train["Loan_Status"]

print("Số dòng X:", X.shape[0])
print("Số cột X :", X.shape[1])
print("Số dòng y:", y.shape[0])


# ============================================================
# B19. CHIA TRAIN / TEST
# ============================================================

print("\n===== B19. TRAIN / TEST SPLIT =====")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================================
# B20. XÁC ĐỊNH CỘT CATEGORICAL VÀ NUMERICAL
# ============================================================

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numerical_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]


# ============================================================
# B21. XỬ LÝ DỮ LIỆU CATEGORICAL
# ============================================================

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


# ============================================================
# B22. XỬ LÝ DỮ LIỆU NUMERICAL
# ============================================================

numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])


# ============================================================
# B23. KẾT HỢP PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    ),
    (
        "numerical",
        numerical_pipeline,
        numerical_features
    )
])


# ============================================================
# B24. LOGISTIC REGRESSION
# ============================================================

print("\n===== B24. LOGISTIC REGRESSION =====")

logistic_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

logistic_model.fit(
    X_train,
    y_train
)

y_pred_logistic = logistic_model.predict(
    X_test
)

logistic_accuracy = accuracy_score(
    y_test,
    y_pred_logistic
)

logistic_precision = precision_score(
    y_test,
    y_pred_logistic,
    pos_label="Y"
)

logistic_recall = recall_score(
    y_test,
    y_pred_logistic,
    pos_label="Y"
)

logistic_f1 = f1_score(
    y_test,
    y_pred_logistic,
    pos_label="Y"
)

print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1-score :", round(logistic_f1, 4))


# ============================================================
# B25. DECISION TREE
# ============================================================

print("\n===== B25. DECISION TREE =====")

decision_tree_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )
    )
])

decision_tree_model.fit(
    X_train,
    y_train
)

y_pred_tree = decision_tree_model.predict(
    X_test
)

tree_accuracy = accuracy_score(
    y_test,
    y_pred_tree
)

tree_precision = precision_score(
    y_test,
    y_pred_tree,
    pos_label="Y"
)

tree_recall = recall_score(
    y_test,
    y_pred_tree,
    pos_label="Y"
)

tree_f1 = f1_score(
    y_test,
    y_pred_tree,
    pos_label="Y"
)

print("Accuracy :", round(tree_accuracy, 4))
print("Precision:", round(tree_precision, 4))
print("Recall   :", round(tree_recall, 4))
print("F1-score :", round(tree_f1, 4))


# ============================================================
# B26. RANDOM FOREST
# ============================================================

print("\n===== B26. RANDOM FOREST =====")

random_forest_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
    )
])

random_forest_model.fit(
    X_train,
    y_train
)

y_pred_forest = random_forest_model.predict(
    X_test
)

forest_accuracy = accuracy_score(
    y_test,
    y_pred_forest
)

forest_precision = precision_score(
    y_test,
    y_pred_forest,
    pos_label="Y"
)

forest_recall = recall_score(
    y_test,
    y_pred_forest,
    pos_label="Y"
)
forest_f1 = f1_score(
    y_test,
    y_pred_forest,
    pos_label="Y"
)
print("Accuracy :", round(forest_accuracy, 4))
print("Precision:", round(forest_precision, 4))
print("Recall   :", round(forest_recall, 4))
print("F1-score :", round(forest_f1, 4))

# ==================================================
# B27. SO SÁNH CÁC MÔ HÌNH
# ==================================================

print("\n===== B27. SO SÁNH CÁC MÔ HÌNH =====")

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        logistic_accuracy,
        tree_accuracy,
        forest_accuracy
    ],

    "Precision": [
        logistic_precision,
        tree_precision,
        forest_precision
    ],

    "Recall": [
        logistic_recall,
        tree_recall,
        forest_recall
    ],

    "F1-score": [
        logistic_f1,
        tree_f1,
        forest_f1
    ]
})

print(results.round(4).to_string(index=False))


# ==================================================
# B28. TỰ ĐỘNG CHỌN MODEL TỐT NHẤT
# ==================================================

print("\n===== B28. CHỌN MODEL TỐT NHẤT =====")

# Chọn model có F1-score cao nhất
best_index = results["F1-score"].idxmax()

best_model_name = results.loc[best_index, "Model"]

print("Model tốt nhất:", best_model_name)
print("F1-score:", round(results.loc[best_index, "F1-score"], 4))


# Tạo danh sách các model
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

# Lấy model tốt nhất
best_model = models[best_model_name]

print("Đã chọn model:", best_model_name)

# ==================================================
# B29. CLASSIFICATION REPORT
# ==================================================

print("\n===== B29. CLASSIFICATION REPORT =====")

if best_model_name == "Logistic Regression":
    best_predictions = y_pred_logistic

elif best_model_name == "Decision Tree":
    best_predictions = y_pred_tree

else:
    best_predictions = y_pred_forest


print(
    classification_report(
        y_test,
        best_predictions
    )
)

# ==================================================
# B30. CONFUSION MATRIX MODEL TỐT NHẤT
# ==================================================

print("\n===== B30. CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["N", "Y"]
)

print("Confusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["N", "Y"]
).plot()

plt.title(
    "Confusion Matrix - " + best_model_name
)

plt.tight_layout()

# ==================================================
# B31. LƯU KẾT QUẢ
# ==================================================

print("\n===== B31. LƯU KẾT QUẢ =====")

os.makedirs(
    "results",
    exist_ok=True
)

results.to_csv(
    "results/model_results.csv",
    index=False
)

print(
    "Đã lưu: results/model_results.csv"
)

# ==================================================
# B32. LƯU MODEL TỐT NHẤT
# ==================================================

print("\n===== B32. LƯU MODEL TỐT NHẤT =====")

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    "models/loan_prediction_model.pkl"
)

print(
    "Đã lưu model:",
    best_model_name
)

print(
    "File: models/loan_prediction_model.pkl"
)

# ==================================================
# B33. DỰ ĐOÁN KHÁCH HÀNG MỚI
# ==================================================

print("\n===== B33. DỰ ĐOÁN KHÁCH HÀNG MỚI =====")

new_customer = pd.DataFrame({
    "Gender": ["Male"],
    "Married": ["Yes"],
    "Dependents": ["0"],
    "Education": ["Graduate"],
    "Self_Employed": ["No"],
    "ApplicantIncome": [5000],
    "CoapplicantIncome": [2000],
    "LoanAmount": [150],
    "Loan_Amount_Term": [360],
    "Credit_History": [1],
    "Property_Area": ["Urban"]
})


# Dự đoán bằng model tốt nhất
new_prediction = best_model.predict(
    new_customer
)

print(
    "Model sử dụng:",
    best_model_name
)

print(
    "Kết quả dự đoán:",
    new_prediction[0]
)


# Xác suất dự đoán
new_probability = best_model.predict_proba(
    new_customer
)

print("\nXác suất dự đoán:")

for class_name, probability in zip(
    best_model.classes_,
    new_probability[0]
):
    print(
        f"{class_name}: {probability:.2%}"
    )

# ==================================================
# B34. DỰ ĐOÁN TEST.CSV
# ==================================================

print("\n===== B34. DỰ ĐOÁN TEST.CSV =====")

# Lấy Loan_ID để đưa vào file kết quả
test_ids = test["Loan_ID"]

# Bỏ Loan_ID trước khi đưa vào model
test_features = test.drop(
    columns=["Loan_ID"]
)

# Dự đoán bằng model tốt nhất
test_predictions = best_model.predict(
    test_features
)

# Tạo file kết quả
submission = pd.DataFrame({
    "Loan_ID": test_ids,
    "Loan_Status": test_predictions
})

# Lưu kết quả
submission.to_csv(
    "results/loan_predictions.csv",
    index=False
)

print(
    "Đã lưu: results/loan_predictions.csv"
)

print("\n5 kết quả đầu tiên:")
print(submission.head())

# ==================================================
# B35. HIỂN THỊ BIỂU ĐỒ
# ==================================================

plt.show()