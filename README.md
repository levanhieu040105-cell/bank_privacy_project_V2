# Privacy-Preserving Banking AI System

Hệ thống AI pipeline đa tác tử (multi-agent) cho dự đoán khách hàng rời bỏ (churn prediction)
với cơ chế bảo vệ quyền riêng tư dữ liệu ngân hàng.

## Tuân thủ pháp luật

- **Nghị định 13/2023/NĐ-CP** — Bảo vệ dữ liệu cá nhân
- **Luật BVDLCN 2025** (Luật số 91/2025/QH15) — Luật Bảo vệ dữ liệu cá nhân

## Pipeline Architecture

```
InputAgent → PIIDetectionAgent → TokenizationAgent → HashingAgent
    → PrivacyMLAgent → FeatureEngineeringAgent → ChurnPredictionAgent
    → SecureResponseAgent
```

## Multi-Agent System

| Agent | Chức năng |
|-------|-----------|
| **InputAgent** | Tải và kiểm tra dữ liệu đầu vào |
| **PIIDetectionAgent** | Phát hiện cột PII (cơ bản, ID, tài chính) |
| **TokenizationAgent** | Thay thế PII text bằng token UUID |
| **HashingAgent** | Hash SHA-256 + salt cột ID, xóa cột gốc |
| **DeTokenizationAgent** | Khôi phục dữ liệu gốc từ token_map |
| **PrivacyMLAgent** | Thêm Differential Privacy noise |
| **FeatureEngineeringAgent** | Tạo feature cho churn prediction |
| **ChurnPredictionAgent** | Train XGBoost model |
| **SecureResponseAgent** | Trả response an toàn (không chứa PII) |

## Kỹ thuật bảo mật

- **PII Tokenization**: Thay thế giá trị PII bằng UUID token
- **SHA-256 Salted Hashing**: Hash cột ID với salt, xóa cột gốc
- **Differential Privacy**: Thêm Gaussian noise vào dữ liệu tài chính nhạy cảm
- **PII Column Removal**: Xóa hoàn toàn cột PII gốc trước khi train
- **Security Verification**: Kiểm tra bảo mật trước khi đưa dữ liệu vào model

## Dataset

Vietnam Bank Churn Dataset 2025  
Kaggle: https://www.kaggle.com/datasets/tranhuunhan/vietnam-bank-churn-dataset-2025

## Kết quả (Optimized Model — F2-score objective)

| Metric | Giá trị |
|--------|---------:|
| Accuracy | 0.6996 |
| Precision | 0.3642 |
| Recall | **0.8976** |
| F1-Score | 0.5182 |
| F2-Score | 0.6943 |
| ROC-AUC | 0.8558 |
| PR-AUC | 0.5391 |
| Threshold | 0.29 |

> Ưu tiên **Recall** và **F2-score** do bài toán churn cần phát hiện tối đa khách hàng có nguy cơ rời bỏ.
> Optuna tối ưu F2-score (beta=2) với scale_pos_weight trong [2.0, 8.0].

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost

## Cách chạy

```bash
python main.py
```