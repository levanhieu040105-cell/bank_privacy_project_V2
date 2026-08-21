from datetime import datetime


class SecureResponseAgent:

    PRIVACY_STATUS = "PII_PROTECTED_OUTPUT"

    APPLIED_TECHNIQUES = [
        "PII_TOKENIZATION",
        "SHA256_SALTED_HASHING",
        "GAUSSIAN_PRIVACY_NOISE",
        "PII_COLUMN_REMOVAL",
    ]

    LEGAL_REFERENCES = [
        "Decree 13/2023/ND-CP on Personal Data Protection",
        "Law on Personal Data Protection 2025 (Law No. 91/2025/QH15)",
    ]

    MIN_TEST_SAMPLES_FOR_CM = 30

    METRICS_WHITELIST = {
        "accuracy", "precision", "recall", "f1_score", "f2_score",
        "roc_auc", "pr_auc",
        "tn", "fp", "fn", "tp",
        "n_features",
        "n_train_samples", "n_test_samples",
        "class_distribution",
        "optimal_threshold", "baseline_metrics",
        "optuna_best_cv_f2", "smote_train_samples",
        "best_params", "selected_model", "feature_columns",
        "classification_report",
    }

    def generate_response(self, metrics: dict) -> dict:
        n_test = metrics.get("n_test_samples", 0)

        safe_metrics = {
            key: metrics[key]
            for key in self.METRICS_WHITELIST
            if key in metrics
        }

        if n_test >= self.MIN_TEST_SAMPLES_FOR_CM:
            safe_metrics["confusion_matrix"] = metrics.get("confusion_matrix")

        response = {
            "status": "SUCCESS",
            "privacy_status": self.PRIVACY_STATUS,
            "message": (
                "Pipeline completed. PII data has been protected "
                "via tokenization, salted hashing, and Gaussian privacy noise. "
                "No personal information in output."
            ),
            "metrics": safe_metrics,
            "security_info": {
                "applied_techniques": self.APPLIED_TECHNIQUES,
                "legal_references": self.LEGAL_REFERENCES,
                "timestamp": datetime.now().isoformat(),
                "pii_in_output": False,
                "token_map_in_output": False,
            },
        }

        return response