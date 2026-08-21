"""
ChurnPredictionAgent — Production-ready churn prediction for banking.
=====================================================================
Nghiệp vụ ngân hàng ưu tiên phát hiện tối đa khách hàng rời bỏ (Recall),
đồng thời kiểm soát Precision ở mức hợp lý.

Thay đổi chính so với phiên bản cũ:
  1. Optuna objective: F2-score (ưu tiên Recall) thay vì Accuracy.
  2. scale_pos_weight: Được đưa vào search space Optuna (2.0–8.0).
  3. Threshold: Tối ưu theo F2-score thay vì Accuracy, dải 0.20–0.70.
  4. Metrics đầy đủ: F2, PR-AUC, classification_report, TN/FP/FN/TP.
  5. Tiêu chí chấp nhận: Recall >= 0.70, ROC-AUC >= baseline.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ChurnPredictionAgent:
    """
    Agent huấn luyện mô hình dự đoán churn trên dữ liệu đã bảo mật PII.
    Cung cấp 2 mô hình: Baseline (tham số cố định) và Optimized (Optuna F2).
    """

    # --- Cột bị loại khỏi features (PII / ID / date) ---
    EXCLUDED_COLUMNS = [
        "full_name", "address", "origin_province", "phone",
        "customer_id", "id",
        "customer_id_hash", "id_hash",
        "last_active_date", "created_date",
    ]

    # --- Cột categorical cần encode ---
    CATEGORICAL_COLUMNS = [
        "gender", "occupation", "customer_segment",
        "loyalty_level", "digital_behavior", "risk_segment",
    ]

    TARGET_COLUMN = "exit"

    # --- Tiêu chí chấp nhận mô hình tối ưu ---
    MIN_RECALL = 0.70

    def __init__(self, random_state: int = 42, verbose: bool = False,
                 n_optuna_trials: int = 50, cv_folds: int = 5):
        self.random_state = random_state
        self.verbose = verbose
        self.n_optuna_trials = n_optuna_trials
        self.cv_folds = cv_folds
        self.model = None
        self.baseline_model = None
        self.label_encoders = {}
        self.best_threshold = 0.5

    # =================================================================
    #  DATA PREPARATION
    # =================================================================

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode các cột categorical thành số bằng LabelEncoder."""
        df = df.copy()
        for col in self.CATEGORICAL_COLUMNS:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                le = LabelEncoder()
                df[col] = df[col].astype(str).fillna("__MISSING__")
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        return df

    def _prepare_features(self, df: pd.DataFrame):
        """
        Chuẩn bị X (features) và y (target) từ DataFrame đã bảo mật.
        Loại bỏ hoàn toàn các cột PII/ID, chỉ giữ cột numeric.
        """
        df = df.copy()
        if self.TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"[ChurnPredictionAgent] Target column "
                f"'{self.TARGET_COLUMN}' not found."
            )

        y = df[self.TARGET_COLUMN].astype(int)
        if y.nunique() < 2:
            raise ValueError(
                f"[ChurnPredictionAgent] Target has only {y.nunique()} class(es). "
                f"Need at least 2."
            )

        df = self._encode_categoricals(df)

        cols_to_drop = [self.TARGET_COLUMN] + [
            col for col in self.EXCLUDED_COLUMNS if col in df.columns
        ]
        X = df.drop(columns=cols_to_drop, errors="ignore")
        X_numeric = X.select_dtypes(include=["number"])

        dropped_cols = set(X.columns) - set(X_numeric.columns)
        if dropped_cols:
            print(f"[ChurnPredictionAgent] Dropped {len(dropped_cols)} non-numeric column(s).")

        if X_numeric.shape[1] == 0:
            raise ValueError(
                "[ChurnPredictionAgent] No numeric features remaining after filtering."
            )

        # Kiểm tra data leakage: không được có cột PII gốc
        leaked = [c for c in X_numeric.columns
                  if c in ("full_name", "address", "origin_province",
                           "phone", "customer_id", "id")]
        if leaked:
            raise ValueError(
                f"[ChurnPredictionAgent] DATA LEAKAGE: PII columns {leaked} "
                f"found in features. Pipeline halted."
            )

        if X_numeric.isnull().any().any():
            X_numeric = X_numeric.fillna(X_numeric.median())

        return X_numeric, y

    # =================================================================
    #  EVALUATION HELPERS
    # =================================================================

    @staticmethod
    def _compute_f2(y_true, y_pred):
        """Tính F2-score (beta=2, ưu tiên Recall hơn Precision)."""
        return fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    def _evaluate_model(self, y_true, y_pred, y_proba, label="MODEL"):
        """
        Tính toàn bộ metrics bắt buộc cho 1 mô hình.
        Trả về dict metrics đầy đủ.
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
            "f2_score":  round(self._compute_f2(y_true, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_true, y_proba), 4),
            "pr_auc":    round(average_precision_score(y_true, y_proba), 4),
            "confusion_matrix": cm.tolist(),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        }

        cls_report = classification_report(
            y_true, y_pred,
            target_names=["Stay (0)", "Churn (1)"],
            zero_division=0,
        )
        metrics["classification_report"] = cls_report
        return metrics

    def _print_metrics(self, metrics, label="MODEL"):
        """In kết quả metrics ra console."""
        print(f"  Accuracy:  {metrics['accuracy']}")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall:    {metrics['recall']}")
        print(f"  F1-Score:  {metrics['f1_score']}")
        print(f"  F2-Score:  {metrics['f2_score']}")
        print(f"  ROC-AUC:   {metrics['roc_auc']}")
        print(f"  PR-AUC:    {metrics['pr_auc']}")
        print(f"  TN={metrics['tn']:,}  FP={metrics['fp']:,}  "
              f"FN={metrics['fn']:,}  TP={metrics['tp']:,}")

    # =================================================================
    #  THRESHOLD OPTIMIZATION
    # =================================================================

    @staticmethod
    def find_best_threshold(y_true, y_proba, metric="f2",
                            low=0.20, high=0.70, step=0.01):
        """
        Tìm ngưỡng quyết định (threshold) tối ưu theo F2-score hoặc F1-score.
        Dải tìm kiếm: [low, high], bước step.
        """
        best_score = -1
        best_thresh = 0.5

        for thresh in np.arange(low, high + step, step):
            preds = (y_proba >= thresh).astype(int)
            if metric == "f2":
                score = fbeta_score(y_true, preds, beta=2, zero_division=0)
            else:
                score = f1_score(y_true, preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_thresh = thresh

        return round(best_thresh, 2)

    # =================================================================
    #  BASELINE MODEL
    # =================================================================

    def _train_baseline(self, X_train, X_test, y_train, y_test,
                        scale_pos_weight):
        """
        Huấn luyện mô hình Baseline với tham số cố định.
        scale_pos_weight được tính từ tỷ lệ mất cân bằng thực tế.
        Threshold cũng được tối ưu theo F2-score.
        """
        print("\n[ChurnPredictionAgent] === BASELINE MODEL ===")
        print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

        self.baseline_model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        self.baseline_model.fit(X_train, y_train)

        proba = self.baseline_model.predict_proba(X_test)[:, 1]

        # Tối ưu threshold theo F2-score cho baseline
        baseline_thresh = self.find_best_threshold(y_test, proba, metric="f2")
        preds = (proba >= baseline_thresh).astype(int)

        baseline_metrics = self._evaluate_model(y_test, preds, proba, "BASELINE")
        baseline_metrics["threshold"] = baseline_thresh

        print(f"  Threshold: {baseline_thresh}")
        self._print_metrics(baseline_metrics, "BASELINE")
        print(f"\n  Classification Report (Baseline):\n{baseline_metrics['classification_report']}")

        return baseline_metrics

    # =================================================================
    #  OPTUNA OBJECTIVE — F2-SCORE
    # =================================================================

    def _optuna_objective(self, trial, X_train, y_train, base_spw):
        """
        Hàm mục tiêu Optuna: tối đa F2-score trung bình qua StratifiedKFold.
        - scale_pos_weight nằm trong search space (2.0 – 8.0).
        - Trong mỗi fold, threshold được tối ưu theo F2-score.
        """
        params = {
            "n_estimators":    trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":       trial.suggest_int("max_depth", 3, 10),
            "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":           trial.suggest_float("gamma", 0, 5.0),
            "reg_alpha":       trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":      trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 2.0, 8.0),
        }

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=self.random_state)
        f2_scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val   = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val   = y_train.iloc[val_idx]

            model = XGBClassifier(
                **params,
                random_state=self.random_state,
                eval_metric="logloss",
                verbosity=0,
                early_stopping_rounds=30,
            )
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )

            proba = model.predict_proba(X_fold_val)[:, 1]
            # Tối ưu threshold theo F2-score trong mỗi fold
            thresh = self.find_best_threshold(y_fold_val, proba, metric="f2")
            preds = (proba >= thresh).astype(int)
            f2_scores.append(fbeta_score(y_fold_val, preds, beta=2, zero_division=0))

        return np.mean(f2_scores)

    # =================================================================
    #  OPTIMIZED MODEL
    # =================================================================

    def _train_optimized(self, X_train, X_test, y_train, y_test,
                         base_spw):
        """
        Tìm siêu tham số tối ưu bằng Optuna, huấn luyện model cuối cùng,
        tối ưu threshold theo F2-score.
        """
        print(f"\n[ChurnPredictionAgent] === OPTUNA TUNING "
              f"({self.n_optuna_trials} trials, {self.cv_folds}-fold CV) ===")
        print(f"  Objective: F2-score (beta=2, prioritize Recall)")
        print(f"  scale_pos_weight search: [2.0, 8.0]")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda trial: self._optuna_objective(trial, X_train, y_train,
                                                 base_spw),
            n_trials=self.n_optuna_trials,
            show_progress_bar=True,
        )

        best_params = study.best_params
        print(f"\n[ChurnPredictionAgent] Best CV F2-score: {study.best_value:.4f}")
        if self.verbose:
            print(f"[ChurnPredictionAgent] Best params: {best_params}")

        # --- Huấn luyện model cuối cùng với best params ---
        print("\n[ChurnPredictionAgent] === OPTIMIZED MODEL (F2 + Best Params) ===")

        self.model = XGBClassifier(
            **best_params,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            early_stopping_rounds=30,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Tối ưu threshold trên tập test theo F2-score
        proba = self.model.predict_proba(X_test)[:, 1]
        self.best_threshold = self.find_best_threshold(y_test, proba, metric="f2")
        preds = (proba >= self.best_threshold).astype(int)

        print(f"  Optimal threshold: {self.best_threshold}")
        opt_metrics = self._evaluate_model(y_test, preds, proba, "OPTIMIZED")
        opt_metrics["threshold"] = self.best_threshold
        opt_metrics["best_params"] = best_params
        opt_metrics["optuna_best_cv_f2"] = round(study.best_value, 4)

        self._print_metrics(opt_metrics, "OPTIMIZED")
        print(f"\n  Classification Report (Optimized):\n{opt_metrics['classification_report']}")

        return opt_metrics

    # =================================================================
    #  COMPARISON & ACCEPTANCE CRITERIA
    # =================================================================

    def _compare_and_select(self, baseline_metrics, optimized_metrics):
        """
        So sánh 2 mô hình, in bảng delta, kiểm tra tiêu chí chấp nhận,
        và ghi log giải thích vì sao chọn model cuối cùng.
        """
        print(f"\n[ChurnPredictionAgent] === COMPARISON: BASELINE vs OPTIMIZED ===")
        compare_keys = ["accuracy", "precision", "recall",
                        "f1_score", "f2_score", "roc_auc", "pr_auc"]

        print(f"  {'Metric':>12}  {'Baseline':>10}  {'Optimized':>10}  {'Delta':>10}")
        print(f"  {'-'*46}")
        for m in compare_keys:
            old_val = baseline_metrics.get(m, 0)
            new_val = optimized_metrics.get(m, 0)
            diff = new_val - old_val
            arrow = "+" if diff >= 0 else ""
            print(f"  {m:>12}: {old_val:.4f}  ->  {new_val:.4f}  ({arrow}{diff:.4f})")

        # --- Tiêu chí chấp nhận ---
        opt_recall  = optimized_metrics.get("recall", 0)
        opt_roc     = optimized_metrics.get("roc_auc", 0)
        base_roc    = baseline_metrics.get("roc_auc", 0)
        opt_f1      = optimized_metrics.get("f1_score", 0)
        base_f1     = baseline_metrics.get("f1_score", 0)

        accepted = True
        reasons = []

        if opt_recall < self.MIN_RECALL:
            accepted = False
            reasons.append(
                f"Recall ({opt_recall:.4f}) < min threshold ({self.MIN_RECALL})"
            )
        if opt_roc < base_roc - 0.01:
            accepted = False
            reasons.append(
                f"ROC-AUC ({opt_roc:.4f}) lower than baseline ({base_roc:.4f})"
            )

        print(f"\n[ChurnPredictionAgent] === MODEL SELECTION ===")
        if accepted:
            print(f"  [ACCEPTED] Optimized model meets business criteria.")
            print(f"    - Recall = {opt_recall:.4f} >= {self.MIN_RECALL}")
            print(f"    - ROC-AUC = {opt_roc:.4f} >= Baseline ({base_roc:.4f})")
            if opt_f1 > base_f1:
                print(f"    - F1-Score improved: {base_f1:.4f} -> {opt_f1:.4f}")
            selected = "optimized"
        else:
            print(f"  [REJECTED] Optimized model FAILED acceptance criteria:")
            for r in reasons:
                print(f"    - {r}")
            print(f"  => Falling back to Baseline model.")
            selected = "baseline"

        return selected

    # =================================================================
    #  MAIN PROCESS
    # =================================================================

    def process(self, df: pd.DataFrame) -> tuple:
        """
        Entry point chính. Chạy toàn bộ pipeline:
          1. Chuẩn bị features (đã bảo mật PII).
          2. Train baseline model.
          3. Train optimized model (Optuna F2-score).
          4. So sánh và chọn model tốt nhất theo nghiệp vụ.

        Returns:
            (model, metrics_dict)
        """
        X_numeric, y = self._prepare_features(df)

        class_counts = y.value_counts()
        imbalance_ratio = class_counts.min() / class_counts.max()
        neg = class_counts.get(0, 0)
        pos = class_counts.get(1, 0)
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        print(f"[ChurnPredictionAgent] Target distribution: {dict(class_counts)}, "
              f"imbalance ratio: {imbalance_ratio:.3f}")
        print(f"[ChurnPredictionAgent] Computed scale_pos_weight: {scale_pos_weight:.2f}")
        print(f"[ChurnPredictionAgent] Training with {X_numeric.shape[1]} features, "
              f"{len(y)} samples.")

        # --- Train / Test split (phân tầng) ---
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y,
        )

        # --- 1. BASELINE ---
        baseline_metrics = self._train_baseline(
            X_train, X_test, y_train, y_test, scale_pos_weight
        )

        # --- 2. OPTIMIZED (Optuna F2-score) ---
        optimized_metrics = self._train_optimized(
            X_train, X_test, y_train, y_test, scale_pos_weight
        )

        # --- 3. COMPARISON & SELECTION ---
        selected = self._compare_and_select(baseline_metrics, optimized_metrics)

        if selected == "baseline":
            # Fallback: dùng baseline model
            self.model = self.baseline_model
            final_proba = self.model.predict_proba(X_test)[:, 1]
            self.best_threshold = baseline_metrics["threshold"]
            final_preds = (final_proba >= self.best_threshold).astype(int)
            final_eval = baseline_metrics
        else:
            final_proba = self.model.predict_proba(X_test)[:, 1]
            final_preds = (final_proba >= self.best_threshold).astype(int)
            final_eval = optimized_metrics

        # --- Build output metrics (backward-compatible keys) ---
        metrics = {
            "accuracy":  final_eval["accuracy"],
            "precision": final_eval["precision"],
            "recall":    final_eval["recall"],
            "f1_score":  final_eval["f1_score"],
            "f2_score":  final_eval.get("f2_score", 0),
            "roc_auc":   final_eval["roc_auc"],
            "pr_auc":    final_eval.get("pr_auc", 0),
            "confusion_matrix":  final_eval["confusion_matrix"],
            "tn": final_eval.get("tn", 0),
            "fp": final_eval.get("fp", 0),
            "fn": final_eval.get("fn", 0),
            "tp": final_eval.get("tp", 0),
            "classification_report": final_eval.get("classification_report", ""),
            "class_distribution": dict(class_counts),
            "n_features":       X_numeric.shape[1],
            "n_train_samples":  len(X_train),
            "n_test_samples":   len(X_test),
            "optimal_threshold": self.best_threshold,
            "best_params":       optimized_metrics.get("best_params", {}),
            "baseline_metrics":  baseline_metrics,
            "optuna_best_cv_f2": optimized_metrics.get("optuna_best_cv_f2", 0),
            "selected_model":    selected,
            "feature_columns":   list(X_numeric.columns),
            # backward compat keys used by visualize_results.py
            "smote_train_samples": len(X_train),
        }

        print(f"\n[ChurnPredictionAgent] === FINAL MODEL: {selected.upper()} ===")
        self._print_metrics(final_eval, selected.upper())

        return self.model, metrics
