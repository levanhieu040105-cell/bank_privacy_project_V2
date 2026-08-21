import numpy as np
import pandas as pd


class FeatureEngineeringAgent:

    PII_BLACKLIST = [
        "full_name", "address", "origin_province", "phone",
        "customer_id", "id",
    ]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        created_features = []

        if self._check_numeric_cols(df, ["balance", "monthly_ir"]):
            df["balance_income_ratio"] = (
                df["balance"] / np.maximum(df["monthly_ir"], 1)
            )
            created_features.append("balance_income_ratio")

        value_cols = ["engagement_score", "credit_sco", "tenure_ye"]
        if self._check_numeric_cols(df, value_cols):
            df["customer_value_score"] = (
                df["engagement_score"] * 0.4 +
                df["credit_sco"] * 0.4 +
                df["tenure_ye"] * 0.2
            )
            created_features.append("customer_value_score")

        if self._check_numeric_cols(df, ["risk_score", "credit_sco"]):
            df["risk_credit_ratio"] = (
                df["risk_score"] / np.maximum(df["credit_sco"], 1)
            )
            created_features.append("risk_credit_ratio")

        if self._check_numeric_cols(df, ["nums_service", "nums_card"]):
            df["service_card_ratio"] = (
                df["nums_service"] / np.maximum(df["nums_card"], 1)
            )
            created_features.append("service_card_ratio")

        if self._check_numeric_cols(df, ["tenure_ye", "age"]):
            df["tenure_age_ratio"] = (
                df["tenure_ye"] / np.maximum(df["age"], 1)
            )
            created_features.append("tenure_age_ratio")

        for feat in created_features:
            if feat in df.columns:
                inf_count = np.isinf(df[feat]).sum()
                nan_count = df[feat].isna().sum()
                if inf_count > 0 or nan_count > 0:
                    df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
                    df[feat] = df[feat].fillna(0)
                    if self.verbose:
                        print(f"[FeatureEngineeringAgent] [DEBUG] {feat}: "
                              f"fixed {inf_count} inf, {nan_count} NaN.")

        print(f"[FeatureEngineeringAgent] Created {len(created_features)} new feature(s).")
        if self.verbose:
            print(f"[FeatureEngineeringAgent] [DEBUG] Features: {created_features}")

        return df

    def _check_numeric_cols(self, df: pd.DataFrame, cols: list) -> bool:
        for col in cols:
            if col not in df.columns:
                return False
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False
            if col in self.PII_BLACKLIST:
                print("[FeatureEngineeringAgent] Warning: "
                      "1 column in PII blacklist, skipping.")
                return False
        return True