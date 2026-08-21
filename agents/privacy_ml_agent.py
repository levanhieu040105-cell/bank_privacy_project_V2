import numpy as np
import pandas as pd


class PrivacyMLAgent:

    DEFAULT_SENSITIVE_COLS = [
        "balance",
        "monthly_ir",
        "credit_sco",
        "risk_score",
    ]

    PROTECTED_COLS = ["exit"]

    def __init__(self, seed: int = 42, noise_scale: float = 0.01,
                 verbose: bool = False):
        self.seed = seed
        self.noise_scale = noise_scale
        self.verbose = verbose
        self.rng = np.random.RandomState(seed)

    def add_noise(self, df: pd.DataFrame,
                  sensitive_cols: list = None) -> pd.DataFrame:
        df = df.copy()

        if sensitive_cols is None:
            sensitive_cols = self.DEFAULT_SENSITIVE_COLS

        processed = []
        for col in sensitive_cols:
            if col not in df.columns:
                continue
            if col in self.PROTECTED_COLS:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            original_dtype = df[col].dtype
            noise = self.rng.normal(0, self.noise_scale, len(df))
            df[col] = df[col] * (1 + noise)

            if original_dtype in [np.int64, np.int32, int]:
                df[col] = df[col].round().astype(original_dtype)

            processed.append(col)

        print(f"[PrivacyMLAgent] Added Gaussian noise (scale={self.noise_scale}) "
              f"to {len(processed)} column(s).")
        if self.verbose:
            print(f"[PrivacyMLAgent] [DEBUG] Processed columns: {processed}")

        return df