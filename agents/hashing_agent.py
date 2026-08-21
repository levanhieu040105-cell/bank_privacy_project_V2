import hashlib
import os
import pandas as pd


class HashingAgent:

    def __init__(self, salt: str = None):
        self.salt = salt or os.getenv("HASH_SALT")
        if not self.salt:
            raise ValueError(
                "[HashingAgent] HASH_SALT is required. "
                "Set environment variable HASH_SALT or pass salt to constructor."
            )
        if len(self.salt) < 16:
            raise ValueError(
                "[HashingAgent] HASH_SALT too short. "
                "Minimum 16 characters required."
            )

    def hash_ids(self, df: pd.DataFrame, id_columns: list = None) -> tuple:
        df = df.copy()
        hash_map = {}

        if id_columns is None:
            id_columns = ["id", "customer_id"]

        for col in id_columns:
            if col not in df.columns:
                continue

            col_hash_map = {}
            unique_vals = df[col].dropna().unique()
            for val in unique_vals:
                col_hash_map[str(val)] = self._hash_value(val)

            hash_map[col] = col_hash_map

            hash_col_name = f"{col}_hash"
            df[hash_col_name] = df[col].apply(self._hash_value)
            df.drop(columns=[col], inplace=True)
            print(f"[HashingAgent] Hashed '{col}' -> '{hash_col_name}' and removed original.")
            print(f"[HashingAgent]   -> {len(col_hash_map)} unique ID(s) mapped.")

        return df, hash_map

    def _hash_value(self, value) -> str:
        if pd.isna(value):
            raw = f"{self.salt}::__NAN__"
        else:
            raw = f"{self.salt}::{value}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()