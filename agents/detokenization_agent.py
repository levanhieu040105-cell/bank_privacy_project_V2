import pandas as pd


class DeTokenizationAgent:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def restore_value(self, tokenized_value, token_map: dict):
        for col_map in token_map.values():
            reverse_map = {token: original for original, token in col_map.items()}
            if tokenized_value in reverse_map:
                return reverse_map[tokenized_value]
        return tokenized_value

    def restore_dataframe(self, df: pd.DataFrame, token_map: dict) -> pd.DataFrame:
        df = df.copy()

        for col, col_map in token_map.items():
            if col not in df.columns:
                print("[DeTokenizationAgent] Warning: 1 column not found, skipping.")
                continue

            reverse_map = {token: original for original, token in col_map.items()}
            df[col] = df[col].map(
                lambda x, rm=reverse_map: rm.get(x, x) if pd.notna(x) else x
            )

        restored_count = sum(1 for col in token_map if col in df.columns)
        print(f"[DeTokenizationAgent] Restored {restored_count} column(s).")
        if self.verbose:
            restored_cols = [col for col in token_map if col in df.columns]
            print(f"[DeTokenizationAgent] [DEBUG] Columns: {restored_cols}")

        return df