import uuid
import pandas as pd


class TokenizationAgent:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def tokenize(self, df: pd.DataFrame, pii_cols: list) -> tuple:
        df = df.copy()
        token_map = {}
        used_tokens = set()

        for col in pii_cols:
            if col not in df.columns:
                print("[TokenizationAgent] Warning: 1 PII column not found, skipping.")
                if self.verbose:
                    print(f"[TokenizationAgent] [DEBUG] Skipped: {col}")
                continue

            token_map[col] = {}
            col_values = df[col].dropna().unique()

            for value in col_values:
                token = self._generate_unique_token(used_tokens)
                token_map[col][value] = token
                used_tokens.add(token)

            df[col] = df[col].map(
                lambda x, mapping=token_map[col]: mapping.get(x, x)
                if pd.notna(x) else x
            )

        tokenized_count = sum(len(v) for v in token_map.values())
        print(f"[TokenizationAgent] Tokenized {tokenized_count} values "
              f"across {len(token_map)} PII column(s).")

        return df, token_map

    @staticmethod
    def _generate_unique_token(used_tokens: set) -> str:
        while True:
            token = f"TOKEN_{uuid.uuid4().hex[:12]}"
            if token not in used_tokens:
                return token