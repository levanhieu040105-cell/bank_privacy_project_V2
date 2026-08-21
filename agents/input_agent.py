import pandas as pd


class InputAgent:

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def load_data(self, path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"[InputAgent] File not found: {path}")
        except Exception as e:
            raise RuntimeError(f"[InputAgent] Error reading file: {e}")

        if df.empty:
            raise ValueError("[InputAgent] DataFrame is empty.")

        print(f"[InputAgent] Loaded {len(df)} rows, {len(df.columns)} columns.")

        if self.verbose:
            print(f"[InputAgent] [DEBUG] Columns: {list(df.columns)}")

        return df