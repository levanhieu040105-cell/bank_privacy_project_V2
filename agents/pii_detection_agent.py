class PIIDetectionAgent:

    PII_COLUMNS_BASIC = [
        "full_name",
        "address",
        "origin_province",
        "phone",
    ]

    ID_COLUMNS = [
        "customer_id",
        "id",
    ]

    SENSITIVE_FINANCIAL_COLUMNS = [
        "balance",
        "monthly_ir",
        "credit_sco",
        "risk_score",
    ]

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def detect_pii_columns(self, df) -> list:
        detected = [col for col in self.PII_COLUMNS_BASIC if col in df.columns]
        print(f"[PIIDetectionAgent] Detected {len(detected)} basic PII column(s).")
        if self.verbose:
            print(f"[PIIDetectionAgent] [DEBUG] Basic PII: {detected}")
        return detected

    def detect_id_columns(self, df) -> list:
        detected = [col for col in self.ID_COLUMNS if col in df.columns]
        print(f"[PIIDetectionAgent] Detected {len(detected)} ID column(s).")
        if self.verbose:
            print(f"[PIIDetectionAgent] [DEBUG] ID: {detected}")
        return detected

    def detect_sensitive_columns(self, df) -> list:
        detected = [col for col in self.SENSITIVE_FINANCIAL_COLUMNS if col in df.columns]
        print(f"[PIIDetectionAgent] Detected {len(detected)} sensitive column(s).")
        if self.verbose:
            print(f"[PIIDetectionAgent] [DEBUG] Sensitive: {detected}")
        return detected