import json
import os
import pandas as pd
from datetime import datetime

from agents.input_agent import InputAgent
from agents.pii_detection_agent import PIIDetectionAgent
from agents.tokenization_agent import TokenizationAgent
from agents.hashing_agent import HashingAgent
from agents.privacy_ml_agent import PrivacyMLAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.churn_prediction_agent import ChurnPredictionAgent
from agents.secure_response_agent import SecureResponseAgent

OUTPUT_DIR = "output"
SECURED_DATA_FILE = os.path.join(OUTPUT_DIR, "secured_data.csv")
SECURE_MAP_FILE = os.path.join(OUTPUT_DIR, "secure_mapping.json")


def main():
    print("=" * 60)
    print("  BANKING PRIVACY MULTI-AGENT PIPELINE")
    print("  Decree 13/2023 & Law on Personal Data Protection 2025")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_agent = InputAgent()
    pii_agent = PIIDetectionAgent()
    token_agent = TokenizationAgent()
    hash_agent = HashingAgent()
    privacy_agent = PrivacyMLAgent(seed=42, noise_scale=0.01)
    feature_agent = FeatureEngineeringAgent()
    ml_agent = ChurnPredictionAgent(random_state=42, n_optuna_trials=50, cv_folds=5)
    response_agent = SecureResponseAgent()

    print("\n[STEP 1] Loading data...")
    df = input_agent.load_data("data/vietnam_bank_churn_2025.csv")

    print("\n[STEP 2] Detecting PII columns...")
    pii_cols = pii_agent.detect_pii_columns(df)
    id_cols = pii_agent.detect_id_columns(df)
    sensitive_cols = pii_agent.detect_sensitive_columns(df)

    print("\n[STEP 3] Tokenizing PII columns...")
    df, token_map = token_agent.tokenize(df, pii_cols)

    print("\n[STEP 3.5] Building row-level PII mapping...")
    row_pii_data = _build_row_pii_map(df, pii_cols, id_cols)

    existing_pii = [col for col in pii_cols if col in df.columns]
    if existing_pii:
        df.drop(columns=existing_pii, inplace=True)
        print(f"[PIPELINE] Removed {len(existing_pii)} PII column(s) from DataFrame.")

    print("\n[STEP 4] Hashing ID columns...")
    df, hash_map = hash_agent.hash_ids(df, id_columns=id_cols)

    print("\n[STEP 4.5] Finalizing row map with hash keys...")
    row_map = _finalize_row_map(row_pii_data, hash_map)

    print("\n[STEP 5] Adding Gaussian Privacy Noise...")
    df = privacy_agent.add_noise(df, sensitive_cols=sensitive_cols)

    print("\n[STEP 6] Feature Engineering...")
    df = feature_agent.transform(df)

    print("\n[SECURITY CHECK] Verifying no PII before training...")
    _verify_no_pii(df)

    print("\n[SAVE] Saving all secured data and unified mapping...")
    _save_secured_data(df)
    _save_secure_mapping(token_map, hash_map, row_map)

    print("\n[STEP 7] Training Churn Prediction Model...")
    model, metrics = ml_agent.process(df)

    print("\n[STEP 8] Generating Secure Response...")
    result = response_agent.generate_response(metrics)

    result["output_files"] = {
        "secured_data": os.path.abspath(SECURED_DATA_FILE),
        "secure_mapping_saved": True,
        "secure_mapping_security": "CONFIDENTIAL_INTERNAL_ONLY",
    }

    print("\n" + "=" * 60)
    print("  PIPELINE RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    return result


def _build_row_pii_map(df, pii_cols, id_cols):
    row_pii_data = {}
    id_col = id_cols[0] if id_cols else None
    if not id_col or id_col not in df.columns:
        print("[WARNING] No ID column found for row mapping.")
        return {}

    active_pii = [c for c in pii_cols if c in df.columns]
    for _, row in df.iterrows():
        original_id = str(row[id_col])
        pii_tokens = {}
        for pii_col in active_pii:
            val = row[pii_col]
            pii_tokens[pii_col] = str(val) if pd.notna(val) else None
        row_pii_data[original_id] = pii_tokens

    print(f"[ROW MAP] Captured PII tokens for {len(row_pii_data)} rows, "
          f"{len(active_pii)} PII column(s).")
    return row_pii_data


def _finalize_row_map(row_pii_data, hash_map):
    id_hash_map = {}
    for col, col_map in hash_map.items():
        id_hash_map.update(col_map)

    row_map = {}
    token_to_row = {}

    for original_id, pii_tokens in row_pii_data.items():
        id_hash = id_hash_map.get(original_id)
        if not id_hash:
            continue
        row_map[id_hash] = pii_tokens
        for pii_col, token_val in pii_tokens.items():
            if token_val and token_val not in token_to_row:
                token_to_row[token_val] = id_hash

    print(f"[ROW MAP] Finalized: {len(row_map)} row mappings, "
          f"{len(token_to_row)} token-to-row index entries.")
    return {"row_pii": row_map, "token_to_row": token_to_row}


def _save_secured_data(df):
    df.to_csv(SECURED_DATA_FILE, index=False, encoding="utf-8-sig")
    print(f"[SAVE] Secured data: {os.path.abspath(SECURED_DATA_FILE)}")
    print(f"[SAVE]   -> {len(df)} rows, {len(df.columns)} columns")


def _save_secure_mapping(token_map, hash_map, row_map):
    serializable_token_map = {}
    for col, col_map in token_map.items():
        serializable_token_map[col] = {
            str(original): token for original, token in col_map.items()
        }

    map_data = {
        "_metadata": {
            "description": "Unified Secure Mapping File - CONFIDENTIAL",
            "warning": "Contains original PII and IDs. Authorized access only.",
            "legal_reference": "Decree 13/2023/ND-CP, Article 26",
            "created_at": datetime.now().isoformat(),
            "total_columns": len(token_map),
            "total_tokens": sum(len(v) for v in token_map.values()),
            "total_hashes": sum(len(v) for v in hash_map.values()),
            "total_rows": len(row_map.get("row_pii", {})),
        },
        "token_map": serializable_token_map,
        "hash_map": hash_map,
        "row_pii": row_map.get("row_pii", {}),
        "token_to_row": row_map.get("token_to_row", {}),
    }

    with open(SECURE_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(map_data, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] Unified secure mapping (CONFIDENTIAL): {os.path.abspath(SECURE_MAP_FILE)}")
    print(f"[SAVE]   -> {map_data['_metadata']['total_tokens']} tokens, "
          f"{map_data['_metadata']['total_hashes']} hashes, "
          f"{map_data['_metadata']['total_rows']} rows")


def _verify_no_pii(df):
    pii_original_cols = [
        "customer_id", "id", "phone",
        "full_name", "address", "origin_province",
    ]

    leaked_cols = [col for col in pii_original_cols if col in df.columns]

    if leaked_cols:
        raise SecurityError(
            f"[SECURITY VIOLATION] Original PII columns still present: "
            f"{leaked_cols}. Pipeline halted."
        )

    print("[SECURITY CHECK] No original PII columns in training data.")


class SecurityError(Exception):
    pass


if __name__ == "__main__":
    main()