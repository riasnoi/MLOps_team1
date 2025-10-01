from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT = RAW_DIR / "sms_spam.csv"

def load_from_hf() -> pd.DataFrame:
    from datasets import load_dataset
    candidates = [
        ("sms_spam", None),
        ("sms_spam_collection", None),
        ("uciml/sms-spam", None),
        ("mrm8488/sms-spam", None),
    ]
    last_err = None
    for name, config in candidates:
        try:
            ds = load_dataset(name, config, split="train")
            df = pd.DataFrame(ds)
            df.columns = [c.lower() for c in df.columns]
            if "v1" in df.columns and "v2" in df.columns:
                df = df.rename(columns={"v1": "label", "v2": "text"})
            if "message" in df.columns and "label" in df.columns:
                df = df.rename(columns={"message": "text"})
            if "sms" in df.columns and "label" in df.columns:
                df = df.rename(columns={"sms": "text"})
            if "category" in df.columns and "label" not in df.columns:
                df = df.rename(columns={"category": "label"})
            if "class" in df.columns and "label" not in df.columns:
                df = df.rename(columns={"class": "label"})
            assert "text" in df.columns and "label" in df.columns
            df = df[["text", "label"]].dropna()
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"HF dataset not found via tried aliases: {last_err}")

def main():
    try:
        df = load_from_hf()
        source = "Hugging Face"
    except Exception as e:
        raise SystemExit(f"Failed to download dataset: {e}")

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT} (source: {source})")

if __name__ == "__main__":
    main()
