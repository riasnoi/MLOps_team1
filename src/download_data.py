from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT = RAW_DIR / "sms_spam.csv"

def main():
    try:
        from datasets import load_dataset
        ds = load_dataset("sms_spam_collection", split="train")
        df = pd.DataFrame(ds)
        df.columns = [c.lower() for c in df.columns]
        if "v1" in df.columns and "v2" in df.columns:
            df = df.rename(columns={"v1": "label", "v2": "text"})
        df = df[["text", "label"]].dropna()
    except Exception:
        print("HF dataset failed — downloading raw csv instead")
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        df = pd.read_csv(url, sep="\t", names=["label", "text"])
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")

if __name__ == "__main__":
    main()
