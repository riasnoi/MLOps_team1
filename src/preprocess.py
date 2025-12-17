from pathlib import Path
import pandas as pd
import numpy as np
import re
import regex as re2
from bs4 import BeautifulSoup
import tldextract

RAW = Path("data/raw/sms_spam.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT = PROCESSED_DIR / "processed.csv"
OUT_PARQUET = PROCESSED_DIR / "processed.parquet"

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
DIGITS_RE = re.compile(r"\d")
UPPER_RE = re2.compile(r"\p{Lu}")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = BeautifulSoup(s, "html.parser").get_text(separator=" ")
    s = s.lower()
    s = URL_RE.sub(" <url> ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def count_urls(s: str) -> int:
    return len(URL_RE.findall(s)) if isinstance(s, str) else 0

def count_digits(s: str) -> int:
    return len(DIGITS_RE.findall(s)) if isinstance(s, str) else 0

def upper_ratio(s: str) -> float:
    if not isinstance(s, str) or not s:
        return 0.0
    upp = len(UPPER_RE.findall(s))
    return float(upp) / max(len(s), 1)

def num_domains(s: str) -> int:
    if not isinstance(s, str):
        return 0
    urls = URL_RE.findall(s)
    domains = set()
    for u in urls:
        ts = tldextract.extract(u)
        dom = ".".join([p for p in [ts.domain, ts.suffix] if p])
        if dom:
            domains.add(dom)
    return len(domains)

def main():
    df = pd.read_csv(RAW)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "text" in df.columns and "label" in df.columns, "Ожидаются колонки text и label"

    df["text_clean"] = df["text"].astype(str).apply(clean_text)

    df["char_len"] = df["text_clean"].str.len()
    df["word_len"] = df["text_clean"].str.split().apply(len)
    df["num_digits"] = df["text"].astype(str).apply(count_digits)
    df["num_urls"] = df["text"].astype(str).apply(count_urls)
    df["num_domains"] = df["text"].astype(str).apply(num_domains)
    df["upper_ratio"] = df["text"].astype(str).apply(upper_ratio)

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["target"] = (df["label"] == "spam").astype(int)
    df = df.reset_index(drop=True)
    df["sms_id"] = df.index.astype(int)

    base_ts = pd.Timestamp("2020-01-01T00:00:00+00:00")
    df["event_timestamp"] = base_ts + pd.to_timedelta(df["sms_id"] % 365, unit="D")

    keep = [
        "sms_id", "event_timestamp",
        "text", "text_clean", "label", "target",
        "char_len", "word_len", "num_digits", "num_urls", "num_domains", "upper_ratio"
    ]
    features = df[keep]
    features.to_csv(OUT, index=False)
    features.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved processed datasets to {OUT} and {OUT_PARQUET} with shape {features.shape}")

if __name__ == "__main__":
    main()
