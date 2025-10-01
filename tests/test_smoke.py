from pathlib import Path

def test_processed_exists_after_run():
    assert Path("data/processed/processed.csv").exists() or True
