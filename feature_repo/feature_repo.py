from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "processed.parquet"

sms_entity = Entity(name="sms_id", join_keys=["sms_id"])

sms_source = FileSource(
    name="sms_processed_source",
    path=str(PROCESSED_PATH),
    timestamp_field="event_timestamp",
)

sms_feature_view = FeatureView(
    name="sms_features",
    entities=[sms_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="char_len", dtype=Int64),
        Field(name="word_len", dtype=Int64),
        Field(name="num_digits", dtype=Int64),
        Field(name="num_urls", dtype=Int64),
        Field(name="num_domains", dtype=Int64),
        Field(name="upper_ratio", dtype=Float32),
    ],
    online=True,
    source=sms_source,
)
