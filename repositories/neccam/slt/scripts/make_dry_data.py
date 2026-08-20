#!/usr/bin/env python3
"""Create tiny, real PHOENIX14T feature splits for the dry-run contract."""

import gzip
import pickle
from pathlib import Path


SOURCE = Path("/datasets/rwth-phoenix-2014-t/features")
DESTINATION = Path("/outputs/neccam-slt/dry-data")
LIMITS = {
    "phoenix14t.pami0.train.protocol4": 32,
    "phoenix14t.pami0.dev.protocol4": 8,
    "phoenix14t.pami0.test.protocol4": 8,
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, limit in LIMITS.items():
        source = SOURCE / name
        if not source.is_file():
            raise FileNotFoundError(source)
        with gzip.open(source, "rb") as handle:
            records = pickle.load(handle)
        if len(records) < limit:
            raise RuntimeError(
                f"{source} contains {len(records)} records; expected >= {limit}"
            )
        sample = records[:limit]
        for record in sample:
            required = {"name", "signer", "sign", "gloss", "text"}
            if set(record) < required:
                raise RuntimeError(f"malformed record in {source}")
            if tuple(record["sign"].shape)[-1] != 1024:
                raise RuntimeError(f"unexpected feature shape {record['sign'].shape}")
        destination = DESTINATION / name
        with gzip.open(destination, "wb") as handle:
            pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"wrote {len(sample)} real records to {destination}")


if __name__ == "__main__":
    main()
