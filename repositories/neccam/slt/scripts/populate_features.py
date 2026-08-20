#!/usr/bin/env python3
"""Populate the checksum-pinned feature artifacts used by neccam/slt."""

import argparse
import gzip
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path


REVISION = "d5c32f2cd1cf27a26083671532a32e75c98dbae3"
FILES = {
    "phoenix14t.pami0.dev": (
        97_747_545,
        "7daa7074035c5617e71aab039ff9d2b7fc8a854efab3b060415baa63bbaa5774",
    ),
    "phoenix14t.pami0.test": (
        113_400_935,
        "db3349323834e2eca80f36d81a2ed0459a53a7b59e4ea298f41fc6c91d72504a",
    ),
    "phoenix14t.pami0.train": (
        1_449_330_684,
        "5418d19653644e7bb7a7579c6a5164b169eb5155101b621f5d23d9f543ddd0fb",
    ),
}
PROTOCOL4_FILES = {
    "phoenix14t.pami0.dev.protocol4": (
        97_747_524,
        "48731f8fec864df137962bfa9987504a88ed766001a7676afccf39366b6eaa3e",
    ),
    "phoenix14t.pami0.test.protocol4": (
        113_400_913,
        "8c0b6d40d66564df4c0fa2f888888c89bfd76b2fb85f6a03cce86a61c5d16401",
    ),
    "phoenix14t.pami0.train.protocol4": (
        1_449_330_661,
        "c45c4cbce46c74586a316876ba2d73eaa0fc19a9e0a0ec08d943f2fa6bb6afbb",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_python37_compatible(source: Path) -> Path:
    """Rewrite only the declared pickle protocol, preserving every payload byte."""
    destination = source.with_name(source.name + ".protocol4")
    expected_size, expected_sha256 = PROTOCOL4_FILES[destination.name]
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size
        and sha256(destination) == expected_sha256
    ):
        print(f"verified existing {destination}")
        return destination
    partial = destination.with_name(destination.name + ".partial")
    with gzip.open(source, "rb") as source_stream:
        protocol = source_stream.read(2)
        if protocol != b"\x80\x05":
            raise RuntimeError(f"unexpected pickle header in {source}: {protocol!r}")
        with partial.open("wb") as compressed_output:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=compressed_output, mtime=0
            ) as destination_stream:
                destination_stream.write(b"\x80\x04")
                shutil.copyfileobj(source_stream, destination_stream, 8 * 1024 * 1024)
    os.replace(partial, destination)
    actual_size = destination.stat().st_size
    actual_sha256 = sha256(destination)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"checksum mismatch for {destination.name}: "
            f"size={actual_size}, sha256={actual_sha256}"
        )
    print(
        f"installed protocol-4 derivative {destination} "
        f"({actual_size} bytes, sha256={actual_sha256})"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/datasets/rwth-phoenix-2014-t/features"),
    )
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)

    for name, (expected_size, expected_sha256) in FILES.items():
        destination = args.destination / name
        if (
            destination.is_file()
            and destination.stat().st_size == expected_size
            and sha256(destination) == expected_sha256
        ):
            print(f"verified existing {destination}")
        else:
            url = (
                "https://huggingface.co/datasets/lavinal712/slt/resolve/"
                f"{REVISION}/{name}?download=true"
            )
            partial = destination.with_suffix(destination.suffix + ".partial")
            print(f"downloading pinned artifact {name}")
            with urllib.request.urlopen(url) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

            actual_size = partial.stat().st_size
            actual_sha256 = sha256(partial)
            if actual_size != expected_size or actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"checksum mismatch for {name}: "
                    f"size={actual_size}, sha256={actual_sha256}"
                )
            os.replace(partial, destination)
            print(
                f"installed {destination} "
                f"({actual_size} bytes, sha256={actual_sha256})"
            )

        make_python37_compatible(destination)


if __name__ == "__main__":
    main()
