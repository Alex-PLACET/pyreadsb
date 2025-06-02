import gzip
from pathlib import Path
from typing import BinaryIO, Final

GZIP_MAGIC: Final[bytes] = b"\x1f\x8b"  # Gzip file header magic


def detect_compression(file_path: Path) -> str:
    """Detect file compression type."""
    if file_path.suffix.lower() == ".gz":
        return "gzip"

    # Check magic bytes
    with open(file_path, "rb") as f:
        magic: bytes = f.read(4)

    if magic.startswith(GZIP_MAGIC):
        return "gzip"
    else:
        return "none"


def open_file(file_path: Path) -> BinaryIO | gzip.GzipFile:
    """Open file with appropriate decompression."""
    compression: Final[str] = detect_compression(file_path)

    if compression == "gzip":
        return gzip.open(file_path, "rb")
    else:
        return open(file_path, "rb")
