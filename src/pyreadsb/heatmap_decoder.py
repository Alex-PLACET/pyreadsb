import logging
import struct
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import (
    Final,
    Protocol,
    runtime_checkable,
)

from .compression_utils import open_file


@runtime_checkable
class FileProtocol(Protocol):
    """Protocol for file-like objects with read, tell, and seek methods."""

    def read(self, size: int = -1, /) -> bytes: ...
    def tell(self) -> int: ...
    def seek(self, offset: int, whence: int = 0, /) -> int: ...


class HeatmapDecoder:
    # Constants from globe_index.h/c
    MAGIC_NUMBER: Final[int] = 0x0E7F7C9D  # Magic number for chunk/timestamp separation

    # Structure sizes
    HEAT_ENTRY_SIZE: Final[int] = 16  # 4 + 4 + 4 + 2 + 2 = 16 bytes (packed)

    # Struct formats for different endianness
    HEAT_ENTRY_LE: Final[struct.Struct] = struct.Struct(
        "<IiiHH"
    )  # Little endian: uint32, int32, int32, uint16, uint16
    HEAT_ENTRY_BE: Final[struct.Struct] = struct.Struct(">IiiHH")  # Big endian

    @dataclass(slots=True)
    class HeatEntry:
        """Represents a decoded aircraft position entry."""

        hex_id: str
        lat: float
        lon: float
        alt: int | str | None
        ground_speed: float | None

    @dataclass(slots=True)
    class CallsignEntry:
        """Represents aircraft callsign information."""

        hex_id: str
        callsign: str | None = None

    @dataclass(slots=True)
    class TimestampSeparator:
        """Represents a timestamp separator between data chunks."""

        timestamp: datetime
        raw_data: bytes

    __slots__ = ("current_timestamp", "logger")

    def __init__(self) -> None:
        self.current_timestamp: datetime | None = None
        self.logger = logging.getLogger(__name__)

    def _check_entry_endianness(self, entry_data: bytes) -> struct.Struct | None:
        """Check a single entry for endianness markers. Returns struct format if found, None otherwise."""
        if len(entry_data) < 4:
            return None

        try:
            # Try little-endian first (more common)
            hex_val_le = struct.unpack("<I", entry_data[:4])[0]
            if hex_val_le == self.MAGIC_NUMBER:
                self.logger.debug("Detected little-endian format")
                return self.HEAT_ENTRY_LE

            # Try big-endian
            hex_val_be = struct.unpack(">I", entry_data[:4])[0]
            if hex_val_be == self.MAGIC_NUMBER:
                self.logger.debug("Detected big-endian format")
                return self.HEAT_ENTRY_BE

            return None
        except struct.error:
            return None

    def _detect_endianness_from_bytes(self, data: bytes) -> struct.Struct:
        """Detect endianness from a bytes object."""
        pos = 0
        while pos + self.HEAT_ENTRY_SIZE <= len(data):
            entry_data = data[pos : pos + self.HEAT_ENTRY_SIZE]
            detected_struct = self._check_entry_endianness(entry_data)

            if detected_struct:
                return detected_struct
            pos += self.HEAT_ENTRY_SIZE

        # Default to little-endian if no magic found
        self.logger.debug("No magic number found, defaulting to little-endian")
        return self.HEAT_ENTRY_LE

    def _detect_endianness_from_file(self, file_handle: FileProtocol) -> struct.Struct:
        """Detect endianness from a file-like object."""
        original_pos: int = file_handle.tell()

        try:
            # Read entries progressively until detecting endianness
            while True:
                entry_data: bytes = file_handle.read(self.HEAT_ENTRY_SIZE)
                if len(entry_data) != self.HEAT_ENTRY_SIZE:
                    self.logger.warning(
                        f"Incomplete entry read at position {original_pos}: {len(entry_data)} bytes"
                    )
                    break

                detected_struct = self._check_entry_endianness(entry_data)
                if detected_struct:
                    return detected_struct

            # Default to little-endian if no magic found
            self.logger.debug("No magic number found, defaulting to little-endian")
            return self.HEAT_ENTRY_LE
        except struct.error as e:
            self.logger.error(f"Error unpacking entry: {e}")
            self.logger.error("File may not be in expected format or corrupted.")
            raise ValueError("Unable to determine endianness") from e
        finally:
            # Restore file position
            file_handle.seek(original_pos)

    def _detect_endianness(self, data_source: FileProtocol | bytes) -> struct.Struct:
        """Detect endianness by reading first few entries."""
        if isinstance(data_source, bytes):
            return self._detect_endianness_from_bytes(data_source)

        return self._detect_endianness_from_file(data_source)

    def _decode_timestamp(self, lat: int, lon: int) -> datetime:
        """Decode timestamp from separator entry."""
        # Based on the original decoder: now = i3u / 1000 + i2u * 4294967.296
        # Where i2u and i3u are the lat and lon values as unsigned integers
        lat_u: Final[int] = lat & 0xFFFFFFFF  # Convert to unsigned
        lon_u: Final[int] = lon & 0xFFFFFFFF  # Convert to unsigned

        timestamp_float: Final[float] = lon_u / 1000.0 + lat_u * 4294967.296
        return datetime.fromtimestamp(timestamp_float, tz=UTC)

    def _decode_heat_entry(
        self, hex_val: int, lat: int, lon: int, alt: int, gs: int
    ) -> HeatEntry | CallsignEntry:
        """Decode a single heat entry."""
        # Check if this is an info entry (bit 30 set in latitude)
        is_info_entry = bool(lat & (1 << 30))

        if is_info_entry:
            # Combine lon and alt for 8-byte callsign
            callsign_bytes: bytes = struct.pack("<IH", lon & 0xFFFFFFFF, alt & 0xFFFF)
            callsign: str = callsign_bytes.rstrip(b"\x00").decode(
                "ascii", errors="ignore"
            )
            addr: int = hex_val & 0xFFFFFF  # Extract address part
            return self.CallsignEntry(
                hex_id=f"{addr:06x}",
                callsign=callsign if callsign else None,
            )
        else:
            # Regular position entry
            # Convert coordinates
            latitude: Final[float] = lat / 1e6
            longitude: Final[float] = lon / 1e6

            # Decode altitude
            altitude: int | str | None
            if alt == -123:
                altitude = "ground"
            elif alt == -124:
                altitude = None
            else:
                altitude = alt * 25  # Altitude in feet

            # Decode ground speed
            if gs == 65535 or gs == -1:  # 0xFFFF or -1
                ground_speed = None
            else:
                ground_speed = gs / 10.0  # Convert to knots

            addr = hex_val & 0xFFFFFF

            return self.HeatEntry(
                hex_id=f"{addr:06x}",
                lat=latitude,
                lon=longitude,
                alt=altitude,
                ground_speed=ground_speed,
            )

    def _decode_entry(
        self, entry_struct: struct.Struct, data: bytes
    ) -> HeatEntry | CallsignEntry | TimestampSeparator:
        """
        Decode a single entry from binary data into the appropriate entry type.

        This method unpacks binary data using the provided struct and determines whether
        the entry represents a heat map entry, callsign entry, or timestamp separator
        based on the magic number detection.

        Args:
            entry_struct: The struct object used for unpacking binary data.
            data: Raw binary data to decode, must be at least HEAT_ENTRY_SIZE bytes.
                Only the first HEAT_ENTRY_SIZE bytes will be processed.

        Returns:
            HeatEntry or CallsignEntry for regular aircraft data, or
            TimestampSeparator when magic number is detected.

        Raises:
            ValueError: If data is insufficient for decoding.

        Note:
            Updates self.current_timestamp when a timestamp separator is encountered.
        """
        if not data or len(data) < self.HEAT_ENTRY_SIZE:
            raise ValueError("Insufficient data for decoding.")

        hex_val, lat, lon, alt, gs = entry_struct.unpack(data[: self.HEAT_ENTRY_SIZE])
        if hex_val == self.MAGIC_NUMBER:
            # Timestamp separator
            timestamp: datetime = self._decode_timestamp(lat, lon)
            self.current_timestamp = timestamp

            separator = self.TimestampSeparator(
                timestamp=timestamp,
                raw_data=data,
            )
            return separator
        else:
            # Heat entry or callsign entry
            entry: HeatmapDecoder.HeatEntry | HeatmapDecoder.CallsignEntry = (
                self._decode_heat_entry(hex_val, lat, lon, alt, gs)
            )
            return entry

    def decode_from_bytes(
        self, data: bytes
    ) -> Generator[HeatEntry | CallsignEntry | TimestampSeparator, None, None]:
        """Decode entries from a bytes object."""
        data_len = len(data)
        if data_len < self.HEAT_ENTRY_SIZE:
            if data:
                self.logger.warning("Insufficient data for decoding.")
            return

        entry_struct: Final[struct.Struct] = self._detect_endianness(data)
        unpack_from = entry_struct.unpack_from  # Cache method lookup
        entry_size = self.HEAT_ENTRY_SIZE
        magic = self.MAGIC_NUMBER

        # Use memoryview to avoid copying
        mv = memoryview(data)
        max_pos = data_len - entry_size + 1

        for pos in range(0, max_pos, entry_size):
            hex_val, lat, lon, alt, gs = unpack_from(data, pos)

            if hex_val == magic:
                # Timestamp separator
                timestamp_float = (lon & 0xFFFFFFFF) / 1000.0 + (lat & 0xFFFFFFFF) * 4294967.296
                timestamp = datetime.fromtimestamp(timestamp_float, tz=UTC)
                self.current_timestamp = timestamp
                yield self.TimestampSeparator(
                    timestamp=timestamp,
                    raw_data=bytes(mv[pos : pos + entry_size]),
                )
            elif lat & (1 << 30):  # Info/callsign entry
                callsign_bytes = struct.pack("<IH", lon & 0xFFFFFFFF, alt & 0xFFFF)
                callsign = callsign_bytes.rstrip(b"\x00").decode("ascii", errors="ignore")
                yield self.CallsignEntry(
                    hex_id=f"{hex_val & 0xFFFFFF:06x}",
                    callsign=callsign if callsign else None,
                )
            else:
                # Regular position entry
                altitude: int | str | None
                if alt == -123:
                    altitude = "ground"
                elif alt == -124:
                    altitude = None
                else:
                    altitude = alt * 25

                yield self.HeatEntry(
                    hex_id=f"{hex_val & 0xFFFFFF:06x}",
                    lat=lat / 1e6,
                    lon=lon / 1e6,
                    alt=altitude,
                    ground_speed=None if gs == 65535 else gs / 10.0,
                )

        # Check for trailing incomplete data
        remaining = data_len % entry_size
        if remaining:
            self.logger.warning(
                f"Incomplete entry at end: {remaining} bytes"
            )

    def decode_from_file(
        self, file_path: Path
    ) -> Generator[HeatEntry | CallsignEntry | TimestampSeparator, None, None]:
        """Memory-efficient decoder that yields entries one by one."""
        self.logger.info(f"Decoding file: {file_path}")

        # Read in chunks for better I/O performance (64KB = 4096 entries)
        buffer_size: Final[int] = 65536

        with open_file(file_path) as f:
            # Detect endianness from file
            entry_struct: Final[struct.Struct] = self._detect_endianness(f)
            unpack_from = entry_struct.unpack_from  # Cache method lookup
            entry_size = self.HEAT_ENTRY_SIZE
            magic = self.MAGIC_NUMBER

            leftover = b""
            while True:
                chunk = f.read(buffer_size)
                if not chunk:
                    if leftover:
                        self.logger.warning(
                            f"Incomplete entry at end: {len(leftover)} bytes"
                        )
                    else:
                        self.logger.info("Reached end of file.")
                    break

                # Combine leftover from previous chunk
                if leftover:
                    chunk = leftover + chunk
                    leftover = b""

                chunk_len = len(chunk)
                max_pos = chunk_len - entry_size + 1

                for pos in range(0, max_pos, entry_size):
                    hex_val, lat, lon, alt, gs = unpack_from(chunk, pos)

                    if hex_val == magic:
                        # Timestamp separator
                        timestamp_float = (lon & 0xFFFFFFFF) / 1000.0 + (lat & 0xFFFFFFFF) * 4294967.296
                        timestamp = datetime.fromtimestamp(timestamp_float, tz=UTC)
                        self.current_timestamp = timestamp
                        yield self.TimestampSeparator(
                            timestamp=timestamp,
                            raw_data=chunk[pos : pos + entry_size],
                        )
                    elif lat & (1 << 30):  # Info/callsign entry
                        callsign_bytes = struct.pack("<IH", lon & 0xFFFFFFFF, alt & 0xFFFF)
                        callsign = callsign_bytes.rstrip(b"\x00").decode("ascii", errors="ignore")
                        yield self.CallsignEntry(
                            hex_id=f"{hex_val & 0xFFFFFF:06x}",
                            callsign=callsign if callsign else None,
                        )
                    else:
                        # Regular position entry
                        altitude: int | str | None
                        if alt == -123:
                            altitude = "ground"
                        elif alt == -124:
                            altitude = None
                        else:
                            altitude = alt * 25

                        yield self.HeatEntry(
                            hex_id=f"{hex_val & 0xFFFFFF:06x}",
                            lat=lat / 1e6,
                            lon=lon / 1e6,
                            alt=altitude,
                            ground_speed=None if gs == 65535 else gs / 10.0,
                        )

                # Save leftover bytes for next iteration
                processed = (chunk_len // entry_size) * entry_size
                if processed < chunk_len:
                    leftover = chunk[processed:]
