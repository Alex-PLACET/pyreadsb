import gzip
import logging
import struct
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import (
    BinaryIO,
    Final,
)

from .compression_utils import open_file


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

    MODE_NON_ICAO_ADDRESS: Final[int] = 1 << 24  # Non-ICAO address type mask

    @dataclass
    class HeatEntry:
        """Represents a decoded aircraft position entry."""

        hex_id: str
        lat: float
        lon: float
        alt: int | str | None
        ground_speed: float | None

    @dataclass
    class CallsignEntry:
        """Represents aircraft callsign information."""

        hex_id: str
        callsign: str | None = None

    @dataclass
    class TimestampSeparator:
        """Represents a timestamp separator between data chunks."""

        timestamp: float
        raw_data: bytes

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.current_timestamp: float | None = None
        self.logger = logging.getLogger(__name__)

    def _log(self, message: str) -> None:
        """Log debug messages if debug mode is enabled."""
        if self.debug:
            self.logger.debug(message)

    def _detect_endianness(
        self, file_handle: BinaryIO | gzip.GzipFile
    ) -> struct.Struct:
        """Detect endianness by reading first few entries."""
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

    def _decode_timestamp(self, hex_val: int, lat: int, lon: int) -> float:
        """Decode timestamp from separator entry."""
        # Based on the original decoder: now = i3u / 1000 + i2u * 4294967.296
        # Where i2u and i3u are the lat and lon values as unsigned integers
        lat_u: Final[int] = lat & 0xFFFFFFFF  # Convert to unsigned
        lon_u: Final[int] = lon & 0xFFFFFFFF  # Convert to unsigned

        timestamp: Final[float] = lon_u / 1000.0 + lat_u * 4294967.296
        return timestamp

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

            # Extract addrtype (top 5 bits)
            # addrtype_5bits = (hex_val >> 27) & 0x1F

            addr = hex_val & 0xFFFFFF

            # non_icao_addr: bool = hex_val & self.MODE_NON_ICAO_ADDRESS != 0

            return self.HeatEntry(
                hex_id=f"{addr:06x}",
                lat=latitude,
                lon=longitude,
                alt=altitude,
                ground_speed=ground_speed,
            )

    def decode(
        self, file_path: Path
    ) -> Generator[HeatEntry | CallsignEntry | TimestampSeparator, None, None]:
        """Memory-efficient decoder that yields entries one by one."""
        self.logger.info(f"Decoding file: {file_path}")

        with open_file(file_path) as f:
            # Detect endianness from file
            entry_struct = self._detect_endianness(f)

            while True:
                # Read one entry at a time
                entry_data = f.read(self.HEAT_ENTRY_SIZE)
                if len(entry_data) != self.HEAT_ENTRY_SIZE:
                    if not entry_data:
                        self.logger.info("Reached end of file.")
                    else:
                        self.logger.warning(
                            f"Incomplete entry read: {len(entry_data)} bytes"
                        )
                    break

                try:
                    hex_val, lat, lon, alt, gs = entry_struct.unpack(entry_data)

                    if hex_val == self.MAGIC_NUMBER:
                        # Timestamp separator
                        timestamp: float = self._decode_timestamp(hex_val, lat, lon)
                        self.current_timestamp = timestamp

                        separator = self.TimestampSeparator(
                            timestamp=timestamp,
                            raw_data=entry_data,
                        )
                        yield separator
                    else:
                        # Heat entry or callsign entry
                        entry: (
                            HeatmapDecoder.HeatEntry | HeatmapDecoder.CallsignEntry
                        ) = self._decode_heat_entry(hex_val, lat, lon, alt, gs)
                        yield entry

                except struct.error as e:
                    self.logger.warning(f"Error parsing entry: {e}")
                    continue
