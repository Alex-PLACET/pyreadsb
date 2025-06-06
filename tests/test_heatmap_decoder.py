import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyreadsb.heatmap_decoder import HeatmapDecoder


class TestHeatmapDecoder:
    """Test suite for HeatmapDecoder class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = HeatmapDecoder(debug=True)

    def test_init(self):
        """Test decoder initialization."""
        decoder = HeatmapDecoder()
        assert not decoder.debug
        assert decoder.current_timestamp is None

        debug_decoder = HeatmapDecoder(debug=True)
        assert debug_decoder.debug

    def test_magic_number_constant(self):
        """Test magic number constant."""
        assert HeatmapDecoder.MAGIC_NUMBER == 0x0E7F7C9D

    def test_heat_entry_size_constant(self):
        """Test heat entry size constant."""
        assert HeatmapDecoder.HEAT_ENTRY_SIZE == 16

    def test_dataclass_creation(self):
        """Test dataclass creation and attributes."""
        # Test HeatEntry
        heat_entry = HeatmapDecoder.HeatEntry(
            hex_id="abc123", lat=37.7749, lon=-122.4194, alt=1000, ground_speed=250.5
        )
        assert heat_entry.hex_id == "abc123"
        assert heat_entry.lat == 37.7749
        assert heat_entry.lon == -122.4194
        assert heat_entry.alt == 1000
        assert heat_entry.ground_speed == 250.5

        # Test CallsignEntry
        callsign_entry = HeatmapDecoder.CallsignEntry(
            hex_id="def456", callsign="UAL123"
        )
        assert callsign_entry.hex_id == "def456"
        assert callsign_entry.callsign == "UAL123"

        # Test TimestampSeparator
        timestamp_sep = HeatmapDecoder.TimestampSeparator(
            timestamp=1609459200.0,
            raw_data=b"\x9d\x7c\x7f\x0e\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        )
        assert timestamp_sep.timestamp == 1609459200.0
        assert len(timestamp_sep.raw_data) == 16

    def test_decode_timestamp(self):
        """Test timestamp decoding logic."""
        # Test with known values
        hex_val = HeatmapDecoder.MAGIC_NUMBER
        lat = 0x12345678  # Example lat value
        lon = 0x9ABCDEF0  # Example lon value

        timestamp = self.decoder._decode_timestamp(hex_val, lat, lon)

        # Verify calculation: lon_u / 1000.0 + lat_u * 4294967.296
        lat_u = lat & 0xFFFFFFFF
        lon_u = lon & 0xFFFFFFFF
        expected = lon_u / 1000.0 + lat_u * 4294967.296

        assert timestamp == expected

    def test_decode_heat_entry_position(self):
        """Test decoding regular position entry."""
        hex_val = 0xABC123
        lat = 37774900  # 37.7749 * 1e6
        lon = -122419400  # -122.4194 * 1e6
        alt = 40  # 40 * 25 = 1000 feet
        gs = 2505  # 250.5 * 10

        entry = self.decoder._decode_heat_entry(hex_val, lat, lon, alt, gs)

        assert isinstance(entry, HeatmapDecoder.HeatEntry)
        assert entry.hex_id == "abc123"
        assert entry.lat == pytest.approx(37.7749, rel=1e-6)
        assert entry.lon == pytest.approx(-122.4194, rel=1e-6)
        assert entry.alt == 1000
        assert entry.ground_speed == pytest.approx(250.5, rel=1e-1)

    def test_decode_heat_entry_ground_altitude(self):
        """Test decoding entry with ground altitude."""
        hex_val = 0xABC123
        lat = 37774900
        lon = -122419400
        alt = -123  # Ground indicator
        gs = 0

        entry = self.decoder._decode_heat_entry(hex_val, lat, lon, alt, gs)

        assert isinstance(entry, HeatmapDecoder.HeatEntry)
        assert entry.alt == "ground"
        assert entry.ground_speed == 0.0

    def test_decode_heat_entry_unknown_altitude(self):
        """Test decoding entry with unknown altitude."""
        hex_val = 0xABC123
        lat = 37774900
        lon = -122419400
        alt = -124  # Unknown altitude indicator
        gs = 65535  # Unknown ground speed

        entry = self.decoder._decode_heat_entry(hex_val, lat, lon, alt, gs)

        assert isinstance(entry, HeatmapDecoder.HeatEntry)
        assert entry.alt is None
        assert entry.ground_speed is None

    def test_decode_heat_entry_callsign(self):
        """Test decoding callsign entry (info entry)."""
        hex_val = 0xABC123
        lat = 37774900 | (1 << 30)  # Set bit 30 to indicate info entry
        lon = 0x41554131  # "AU1A" in bytes
        alt = 0x3332  # "32" in bytes
        gs = 0

        entry = self.decoder._decode_heat_entry(hex_val, lat, lon, alt, gs)

        assert isinstance(entry, HeatmapDecoder.CallsignEntry)
        assert entry.hex_id == "abc123"
        # Callsign should be extracted from lon and alt bytes
        assert entry.callsign is not None

    def test_detect_endianness_little_endian(self):
        """Test endianness detection with little-endian data."""
        # Create mock data with magic number in little-endian format
        magic_bytes = struct.pack("<I", HeatmapDecoder.MAGIC_NUMBER)
        mock_data = magic_bytes + b"\x00" * 12  # 16-byte entry

        mock_file = MagicMock()
        mock_file.tell.return_value = 0
        mock_file.read.return_value = mock_data

        result = self.decoder._detect_endianness(mock_file)

        assert result == HeatmapDecoder.HEAT_ENTRY_LE
        mock_file.seek.assert_called_with(0)

    def test_detect_endianness_big_endian(self):
        """Test endianness detection with big-endian data."""
        # Create mock data with magic number in big-endian format
        magic_bytes = struct.pack(">I", HeatmapDecoder.MAGIC_NUMBER)
        mock_data = magic_bytes + b"\x00" * 12

        mock_file = MagicMock()
        mock_file.tell.return_value = 0
        mock_file.read.return_value = mock_data

        result = self.decoder._detect_endianness(mock_file)

        assert result == HeatmapDecoder.HEAT_ENTRY_BE
        mock_file.seek.assert_called_with(0)

    def test_detect_endianness_no_magic_defaults_to_little(self):
        """Test endianness detection defaults to little-endian when no magic found."""
        # Create mock data without magic number
        mock_data = b"\x00" * 16

        mock_file = MagicMock()
        mock_file.tell.return_value = 0
        mock_file.read.side_effect = [mock_data, b""]  # Return empty on second read

        result = self.decoder._detect_endianness(mock_file)

        assert result == HeatmapDecoder.HEAT_ENTRY_LE

    def test_detect_endianness_struct_error(self):
        """Test endianness detection with struct error."""
        mock_file = MagicMock()
        mock_file.tell.return_value = 0
        mock_file.read.return_value = b"\x00" * 3  # Too short for struct
        # This should just log a warning and continue, not raise ValueError
        self.decoder._detect_endianness(mock_file)

    def test_log_debug_enabled(self):
        """Test logging when debug is enabled."""
        debug_decoder = HeatmapDecoder(debug=True)

        with patch.object(debug_decoder.logger, "debug") as mock_debug:
            debug_decoder._log("test message")
            mock_debug.assert_called_once_with("test message")

    def test_log_debug_disabled(self):
        """Test logging when debug is disabled."""
        no_debug_decoder = HeatmapDecoder(debug=False)

        with patch.object(no_debug_decoder.logger, "debug") as mock_debug:
            no_debug_decoder._log("test message")
            mock_debug.assert_not_called()


class TestHeatmapDecoderWithRealFile:
    """Test suite using the actual heatmap file."""

    def setup_method(self):
        """Set up test fixtures."""
        self.decoder = HeatmapDecoder(debug=True)
        self.test_file_path = Path("tests/resources/16.bin.ttf")

    def test_file_exists(self):
        """Test that the test file exists."""
        assert self.test_file_path.exists()
        assert self.test_file_path.is_file()
        assert self.test_file_path.stat().st_size > 0

    def test_decode_real_file_basic(self):
        """Test basic decoding of real heatmap file."""

        # generator = self.decoder.decode(self.test_file_path)
        # for entry in generator:
        #     if isinstance(entry, (HeatmapDecoder.TimestampSeparator)):
        #         print(f"Timestamp: {entry.timestamp}")
        #     # Check that each entry is an instance of expected types
        #     print(f"Entry: {entry}")
        entries = list(self.decoder.decode_from_file(self.test_file_path))

        # Basic assertions
        assert len(entries) > 0

        # Check that we have different types of entries
        entry_types = {type(entry) for entry in entries}
        assert HeatmapDecoder.TimestampSeparator in entry_types
        assert HeatmapDecoder.HeatEntry in entry_types

    def test_decode_real_file_first_entries(self):
        """Test the first few entries of the real file."""
        entries = []
        count = 0

        for entry in self.decoder.decode_from_file(self.test_file_path):
            entries.append(entry)
            count += 1
            if count >= 10:  # Only process first 10 entries
                break

        # Should have at least some entries
        assert len(entries) > 0

        # First entry should typically be a timestamp separator
        if entries:
            first_entry = entries[0]
            # Could be either timestamp or heat entry depending on file format
            assert isinstance(
                first_entry,
                HeatmapDecoder.TimestampSeparator | HeatmapDecoder.HeatEntry,
            )

    def test_decode_real_file_data_validity(self):
        """Test data validity of decoded entries from real file."""
        entry_count = 0
        timestamp_count = 0
        heat_count = 0
        callsign_count = 0

        for entry in self.decoder.decode_from_file(self.test_file_path):
            entry_count += 1

            if isinstance(entry, HeatmapDecoder.TimestampSeparator):
                timestamp_count += 1
                assert entry.timestamp > 0
                assert len(entry.raw_data) == 16

            elif isinstance(entry, HeatmapDecoder.HeatEntry):
                heat_count += 1
                assert entry.hex_id is not None
                assert len(entry.hex_id) == 6  # 6-character hex ID

                # Validate latitude and longitude ranges
                assert -90 <= entry.lat <= 90
                assert -180 <= entry.lon <= 180

                # Validate altitude
                if entry.alt is not None and entry.alt != "ground":
                    assert isinstance(entry.alt, int)
                    assert entry.alt >= -1

                # Validate ground speed
                if entry.ground_speed is not None:
                    assert 0 <= entry.ground_speed <= 1000  # Reasonable speed range

            elif isinstance(entry, HeatmapDecoder.CallsignEntry):
                callsign_count += 1
                assert entry.hex_id is not None
                assert len(entry.hex_id) == 6

            # Limit processing to avoid long test times
            if entry_count >= 1000:
                break

        # Verify we processed some entries
        assert entry_count > 0
        assert heat_count > 0  # Should have at least some heat entries

        print(f"Processed {entry_count} entries:")
        print(f"  - Timestamp separators: {timestamp_count}")
        print(f"  - Heat entries: {heat_count}")
        print(f"  - Callsign entries: {callsign_count}")

    def test_decode_real_file_memory_efficiency(self):
        """Test that decoder processes file efficiently without loading everything into memory."""
        # This test verifies the generator works correctly
        decoder_gen = self.decoder.decode_from_file(self.test_file_path)

        # Get first entry
        first_entry = next(decoder_gen)
        assert first_entry is not None

        # Get a few more entries to ensure generator continues working
        for i, entry in enumerate(decoder_gen):
            if i >= 10:  # Just test first 10 additional entries
                break
            assert entry is not None
