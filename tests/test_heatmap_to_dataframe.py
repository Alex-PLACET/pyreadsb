import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl
import pytest

from pyreadsb.heatmap_decoder import HeatmapDecoder
from pyreadsb.heatmap_to_dataframe import convert_to_dataframes, export_to_parquet


class TestConvertToDataframes:
    """Test suite for convert_to_dataframes function."""

    def test_convert_empty_generator(self):
        """Test conversion with empty generator."""

        def empty_generator():
            return
            yield  # Unreachable, but makes it a generator

        start_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        heat_df, callsign_df = convert_to_dataframes(empty_generator(), start_timestamp)

        assert len(heat_df) == 0
        assert len(callsign_df) == 0
        assert heat_df.schema == {
            "hex_id": pl.String,
            "lat": pl.Float32,
            "lon": pl.Float32,
            "alt": pl.Int32,
            "ground_speed": pl.Float32,
            "timestamp": pl.Datetime("ms", "UTC"),
        }
        assert callsign_df.schema == {
            "hex_id": pl.String,
            "callsign": pl.String,
        }

    def test_convert_timestamp_separator_only(self):
        """Test conversion with only timestamp separators."""

        def timestamp_generator():
            yield HeatmapDecoder.TimestampSeparator(
                timestamp=3600.0,
                raw_data=b"\x00" * 16,  # 1 hour offset
            )
            yield HeatmapDecoder.TimestampSeparator(
                timestamp=7200.0,
                raw_data=b"\x00" * 16,  # 2 hour offset
            )

        start_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        heat_df, callsign_df = convert_to_dataframes(
            timestamp_generator(), start_timestamp
        )

        assert len(heat_df) == 0
        assert len(callsign_df) == 0

    def test_convert_heat_entries_only(self):
        """Test conversion with only heat entries."""

        def heat_generator():
            yield HeatmapDecoder.HeatEntry(
                hex_id="abc123",
                lat=37.7749,
                lon=-122.4194,
                alt=1000,
                ground_speed=250.5,
            )
            yield HeatmapDecoder.HeatEntry(
                hex_id="def456", lat=40.7128, lon=-74.0060, alt=2000, ground_speed=300.0
            )

        start_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        heat_df, callsign_df = convert_to_dataframes(heat_generator(), start_timestamp)

        assert len(heat_df) == 2
        assert len(callsign_df) == 0

        # Check data content
        assert heat_df["hex_id"].to_list() == ["abc123", "def456"]
        assert heat_df["lat"].to_list() == [
            pytest.approx(37.7749),
            pytest.approx(40.7128),
        ]
        assert heat_df["lon"].to_list() == [
            pytest.approx(-122.4194),
            pytest.approx(-74.0060),
        ]
        assert heat_df["alt"].to_list() == [1000, 2000]
        assert heat_df["ground_speed"].to_list() == [250.5, 300.0]

        # All timestamps should be the start timestamp
        expected_timestamps = [start_timestamp] * 2
        assert heat_df["timestamp"].to_list() == expected_timestamps

    def test_convert_callsign_entries_only(self):
        """Test conversion with only callsign entries."""

        def callsign_generator():
            yield HeatmapDecoder.CallsignEntry(hex_id="abc123", callsign="UAL123")
            yield HeatmapDecoder.CallsignEntry(hex_id="def456", callsign="DAL456")

        start_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        heat_df, callsign_df = convert_to_dataframes(
            callsign_generator(), start_timestamp
        )

        assert len(heat_df) == 0
        assert len(callsign_df) == 2

        # Check data content
        assert callsign_df["hex_id"].to_list() == ["abc123", "def456"]
        assert callsign_df["callsign"].to_list() == ["UAL123", "DAL456"]

    def test_convert_mixed_entries(self):
        """Test conversion with mixed entry types."""

        def mixed_generator():
            # Start with a timestamp
            yield HeatmapDecoder.TimestampSeparator(
                timestamp=3600.0,
                raw_data=b"\x00" * 16,  # 1 hour offset
            )

            # Add heat entry
            yield HeatmapDecoder.HeatEntry(
                hex_id="abc123",
                lat=37.7749,
                lon=-122.4194,
                alt=1000,
                ground_speed=250.5,
            )

            # Add callsign entry
            yield HeatmapDecoder.CallsignEntry(hex_id="abc123", callsign="UAL123")

            # Another timestamp
            yield HeatmapDecoder.TimestampSeparator(
                timestamp=7200.0,
                raw_data=b"\x00" * 16,  # 2 hour offset
            )

            # Another heat entry with new timestamp
            yield HeatmapDecoder.HeatEntry(
                hex_id="def456", lat=40.7128, lon=-74.0060, alt=2000, ground_speed=300.0
            )

        start_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        heat_df, callsign_df = convert_to_dataframes(mixed_generator(), start_timestamp)

        assert len(heat_df) == 2
        assert len(callsign_df) == 1

        # Check heat data
        assert heat_df["hex_id"].to_list() == ["abc123", "def456"]

        # Check timestamps are updated correctly
        first_timestamp = datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC)  # +1 hour
        second_timestamp = datetime(2024, 1, 1, 2, 0, 0, tzinfo=UTC)  # +2 hours
        assert heat_df["timestamp"].to_list() == [first_timestamp, second_timestamp]

        # Check callsign data
        assert callsign_df["hex_id"].to_list() == ["abc123"]
        assert callsign_df["callsign"].to_list() == ["UAL123"]

    def test_convert_duplicate_callsigns(self):
        """Test that duplicate callsigns are handled correctly (keeping last)."""

        def duplicate_callsign_generator():
            yield HeatmapDecoder.CallsignEntry(hex_id="abc123", callsign="UAL123")
            yield HeatmapDecoder.CallsignEntry(
                hex_id="abc123", callsign="UAL456"
            )  # Updated callsign
            yield HeatmapDecoder.CallsignEntry(hex_id="def456", callsign="DAL789")

        start_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        heat_df, callsign_df = convert_to_dataframes(
            duplicate_callsign_generator(), start_timestamp
        )

        assert len(callsign_df) == 2

        # Should keep the last occurrence of each hex_id
        hex_ids = callsign_df["hex_id"].to_list()
        callsigns = callsign_df["callsign"].to_list()

        abc123_idx = hex_ids.index("abc123")
        assert callsigns[abc123_idx] == "UAL456"  # Should be the updated callsign

    def test_convert_special_altitude_values(self):
        """Test conversion with special altitude values."""

        def special_altitude_generator():
            yield HeatmapDecoder.HeatEntry(
                hex_id="abc123",
                lat=37.7749,
                lon=-122.4194,
                alt="ground",
                ground_speed=0.0,
            )
            yield HeatmapDecoder.HeatEntry(
                hex_id="def456", lat=40.7128, lon=-74.0060, alt=None, ground_speed=None
            )

        start_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

        # This should raise an error due to mixed types in alt column
        with pytest.raises(
            (pl.SchemaError, pl.ComputeError, TypeError, ValueError)
        ):  # Polars will raise an error for mixed types
            convert_to_dataframes(special_altitude_generator(), start_timestamp)

    def test_convert_none_ground_speed(self):
        """Test conversion with None ground speed values."""

        def none_speed_generator():
            yield HeatmapDecoder.HeatEntry(
                hex_id="abc123", lat=37.7749, lon=-122.4194, alt=1000, ground_speed=None
            )

        start_timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        heat_df, callsign_df = convert_to_dataframes(
            none_speed_generator(), start_timestamp
        )

        assert len(heat_df) == 1
        assert heat_df["ground_speed"][0] is None


class TestExportToParquet:
    """Test suite for export_to_parquet function."""

    def test_export_list_entries(self):
        """Test export with list of entries (backward compatibility)."""
        entries = [
            HeatmapDecoder.HeatEntry(
                hex_id="abc123",
                lat=37.7749,
                lon=-122.4194,
                alt=1000,
                ground_speed=250.5,
            ),
            HeatmapDecoder.CallsignEntry(hex_id="abc123", callsign="UAL123"),
            HeatmapDecoder.HeatEntry(
                hex_id="def456", lat=40.7128, lon=-74.0060, alt=2000, ground_speed=300.0
            ),
            HeatmapDecoder.CallsignEntry(
                hex_id="abc123", callsign="UAL456"
            ),  # Duplicate, should keep last
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export.parquet"
            export_to_parquet(entries, output_path)

    def test_export_list_heat_entries_only(self):
        """Test export with list containing only heat entries."""
        entries: Final[list[HeatmapDecoder.HeatEntry]] = [
            HeatmapDecoder.HeatEntry(
                hex_id="abc123",
                lat=37.7749,
                lon=-122.4194,
                alt=1000,
                ground_speed=250.5,
            ),
            HeatmapDecoder.HeatEntry(
                hex_id="def456", lat=40.7128, lon=-74.0060, alt=2000, ground_speed=300.0
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export.parquet"
            export_to_parquet(entries, output_path)

    def test_export_list_callsign_entries_only(self):
        """Test export with list containing only callsign entries."""
        entries: Final[list[HeatmapDecoder.CallsignEntry]] = [
            HeatmapDecoder.CallsignEntry(hex_id="abc123", callsign="UAL123"),
            HeatmapDecoder.CallsignEntry(hex_id="def456", callsign="DAL456"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_export.parquet"
            export_to_parquet(entries, output_path)


class TestHeatmapToDataframeIntegration:
    """Integration tests combining decoder and dataframe functions."""

    def test_pipeline_with_real_file_sample(self):
        """Test pipeline with a small sample from the real file."""
        decoder = HeatmapDecoder()
        test_file_path = Path("tests/resources/16.bin.ttf")

        if not test_file_path.exists():
            pytest.skip("Test file not available")

        # Process only first 100 entries to keep test fast
        def limited_generator():
            count = 0
            for entry in decoder.decode_from_file(test_file_path):
                yield entry
                count += 1
                if count >= 100:
                    break

        start_timestamp = datetime(2024, 12, 30, 0, 0, 0, tzinfo=UTC)
        heat_df, callsign_df = convert_to_dataframes(
            limited_generator(), start_timestamp
        )

        # Basic validation
        assert isinstance(heat_df, pl.DataFrame)
        assert isinstance(callsign_df, pl.DataFrame)

        # Should have some data
        total_entries = len(heat_df) + len(callsign_df)
        assert total_entries > 0

        print(
            f"Processed {len(heat_df)} heat entries and {len(callsign_df)} callsign entries"
        )
