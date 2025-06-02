from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyreadsb.traces_decoder import get_aircraft_record, process_traces


class TestGetAircraftRecord:
    def test_get_aircraft_record_success(self):
        """Test successful extraction of aircraft record."""
        mock_data = {
            "icao": "ABC123",
            "r": "registration",
            "t": "type",
            "db_flags": 1,
            "description": "Test Aircraft",
            "own_op": "Test Operator",
            "year": "2020",
            "timestamp_casted": 1609459200.0,  # 2021-01-01 00:00:00
        }

        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.return_value = mock_data

            result = get_aircraft_record(Path("test.json.gz"))

            assert result.icao == "ABC123"
            assert result.r == "registration"
            assert result.t == "type"
            assert result.db_flags == 1
            assert result.description == "Test Aircraft"
            assert result.own_op == "Test Operator"
            assert result.year == 2020
            assert result.timestamp == datetime.fromtimestamp(1609459200.0)

    def test_get_aircraft_record_year_0000(self):
        """Test handling of year '0000' case."""
        mock_data = {
            "icao": "ABC123",
            "r": "registration",
            "t": "type",
            "db_flags": 1,
            "description": "Test Aircraft",
            "own_op": "Test Operator",
            "year": "0000",
            "timestamp_casted": 1609459200.0,
        }

        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.return_value = mock_data

            result = get_aircraft_record(Path("test.json.gz"))
            assert result.year == 0

    def test_get_aircraft_record_year_none(self):
        """Test handling of missing year field."""
        mock_data = {
            "icao": "ABC123",
            "r": "registration",
            "t": "type",
            "db_flags": 1,
            "description": "Test Aircraft",
            "own_op": "Test Operator",
            "timestamp_casted": 1609459200.0,
        }

        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.return_value = mock_data

            result = get_aircraft_record(Path("test.json.gz"))
            assert result.year is None

    def test_get_aircraft_record_year_empty_string(self):
        """Test handling of empty string year."""
        mock_data = {
            "icao": "ABC123",
            "r": "registration",
            "t": "type",
            "db_flags": 1,
            "description": "Test Aircraft",
            "own_op": "Test Operator",
            "year": "",
            "timestamp_casted": 1609459200.0,
        }

        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.return_value = mock_data

            result = get_aircraft_record(Path("test.json.gz"))
            assert result.year is None

    def test_get_aircraft_record_file_error(self):
        """Test handling of file opening errors."""
        with patch("pyreadsb.traces_decoder.open_file") as mock_open_file:
            mock_open_file.side_effect = FileNotFoundError("File not found")

            with pytest.raises(FileNotFoundError):
                get_aircraft_record(Path("nonexistent.json.gz"))

    def test_get_aircraft_record_json_error(self):
        """Test handling of JSON parsing errors."""
        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.side_effect = KeyError("Missing required field")

            with pytest.raises(KeyError):
                get_aircraft_record(Path("invalid.json.gz"))


class TestProcessTraces:
    def test_process_traces_success(self):
        """Test successful processing of traces."""
        mock_timestamp = 1609459200.0  # 2021-01-01 00:00:00
        mock_traces = [
            [
                5.0,
                40.7128,
                -74.0060,
                35000,
                120.5,
                180.0,
                1,
                0,
                {"key": "value"},
                "source1",
                35100,
                50,
                250,
                0,
            ],
            [
                10.0,
                40.7130,
                -74.0062,
                "ground",
                0.0,
                None,
                2,
                None,
                {"key": "value2"},
                None,
                None,
                None,
                None,
                None,
            ],
        ]

        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.side_effect = [mock_timestamp, mock_traces]

            traces = list(process_traces(Path("test.json.gz")))

            assert len(traces) == 2

            # First trace
            assert traces[0].latitude == 40.7128
            assert traces[0].longitude == -74.0060
            assert traces[0].altitude == 35000
            assert traces[0].ground_speed == 120.5
            assert traces[0].track == 180.0
            assert traces[0].flags == 1
            assert traces[0].vertical_rate == 0
            assert traces[0].aircraft == {"key": "value"}
            assert traces[0].source == "source1"
            assert traces[0].geometric_altitude == 35100
            assert traces[0].geometric_vertical_rate == 50
            assert traces[0].indicated_airspeed == 250
            assert traces[0].roll_angle == 0
            assert traces[0].timestamp == datetime.fromtimestamp(1609459200.0 + 5.0)

            # Second trace with "ground" altitude and None values
            assert traces[1].altitude == -1  # "ground" converted to -1
            assert traces[1].track is None
            assert traces[1].vertical_rate is None
            assert traces[1].source is None
            assert traces[1].geometric_altitude is None
            assert traces[1].geometric_vertical_rate is None
            assert traces[1].indicated_airspeed is None
            assert traces[1].roll_angle is None
            assert traces[1].timestamp == datetime.fromtimestamp(1609459200.0 + 10.0)

    def test_process_traces_empty_traces(self):
        """Test processing with empty traces array."""
        mock_timestamp = 1609459200.0
        mock_traces = []

        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.side_effect = [mock_timestamp, mock_traces]

            traces = list(process_traces(Path("test.json.gz")))
            assert len(traces) == 0

    def test_process_traces_ground_altitude(self):
        """Test handling of 'ground' altitude specifically."""
        mock_timestamp = 1609459200.0
        mock_traces = [
            [
                0.0,
                40.0,
                -74.0,
                "ground",
                0.0,
                0.0,
                0,
                0,
                {},
                None,
                None,
                None,
                None,
                None,
            ]
        ]

        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.side_effect = [mock_timestamp, mock_traces]

            traces = list(process_traces(Path("test.json.gz")))
            assert traces[0].altitude == -1

    def test_process_traces_file_error(self):
        """Test handling of file opening errors."""
        with patch("pyreadsb.traces_decoder.open_file") as mock_open_file:
            mock_open_file.side_effect = FileNotFoundError("File not found")

            with pytest.raises(FileNotFoundError):
                list(process_traces(Path("nonexistent.json.gz")))

    def test_process_traces_json_error(self):
        """Test handling of JSON parsing errors."""
        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.side_effect = ValueError("Invalid JSON")

            with pytest.raises(ValueError):
                list(process_traces(Path("invalid.json.gz")))

    def test_process_traces_timestamp_calculation(self):
        """Test timestamp calculation with seconds offset."""
        mock_timestamp = 1609459200.0  # 2021-01-01 00:00:00
        mock_traces = [
            [
                15.5,
                40.0,
                -74.0,
                1000,
                100.0,
                90.0,
                0,
                0,
                {},
                None,
                None,
                None,
                None,
                None,
            ]
        ]

        with (
            patch("pyreadsb.traces_decoder.open_file") as mock_open_file,
            patch("ijson.items") as mock_ijson_items,
        ):
            mock_file = MagicMock()
            mock_open_file.return_value.__enter__.return_value = mock_file
            mock_ijson_items.side_effect = [mock_timestamp, mock_traces]

            traces = list(process_traces(Path("test.json.gz")))
            expected_timestamp = datetime.fromtimestamp(1609459200.0 + 15.5)
            assert traces[0].timestamp == expected_timestamp
