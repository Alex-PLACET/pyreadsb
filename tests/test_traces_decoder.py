from datetime import datetime, timedelta
from pathlib import Path

import pytest

from pyreadsb.traces_decoder import get_aircraft_record, process_traces_from_file


class TestGetAircraftRecord:
    @pytest.fixture
    def test_data_path(self):
        """Fixture providing path to test data file."""
        return Path(__file__).parent / "resources" / "trace_full_ac134a.json"

    def test_get_aircraft_record_with_real_data(self, test_data_path):
        """Test extraction of aircraft record using real test data."""
        result = get_aircraft_record(test_data_path)

        assert result.icao == "ac134a"
        assert result.r == "N8774Q"
        assert result.t == "B38M"
        assert result.db_flags == 0
        assert result.description == "BOEING 737 MAX 8"
        assert result.own_op == "SOUTHWEST AIRLINES CO"
        assert result.year == 2023
        assert result.timestamp == datetime.fromtimestamp(1723420800.000)

    def test_get_aircraft_record_file_not_found(self):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError):
            get_aircraft_record(Path("nonexistent_file.json"))


class TestProcessTraces:
    @pytest.fixture
    def test_data_path(self):
        """Fixture providing path to test data file."""
        return Path(__file__).parent / "resources" / "trace_full_ac134a.json"

    def test_process_traces_with_real_data(self, test_data_path):
        """Test processing of traces using real test data."""
        traces = list(process_traces_from_file(test_data_path))

        # Check that we get the expected number of traces
        assert len(traces) > 0

        # Test first trace
        first_trace = traces[0]
        assert first_trace.latitude == 40.134593
        assert first_trace.longitude == -75.330817
        assert first_trace.altitude == 36000
        assert first_trace.ground_speed == 384.3
        assert first_trace.track == 263.7
        assert first_trace.flags == 0
        assert first_trace.vertical_rate == 0
        assert first_trace.aircraft is None
        assert first_trace.source == "adsb_icao"
        assert first_trace.geometric_altitude == 37375
        assert first_trace.geometric_vertical_rate is None
        assert first_trace.indicated_airspeed is None
        assert first_trace.roll_angle is None
        assert first_trace.timestamp == datetime.fromtimestamp(1723420800.000 + 5.06)

        # Test a trace with aircraft data (4th trace, index 3)
        aircraft_trace = traces[3]
        assert aircraft_trace.aircraft is not None
        assert isinstance(aircraft_trace.aircraft, dict)
        assert aircraft_trace.aircraft["type"] == "adsb_icao"
        assert aircraft_trace.aircraft["flight"] == "SWA506  "

        # Test a trace with negative vertical rate (2nd trace, index 1)
        negative_vr_trace = traces[1]
        assert negative_vr_trace.vertical_rate == -64

        # Test a trace with positive vertical rate
        positive_vr_traces = [
            t for t in traces if t.vertical_rate and t.vertical_rate > 0
        ]
        assert len(positive_vr_traces) > 0
        assert positive_vr_traces[0].vertical_rate == 64

    def test_process_traces_timestamp_calculation(self, test_data_path):
        """Test that timestamps are calculated correctly."""
        traces = list(process_traces_from_file(test_data_path))

        base_timestamp = datetime.fromtimestamp(1723420800.000)

        # Check that timestamps increase properly
        for i, trace in enumerate(traces[:5]):  # Check first 5 traces
            expected_offsets = [5.06, 24.91, 31.52, 36.24, 41.05]
            expected_timestamp = base_timestamp.replace(microsecond=0) + timedelta(
                seconds=expected_offsets[i]
            )
            # Allow for small floating point differences
            time_diff = abs((trace.timestamp - expected_timestamp).total_seconds())
            assert time_diff < 0.1, f"Timestamp mismatch for trace {i}"

    def test_process_traces_file_not_found(self):
        """Test handling of missing file."""
        with pytest.raises(FileNotFoundError):
            list(process_traces_from_file(Path("nonexistent_file.json")))

    def test_process_traces_data_integrity(self, test_data_path):
        """Test that all traces have valid data types."""
        traces = list(process_traces_from_file(test_data_path))

        for i, trace in enumerate(traces[:10]):  # Check first 10 traces
            assert isinstance(trace.latitude, float), (
                f"Trace {i}: latitude should be float"
            )
            assert isinstance(trace.longitude, float), (
                f"Trace {i}: longitude should be float"
            )
            assert isinstance(trace.altitude, int), f"Trace {i}: altitude should be int"
            assert isinstance(trace.ground_speed, float), (
                f"Trace {i}: ground_speed should be float"
            )
            assert trace.track is None or isinstance(trace.track, float), (
                f"Trace {i}: track should be float or None"
            )
            assert isinstance(trace.flags, int), f"Trace {i}: flags should be int"
            assert trace.vertical_rate is None or isinstance(
                trace.vertical_rate, int
            ), f"Trace {i}: vertical_rate should be int or None"
            assert trace.aircraft is None or isinstance(trace.aircraft, dict), (
                f"Trace {i}: aircraft should be dict or None"
            )
            assert trace.source is None or isinstance(trace.source, str), (
                f"Trace {i}: source should be str or None"
            )
            assert isinstance(trace.timestamp, datetime), (
                f"Trace {i}: timestamp should be datetime"
            )

    def test_process_traces_ground_altitude_handling(self, test_data_path):
        """Test that ground altitude is handled correctly."""
        # Add a trace with ground altitude to test data if needed
        # For now, test with regular altitudes since our test data doesn't have "ground"
        traces = list(process_traces_from_file(test_data_path))

        # All traces in our test data should have numeric altitudes
        for trace in traces[:5]:
            assert isinstance(trace.altitude, int)
            assert trace.altitude > 0  # All our test data is at cruise altitude
