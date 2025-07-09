from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import ijson

from .compression_utils import open_file


@dataclass
class AircraftRecord:
    """Dataclass to hold aircraft record information."""

    icao: str
    r: str
    t: str
    db_flags: int
    description: str
    own_op: str
    year: int | None
    timestamp: datetime


@dataclass
class TraceEntry:
    """Dataclass to hold trace entry information."""

    latitude: float
    longitude: float
    altitude: int
    ground_speed: float | None
    track: float | None
    flags: int
    vertical_rate: int | None
    aircraft: dict[str, Any]
    source: str | None
    geometric_altitude: int | None
    geometric_vertical_rate: int | None
    indicated_airspeed: int | None
    roll_angle: int | None
    timestamp: datetime


TRACE_FLAG_STALE: Final[int] = 1
TRACE_FLAG_NEW_LEG: Final[int] = 2
TRACE_FLAG_VERTICAL_RATE_GEOMETRIC: Final[int] = 4
TRACE_FLAG_ALTITUDE_GEOMETRIC: Final[int] = 8

TRACE_FLAGS = {
    TRACE_FLAG_STALE,
    TRACE_FLAG_NEW_LEG,
    TRACE_FLAG_VERTICAL_RATE_GEOMETRIC,
    TRACE_FLAG_ALTITUDE_GEOMETRIC,
}


def get_aircraft_record(trace_file: Path) -> AircraftRecord:
    """Extract aircraft record from a gzipped JSON file."""
    with open_file(trace_file) as f:
        # Parse the top-level JSON object
        data = {}
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if event in ("string", "number", "boolean", "null"):
                data[prefix] = value

        return AircraftRecord(
            icao=data["icao"],
            r=data["r"],
            t=data["t"],
            db_flags=data["dbFlags"],
            description=data["desc"],
            own_op=data["ownOp"],
            year=(
                0
                if data.get("year") == "0000"
                else int(data.get("year", 0))
                if data.get("year")
                else None
            ),
            timestamp=datetime.fromtimestamp(float(data["timestamp"])),
        )


def _parse_timestamp_from_source(json_source: Any) -> datetime:
    """Parse timestamp from a JSON source (file handle or bytes)."""
    parser = ijson.parse(json_source)
    timestamp_value = None
    for prefix, event, value in parser:
        if prefix == "timestamp" and event == "number":
            timestamp_value = value
            break

    if timestamp_value is None:
        raise ValueError("No timestamp found in JSON")

    return datetime.fromtimestamp(float(timestamp_value))


def _create_trace_entry(trace: list[Any], timestamp_dt: datetime) -> TraceEntry:
    """Create a TraceEntry from trace data and base timestamp."""
    second_after_timestamp: float = float(trace[0])
    altitude = trace[3] if trace[3] != "ground" else -1

    return TraceEntry(
        latitude=float(trace[1]),
        longitude=float(trace[2]),
        altitude=altitude,
        ground_speed=float(trace[4]) if trace[4] is not None else None,
        track=float(trace[5]) if trace[5] is not None else None,
        flags=int(trace[6]),
        vertical_rate=int(trace[7]) if trace[7] is not None else None,
        aircraft=trace[8],
        source=trace[9],
        geometric_altitude=int(trace[10]) if trace[10] is not None else None,
        geometric_vertical_rate=int(trace[11]) if trace[11] is not None else None,
        indicated_airspeed=int(trace[12]) if trace[12] is not None else None,
        roll_angle=int(trace[13]) if trace[13] is not None else None,
        timestamp=timestamp_dt + timedelta(seconds=second_after_timestamp),
    )


def process_traces_from_json_bytes(trace_bytes: bytes) -> Generator[TraceEntry]:
    """Process traces from JSON bytes."""
    timestamp_dt: Final[datetime] = _parse_timestamp_from_source(trace_bytes)

    # Parse traces
    traces = ijson.items(trace_bytes, "trace.item")
    for trace in traces:
        yield _create_trace_entry(trace, timestamp_dt)


def process_traces_from_file(trace_file: Path) -> Generator[TraceEntry]:
    """Process traces from a gzipped JSON file."""
    with open_file(trace_file) as f:
        timestamp_dt: Final[datetime] = _parse_timestamp_from_source(f)

        # Reset file and parse traces
        f.seek(0)
        traces = ijson.items(f, "trace.item")
        for trace in traces:
            yield _create_trace_entry(trace, timestamp_dt)
