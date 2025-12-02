from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import ijson

from .compression_utils import open_file


@dataclass(slots=True)
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


@dataclass(slots=True)
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

TRACE_FLAGS = frozenset(
    {
        TRACE_FLAG_STALE,
        TRACE_FLAG_NEW_LEG,
        TRACE_FLAG_VERTICAL_RATE_GEOMETRIC,
        TRACE_FLAG_ALTITUDE_GEOMETRIC,
    }
)


def get_aircraft_record(trace_file: Path) -> AircraftRecord:
    """Extract aircraft record from a gzipped JSON file."""
    # Fields we need to extract
    required_fields = {
        "icao",
        "r",
        "t",
        "dbFlags",
        "desc",
        "ownOp",
        "year",
        "timestamp",
    }

    with open_file(trace_file) as f:
        data: dict[str, Any] = {}
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if prefix in required_fields and event in (
                "string",
                "number",
                "boolean",
                "null",
            ):
                data[prefix] = value
                # Early exit once we have all required fields
                if len(data) == len(required_fields):
                    break

        year_val = data.get("year")
        return AircraftRecord(
            icao=data["icao"],
            r=data["r"],
            t=data["t"],
            db_flags=data["dbFlags"],
            description=data["desc"],
            own_op=data["ownOp"],
            year=None if year_val is None or year_val == "0000" else int(year_val),
            timestamp=datetime.fromtimestamp(float(data["timestamp"]), tz=UTC),
        )


def _create_trace_entry(trace: list[Any], timestamp_dt: datetime) -> TraceEntry:
    """Create a TraceEntry from trace data and base timestamp."""
    offset_seconds: float = float(trace[0])
    altitude: int = trace[3] if trace[3] != "ground" else -1

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
        timestamp=timestamp_dt + timedelta(seconds=offset_seconds),
    )


def process_traces_from_json_bytes(trace_bytes: bytes) -> Generator[TraceEntry]:
    """Process traces from JSON bytes."""
    # Use kvitems to get timestamp and trace in single pass
    timestamp_dt: datetime | None = None

    for key, value in ijson.kvitems(trace_bytes, ""):
        if key == "timestamp":
            timestamp_dt = datetime.fromtimestamp(float(value), tz=UTC)
        elif key == "trace" and timestamp_dt is not None:
            for trace in value:
                yield _create_trace_entry(trace, timestamp_dt)
            return

    if timestamp_dt is None:
        raise ValueError("No timestamp found in JSON")


def process_traces_from_file(trace_file: Path) -> Generator[TraceEntry]:
    """Process traces from a gzipped JSON file."""
    with open_file(trace_file) as f:
        # Use kvitems to get timestamp and trace in single pass
        timestamp_dt: datetime | None = None

        for key, value in ijson.kvitems(f, ""):
            if key == "timestamp":
                timestamp_dt = datetime.fromtimestamp(float(value), tz=UTC)
            elif key == "trace" and timestamp_dt is not None:
                for trace in value:
                    yield _create_trace_entry(trace, timestamp_dt)
                return

        if timestamp_dt is None:
            raise ValueError("No timestamp found in JSON")
