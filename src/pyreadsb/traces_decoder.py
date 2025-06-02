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
    ground_speed: float
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


def get_aircraft_record(trace_file: Path) -> AircraftRecord:
    """Extract aircraft record from a gzipped JSON file."""
    with open_file(trace_file) as f:
        data = ijson.items(f, "item")
        return AircraftRecord(
            icao=data["icao"],
            r=data["r"],
            t=data["t"],
            db_flags=data["db_flags"],
            description=data["description"],
            own_op=data["own_op"],
            year=(
                0
                if data.get("year") == "0000"
                else int(data.get("year", 0))
                if data.get("year")
                else None
            ),
            timestamp=datetime.fromtimestamp(data["timestamp_casted"]),
        )


def process_traces(trace_file: Path) -> Generator[TraceEntry, Any, None]:
    """Process traces from a gzipped JSON file."""
    with open_file(trace_file) as f:
        data_timestamp = ijson.items(f, "item.timestamp")
        timestamp_dt: Final[datetime] = datetime.fromtimestamp(data_timestamp)
        traces = ijson.items(f, "item.traces")
        for trace in traces:
            second_after_timestamp: float = trace[0]
            altitude = trace[3] if trace[3] != "ground" else -1
            yield TraceEntry(
                latitude=trace[1],
                longitude=trace[2],
                altitude=altitude,
                ground_speed=trace[4],
                track=trace[5],
                flags=trace[6],
                vertical_rate=trace[7],
                aircraft=trace[8],
                source=trace[9],
                geometric_altitude=trace[10],
                geometric_vertical_rate=trace[11],
                indicated_airspeed=trace[12],
                roll_angle=trace[13],
                timestamp=timestamp_dt + timedelta(seconds=second_after_timestamp),
            )
