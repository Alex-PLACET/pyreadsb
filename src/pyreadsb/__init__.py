"""pyreadsb - Python library for decoding readsb data formats."""

from .compression_utils import detect_compression, open_file
from .heatmap_decoder import HeatmapDecoder
from .heatmap_to_dataframe import convert_to_dataframes, export_to_parquet
from .traces_decoder import (
    TRACE_FLAG_ALTITUDE_GEOMETRIC,
    TRACE_FLAG_NEW_LEG,
    TRACE_FLAG_STALE,
    TRACE_FLAG_VERTICAL_RATE_GEOMETRIC,
    TRACE_FLAGS,
    AircraftRecord,
    TraceEntry,
    get_aircraft_record,
    process_traces_from_file,
    process_traces_from_json_bytes,
)

__all__ = [
    # Heatmap decoder
    "HeatmapDecoder",
    "convert_to_dataframes",
    "export_to_parquet",
    # Traces decoder
    "AircraftRecord",
    "TraceEntry",
    "get_aircraft_record",
    "process_traces_from_file",
    "process_traces_from_json_bytes",
    "TRACE_FLAG_STALE",
    "TRACE_FLAG_NEW_LEG",
    "TRACE_FLAG_VERTICAL_RATE_GEOMETRIC",
    "TRACE_FLAG_ALTITUDE_GEOMETRIC",
    "TRACE_FLAGS",
    # Compression utilities
    "detect_compression",
    "open_file",
]

__version__ = "0.1.0"
