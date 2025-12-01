import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from .heatmap_decoder import HeatmapDecoder

logger: Final[logging.Logger] = logging.getLogger(__name__)


def convert_to_dataframes(
    entries: Iterable[
        HeatmapDecoder.HeatEntry
        | HeatmapDecoder.CallsignEntry
        | HeatmapDecoder.TimestampSeparator
    ],
    start_timestamp: datetime,
    batch_size: int = 10000,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convert entries to separate Polars DataFrames with batched processing."""
    # Collect data in lists (much faster than row-by-row DataFrame concat)
    heat_data: list[dict] = []
    callsign_dict: dict[str, str | None] = {}  # Deduplicate by hex_id

    current_timestamp: datetime = start_timestamp

    for entry in entries:
        if isinstance(entry, HeatmapDecoder.TimestampSeparator):
            # Use the decoded timestamp directly (already a datetime)
            current_timestamp = entry.timestamp
        elif isinstance(entry, HeatmapDecoder.HeatEntry):
            heat_data.append(
                {
                    "hex_id": entry.hex_id,
                    "lat": entry.lat,
                    "lon": entry.lon,
                    "alt": entry.alt,
                    "ground_speed": entry.ground_speed,
                    "timestamp": current_timestamp,
                }
            )
        elif isinstance(entry, HeatmapDecoder.CallsignEntry):
            # Keep last occurrence of each hex_id
            callsign_dict[entry.hex_id] = entry.callsign

    # Build DataFrames from collected data
    heat_schema = {
        "hex_id": pl.String,
        "lat": pl.Float32,
        "lon": pl.Float32,
        "alt": pl.Int32,
        "ground_speed": pl.Float32,
        "timestamp": pl.Datetime("ms", "UTC"),
    }

    callsign_schema = {
        "hex_id": pl.String,
        "callsign": pl.String,
    }

    heat_df: pl.DataFrame
    callsign_df: pl.DataFrame

    if heat_data:
        heat_df = pl.DataFrame(heat_data, schema=heat_schema)  # type: ignore[arg-type]
    else:
        heat_df = pl.DataFrame(schema=heat_schema)  # type: ignore[arg-type]

    if callsign_dict:
        callsign_data = [
            {"hex_id": hex_id, "callsign": callsign}
            for hex_id, callsign in callsign_dict.items()
        ]
        callsign_df = pl.DataFrame(callsign_data, schema=callsign_schema)
    else:
        callsign_df = pl.DataFrame(schema=callsign_schema)

    return heat_df, callsign_df


def export_to_parquet(
    entries: Iterable[
        HeatmapDecoder.HeatEntry
        | HeatmapDecoder.CallsignEntry
        | HeatmapDecoder.TimestampSeparator
    ],
    output_path: Path,
) -> None:
    """Export decoded entries to separate Parquet files."""

    # Convert list to DataFrames (backward compatibility)
    heat_data = []
    callsign_data = []
    callsign_dict = {}  # To deduplicate callsigns

    for entry in entries:
        if isinstance(entry, HeatmapDecoder.HeatEntry):
            heat_data.append(
                {
                    "hex_id": entry.hex_id,
                    "lat": entry.lat,
                    "lon": entry.lon,
                    "alt": entry.alt,
                    "ground_speed": entry.ground_speed,
                }
            )
        elif isinstance(entry, HeatmapDecoder.CallsignEntry):
            # Keep last occurrence of each hex_id
            callsign_dict[entry.hex_id] = {
                "hex_id": entry.hex_id,
                "callsign": entry.callsign,
            }

    callsign_data = list(callsign_dict.values())

    # Create output paths
    heat_output = output_path.with_stem(f"{output_path.stem}_positions")
    callsign_output = output_path.with_stem(f"{output_path.stem}_callsigns")

    if heat_data:
        heat_df = pl.DataFrame(
            heat_data,
            schema={
                "hex_id": pl.String,
                "lat": pl.Float32,
                "lon": pl.Float32,
                "alt": pl.Int32,
                "ground_speed": pl.Float32,
            },
        )
        heat_df.write_parquet(heat_output, compression="brotli", use_pyarrow=True)
        logger.info(f"Exported {len(heat_data)} position entries to {heat_output}")

    if callsign_data:
        callsign_df = pl.DataFrame(
            callsign_data,
            schema={
                "hex_id": pl.String,
                "callsign": pl.String,
            },
        )
        callsign_df.write_parquet(
            callsign_output, compression="brotli", use_pyarrow=True
        )
        logger.info(
            f"Exported {len(callsign_data)} callsign entries to {callsign_output}"
        )
