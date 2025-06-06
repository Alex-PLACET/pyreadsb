import logging
from collections.abc import Generator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from .heatmap_decoder import HeatmapDecoder

logger: logging.Logger = logging.getLogger("pyreadsb")


def convert_to_dataframes(
    entries: Generator[
        HeatmapDecoder.HeatEntry
        | HeatmapDecoder.CallsignEntry
        | HeatmapDecoder.TimestampSeparator,
        None,
        None,
    ],
    start_timestamp: datetime,
    batch_size: int = 10000,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Convert generator entries to separate Polars DataFrames with iterative processing."""
    # Initialize empty dataframes with proper schemas
    heat_df = pl.DataFrame(
        schema={
            "hex_id": pl.String,
            "lat": pl.Float32,
            "lon": pl.Float32,
            "alt": pl.Int32,
            "ground_speed": pl.Float32,
            "timestamp": pl.Datetime("ms", "UTC"),
        }
    )

    callsign_df = pl.DataFrame(
        schema={
            "hex_id": pl.String,
            "callsign": pl.String,
        }
    )

    current_timestamp: datetime = start_timestamp

    for entry in entries:
        if isinstance(entry, HeatmapDecoder.TimestampSeparator):
            # Convert float timestamp to datetime relative to start_timestamp
            offset_seconds = entry.timestamp
            current_timestamp = datetime.fromtimestamp(
                start_timestamp.timestamp() + offset_seconds, tz=UTC
            )
        elif isinstance(entry, HeatmapDecoder.HeatEntry):
            # Create new row with current timestamp and add to dataframe
            new_row = pl.DataFrame(
                {
                    "hex_id": [entry.hex_id],
                    "lat": [entry.lat],
                    "lon": [entry.lon],
                    "alt": [entry.alt],
                    "ground_speed": [entry.ground_speed],
                    "timestamp": [current_timestamp],
                },
                schema={
                    "hex_id": pl.String,
                    "lat": pl.Float32,
                    "lon": pl.Float32,
                    "alt": pl.Int32,
                    "ground_speed": pl.Float32,
                    "timestamp": pl.Datetime("ms", "UTC"),
                },
            )
            heat_df = pl.concat([heat_df, new_row], how="vertical")

        elif isinstance(entry, HeatmapDecoder.CallsignEntry):
            # Create new row and add to callsign dataframe
            new_row = pl.DataFrame(
                {
                    "hex_id": [entry.hex_id],
                    "callsign": [entry.callsign],
                },
                schema={
                    "hex_id": pl.String,
                    "callsign": pl.String,
                },
            )
            callsign_df = pl.concat([callsign_df, new_row], how="vertical")

    # Remove duplicates from callsign dataframe, keeping the last occurrence
    if len(callsign_df) > 0:
        callsign_df = callsign_df.unique(subset=["hex_id"], keep="last")

    return heat_df, callsign_df


def export_to_parquet(
    entries: Sequence[
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
