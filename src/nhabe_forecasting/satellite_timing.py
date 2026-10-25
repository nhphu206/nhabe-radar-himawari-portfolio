"""Read bounded HSD headers; never modify/decode full satellite image products."""

from __future__ import annotations

import bz2
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_header_timing(data: bytes) -> dict[str, Any]:
    """Read Blocks 1/5/7/9 using the pinned SatPy dtype definitions."""
    from satpy.readers import ahi_hsd

    types = {1: ahi_hsd._BASIC_INFO_TYPE, 5: ahi_hsd._CAL_INFO_TYPE,
             7: ahi_hsd._SEGMENT_INFO_TYPE, 9: ahi_hsd._OBSERVATION_TIME_INFO_TYPE}
    blocks: dict[int, Any] = {}
    offset = 0
    for expected in range(1, 10):
        if offset + 3 > len(data):
            raise ValueError("Truncated HSD block header")
        number = data[offset]
        length = int.from_bytes(data[offset + 1:offset + 3], "little")
        if number != expected or length < 3 or offset + length > len(data):
            raise ValueError("Invalid HSD block sequence/length")
        if number in types:
            dtype = types[number]
            if length < dtype.itemsize:
                raise ValueError("HSD block smaller than its dtype")
            blocks[number] = np.frombuffer(data, dtype=dtype, count=1, offset=offset)[0]
        if number == 9:
            count = int(blocks[9]["number_of_observation_times"])
            dtype = ahi_hsd._OBSERVATION_LINE_TIME_INFO_TYPE
            start = offset + types[9].itemsize
            if count < 1 or start + count * dtype.itemsize > offset + length:
                raise ValueError("Invalid Block 9 record count")
            observations = np.frombuffer(data, dtype=dtype, count=count, offset=start)
        offset += length
    basic = blocks[1]
    if int(basic["byte_order"]) != 0:
        raise ValueError("Only little-endian HSD is supported")
    epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)

    def timestamp(value: Any) -> datetime:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Non-finite HSD timestamp")
        return epoch + timedelta(days=value)

    start = timestamp(basic["observation_start_time"])
    end = timestamp(basic["observation_end_time"])
    creation = timestamp(basic["file_creation_time"])
    line_start = timestamp(np.min(observations["observation_time"]))
    line_end = timestamp(np.max(observations["observation_time"]))
    if not start <= line_start <= line_end <= end <= creation:
        raise ValueError("Inconsistent observation/creation timestamps")
    if int(blocks[7]["total_number_of_segments"]) != 10:
        raise ValueError("Expected ten full-disk segments")
    result = {
        "band": f"B{int(blocks[5]['band_number']):02d}",
        "segment": f"{int(blocks[7]['segment_sequence_number']):02d}",
        "satellite": bytes(basic["satellite"]).decode("ascii").strip("\x00 "),
        "observation_area": bytes(basic["observation_area"]).decode("ascii").strip("\x00 "),
        "observation_timeline": int(basic["observation_timeline"]),
        "header_through_block9_sha256": hashlib.sha256(data[:offset]).hexdigest(),
    }
    for name, value in (("observation_start_utc", start), ("observation_end_utc", end),
                        ("file_creation_utc", creation)):
        result[name] = value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return result


def read_header_timing(path: Path) -> dict[str, Any]:
    """Read at most 1 MB compressed and expose at most 64 KiB decompressed."""
    with path.open("rb") as handle:
        prefix = handle.read(1_000_000)
    data = bz2.BZ2Decompressor().decompress(prefix, max_length=65_536)
    result = parse_header_timing(data)
    result["file_size_bytes"] = path.stat().st_size
    return result
