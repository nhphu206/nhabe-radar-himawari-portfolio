"""Đọc HSD, thời gian dòng quét và nhiệt độ sáng Himawari--9."""

from __future__ import annotations

import bz2
import gc
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from satpy import DataQuery, Scene
from satpy.readers import ahi_hsd

from .common_grid import CommonGrid


UTC = timezone.utc
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)


def iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def hsd_filename(date: str, cycle: str, band: str, segment: str) -> str:
    return f"HS_H09_{date}_{cycle}_{band}_FLDK_R20_S{segment}10.DAT.bz2"


def cycle_files(root: Path, date: str, cycle: str, config: dict[str, Any]) -> list[Path]:
    files: list[Path] = []
    for band in config["himawari"]["bands"]:
        for segment in config["himawari"]["segments"]:
            path = root / date / cycle / hsd_filename(date, cycle, band, segment)
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(path)
    return files


def first_bzip_stream(compressed: bytes) -> bytes:
    decompressor = bz2.BZ2Decompressor()
    value = decompressor.decompress(compressed)
    if not decompressor.eof:
        raise EOFError("Dòng bzip2 đầu tiên chưa hoàn chỉnh")
    return value


def parse_hsd_header(
    data: bytes,
    date: str,
    cycle: str,
    full_disk_line: float,
) -> dict[str, Any]:
    """Đọc Block 1, 7 và 9 để lấy thời gian quan trắc tại Nhà Bè."""
    offset = 0
    block1 = None
    observations = None
    segment = None
    for expected_block in range(1, 10):
        if len(data) < offset + 3:
            raise ValueError("HSD header bị cắt ngắn")
        block_number = data[offset]
        block_length = int.from_bytes(data[offset + 1:offset + 3], "little")
        if block_number != expected_block or block_length <= 0:
            raise ValueError(f"HSD block không hợp lệ tại byte {offset}")
        if expected_block == 1:
            block1 = np.frombuffer(data, dtype=ahi_hsd._BASIC_INFO_TYPE, count=1, offset=offset)[0]
        elif expected_block == 7:
            segment = np.frombuffer(
                data, dtype=ahi_hsd._SEGMENT_INFO_TYPE, count=1, offset=offset
            )[0]
        elif expected_block == 9:
            header = np.frombuffer(
                data, dtype=ahi_hsd._OBSERVATION_TIME_INFO_TYPE, count=1, offset=offset
            )[0]
            count = int(header["number_of_observation_times"])
            start = offset + ahi_hsd._OBSERVATION_TIME_INFO_TYPE.itemsize
            observations = np.frombuffer(
                data,
                dtype=ahi_hsd._OBSERVATION_LINE_TIME_INFO_TYPE,
                count=count,
                offset=start,
            )
        offset += block_length
    if block1 is None or segment is None or observations is None or observations.size == 0:
        raise ValueError("Thiếu HSD Block 1, 7 hoặc 9")
    lines = np.asarray(observations["line_number"], dtype=np.float64)
    times = np.asarray(observations["observation_time"], dtype=np.float64)
    if not float(lines.min()) <= full_disk_line <= float(lines.max()):
        raise ValueError("Dòng Nhà Bè nằm ngoài phân đoạn thời gian")
    local_mjd = float(np.interp(full_disk_line, lines, times))
    nominal = datetime.strptime(date + cycle, "%Y%m%d%H%M").replace(tzinfo=UTC)
    observation_start = MJD_EPOCH + timedelta(days=float(block1["observation_start_time"]))
    observation_end = MJD_EPOCH + timedelta(days=float(block1["observation_end_time"]))
    local = MJD_EPOCH + timedelta(days=local_mjd)
    if not observation_start <= local <= observation_end:
        raise ValueError("Thời gian dòng Nhà Bè nằm ngoài khoảng quan trắc")
    return {
        "date": date,
        "cycle": cycle,
        "nominal_start_utc": iso_utc(nominal),
        "observation_start_utc": iso_utc(observation_start),
        "observation_end_utc": iso_utc(observation_end),
        "local_nhabe_line_time_utc": iso_utc(local),
        "local_minus_nominal_seconds": float((local - nominal).total_seconds()),
        "local_nhabe_full_disk_line_one_based": float(full_disk_line),
        "segment_sequence_number": int(segment["segment_sequence_number"]),
        "total_segments": int(segment["total_number_of_segments"]),
    }


def read_local_observation_time(
    path: Path,
    date: str,
    cycle: str,
    full_disk_line: float,
    maximum_compressed_header_bytes: int = 1_000_000,
) -> dict[str, Any]:
    """Đọc HSD Block 9 từ tệp đầy đủ mà không giải nén phần ảnh."""
    with path.open("rb") as handle:
        compressed = handle.read(maximum_compressed_header_bytes)
    return parse_hsd_header(first_bzip_stream(compressed), date, cycle, full_disk_line)


def resample_brightness_temperature(
    root: Path,
    date: str,
    cycle: str,
    common_grid: CommonGrid,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hiệu chuẩn năm băng thành Kelvin và lấy mẫu lại lên lưới chung."""
    files = cycle_files(root, date, cycle, config)
    bands = tuple(config["himawari"]["bands"])
    queries = [DataQuery(name=band, calibration="brightness_temperature") for band in bands]
    scene = Scene(reader=config["himawari"]["reader"], filenames=[str(path) for path in files])
    scene.load(queries)
    settings = config["himawari"]
    target = scene.resample(
        common_grid.area,
        datasets=queries,
        resampler=settings["resampler"],
        radius_of_influence=settings["radius_of_influence_m"],
        neighbours=settings["neighbours"],
        epsilon=settings["epsilon_m"],
        fill_value=np.nan,
        reduce_data=settings["reduce_data"],
        unload=False,
    )
    layers: list[np.ndarray] = []
    for query in queries:
        result = target[query].compute()
        if result.attrs.get("calibration") != "brightness_temperature":
            raise ValueError("Himawari chưa được hiệu chuẩn thành nhiệt độ sáng")
        if result.attrs.get("units") != "K":
            raise ValueError("Đơn vị nhiệt độ sáng không phải Kelvin")
        if not np.array_equal(np.asarray(result.coords["x"].data), common_grid.x):
            raise ValueError("Tọa độ x Himawari không khớp lưới chung")
        if not np.array_equal(np.asarray(result.coords["y"].data), common_grid.y):
            raise ValueError("Tọa độ y Himawari không khớp lưới chung")
        values = np.asarray(result.values, dtype=np.float32)
        values[~np.isfinite(values)] = np.nan
        layers.append(values)
    stack = np.stack(layers)
    valid = np.isfinite(stack)
    all_valid = np.all(valid, axis=0)
    del target, scene
    gc.collect()
    return stack, valid.astype(bool), all_valid.astype(bool)
