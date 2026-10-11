"""Định nghĩa lưới AEQD chung cho radar và Himawari."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyresample.geometry import AreaDefinition


@dataclass(frozen=True)
class CommonGrid:
    area: AreaDefinition
    x: np.ndarray
    y: np.ndarray
    radius: np.ndarray


def build_common_grid(config: dict[str, Any]) -> CommonGrid:
    """Tạo AreaDefinition và tọa độ tâm pixel từ một nguồn cấu hình duy nhất."""
    grid = config["grid"]
    height, width = map(int, grid["shape_yx"])
    area = AreaDefinition(
        area_id=grid["grid_id"],
        description="Lưới chung radar Nhà Bè và Himawari-9",
        proj_id="nhabe_aeqd",
        projection=grid["projection"],
        width=width,
        height=height,
        area_extent=tuple(map(float, grid["area_extent_m"])),
    )
    x, y = (np.asarray(values, dtype=np.float64) for values in area.get_proj_vectors())
    if x.shape != (width,) or y.shape != (height,):
        raise ValueError("Tọa độ lưới không khớp shape_yx")
    expected_spacing = float(grid["pixel_size_m"])
    if not np.allclose(np.diff(x), expected_spacing):
        raise ValueError("Bước lưới x không đúng cấu hình")
    if not np.allclose(np.diff(y), -expected_spacing):
        raise ValueError("Bước lưới y không đúng cấu hình")
    xx, yy = np.meshgrid(x, y)
    return CommonGrid(area=area, x=x, y=y, radius=np.hypot(xx, yy))
