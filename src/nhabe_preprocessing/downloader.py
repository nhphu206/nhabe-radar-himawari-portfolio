"""Tải chọn lọc HSD theo chu kỳ đã ghép, có thể tiếp tục từ tệp `.part`."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Iterable

import aiohttp

from .himawari import hsd_filename


def planned_objects(
    date: str,
    cycles: Iterable[str],
    staging_root: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    settings = config["himawari"]
    objects: list[dict[str, Any]] = []
    for cycle in sorted(set(cycles)):
        for band in settings["bands"]:
            for segment in settings["segments"]:
                name = hsd_filename(date, cycle, band, segment)
                key = (
                    f"{settings['product']}/{date[:4]}/{date[4:6]}/{date[6:8]}/"
                    f"{cycle}/{name}"
                )
                objects.append({
                    "date": date,
                    "cycle": cycle,
                    "band": band,
                    "segment": f"S{segment}10",
                    "name": name,
                    "key": key,
                    "url": f"https://s3.amazonaws.com/{settings['bucket']}/{key}",
                    "path": staging_root / date / cycle / name,
                })
    return objects


def _total_size(response: aiohttp.ClientResponse, start: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    length = response.headers.get("Content-Length")
    return start + int(length) if length is not None else None


async def download_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    item: dict[str, Any],
) -> dict[str, Any]:
    path = Path(item["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return {**item, "status": "REUSED_EXISTING", "size_bytes": path.stat().st_size,
                "attempts": 0, "error": ""}
    partial = path.with_suffix(path.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(1, 7):
        start = partial.stat().st_size if partial.is_file() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        try:
            async with semaphore:
                async with session.get(item["url"], headers=headers) as response:
                    if response.status == 404:
                        raise FileNotFoundError(item["url"])
                    if response.status not in (200, 206):
                        raise RuntimeError(f"HTTP {response.status}: {item['url']}")
                    if start and response.status == 200:
                        start = 0
                    expected = _total_size(response, start)
                    with partial.open("ab" if start else "wb") as handle:
                        async for chunk in response.content.iter_chunked(1024 * 1024):
                            handle.write(chunk)
            actual = partial.stat().st_size
            if expected is not None and actual != expected:
                raise IOError(f"Kích thước {actual} khác {expected}")
            os.replace(partial, path)
            return {**item, "status": "DOWNLOADED", "size_bytes": actual,
                    "attempts": attempt, "error": ""}
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as error:
            last_error = error
            if attempt < 6:
                await asyncio.sleep(min(2 ** attempt, 30))
    return {**item, "status": "FAILED", "size_bytes": 0, "attempts": 6,
            "error": repr(last_error)}


async def download_all(
    objects: Iterable[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    values = list(objects)
    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=180)
    connector = aiohttp.TCPConnector(limit=workers + 4, limit_per_host=workers + 4)
    semaphore = asyncio.Semaphore(workers)
    headers = {"User-Agent": "nha-be-radar-himawari-research/2.0"}
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        tasks = [asyncio.create_task(download_one(session, semaphore, item)) for item in values]
        return [await task for task in asyncio.as_completed(tasks)]
