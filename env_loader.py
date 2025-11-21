"""간단한 .env 로더 유틸.

모듈 내에서 `ensure_env_loaded(start=__file__)` 정도로 호출하면
가까운 상위 디렉터리에서 `.env` 파일을 찾아 한 번만 로드한다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

_LOADED: Dict[Tuple[str, str], bool] = {}


def find_env_file(filename: str = ".env", search_from: Optional[str | Path] = None) -> Optional[Path]:
    """지정된 위치에서 시작해 상위 디렉터리를 탐색하며 .env 파일을 찾는다."""
    start_path = Path(search_from).resolve() if search_from else Path.cwd().resolve()
    if start_path.is_file():
        start_path = start_path.parent

    for root in [start_path, *start_path.parents]:
        candidate = root / filename
        if candidate.is_file():
            return candidate
    return None


def load_env_file(env_path: Path) -> None:
    """env 파일 내용을 읽어 환경 변수에 채운다."""
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def ensure_env_loaded(filename: str = ".env", search_from: Optional[str | Path] = None) -> None:
    """동일한 파일을 중복 로드하지 않도록 보장한다."""
    cache_key = (filename, str(Path(search_from).resolve()) if search_from else "")
    if _LOADED.get(cache_key):
        return
    env_path = find_env_file(filename=filename, search_from=search_from)
    if env_path:
        load_env_file(env_path)
    _LOADED[cache_key] = True
