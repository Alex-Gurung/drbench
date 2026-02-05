from __future__ import annotations

from typing import Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")

try:
    from tqdm import tqdm  # type: ignore

    def progress(
        iterable: Iterable[T],
        total: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> Iterator[T]:
        return tqdm(iterable, total=total, desc=desc, ncols=100)

except Exception:  # pragma: no cover

    def progress(
        iterable: Iterable[T],
        total: Optional[int] = None,
        desc: Optional[str] = None,
    ) -> Iterator[T]:
        label = desc or "Progress"
        if total:
            step = max(1, total // 20)
        else:
            step = 100
        for idx, item in enumerate(iterable, 1):
            if idx == 1 or idx % step == 0:
                if total:
                    print(f"{label}: {idx}/{total}")
                else:
                    print(f"{label}: {idx}")
            yield item
