"""Utilities for parsing label cluster summary reports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class LabelClusterEntry:
    name: str
    status: str
    p_value: Optional[float] = None
    t_sum: Optional[float] = None
    sign: Optional[str] = None
    time_start_ms: Optional[float] = None
    time_end_ms: Optional[float] = None
    peak_ms: Optional[float] = None


@dataclass
class LabelClusterSummary:
    header: dict
    entries: List[LabelClusterEntry]

    @property
    def significant(self) -> List[LabelClusterEntry]:
        return [entry for entry in self.entries if entry.status == "significant"]


_HEADER_KV_RE = re.compile(r"(?P<key>[a-zA-Z_]+)=(?P<value>[^;]+)")
_SIG_LINE_RE = re.compile(
    r"^(?P<name>[^:]+):\s+SIGNIFICANT cluster p=(?P<pval>[0-9eE+\-.]+)"
    r"(?:\s*\((?P<sign>positive|negative|mixed),\s*t-sum=(?P<t_sum>[0-9eE+\-.]+)\))?"
    r"\s+at\s+(?P<start>[0-9.]+)-(?P<end>[0-9.]+)\s*ms"
    r"\s*\(peak\s+(?P<peak>[0-9.]+)\s*ms\)\s*$",
    re.IGNORECASE,
)


def _clean_value(value: str) -> str:
    return value.strip().strip('.')


def parse_label_summary(path: Path) -> LabelClusterSummary:
    """Parse an aux/label_cluster_summary.txt file."""

    if not path or not Path(path).exists():
        return LabelClusterSummary(header={}, entries=[])

    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    header: dict = {}
    entries: List[LabelClusterEntry] = []

    if len(lines) >= 2:
        for match in _HEADER_KV_RE.finditer(lines[1]):
            key = match.group("key").strip()
            value = _clean_value(match.group("value"))
            header[key] = value

        window_str = header.get("window")
        if window_str and "-" in window_str:
            window_clean = window_str.rstrip('s')
            try:
                start_s, end_s = [float(v) for v in window_clean.split('-')[:2]]
                header["window_start_s"] = start_s
                header["window_end_s"] = end_s
            except ValueError:
                pass
        try:
            header["n_permutations"] = int(float(header.get("n_permutations", "nan")))
        except ValueError:
            pass

    for line in lines[2:]:
        sig_match = _SIG_LINE_RE.match(line)
        if sig_match:
            name = sig_match.group("name").strip()
            p_val = float(sig_match.group("pval"))
            sign = sig_match.group("sign")
            t_sum_raw = sig_match.group("t_sum")
            t_sum = float(t_sum_raw) if t_sum_raw else None
            start_ms = float(sig_match.group("start"))
            end_ms = float(sig_match.group("end"))
            peak_ms = float(sig_match.group("peak"))
            entries.append(
                LabelClusterEntry(
                    name=name,
                    status="significant",
                    p_value=p_val,
                    t_sum=t_sum,
                    sign=sign,
                    time_start_ms=start_ms,
                    time_end_ms=end_ms,
                    peak_ms=peak_ms,
                )
            )
            continue

        if ":" in line:
            name, rest = line.split(":", 1)
            entries.append(
                LabelClusterEntry(
                    name=name.strip(),
                    status=_clean_value(rest).lower(),
                )
            )
        else:
            entries.append(LabelClusterEntry(name=line, status="info"))

    return LabelClusterSummary(header=header, entries=entries)


