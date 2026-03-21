# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 17:05:10 2026

@author: Xiguan Liang @SKKU
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
import json


# ---------------------------
# User inputs (edit as needed)
# ---------------------------
MEASURED_MONTHLY_KWH: Dict[str, float] = {
    "1": 5530.4167,
    "2": 5716.1806,
    "3": 5002.9167,
    "4": 3540.625,
    "10": 126.875,
    "11": 1469.7222,
    "12": 3137.6389,
}

BUILDING_TYPE: str = "residential"  # used later (occupancy will be added in next step)

INPUT_IDF_PATH: Path = Path("./generated_idfs/geom_multizone_modified_RFH.idf")
OUTPUT_IDF_PATH: Path = Path("./Calibration/Before_Cali_RFH.idf")

DEFAULT_INFILTRATION_ACH: float = 0.50  # quick-mode default for residential


# ---------------------------
# IDF parsing helpers (lightweight)
# ---------------------------
def _strip_inline_comment(s: str) -> str:
    # Remove inline comment starting with '!'
    return s.split("!")[0].strip()


def _split_objects(idf_text: str) -> List[str]:
    """
    Split an IDF into raw object strings (each ending with ';').
    Keeps comments/formatting inside each object.
    """
    objs = []
    buff = []
    for line in idf_text.splitlines(True):
        buff.append(line)
        if ";" in line:
            # object terminator may appear once per object in typical IDF formatting
            # handle rare cases with multiple ';' by splitting conservatively
            joined = "".join(buff)
            parts = joined.split(";")
            for part in parts[:-1]:
                objs.append(part + ";\n")
            buff = [parts[-1]] if parts[-1].strip() else []
    if "".join(buff).strip():
        # dangling text (shouldn't happen in valid IDF), keep it
        objs.append("".join(buff))
    return objs


def _object_type(obj: str) -> str:
    """
    Return the object type token (e.g., 'RunPeriod', 'Zone', 'Output:Meter').
    """
    # find first non-empty, non-comment token up to first comma
    for line in obj.splitlines():
        s = line.strip()
        if not s or s.startswith("!"):
            continue
        s = _strip_inline_comment(s)
        if not s:
            continue
        # object type is before first comma
        if "," in s:
            return s.split(",", 1)[0].strip()
        return s.strip()
    return ""


def _object_fields(obj: str) -> List[str]:
    """
    Extract comma-separated fields (without comments). Not perfect but adequate for:
    - Zone name extraction
    - Infiltration schedule name extraction
    """
    # Remove full-line comments and strip inline comments, then join
    cleaned_lines = []
    for line in obj.splitlines():
        st = line.strip()
        if not st or st.startswith("!"):
            continue
        cleaned_lines.append(_strip_inline_comment(line))
    joined = " ".join(cleaned_lines)
    # Remove trailing ';'
    joined = joined.replace(";", " ")
    # Split by commas
    fields = [f.strip() for f in joined.split(",")]
    # Drop empty tail fields
    while fields and fields[-1] == "":
        fields.pop()
    return fields


def _has_schedule_on(objs: List[str]) -> bool:
    for o in objs:
        if _object_type(o).lower() == "schedule:compact":
            f = _object_fields(o)
            # f[0] = 'Schedule:Compact', f[1] = name
            if len(f) > 1 and f[1].strip().strip('"').upper() == "ON":
                return True
    return False


def _list_zone_names(objs: List[str]) -> List[str]:
    zones = []
    for o in objs:
        if _object_type(o).lower() == "zone":
            f = _object_fields(o)
            # Zone, Name, ...
            if len(f) > 1 and f[1]:
                zones.append(f[1].strip().strip('"'))
    return zones


def _any_zone_infiltration(objs: List[str]) -> bool:
    for o in objs:
        if _object_type(o).lower().startswith("zoneinfiltration:"):
            return True
    return False


def _remove_outputs(objs: List[str]) -> List[str]:
    """
    Remove output-related objects so the file stays lightweight.
    """
    drop_prefixes = (
        "output:",
        "outputcontrol:",
    )
    drop_exact = {
        "output:meter",  # will be re-added
        "output:variable",
        "output:table:summaryreports",
        "output:variabledictionary",
        "output:surfaces:drawing",
        "output:sqlite",
        "output:diagnostics",
        "output:constructions",
        "output:environmentalimpactfactors",
    }
    kept = []
    for o in objs:
        t = _object_type(o)
        tl = t.lower()
        if tl in drop_exact:
            continue
        if any(tl.startswith(p) for p in drop_prefixes):
            continue
        kept.append(o)
    return kept


def _make_output_meters() -> str:
    return (
        "\n"
        "  Output:Meter,\n"
        "    Electricity:Heating,   !- Name\n"
        "    Monthly;               !- Reporting Frequency\n"
        "\n"
        "  Output:Meter,\n"
        "    Gas:Heating,\n"
        "    Monthly;\n"
        "\n"
        "  Output:Meter,\n"
        "    DistrictHeating:Facility,\n"
        "    Monthly;\n"
    )


def _remove_runperiods(objs: List[str]) -> List[str]:
    kept = []
    for o in objs:
        if _object_type(o).lower() == "runperiod":
            continue
        kept.append(o)
    return kept


def _month_set_to_runperiods(months: List[int]) -> List[Tuple[int, int, int, int, str]]:
    """
    Convert a set of months to up to two RunPeriods to avoid discontinuous simulation.
    For your current case {1,2,3,4,10,11,12} -> two periods:
      01/01-04/30 and 10/01-12/31

    Returns list of tuples: (begin_month, begin_day, end_month, end_day, name)
    """
    mset = sorted(set(months))
    if not mset:
        raise ValueError("No months provided.")

    # Simple heuristic:
    # - If months are a single contiguous block -> one RunPeriod covering min..max.
    # - If two blocks separated by a gap -> two RunPeriods for each block.
    # We do not support >2 blocks in quick mode (can be added later).
    blocks = []
    cur = [mset[0]]
    for m in mset[1:]:
        if m == cur[-1] + 1:
            cur.append(m)
        else:
            blocks.append(cur)
            cur = [m]
    blocks.append(cur)

    if len(blocks) > 2:
        raise ValueError(
            f"Quick mode supports at most 2 month blocks; got {len(blocks)} blocks: {blocks}. "
            "Consider simulating the full year and ignoring unmeasured months."
        )

    def _end_day(mm: int) -> int:
        # Use common month end days; leap day irrelevant for month-level results.
        return {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}[mm]

    runperiods = []
    for i, b in enumerate(blocks, start=1):
        bm = b[0]
        em = b[-1]
        runperiods.append((bm, 1, em, _end_day(em), f"Cali_RunPeriod_{i}"))
    return runperiods


def _make_runperiod(bm: int, bd: int, em: int, ed: int, name: str) -> str:
    """
    EnergyPlus 8.9 RunPeriod fields:
      Name,
      Begin Month, Begin Day,
      End Month, End Day,
      Day of Week for Start Day,
      Use Weather File Holidays and Special Days,
      Use Weather File Daylight Saving Period,
      Apply Weekend Holiday Rule,
      Use Weather File Rain Indicators,
      Use Weather File Snow Indicators
    """
    return (
        "\n"
        "  RunPeriod,\n"
        f"    {name},              !- Name\n"
        f"    {bm},                   !- Begin Month\n"
        f"    {bd},                   !- Begin Day of Month\n"
        f"    {em},                   !- End Month\n"
        f"    {ed},                   !- End Day of Month\n"
        "    UseWeatherFile,      !- Day of Week for Start Day\n"
        "    Yes,                 !- Use Weather File Holidays and Special Days\n"
        "    Yes,                 !- Use Weather File Daylight Saving Period\n"
        "    No,                  !- Apply Weekend Holiday Rule\n"
        "    Yes,                 !- Use Weather File Rain Indicators\n"
        "    Yes;                 !- Use Weather File Snow Indicators\n"
    )


def _make_schedule_on() -> str:
    return (
        "\n"
        "  Schedule:Compact,\n"
        "    ON,                  !- Name\n"
        "    Fraction,            !- Schedule Type Limits Name\n"
        "    Through: 12/31,\n"
        "    For: AllDays,\n"
        "    Until: 24:00, 1.0;\n"
    )


def _make_infiltration_objects(zone_names: List[str], ach: float) -> str:
    """
    Add one ZoneInfiltration:DesignFlowRate per zone.
    Uses AirChanges/Hour for clarity and calibration friendliness.
    """
    blocks = ["\n  ! ==== Added by preprocess_idf_for_calibration.py: Infiltration (Quick Mode) ====\n"]
    for zn in zone_names:
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", zn).strip("_")
        blocks.append(
            "  ZoneInfiltration:DesignFlowRate,\n"
            f"    {safe_name}_Infiltration,    !- Name\n"
            f"    {zn},                        !- Zone or ZoneList or Space or SpaceList Name\n"
            "    ON,                          !- Schedule Name\n"
            "    AirChanges/Hour,             !- Design Flow Rate Calculation Method\n"
            "    ,                            !- Design Flow Rate {m3/s}\n"
            "    ,                            !- Flow Rate per Floor Area {m3/s-m2}\n"
            "    ,                            !- Flow Rate per Exterior Surface Area {m3/s-m2}\n"
            f"    {ach:.3f},                    !- Air Changes per Hour {1/hr}\n"
            "    1.0,                          !- Constant Term Coefficient\n"
            "    0.0,                          !- Temperature Term Coefficient\n"
            "    0.0,                          !- Velocity Term Coefficient\n"
            "    0.0;                          !- Velocity Squared Term Coefficient\n\n"
        )
    return "".join(blocks)


def preprocess_idf(
    input_idf: Path,
    output_idf: Path,
    measured_monthly_kwh: Dict[str, float],
    default_infiltration_ach: float = 0.50,
) -> None:
    if not input_idf.exists():
        raise FileNotFoundError(f"Input IDF not found: {input_idf.resolve()}")

    idf_text = input_idf.read_text(encoding="utf-8", errors="ignore")
    objs = _split_objects(idf_text)

    # 1) Remove all output requests
    objs = _remove_outputs(objs)

    # 2) Remove existing RunPeriod(s) and add periods covering measured months
    months = [int(m) for m in measured_monthly_kwh.keys()]
    runperiods = _month_set_to_runperiods(months)
    objs = _remove_runperiods(objs)

    # 3) Ensure schedule ON exists
    if not _has_schedule_on(objs):
        objs.append(_make_schedule_on())

    # 4) Add infiltration if none exists
    if not _any_zone_infiltration(objs):
        zone_names = _list_zone_names(objs)
        if not zone_names:
            raise RuntimeError("No Zone objects found; cannot add infiltration.")
        objs.append(_make_infiltration_objects(zone_names, default_infiltration_ach))

    # 5) Append minimal monthly meters
    objs.append(_make_output_meters())

    # 6) Append RunPeriods at the end (keeps edits obvious)
    for (bm, bd, em, ed, name) in runperiods:
        objs.append(_make_runperiod(bm, bd, em, ed, name))

    # 7) Write output
    output_idf.parent.mkdir(parents=True, exist_ok=True)
    output_idf.write_text("".join(objs), encoding="utf-8")
    print(f"[OK] Wrote preprocessed IDF to: {output_idf.resolve()}")


if __name__ == "__main__":
    print("=== Preprocess IDF for Calibration (Quick Mode) ===")
    print(f"Input IDF : {INPUT_IDF_PATH.resolve()}")
    print(f"Output IDF: {OUTPUT_IDF_PATH.resolve()}")
    print(f"Measured months: {sorted(int(k) for k in MEASURED_MONTHLY_KWH.keys())}")
    preprocess_idf(
        input_idf=INPUT_IDF_PATH,
        output_idf=OUTPUT_IDF_PATH,
        measured_monthly_kwh=MEASURED_MONTHLY_KWH,
        default_infiltration_ach=DEFAULT_INFILTRATION_ACH,
    )
