# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 14:00:46 2026

@author: Xiguan Liang @SKKU
"""


# Add_Cali.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import subprocess
import time
import re
import csv

import numpy as np
import pandas as pd
from eppy.modeleditor import IDF

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.optimize import minimize


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

# IMPORTANT: set EnergyPlus v8.9 IDD path
IDD_PATH: Path = Path(r"C:/EnergyPlusV8-9-0/Energy+.idd")  # <-- change

# Weather
EPW_PATH: Path = Path(r"C:\EnergyPlusV8-9-0\WeatherData\KOR_INCH'ON_IWEC.epw")

# IDF IO
INPUT_IDF_PATH: Path = Path("./Calibration/Ready1_Cali_RFH.idf")
OUTPUT_IDF_PATH: Path = Path("./Calibration/After_Cali_RFH.idf")

# Working directory for repeated simulations
WORK_DIR: Path = Path("./Calibration/_cali_runs")

# NSGA-II settings (start small; EnergyPlus evaluations are expensive)
POP_SIZE: int = 16
N_GEN: int = 8
SEED: int = 42

# Choose which meters to use for "heating energy"
# We will sum what is available among these (Monthly, J) and convert to kWh.
HEATING_METERS_PRIORITY: List[str] = [
    "DistrictHeating:Facility",
    "Electricity:Heating",
    "Gas:Heating",
]

# ---------------------------
# Calibration targets in your IDF (from your snippets)
# ---------------------------
WALL_INSUL_MAT_NAME = "Ext_Insul"
ROOF_MAT_NAME = "DefaultMaterial"
FLOOR_INSUL_MAT_NAME = "INS - EXPANDED EXT POLYSTYRENE R12 2 IN"
WINDOW_SIMPLE_GLAZING_NAME = "SG_2p0"

# Base values (used only for bounds sanity checks; we read actual from IDF too)
# These should match your current IDF (if different, code will use IDF values anyway).
BASE_WALL_K = 0.035
BASE_ROOF_K = 0.1
BASE_FLOOR_K = 0.035
BASE_WINDOW_U = 2.0

# Bounds on multipliers (broad but physically reasonable for quick-mode calibration)
# (You can tighten later after you see sensitivity.)
BOUNDS = {
    "m_wall_k": (0.5, 2.0),     # k 0.0175..0.07 if base 0.035
    "m_roof_k": (0.3, 3.0),     # effective layer
    "m_floor_k": (0.5, 2.0),    # EPS k
    "m_window_u": (0.6, 1.8),   # U 1.2..3.6 if base 2.0
}
#%%
MONTH_NAME_TO_INT = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

def _month_cell_to_int(x) -> int | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    # handle "January" or "Jan" (optional)
    if s in MONTH_NAME_TO_INT:
        return MONTH_NAME_TO_INT[s]
    # allow "Jan", "Feb" if they appear
    for k, v in MONTH_NAME_TO_INT.items():
        if k.startswith(s) and len(s) >= 3:
            return v
    # allow numeric month "1", "01"
    if s.isdigit():
        mm = int(s)
        return mm if 1 <= mm <= 12 else None
    return None


def read_monthly_meter_j_from_meter_csv(meter_csv: Path) -> dict[str, dict[int, float]]:
    """
    Parse EnergyPlus '*Meter.csv' / 'baselineMeter.csv' format like:
      Date/Time    DistrictHeating:Facility [J](Monthly)
      January      6966423054
      February     5545826735
      ...

    Returns:
      { "DistrictHeating:Facility": {1:valJ, 2:valJ, ...} }
    """
    df = pd.read_csv(meter_csv)

    # Identify month column
    month_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("date/time", "date", "time"):
            month_col = c
            break
    if month_col is None:
        # fallback: first column
        month_col = df.columns[0]

    # Identify meter columns; normalize by stripping units/freq annotation
    meters: dict[str, dict[int, float]] = {}

    for col in df.columns:
        if col == month_col:
            continue

        # Example col: "DistrictHeating:Facility [J](Monthly)"
        base_name = str(col).split("[", 1)[0].strip()

        for _, row in df.iterrows():
            mm = _month_cell_to_int(row[month_col])
            if mm is None:
                continue
            try:
                val_j = float(row[col])
            except Exception:
                continue
            meters.setdefault(base_name, {})[mm] = val_j

    if not meters:
        raise RuntimeError(f"No meter columns parsed from {meter_csv.name}. Columns={list(df.columns)}")

    return meters

# ============================================================
# Metrics: ASHRAE Guideline 14 style (monthly)
# ============================================================
def nmbe_percent(meas: np.ndarray, sim: np.ndarray) -> float:
    """
    NMBE (%) for monthly data:
      NMBE = 100 * sum(sim - meas) / ((n - 1) * mean(meas))
    Commonly uses (n - p) in denominator; for quick-mode monthly, use (n - 1).
    """
    n = len(meas)
    if n < 2:
        return float("nan")
    denom = (n - 1) * np.mean(meas)
    if denom == 0:
        return float("nan")
    return 100.0 * np.sum(sim - meas) / denom


def cvrmse_percent(meas: np.ndarray, sim: np.ndarray) -> float:
    """
    CVRMSE (%) for monthly data:
      CVRMSE = 100 * RMSE / mean(meas)
    where RMSE = sqrt( sum((sim-meas)^2) / (n - 1) ) for Guideline 14 monthly convention.
    """
    n = len(meas)
    if n < 2:
        return float("nan")
    denom = np.mean(meas)
    if denom == 0:
        return float("nan")
    rmse = np.sqrt(np.sum((sim - meas) ** 2) / (n - 1))
    return 100.0 * rmse / denom


# ============================================================
# EnergyPlus runner
# ============================================================
def guess_energyplus_exe(idd_path: Path) -> Path:
    """
    If IDD is C:/EnergyPlusV8-9-0/Energy+.idd,
    EnergyPlus exe is typically C:/EnergyPlusV8-9-0/energyplus.exe
    """
    candidate = idd_path.parent / "energyplus.exe"
    if candidate.exists():
        return candidate
    # fallback: rely on PATH
    return Path("energyplus")


def run_energyplus(energyplus_exe: Path, idf_path: Path, epw_path: Path, out_dir: Path, timeout_s: int = 3600) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(energyplus_exe),
        "-w", str(epw_path),
        "-d", str(out_dir),
        str(idf_path),
    ]
    # On Windows, avoid console popups by using creationflags if desired (optional).
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        # Surface useful parts of stderr/stdout
        msg = (proc.stdout[-4000:] if proc.stdout else "") + "\n" + (proc.stderr[-4000:] if proc.stderr else "")
        raise RuntimeError(f"EnergyPlus failed (code {proc.returncode}). Tail output:\n{msg}")


# ============================================================
# Parse monthly meters (J) from eplusout.mtr (preferred)
# ============================================================
def _parse_month_from_timestamp(ts: str) -> Optional[int]:
    """
    Typical meter timestamp looks like '01/31 24:00:00' or '1/31 24:00:00'
    """
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})", ts)
    if not m:
        return None
    return int(m.group(1))


def read_monthly_meter_j_from_mtr(mtr_path: Path) -> Dict[str, Dict[int, float]]:
    """
    EnergyPlus 8.9 eplusout.mtr parsing (Monthly meters).

    Dictionary line example:
      978,9,DistrictHeating:Facility [J] !Monthly [Value,Min,...]
    Data section example:
      4,31, 1
      978,6966423054.207052,0.0, ...
      4,59, 2
      978,5545826735.155864,0.0, ...

    Returns:
      { meter_name: { month:int -> value_J:float } }
    """
    lines = mtr_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 1) Split dictionary vs data
    end_idx = None
    for i, line in enumerate(lines):
        if "End of Data Dictionary" in line:
            end_idx = i
            break
    if end_idx is None:
        raise RuntimeError("Could not find 'End of Data Dictionary' in eplusout.mtr")

    dict_lines = lines[: end_idx + 1]
    data_lines = lines[end_idx + 1 :]

    # 2) Parse monthly meter indices from dictionary
    # Map: meter_index(int) -> meter_name(str)
    monthly_idx_to_name: Dict[int, str] = {}
    for line in dict_lines:
        # interested in lines like: "978,9,DistrictHeating:Facility [J] !Monthly ..."
        if "!Monthly" not in line and "! Monthly" not in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except Exception:
            continue

        # meter name is in the 3rd field and may include units [J]
        name_with_units = parts[2]
        meter_name = name_with_units.split("[", 1)[0].strip()
        monthly_idx_to_name[idx] = meter_name

    if not monthly_idx_to_name:
        raise RuntimeError(
            "No Monthly meters detected in eplusout.mtr dictionary. "
            "Check that your Output:Meter objects request Monthly frequency."
        )

    # 3) Parse data: month comes from record-type '4,...,<month>'
    meters_monthly: Dict[str, Dict[int, float]] = {n: {} for n in monthly_idx_to_name.values()}
    current_month: Optional[int] = None

    for line in data_lines:
        s = line.strip()
        if not s:
            continue
        parts = [p.strip() for p in s.split(",")]

        # Monthly record header line starts with "4,"
        # Format in your file: "4,31, 1"  -> month is the 3rd token
        if parts[0] == "4" and len(parts) >= 3:
            try:
                mm = int(parts[2])
                if 1 <= mm <= 12:
                    current_month = mm
                else:
                    current_month = None
            except Exception:
                current_month = None
            continue

        # Meter value line starts with meter index, e.g., "978,6966423054.20,..."
        try:
            idx = int(parts[0])
        except Exception:
            continue

        if idx in monthly_idx_to_name and current_month is not None and len(parts) >= 2:
            try:
                val_j = float(parts[1])
            except Exception:
                continue
            mname = monthly_idx_to_name[idx]
            meters_monthly[mname][current_month] = val_j

    # Remove empty meters (if any)
    meters_monthly = {k: v for k, v in meters_monthly.items() if v}
    if not meters_monthly:
        raise RuntimeError("Monthly meter indices found, but no monthly values parsed from data section.")

    return meters_monthly


def read_monthly_meter_j_from_csv(csv_path: Path) -> Dict[str, Dict[int, float]]:
    """
    Fallback parser for eplusout.csv-like tables that already contain Monthly meter results.
    Tries to find a 'Month' column and meter columns matching names.
    """
    df = pd.read_csv(csv_path)
    month_col = None
    for c in df.columns:
        if str(c).strip().lower() in ("month", "months"):
            month_col = c
            break
    if month_col is None:
        # Try DateTime column containing month
        for c in df.columns:
            if "date" in str(c).lower() or "time" in str(c).lower():
                month_col = c
                break
    if month_col is None:
        raise RuntimeError(f"Could not identify a month/date column in {csv_path.name}")

    meters: Dict[str, Dict[int, float]] = {}
    for m in HEATING_METERS_PRIORITY:
        # find matching column
        col = None
        for c in df.columns:
            if str(c).strip().lower() == m.strip().lower():
                col = c
                break
        if col is None:
            continue
        for _, row in df.iterrows():
            mm = None
            if str(month_col).lower() in ("month", "months"):
                try:
                    mm = int(row[month_col])
                except Exception:
                    mm = None
            else:
                mm = _parse_month_from_timestamp(str(row[month_col]))
            if mm is None:
                continue
            try:
                val = float(row[col])
            except Exception:
                continue
            meters.setdefault(m, {})[mm] = val
    if not meters:
        raise RuntimeError("No heating meters found in CSV fallback.")
    return meters


def read_sim_monthly_heating_kwh(out_dir: Path) -> Dict[int, float]:
    """
    Returns {month:int -> heating_kwh:float} by summing available heating meters.

    Priority:
      1) *Meter.csv (e.g., baselineMeter.csv, eplusoutMeter.csv, etc.)
      2) eplusout.mtr
      3) other CSV fallback (legacy)
    """
    # 1) Prefer *Meter.csv
    meter_csvs = sorted(out_dir.glob("*Meter.csv"))
    if meter_csvs:
        meters = read_monthly_meter_j_from_meter_csv(meter_csvs[0])
    else:
        # 2) Prefer .mtr
        mtr = out_dir / "eplusout.mtr"
        if mtr.exists():
            meters = read_monthly_meter_j_from_mtr(mtr)
        else:
            # 3) Last resort: any csv
            csv_files = list(out_dir.glob("*.csv"))
            if not csv_files:
                raise RuntimeError("No *Meter.csv, no eplusout.mtr, and no CSV files found in output directory.")
            meters = read_monthly_meter_j_from_csv(csv_files[0])

    # Sum heating meters that exist
    monthly_j: Dict[int, float] = {}
    found_any = False

    for meter_name in HEATING_METERS_PRIORITY:
        key = None
        for k in meters.keys():
            if k.strip().lower() == meter_name.strip().lower():
                key = k
                break
        if key is None:
            continue

        found_any = True
        for mm, val_j in meters[key].items():
            monthly_j[mm] = monthly_j.get(mm, 0.0) + float(val_j)

    if not found_any:
        raise RuntimeError(
            f"None of HEATING_METERS_PRIORITY found: {HEATING_METERS_PRIORITY}. "
            f"Available meters: {list(meters.keys())}"
        )

    return {mm: (val_j / 3.6e6) for mm, val_j in monthly_j.items()}  # J -> kWh



# ============================================================
# IDF editing utilities (eppy)
# ============================================================
def _get_material(idf: IDF, name: str):
    mats = idf.idfobjects.get("MATERIAL", [])
    for m in mats:
        if getattr(m, "Name", "").strip() == name:
            return m
    raise KeyError(f"Material not found: {name}")


def _get_simple_glazing(idf: IDF, name: str):
    objs = idf.idfobjects.get("WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", [])
    for o in objs:
        if getattr(o, "Name", "").strip() == name:
            return o
    raise KeyError(f"WindowMaterial:SimpleGlazingSystem not found: {name}")


def read_base_params_from_idf(idf: IDF) -> Dict[str, float]:
    wall = _get_material(idf, WALL_INSUL_MAT_NAME)
    roof = _get_material(idf, ROOF_MAT_NAME)
    floor = _get_material(idf, FLOOR_INSUL_MAT_NAME)
    win = _get_simple_glazing(idf, WINDOW_SIMPLE_GLAZING_NAME)

    return {
        "wall_k": float(getattr(wall, "Conductivity")),
        "roof_k": float(getattr(roof, "Conductivity")),
        "floor_k": float(getattr(floor, "Conductivity")),
        "window_u": float(getattr(win, "UFactor")),
    }


def apply_envelope_params(idf: IDF, m_wall_k: float, m_roof_k: float, m_floor_k: float, m_window_u: float) -> None:
    wall = _get_material(idf, WALL_INSUL_MAT_NAME)
    roof = _get_material(idf, ROOF_MAT_NAME)
    floor = _get_material(idf, FLOOR_INSUL_MAT_NAME)
    win = _get_simple_glazing(idf, WINDOW_SIMPLE_GLAZING_NAME)

    # Use current values as base (safer if IDF changed)
    wall_base = float(getattr(wall, "Conductivity"))
    roof_base = float(getattr(roof, "Conductivity"))
    floor_base = float(getattr(floor, "Conductivity"))
    win_base = float(getattr(win, "UFactor"))

    # Apply multipliers
    wall.Conductivity = wall_base * float(m_wall_k)
    roof.Conductivity = roof_base * float(m_roof_k)
    floor.Conductivity = floor_base * float(m_floor_k)
    win.UFactor = win_base * float(m_window_u)


# ============================================================
# Evaluation: simulate & compute metrics
# ============================================================
def align_months(measured_kwh: Dict[str, float], simulated_kwh: Dict[int, float]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    months = sorted(int(k) for k in measured_kwh.keys())
    meas = []
    sim = []
    kept_months = []
    for mm in months:
        if mm in simulated_kwh:
            meas.append(float(measured_kwh[str(mm)]))
            sim.append(float(simulated_kwh[mm]))
            kept_months.append(mm)
    if len(meas) < 2:
        raise RuntimeError(f"Not enough overlapping months. Measured={months}, SimAvailable={sorted(simulated_kwh.keys())}")
    return np.array(meas), np.array(sim), kept_months


def compute_metrics(measured_kwh: Dict[str, float], simulated_kwh: Dict[int, float]) -> Tuple[float, float, Dict[int, float]]:
    meas, sim, months = align_months(measured_kwh, simulated_kwh)
    cvr = cvrmse_percent(meas, sim)
    nb = nmbe_percent(meas, sim)
    sim_used = {m: float(simulated_kwh[m]) for m in months}
    return cvr, nb, sim_used


# ============================================================
# Pymoo Problem
# ============================================================
class EnvelopeCalibrationProblem(ElementwiseProblem):
    def __init__(
        self,
        idd_path: Path,
        input_idf_path: Path,
        epw_path: Path,
        work_dir: Path,
        measured_monthly_kwh: Dict[str, float],
        energyplus_exe: Path,
        base_idf_params: Dict[str, float],
        log_csv_path: Path,
    ):
        xl = np.array([BOUNDS["m_wall_k"][0], BOUNDS["m_roof_k"][0], BOUNDS["m_floor_k"][0], BOUNDS["m_window_u"][0]], dtype=float)
        xu = np.array([BOUNDS["m_wall_k"][1], BOUNDS["m_roof_k"][1], BOUNDS["m_floor_k"][1], BOUNDS["m_window_u"][1]], dtype=float)
        super().__init__(n_var=4, n_obj=2, xl=xl, xu=xu)

        self.idd_path = idd_path
        self.input_idf_path = input_idf_path
        self.epw_path = epw_path
        self.work_dir = work_dir
        self.measured = measured_monthly_kwh
        self.energyplus_exe = energyplus_exe
        self.base_idf_params = base_idf_params
        self.log_csv_path = log_csv_path

        self.eval_counter = 0

        # Prepare log file header
        if not self.log_csv_path.exists():
            self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "eval_id",
                    "m_wall_k", "m_roof_k", "m_floor_k", "m_window_u",
                    "CVRMSE_%", "NMBE_%", "absNMBE_%",
                    "runtime_s",
                ])

    def _evaluate(self, x, out, *args, **kwargs):
        self.eval_counter += 1
        eval_id = self.eval_counter

        m_wall_k, m_roof_k, m_floor_k, m_window_u = map(float, x.tolist())

        # Create a temp IDF for this evaluation
        run_dir = self.work_dir / f"run_{eval_id:05d}"
        run_idf = run_dir / "in.idf"
        out_dir = run_dir / "out"

        run_dir.mkdir(parents=True, exist_ok=True)

        # Load base IDF fresh each time (avoids cumulative drift)
        IDF.setiddname(str(self.idd_path))
        idf = IDF(str(self.input_idf_path))

        # Apply envelope multipliers (relative to *current IDF* values)
        apply_envelope_params(idf, m_wall_k, m_roof_k, m_floor_k, m_window_u)

        # Save
        idf.saveas(str(run_idf))

        # Simulate
        t0 = time.time()
        try:
            run_energyplus(self.energyplus_exe, run_idf, self.epw_path, out_dir)
            sim_monthly_kwh = read_sim_monthly_heating_kwh(out_dir)
            cvr, nb, _sim_used = compute_metrics(self.measured, sim_monthly_kwh)
            absnb = abs(nb)
        except Exception as e:
            # Penalize failed runs heavily
            cvr, nb, absnb = 1e6, 1e6, 1e6
        runtime = time.time() - t0

        # Objectives: minimize CVRMSE and |NMBE|
        out["F"] = np.array([cvr, absnb], dtype=float)

        # Log this evaluation
        with self.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([eval_id, m_wall_k, m_roof_k, m_floor_k, m_window_u, cvr, nb, absnb, runtime])


# ============================================================
# Main workflow
# ============================================================
def baseline_report(idd_path: Path, input_idf_path: Path, epw_path: Path, energyplus_exe: Path) -> Tuple[float, float]:
    """
    Run once with unchanged parameters and report baseline CVRMSE/NMBE.
    """
    baseline_dir = WORK_DIR / "baseline"
    baseline_idf = baseline_dir / "baseline.idf"
    out_dir = baseline_dir / "out"

    baseline_dir.mkdir(parents=True, exist_ok=True)

    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))
    idf.saveas(str(baseline_idf))

    run_energyplus(energyplus_exe, baseline_idf, epw_path, out_dir)

    sim_monthly_kwh = read_sim_monthly_heating_kwh(out_dir)
    cvr, nb, sim_used = compute_metrics(MEASURED_MONTHLY_KWH, sim_monthly_kwh)

    # Print per-month table
    months = sorted(int(k) for k in MEASURED_MONTHLY_KWH.keys())
    print("\n=== Baseline Monthly Comparison (kWh) ===")
    print("Month | Measured | Simulated | Residual (Sim-Meas)")
    for m in months:
        meas = MEASURED_MONTHLY_KWH.get(str(m))
        sim = sim_used.get(m, float("nan"))
        if meas is None:
            continue
        print(f"{m:>5} | {meas:>8.2f} | {sim:>9.2f} | {sim - float(meas):>+12.2f}")

    print("\n=== Baseline Metrics (Monthly) ===")
    print(f"CVRMSE: {cvr:.3f} %")
    print(f"NMBE  : {nb:.3f} %\n")
    return cvr, nb


def select_best_from_log(log_csv: Path) -> Tuple[Dict[str, float], float, float]:
    """
    Pick the single "best" point as:
      minimize CVRMSE first, then minimize |NMBE|
    (You still have the full Pareto set in the NSGA-II result; this is a practical single-pick.)
    """
    df = pd.read_csv(log_csv)
    df["absNMBE_%"] = df["absNMBE_%"].astype(float)
    df["CVRMSE_%"] = df["CVRMSE_%"].astype(float)
    df = df.sort_values(["CVRMSE_%", "absNMBE_%"], ascending=[True, True]).reset_index(drop=True)
    best = df.iloc[0].to_dict()
    params = {
        "m_wall_k": float(best["m_wall_k"]),
        "m_roof_k": float(best["m_roof_k"]),
        "m_floor_k": float(best["m_floor_k"]),
        "m_window_u": float(best["m_window_u"]),
    }
    return params, float(best["CVRMSE_%"]), float(best["NMBE_%"])


def write_final_idf_with_best_params(
    idd_path: Path,
    input_idf_path: Path,
    output_idf_path: Path,
    best_params: Dict[str, float],
) -> None:
    IDF.setiddname(str(idd_path))
    idf = IDF(str(input_idf_path))
    apply_envelope_params(
        idf,
        best_params["m_wall_k"],
        best_params["m_roof_k"],
        best_params["m_floor_k"],
        best_params["m_window_u"],
    )
    output_idf_path.parent.mkdir(parents=True, exist_ok=True)
    idf.saveas(str(output_idf_path))


def main() -> None:
    # Basic checks
    if not IDD_PATH.exists():
        raise FileNotFoundError(f"IDD not found: {IDD_PATH.resolve()}")
    if not EPW_PATH.exists():
        raise FileNotFoundError(f"EPW not found: {EPW_PATH.resolve()}")
    if not INPUT_IDF_PATH.exists():
        raise FileNotFoundError(f"IDF not found: {INPUT_IDF_PATH.resolve()}")

    energyplus_exe = guess_energyplus_exe(IDD_PATH)
    print(f"[INFO] EnergyPlus exe: {energyplus_exe}")

    # Ensure work dir
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # Baseline
    baseline_report(IDD_PATH, INPUT_IDF_PATH, EPW_PATH, energyplus_exe)

    # Read base param values (informational)
    IDF.setiddname(str(IDD_PATH))
    base_idf = IDF(str(INPUT_IDF_PATH))
    base_params = read_base_params_from_idf(base_idf)
    print("[INFO] Base envelope parameters (from IDF):")
    for k, v in base_params.items():
        print(f"  - {k}: {v}")

    # Optimization
    log_csv = WORK_DIR / "eval_log.csv"
    problem = EnvelopeCalibrationProblem(
        idd_path=IDD_PATH,
        input_idf_path=INPUT_IDF_PATH,
        epw_path=EPW_PATH,
        work_dir=WORK_DIR,
        measured_monthly_kwh=MEASURED_MONTHLY_KWH,
        energyplus_exe=energyplus_exe,
        base_idf_params=base_params,
        log_csv_path=log_csv,
    )

    algorithm = NSGA2(pop_size=POP_SIZE)
    termination = get_termination("n_gen", N_GEN)

    print("\n[INFO] Starting NSGA-II optimization...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=SEED,
        save_history=False,
        verbose=True,
    )

    # Pick a single best solution from the evaluation log
    best_params, best_cvr, best_nb = select_best_from_log(log_csv)
    print("\n=== Selected Best (single-point pick) ===")
    print("Params:")
    for k, v in best_params.items():
        print(f"  {k} = {v:.4f}")
    print(f"CVRMSE = {best_cvr:.3f} %")
    print(f"NMBE   = {best_nb:.3f} %")
    print(f"[INFO] Full evaluation log saved to: {log_csv.resolve()}")

    # Write final calibrated IDF
    write_final_idf_with_best_params(IDD_PATH, INPUT_IDF_PATH, OUTPUT_IDF_PATH, best_params)
    print(f"[OK] Calibrated IDF saved to: {OUTPUT_IDF_PATH.resolve()}")


if __name__ == "__main__":
    main()


