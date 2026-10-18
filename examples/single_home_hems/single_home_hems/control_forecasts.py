"""Causal external single-path forecasts and paired EV scenario containers."""
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
from examples.five_home_block.five_home_block.ev_schedule import EVSchedule

UNITS = {"electricity_price_usd_per_kwh": "USD/kWh", "backed_load_kw": "kW",
         "Water Heating (L/min)": "L/min", "Clothes Washer (L/min)": "L/min",
         "Dishwasher (L/min)": "L/min", "Mains Temperature (C)": "C"}


@dataclass(frozen=True)
class EVScenarioSet:
    """Equal-probability EV paths; forecasts preserve price/load pairing."""
    issued_at: pd.Timestamp
    paths: tuple[EVSchedule, ...]
    backed_load_paths: tuple[pd.Series, ...]
    forecasts: tuple = ()
    candidate_ids: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.paths or len(self.paths) != len(self.backed_load_paths):
            raise ValueError("EV scenarios require paired mobility and backed-load paths")
        if self.forecasts and len(self.forecasts) != len(self.paths):
            raise ValueError("EV scenario forecasts must match mobility paths")
        if self.candidate_ids and len(self.candidate_ids) != len(self.paths):
            raise ValueError("EV candidate IDs must match mobility paths")

    @property
    def probabilities(self):
        """Uniform scenario weights, consistent with the joint mean-cost objective."""
        return np.full(len(self.paths), 1 / len(self.paths))


class ExternalForecast:
    """JSON snapshots with issued_at, data rows, explicit units, and EV trips.

    The latest issuance available at the decision is used. A selected snapshot
    must cover every native plant step through the configured forecast length.
    No future realized columns are copied into a deterministic forecast.
    """
    def __init__(self, path, device, forecast_hours):
        self.device = device
        self.length = pd.Timedelta(hours=forecast_hours)
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        self.required = (list(UNITS)[:2] if device.startswith("EV_") else
                         [k for k in UNITS if k != "backed_load_kw"])
        for name in self.required:
            if doc.get("units", {}).get(name) != UNITS[name]:
                raise ValueError(f"External forecast must declare {name}: {UNITS[name]}")
        if device.startswith("EV_") and doc.get("units", {}).get("distance_km") != "km":
            raise ValueError("External EV trips must declare distance_km: km")
        self.snapshots = sorted(doc["snapshots"], key=lambda r: pd.Timestamp(r["issued_at"]))
        times = [pd.Timestamp(s["issued_at"]) for s in self.snapshots]
        if len(set(times)) != len(times):
            raise ValueError("Duplicate forecast issuance time")
        self.history = []

    def at(self, timestamp, realized, horizon):
        """Select a causal snapshot with full coverage; never fill from future actuals."""
        timestamp = pd.Timestamp(timestamp)
        candidates = [s for s in self.snapshots if pd.Timestamp(s["issued_at"]) <= timestamp]
        if not candidates:
            raise ValueError(f"No causal forecast for {self.device} at {timestamp}")
        snapshot = candidates[-1]
        rows = pd.DataFrame(snapshot["data"])
        rows["timestamp"] = pd.to_datetime(rows["timestamp"])
        rows = rows.set_index("timestamp")
        if rows.index.has_duplicates or not rows.index.is_monotonic_increasing:
            raise ValueError("External forecast targets must be unique and ordered")
        step = realized.data.index[1]-realized.data.index[0]
        index = pd.date_range(timestamp, timestamp+max(horizon, self.length), freq=step, inclusive="left")
        rows = rows.reindex(index)[self.required]
        if not np.isfinite(rows.to_numpy(dtype=float)).all():
            raise ValueError("External forecast lacks full finite target coverage")
        if "backed_load_kw" in rows and rows.backed_load_kw.lt(0).any():
            raise ValueError("External backed-load forecast cannot be negative")
        # Past realized observations are allowed; future realized data are not.
        past = realized.data.loc[realized.data.index < timestamp]
        data = pd.concat([past, rows.drop(columns="backed_load_kw", errors="ignore")])
        mobility = realized.ev_schedules
        backed = None
        if self.device.startswith("EV_"):
            trips = pd.DataFrame(snapshot["trips"], columns=["departure", "arrival", "distance_km"])
            for name in ("departure", "arrival"):
                trips[name] = pd.to_datetime(trips[name])
            mobility = {self.device[3:]: EVSchedule(trips, timestamp, index[-1]+step, {"forecast": True})}
            backed = rows.backed_load_kw
        self.history.append({"timestamp": timestamp, "issued_at": snapshot["issued_at"], "device": self.device})
        return replace(realized, data=data, ev_schedules=mobility, backed_load_kw=backed, providers={})
