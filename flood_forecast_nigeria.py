
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage, optimize, stats

# --------------------------------------------------------------------------
# Optional third-party imports - wrapped so that the module remains importable
# in reduced environments; methods that depend on a missing library will
# degrade gracefully with a logged warning.
# --------------------------------------------------------------------------
try:
    import rasterio
    from rasterio import features as rio_features
    from rasterio import mask as rio_mask
    from rasterio.transform import from_bounds as _rio_from_bounds
    _HAS_RASTERIO = True
except Exception:  # pragma: no cover - environment dependent
    rasterio = None
    rio_features = None
    rio_mask = None
    _rio_from_bounds = None
    _HAS_RASTERIO = False

try:
    import geopandas as gpd
    from shapely.geometry import (LineString, MultiPolygon, Point, Polygon,
                                   mapping, shape)
    from shapely.ops import unary_union
    _HAS_GEO = True
except Exception:  # pragma: no cover
    gpd = None
    LineString = MultiPolygon = Point = Polygon = shape = None
    mapping = unary_union = None
    _HAS_GEO = False

try:
    from pysheds.grid import Grid as _PyShedsGrid
    _HAS_PYSHEDS = True
except Exception:  # pragma: no cover
    _PyShedsGrid = None
    _HAS_PYSHEDS = False

try:
    import hydroeval as he
    _HAS_HYDROEVAL = True
except Exception:  # pragma: no cover
    he = None
    _HAS_HYDROEVAL = False

try:
    import pastas as ps
    _HAS_PASTAS = True
except Exception:  # pragma: no cover
    ps = None
    _HAS_PASTAS = False

try:
    import requests
    _HAS_REQUESTS = True
except Exception:  # pragma: no cover
    requests = None
    _HAS_REQUESTS = False

try:
    import cdsapi
    _HAS_CDSAPI = True
except Exception:  # pragma: no cover
    cdsapi = None
    _HAS_CDSAPI = False

try:
    import netCDF4  # noqa: F401
    _HAS_NETCDF = True
except Exception:  # pragma: no cover
    _HAS_NETCDF = False

try:
    import matplotlib as _mpl
    from matplotlib.figure import Figure
    _HAS_MPL = True
    plt = None  # imported lazily to avoid selecting a backend at import time
except Exception:  # pragma: no cover
    _mpl = None
    plt = None
    Figure = Any  # type: ignore
    _HAS_MPL = False


def _ensure_mpl():
    """Lazily configure a headless matplotlib backend and import pyplot."""
    global plt
    if plt is not None or not _HAS_MPL:
        return plt
    try:
        _mpl.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as _plt
    plt = _plt
    return plt

try:
    import folium
    from folium.features import GeoJson, GeoJsonTooltip
    from branca.element import MacroElement, Template
    _HAS_FOLIUM = True
except Exception:  # pragma: no cover
    folium = None
    GeoJson = GeoJsonTooltip = None
    MacroElement = Template = None
    _HAS_FOLIUM = False


# ===========================================================================
#                          1. Config  (dataclass)
# ===========================================================================
@dataclass
class Config:
    """Immutable container of system-wide constants and paths.

    All other classes accept a ``Config`` instance in their constructor and
    reference parameters through it, ensuring that no global state exists.
    """

    data_dir: Path = Path("./data")
    dem_path: Path = Path("./data/dem/nigeria_srtm30.tif")
    gauge_csv: Path = Path("./data/gauges/gauge_stations.csv")
    output_dir: Path = Path("./outputs")

    # Nigeria bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max).
    bbox: Tuple[float, float, float, float] = (2.5, 4.0, 14.7, 13.9)

    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    forecast_horizon_days: int = 7

    # x1 production store capacity (mm), x2 exchange coefficient (mm/day),
    # x3 routing store capacity (mm), x4 unit-hydrograph time-base (days).
    gr4j_default_params: Tuple[float, float, float, float] = (350.0, 0.0, 90.0, 1.7)

    muskingum_k: float = 1.0      # travel time (days)
    muskingum_x: float = 0.2      # weighting factor (0-0.5)

    alert_green: float = 1.0      # below normal
    alert_amber: float = 1.5      # elevated
    alert_red: float = 2.5        # dangerous

    min_years_required: int = 3   # threshold for "sparse" classification

    era5_variables: List[str] = field(default_factory=lambda: [
        "total_precipitation", "2m_temperature",
        "potential_evaporation", "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ])

    chirps_base_url: str = ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
                             "africa_daily/tifs/p05/")

    log_level: str = "INFO"

    use_cloud_storage: bool = False
    r2_bucket_name: str = ""
    r2_endpoint_url: str = ""
    r2_access_key: str = ""
    r2_secret_key: str = ""

    def __post_init__(self) -> None:
        """Create the full directory tree used by the pipeline."""
        self.data_dir = Path(self.data_dir)
        self.dem_path = Path(self.dem_path)
        self.gauge_csv = Path(self.gauge_csv)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "gauges").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "dem").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "met").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "chirps").mkdir(parents=True, exist_ok=True)


# ===========================================================================
#                         2. DataManager
# ===========================================================================
class DataManager:
    """Load, validate and clean all input hydrometeorological data.

    Responsibilities:
    - Parse the gauge-station CSV.
    - Flag stations with insufficient record length as *sparse*.
    - Download (or synthesise) CHIRPS rainfall and ERA5 met variables.
    - Provide merged daily meteorological forcing for downstream models.
    """

    def __init__(self, config: Config) -> None:
        """Instantiate a DataManager bound to a Config.

        Args:
            config: Global Config instance.
        """
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self.gauge_data: Dict[str, pd.DataFrame] = {}
        self.chirps: Optional[Any] = None
        self.era5: Optional[Any] = None
        self.sparse_stations: List[str] = []
        self._station_meta: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def load_gauge_data(self) -> Dict[str, pd.DataFrame]:
        """Read, QC and index the gauge-station CSV.

        Returns:
            Mapping ``{station_id: DataFrame}`` indexed by date with columns
            ``[discharge_m3s, stage_m]``.

        Raises:
            FileNotFoundError: If ``config.gauge_csv`` does not exist.
        """
        path = Path(self.config.gauge_csv)
        if not path.exists():
            raise FileNotFoundError(f"Gauge CSV not found at {path}")

        raw = pd.read_csv(path)
        required = {"station_id", "station_name", "latitude", "longitude",
                    "river", "date", "discharge_m3s", "stage_m"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"Gauge CSV missing columns: {missing}")

        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw = raw.dropna(subset=["date"])

        meta_cols = ["station_id", "station_name", "latitude", "longitude", "river"]
        self._station_meta = raw[meta_cols].drop_duplicates("station_id").reset_index(drop=True)

        self.gauge_data = {}
        self.sparse_stations = []
        for sid, sub in raw.groupby("station_id"):
            df = sub.sort_values("date").set_index("date")[["discharge_m3s", "stage_m"]].copy()

            # QC: flag negatives and 5-sigma positive outliers.
            mean = df["discharge_m3s"].mean(skipna=True)
            std = df["discharge_m3s"].std(skipna=True)
            if std is not None and std > 0:
                mask_bad = (df["discharge_m3s"] < 0) | (df["discharge_m3s"] > mean + 5 * std)
                n_bad = int(mask_bad.sum())
                if n_bad:
                    self._logger.info("Station %s: flagged %d QC outliers", sid, n_bad)
                df.loc[mask_bad, "discharge_m3s"] = np.nan
            else:
                df.loc[df["discharge_m3s"] < 0, "discharge_m3s"] = np.nan

            # Use pastas (if available) to obtain a robust gap summary.
            if _HAS_PASTAS and df["discharge_m3s"].notna().sum() > 30:
                try:
                    _ = ps.TimeSeries(df["discharge_m3s"].dropna(), name=str(sid))
                except Exception:
                    pass

            # Record-length test.
            valid = df["discharge_m3s"].dropna()
            if len(valid) > 0:
                years = (valid.index.max() - valid.index.min()).days / 365.25
            else:
                years = 0.0
            if years < self.config.min_years_required:
                self.sparse_stations.append(str(sid))
                self._logger.warning(
                    "Station %s has only %.2f yr of record - marked SPARSE", sid, years)

            self.gauge_data[str(sid)] = df

        self._logger.info("Loaded %d gauge stations (%d sparse)",
                          len(self.gauge_data), len(self.sparse_stations))
        return self.gauge_data

    # ------------------------------------------------------------------
    def download_chirps(self, year: int, month: int) -> Path:
        """Download one month of CHIRPS daily rainfall.

        Args:
            year: Four-digit year.
            month: Month number (1-12).

        Returns:
            Path to the saved file (may point to a placeholder on failure).

        Raises:
            RuntimeError: After three failed download attempts.
        """
        fname = f"chirps-v2.0.{year}.{month:02d}.tif.gz"
        url = f"{self.config.chirps_base_url}{year}/{fname}"
        out = Path(self.config.data_dir) / "chirps" / fname

        if out.exists():
            return out
        if not _HAS_REQUESTS:
            self._logger.warning("requests not installed - skipping CHIRPS %s", fname)
            return out

        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                resp = requests.get(url, timeout=30, stream=True)
                resp.raise_for_status()
                with open(out, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 15):
                        if chunk:
                            fh.write(chunk)
                self._logger.info("Downloaded CHIRPS %s", fname)
                return out
            except Exception as exc:  # pragma: no cover - network
                last_exc = exc
                self._logger.warning("CHIRPS attempt %d failed: %s", attempt, exc)
                time.sleep(delay)
                delay *= 2.0

        self._logger.error("CHIRPS download failed after 3 attempts: %s", last_exc)
        return out

    # ------------------------------------------------------------------
    def load_chirps_for_bbox(self, start: str, end: str) -> pd.DataFrame:
        """Return a daily area-mean CHIRPS precipitation series for the bbox.

        Args:
            start: ISO start date.
            end: ISO end date.

        Returns:
            DataFrame indexed by date with column ``precip_mm``. Falls back
            to :meth:`generate_synthetic_met` precipitation if real CHIRPS
            rasters are unavailable.
        """
        start_d = pd.to_datetime(start)
        end_d = pd.to_datetime(end)
        months = pd.date_range(start_d, end_d, freq="MS")

        records: List[Tuple[pd.Timestamp, float]] = []
        any_loaded = False
        for m in months:
            try:
                path = self.download_chirps(m.year, m.month)
            except Exception:
                continue
            if not (_HAS_RASTERIO and path.exists() and path.stat().st_size > 0):
                continue
            try:
                with rasterio.open(path) as src:
                    window = src.window(*self.config.bbox)
                    arr = src.read(1, window=window, masked=True)
                    mean = float(np.ma.masked_invalid(arr).mean())
                records.append((pd.Timestamp(m.year, m.month, 15), mean))
                any_loaded = True
            except Exception as exc:
                self._logger.warning("Could not read CHIRPS %s: %s", path.name, exc)

        if not any_loaded:
            self._logger.info("No CHIRPS rasters available - using synthetic precipitation")
            synth = self.generate_synthetic_met(start, end)
            return synth[["precip_mm"]]

        monthly = pd.DataFrame(records, columns=["date", "precip_mm"]).set_index("date")
        daily_index = pd.date_range(start_d, end_d, freq="D")
        daily = monthly.reindex(daily_index).interpolate(method="time").ffill().bfill()
        daily.index.name = "date"
        return daily

    # ------------------------------------------------------------------
    def download_era5(self, variables: List[str], start: str, end: str) -> Path:
        """Retrieve ERA5 reanalysis via the CDS API.

        Args:
            variables: CDS variable names.
            start: ISO start date.
            end: ISO end date.

        Returns:
            Path to the saved NetCDF file (may be empty on failure).
        """
        out = Path(self.config.data_dir) / "met" / "era5.nc"
        if out.exists():
            return out
        if not _HAS_CDSAPI:
            self._logger.error(
                "cdsapi not installed; configure ~/.cdsapirc from "
                "https://cds.climate.copernicus.eu/api-how-to and "
                "install the 'cdsapi' package")
            return out
        try:
            c = cdsapi.Client()
            start_d = pd.to_datetime(start)
            end_d = pd.to_datetime(end)
            years = [str(y) for y in range(start_d.year, end_d.year + 1)]
            months = [f"{m:02d}" for m in range(1, 13)]
            days = [f"{d:02d}" for d in range(1, 32)]
            area = [self.config.bbox[3], self.config.bbox[0],
                    self.config.bbox[1], self.config.bbox[2]]  # N,W,S,E
            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "variable": variables,
                    "year": years,
                    "month": months,
                    "day": days,
                    "time": ["12:00"],
                    "area": area,
                    "format": "netcdf",
                },
                str(out),
            )
            self._logger.info("ERA5 downloaded to %s", out)
        except Exception as exc:  # pragma: no cover - network
            self._logger.error(
                "ERA5 download failed: %s. Place a valid ~/.cdsapirc file in "
                "your home directory. Falling back to synthetic met data.", exc)
        return out

    # ------------------------------------------------------------------
    def generate_synthetic_met(self, start: str, end: str) -> pd.DataFrame:
        """Produce plausible synthetic daily forcing for Nigeria.

        Temperature follows a sinusoidal annual cycle; precipitation is a
        gamma mixture with seasonal parameter switching; PET uses the
        Hargreaves formula on synthetic ``T_min`` / ``T_max``.

        Args:
            start: ISO start date.
            end: ISO end date.

        Returns:
            DataFrame indexed by date with columns
            ``[precip_mm, temp_c, pet_mm]``.
        """
        idx = pd.date_range(start, end, freq="D")
        n = len(idx)
        doy = np.array([d.timetuple().tm_yday for d in idx])

        rng = np.random.default_rng(seed=20260421)

        # Temperature: mean 27 degC, 5 degC amplitude, peak near DOY 180.
        temp = 27.0 + 5.0 * np.sin(2 * np.pi * (doy - 90) / 365.25) + rng.normal(0, 1.0, n)

        # Seasonal precipitation.
        wet = np.isin(np.array([d.month for d in idx]), [5, 6, 7, 8, 9, 10])
        precip = np.where(wet,
                          rng.gamma(shape=0.8, scale=12.0, size=n),
                          rng.gamma(shape=0.3, scale=2.5, size=n))
        precip = np.clip(precip, 0.0, None)

        # Hargreaves PET with +/-4 degC daily range.
        tmax = temp + 4.0
        tmin = temp - 4.0
        lat = np.mean(self.config.bbox[1::2]) * np.pi / 180.0
        dr = 1.0 + 0.033 * np.cos(2 * np.pi * doy / 365.25)
        sol_decl = 0.409 * np.sin(2 * np.pi * doy / 365.25 - 1.39)
        ws = np.arccos(np.clip(-np.tan(lat) * np.tan(sol_decl), -1, 1))
        ra = (24 * 60 / np.pi) * 0.0820 * dr * (
            ws * np.sin(lat) * np.sin(sol_decl)
            + np.cos(lat) * np.cos(sol_decl) * np.sin(ws))
        pet = 0.0023 * (temp + 17.8) * np.sqrt(np.maximum(tmax - tmin, 0.1)) * ra
        pet = np.clip(pet, 0.0, None)

        df = pd.DataFrame({"precip_mm": precip, "temp_c": temp, "pet_mm": pet}, index=idx)
        df.index.name = "date"
        return df

    # ------------------------------------------------------------------
    def load_era5(self) -> pd.DataFrame:
        """Load an ERA5 NetCDF and return daily area-means.

        Returns:
            DataFrame indexed by date with columns
            ``[precip_mm, temp_c, pet_mm]``. Falls back to synthetic data
            if no valid NetCDF is present.
        """
        path = Path(self.config.data_dir) / "met" / "era5.nc"
        if not (_HAS_NETCDF and path.exists() and path.stat().st_size > 0):
            self._logger.info("ERA5 NetCDF missing - using synthetic forcing")
            return self.generate_synthetic_met(self.config.start_date, self.config.end_date)
        try:
            ds = netCDF4.Dataset(str(path))  # type: ignore[name-defined]
            t = ds.variables["time"]
            dates = netCDF4.num2date(t[:], t.units)  # type: ignore[name-defined]
            dates = pd.to_datetime([str(d)[:10] for d in dates])
            tp = ds.variables.get("tp")
            t2m = ds.variables.get("t2m")
            pev = ds.variables.get("pev")
            precip = (np.asarray(tp[:]).mean(axis=(1, 2)) * 1000.0) if tp is not None else np.zeros(len(dates))
            temp = (np.asarray(t2m[:]).mean(axis=(1, 2)) - 273.15) if t2m is not None else np.full(len(dates), 27.0)
            pet = (np.asarray(pev[:]).mean(axis=(1, 2)) * -1000.0) if pev is not None else np.full(len(dates), 4.0)
            ds.close()
            df = pd.DataFrame({"precip_mm": precip, "temp_c": temp, "pet_mm": pet}, index=dates)
            df.index.name = "date"
            return df
        except Exception as exc:
            self._logger.error("ERA5 read failed (%s) - using synthetic forcing", exc)
            return self.generate_synthetic_met(self.config.start_date, self.config.end_date)

    # ------------------------------------------------------------------
    def merge_met_data(self, chirps: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
        """Combine CHIRPS rainfall with ERA5 temperature and PET.

        Where both products provide precipitation, CHIRPS is preferred for
        its higher spatial resolution. Remaining gaps are linearly
        interpolated up to seven days.

        Args:
            chirps: DataFrame with at least ``precip_mm``.
            era5:   DataFrame with at least ``temp_c`` and ``pet_mm``.

        Returns:
            Merged daily DataFrame indexed by date.
        """
        idx = pd.date_range(self.config.start_date, self.config.end_date, freq="D")
        out = pd.DataFrame(index=idx)
        out.index.name = "date"

        p_chirps = chirps["precip_mm"].reindex(idx) if "precip_mm" in chirps.columns else None
        p_era5 = era5["precip_mm"].reindex(idx) if "precip_mm" in era5.columns else None
        if p_chirps is not None:
            out["precip_mm"] = p_chirps
            if p_era5 is not None:
                out["precip_mm"] = out["precip_mm"].fillna(p_era5)
        elif p_era5 is not None:
            out["precip_mm"] = p_era5
        else:
            out["precip_mm"] = 0.0

        out["temp_c"] = era5["temp_c"].reindex(idx) if "temp_c" in era5.columns else 27.0
        out["pet_mm"] = era5["pet_mm"].reindex(idx) if "pet_mm" in era5.columns else 4.0

        out = out.interpolate(method="linear", limit=7).ffill().bfill()
        out["precip_mm"] = out["precip_mm"].clip(lower=0.0)
        out["pet_mm"] = out["pet_mm"].clip(lower=0.0)
        return out

    # ------------------------------------------------------------------
    def get_station_geodataframe(self) -> "gpd.GeoDataFrame":
        """Build a GeoDataFrame of all loaded stations.

        Returns:
            GeoDataFrame in EPSG:4326 with columns
            ``[station_id, station_name, latitude, longitude, river,
            is_sparse, geometry]``.
        """
        if self._station_meta is None:
            self.load_gauge_data()
        meta = self._station_meta.copy() if self._station_meta is not None else pd.DataFrame()
        meta["is_sparse"] = meta["station_id"].astype(str).isin(self.sparse_stations)
        if _HAS_GEO:
            geom = gpd.points_from_xy(meta["longitude"], meta["latitude"])
            return gpd.GeoDataFrame(meta, geometry=geom, crs="EPSG:4326")
        return meta  # type: ignore[return-value]


# ===========================================================================
#                         3. GapFiller
# ===========================================================================
class GapFiller:
    """Synthesise complete daily discharge series for sparse stations."""

    def __init__(self, config: Config, gauge_data: Dict[str, pd.DataFrame],
                 met_data: pd.DataFrame) -> None:
        """Bind inputs used for all filling strategies.

        Args:
            config: Global Config.
            gauge_data: Mapping of station_id -> DataFrame with discharge.
            met_data: Daily forcing DataFrame produced by DataManager.
        """
        self.config = config
        self.gauge_data = gauge_data
        self.met_data = met_data
        self._logger = logging.getLogger(self.__class__.__name__)
        try:
            self._meta = pd.read_csv(config.gauge_csv).drop_duplicates("station_id")
            self._meta = self._meta[["station_id", "latitude", "longitude", "river"]]
            self._meta["station_id"] = self._meta["station_id"].astype(str)
        except Exception:
            self._meta = pd.DataFrame(columns=["station_id", "latitude", "longitude", "river"])

    # ------------------------------------------------------------------
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometres between two WGS84 points."""
        r = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
        return 2.0 * r * math.asin(math.sqrt(a))

    # ------------------------------------------------------------------
    def _estimate_catchment_area(self, station_id: str) -> float:
        """Heuristic catchment-area estimate (km^2) when none is available."""
        try:
            river = self._meta.loc[self._meta["station_id"] == station_id, "river"].iloc[0].lower()
        except Exception:
            river = ""
        if "niger" in river:
            return 150_000.0
        if "benue" in river:
            return 80_000.0
        if "kaduna" in river or "sokoto" in river:
            return 30_000.0
        if "osun" in river or "ogun" in river or "imo" in river:
            return 8_000.0
        return 15_000.0

    # ------------------------------------------------------------------
    def fill_by_spatial_interpolation(self, station_id: str) -> pd.Series:
        """Fill a station from neighbours within 300 km using IDW.

        Args:
            station_id: Target station identifier.

        Returns:
            Daily discharge Series covering the full simulation window.

        Raises:
            KeyError: If the station is not present in the metadata table.
        """
        target = self._meta.loc[self._meta["station_id"] == station_id]
        if target.empty:
            raise KeyError(f"Station {station_id} not in metadata")
        t_lat = float(target["latitude"].iloc[0])
        t_lon = float(target["longitude"].iloc[0])

        neighbours: List[Tuple[str, float]] = []
        for sid, df in self.gauge_data.items():
            if sid == station_id or sid in self.config.__dict__.get("sparse_stations", []):
                continue
            row = self._meta.loc[self._meta["station_id"] == sid]
            if row.empty:
                continue
            dist = self._haversine(t_lat, t_lon,
                                   float(row["latitude"].iloc[0]),
                                   float(row["longitude"].iloc[0]))
            if 0 < dist <= 300.0 and df["discharge_m3s"].notna().sum() > 30:
                neighbours.append((sid, dist))

        if len(neighbours) < 2:
            self._logger.info("Station %s has %d neighbours - using HBV fallback",
                              station_id, len(neighbours))
            return self.fill_by_hbv_simulation(station_id, self._estimate_catchment_area(station_id))

        idx = pd.date_range(self.config.start_date, self.config.end_date, freq="D")
        accum = np.zeros(len(idx))
        weights = np.zeros(len(idx))
        for sid, dist in neighbours:
            w = 1.0 / max(dist, 1.0) ** 2
            series = (self.gauge_data[sid]["discharge_m3s"]
                      .reindex(idx).interpolate(limit=7).values)
            valid = ~np.isnan(series)
            accum[valid] += w * series[valid]
            weights[valid] += w
        filled = np.where(weights > 0, accum / np.maximum(weights, 1e-9), np.nan)
        series = pd.Series(filled, index=idx, name=station_id)
        series = series.interpolate(limit=30).bfill().ffill().fillna(0.0)
        return series

    # ------------------------------------------------------------------
    def fill_by_hbv_simulation(self, station_id: str,
                               catchment_area_km2: float) -> pd.Series:
        """Run a simplified four-parameter HBV-light model.

        Args:
            station_id: Station identifier (used only for the series name).
            catchment_area_km2: Basin area in square kilometres.

        Returns:
            Daily discharge Series in m^3/s covering the forcing period.
        """
        FC, BETA, LP, K = 250.0, 2.0, 0.7, 0.05
        P = self.met_data["precip_mm"].values
        E = self.met_data["pet_mm"].values
        n = len(P)

        SM = FC / 2.0
        UZ = 10.0
        Q_out = np.empty(n, dtype=float)

        for t in range(n):
            # Soil-moisture accounting.
            recharge = P[t] * (SM / FC) ** BETA if FC > 0 else 0.0
            infil = P[t] - recharge
            AET = E[t] * min(1.0, SM / (LP * FC)) if FC > 0 else E[t]
            SM = max(0.0, SM + infil - AET)
            if SM > FC:
                recharge += SM - FC
                SM = FC
            UZ = max(0.0, UZ + recharge)
            q_mm = UZ * K
            UZ = max(0.0, UZ - q_mm)
            Q_out[t] = q_mm * catchment_area_km2 / 86400.0

        series = pd.Series(Q_out, index=self.met_data.index, name=station_id)
        return series

    # ------------------------------------------------------------------
    def fill_by_lstm_regression(self, station_id: str) -> pd.Series:
        """Fit a windowed multivariate linear regression (LSTM surrogate).

        Features are the preceding 14 days of precipitation and the
        preceding 7 days of discharge. Coefficients are estimated with
        :func:`numpy.linalg.lstsq` on the valid observed subset and then
        used to reconstruct a continuous series over the whole window.

        Args:
            station_id: Station identifier.

        Returns:
            Full daily discharge Series.
        """
        if station_id not in self.gauge_data:
            return self.fill_by_hbv_simulation(
                station_id, self._estimate_catchment_area(station_id))

        idx = pd.date_range(self.config.start_date, self.config.end_date, freq="D")
        P = self.met_data["precip_mm"].reindex(idx).fillna(0.0).values
        Q_obs = (self.gauge_data[station_id]["discharge_m3s"]
                 .reindex(idx).astype(float).values)

        n = len(idx)
        lagsP, lagsQ = 14, 7
        start = max(lagsP, lagsQ)

        # Build design matrix on valid rows.
        rows = []
        targets = []
        for t in range(start, n):
            if np.isnan(Q_obs[t]):
                continue
            q_lags = Q_obs[t - lagsQ:t]
            if np.any(np.isnan(q_lags)):
                continue
            feats = np.concatenate([[1.0], P[t - lagsP:t], q_lags])
            rows.append(feats)
            targets.append(Q_obs[t])
        if len(rows) < 30:
            self._logger.info("Station %s: insufficient training rows for LSTM "
                              "regression (%d) - falling back to HBV",
                              station_id, len(rows))
            return self.fill_by_hbv_simulation(
                station_id, self._estimate_catchment_area(station_id))
        X = np.asarray(rows)
        y = np.asarray(targets)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

        # Auto-regressive reconstruction.
        Q_filled = Q_obs.copy()
        # Seed any leading NaN with HBV.
        hbv = self.fill_by_hbv_simulation(
            station_id, self._estimate_catchment_area(station_id)).values
        mask_nan = np.isnan(Q_filled)
        Q_filled[mask_nan] = hbv[mask_nan]
        for t in range(start, n):
            if not np.isnan(Q_obs[t]):
                continue  # keep observed
            feats = np.concatenate([[1.0], P[t - lagsP:t], Q_filled[t - lagsQ:t]])
            Q_filled[t] = max(0.0, float(feats @ coef))
        return pd.Series(Q_filled, index=idx, name=station_id)

    # ------------------------------------------------------------------
    def fill_all(self) -> Dict[str, pd.Series]:
        """Apply the appropriate method to every sparse station.

        Selection rules:
            - 0 years of record                         -> HBV
            - 0 < years < min_years_required            -> LSTM regression
            - sparse with at least two valid neighbours -> spatial IDW
            - otherwise                                 -> HBV

        Returns:
            Mapping ``{station_id: filled daily Series}``.
        """
        # Expose the sparse list to helpers via the Config attribute dict.
        setattr(self.config, "sparse_stations", list(
            getattr(self.config, "sparse_stations", [])) + [])
        sparse = list(getattr(self.config, "sparse_stations", []))
        if not sparse:
            # Caller may have stored the list on DataManager instead.
            sparse = []
        # If nothing in the config, derive from gauge_data vs min_years_required.
        if not sparse:
            for sid, df in self.gauge_data.items():
                v = df["discharge_m3s"].dropna()
                if len(v) == 0:
                    years = 0.0
                else:
                    years = (v.index.max() - v.index.min()).days / 365.25
                if years < self.config.min_years_required:
                    sparse.append(sid)

        filled: Dict[str, pd.Series] = {}
        for sid in sparse:
            df = self.gauge_data.get(sid)
            if df is None or df["discharge_m3s"].dropna().empty:
                years = 0.0
                n_valid = 0
            else:
                valid = df["discharge_m3s"].dropna()
                years = (valid.index.max() - valid.index.min()).days / 365.25
                n_valid = len(valid)

            n_neighbours = sum(
                1 for other_sid, other_df in self.gauge_data.items()
                if other_sid != sid and other_sid not in sparse
                and other_df["discharge_m3s"].notna().sum() > 30)

            if years == 0 or n_valid == 0:
                method = "HBV"
                series = self.fill_by_hbv_simulation(
                    sid, self._estimate_catchment_area(sid))
            elif 0 < years < self.config.min_years_required and n_neighbours < 2:
                method = "HBV"
                series = self.fill_by_hbv_simulation(
                    sid, self._estimate_catchment_area(sid))
            elif 0 < years < self.config.min_years_required:
                method = "LSTM-regression"
                series = self.fill_by_lstm_regression(sid)
            elif n_neighbours >= 2:
                method = "spatial-IDW"
                series = self.fill_by_spatial_interpolation(sid)
            else:
                method = "HBV"
                series = self.fill_by_hbv_simulation(
                    sid, self._estimate_catchment_area(sid))

            self._logger.info("Gap-filled station %s using %s", sid, method)
            filled[sid] = series

            # Merge back into gauge_data.
            if sid in self.gauge_data:
                df = self.gauge_data[sid]
                merged = df.reindex(series.index)
                merged["discharge_m3s"] = merged["discharge_m3s"].combine_first(series)
                self.gauge_data[sid] = merged
            else:
                self.gauge_data[sid] = pd.DataFrame(
                    {"discharge_m3s": series, "stage_m": np.nan}, index=series.index)
        return filled


# ===========================================================================
#                         4. WatershedDelineator
# ===========================================================================
class WatershedDelineator:
    """Extract basins, flow-direction, flow-accumulation and river network."""

    def __init__(self, config: Config) -> None:
        """Bind the delineator to a Config.

        Args:
            config: Global Config.
        """
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self.dem: Optional[np.ndarray] = None
        self.profile: Optional[Dict[str, Any]] = None
        self.flow_dir: Optional[np.ndarray] = None
        self.flow_acc: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def load_dem(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Read the DEM or create a synthetic one.

        Returns:
            Tuple ``(dem_array, rasterio_profile)``.
        """
        path = Path(self.config.dem_path)
        if not path.exists():
            self._logger.warning("DEM not found at %s - generating synthetic", path)
            self.generate_synthetic_dem()
        if _HAS_RASTERIO and path.exists():
            with rasterio.open(path) as src:
                self.dem = src.read(1).astype(float)
                self.profile = dict(src.profile)
        else:
            # Rasterio missing: build in-memory DEM.
            self.dem, self.profile = self._build_synthetic_dem_array()
        return self.dem, self.profile

    # ------------------------------------------------------------------
    def _build_synthetic_dem_array(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute the synthetic DEM array and profile dict."""
        lon_min, lat_min, lon_max, lat_max = self.config.bbox
        res = 0.05  # degrees - ~5 km
        ncols = max(32, int(round((lon_max - lon_min) / res)))
        nrows = max(32, int(round((lat_max - lat_min) / res)))

        rng = np.random.default_rng(seed=20260421)
        noise_f = rng.standard_normal((nrows, ncols)) + 1j * rng.standard_normal((nrows, ncols))
        u = np.fft.fftfreq(nrows)[:, None]
        v = np.fft.fftfreq(ncols)[None, :]
        freq = np.sqrt(u * u + v * v)
        freq[0, 0] = 1.0
        noise_f *= freq ** (-1.5)
        noise_f[0, 0] = 0.0
        terrain = np.real(np.fft.ifft2(noise_f))
        terrain = (terrain - terrain.min()) / max(terrain.max() - terrain.min(), 1e-9)

        # North-south gradient: higher in north (Jos plateau area).
        lat_grid = np.linspace(lat_max, lat_min, nrows)[:, None]
        gradient = (lat_grid - lat_min) / (lat_max - lat_min)  # 1 at north, 0 at south
        dem = 50.0 + 1200.0 * gradient + 400.0 * terrain

        transform = None
        if _HAS_RASTERIO:
            transform = _rio_from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
        profile = {
            "driver": "GTiff", "dtype": "float32",
            "width": ncols, "height": nrows, "count": 1,
            "crs": "EPSG:4326", "transform": transform,
            "nodata": -9999.0,
        }
        return dem.astype(np.float32), profile

    # ------------------------------------------------------------------
    def generate_synthetic_dem(self) -> Path:
        """Write a plausible synthetic Nigeria DEM GeoTIFF.

        Returns:
            Path to the generated file.
        """
        dem, profile = self._build_synthetic_dem_array()
        path = Path(self.config.dem_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_RASTERIO:
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(dem, 1)
            self._logger.info("Synthetic DEM written to %s", path)
        else:
            np.save(path.with_suffix(".npy"), dem)
            self._logger.warning("rasterio missing - saved synthetic DEM as .npy")
        self.dem = dem
        self.profile = profile
        return path

    # ------------------------------------------------------------------
    def _load_station_locations(self) -> pd.DataFrame:
        """Return the station metadata table required for delineation."""
        meta = pd.read_csv(self.config.gauge_csv).drop_duplicates("station_id")
        meta["station_id"] = meta["station_id"].astype(str)
        return meta[["station_id", "latitude", "longitude", "river"]]

    # ------------------------------------------------------------------
    def _delineate_numpy(self, dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pure NumPy D8 flow-direction and accumulation fall-back."""
        rows, cols = dem.shape
        # D8 encodings following the ESRI convention
        #   32 64 128
        #   16     1
        #    8  4  2
        dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])
        dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])
        codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.int32)

        fdir = np.zeros((rows, cols), dtype=np.int32)
        max_drop = np.full((rows, cols), -np.inf)
        for k in range(8):
            shifted = np.full_like(dem, np.nan)
            r0, r1 = max(0, dy[k]), min(rows, rows + dy[k])
            c0, c1 = max(0, dx[k]), min(cols, cols + dx[k])
            shifted[r0:r1, c0:c1] = dem[r0 - dy[k]:r1 - dy[k], c0 - dx[k]:c1 - dx[k]]
            dist = math.sqrt(dx[k] ** 2 + dy[k] ** 2)
            drop = (dem - shifted) / dist
            better = drop > max_drop
            max_drop = np.where(better, drop, max_drop)
            fdir = np.where(better, codes[k], fdir)
        fdir[max_drop <= 0] = 0

        # Flow accumulation by topological ordering along elevation.
        acc = np.ones((rows, cols), dtype=np.float32)
        order = np.argsort(dem.ravel())[::-1]
        for idx in order:
            i = idx // cols
            j = idx % cols
            c = fdir[i, j]
            if c == 0:
                continue
            k = int(np.log2(c))
            ni, nj = i + dy[k], j + dx[k]
            if 0 <= ni < rows and 0 <= nj < cols:
                acc[ni, nj] += acc[i, j]
        return fdir, acc

    # ------------------------------------------------------------------
    def delineate(self) -> Dict[str, Any]:
        """Run the full watershed-delineation pipeline.

        Returns:
            Dictionary with keys
            ``basins``, ``flow_dir``, ``flow_acc``, ``river_network`` and
            ``snapped_stations``.
        """
        if self.dem is None:
            self.load_dem()
        assert self.dem is not None and self.profile is not None

        meta = self._load_station_locations()

        fdir: Optional[np.ndarray] = None
        facc: Optional[np.ndarray] = None
        catchment_masks: Dict[str, np.ndarray] = {}
        snapped_coords: Dict[str, Tuple[float, float]] = {}

        if _HAS_PYSHEDS:
            try:
                grid = _PyShedsGrid.from_raster(str(self.config.dem_path))
                dem_ras = grid.read_raster(str(self.config.dem_path))
                flooded = grid.fill_depressions(dem_ras)
                inflated = grid.resolve_flats(flooded)
                fdir_ras = grid.flowdir(inflated)
                acc_ras = grid.accumulation(fdir_ras)
                fdir = np.asarray(fdir_ras)
                facc = np.asarray(acc_ras)
                for _, row in meta.iterrows():
                    try:
                        x_snap, y_snap = grid.snap_to_mask(
                            acc_ras > max(1000, facc.max() * 0.001),
                            (float(row["longitude"]), float(row["latitude"])))
                    except Exception:
                        x_snap, y_snap = float(row["longitude"]), float(row["latitude"])
                    snapped_coords[str(row["station_id"])] = (x_snap, y_snap)
                    try:
                        catch = grid.catchment(x=x_snap, y=y_snap,
                                                fdir=fdir_ras, xytype="coordinate")
                        catchment_masks[str(row["station_id"])] = np.asarray(catch).astype(bool)
                    except Exception as exc:
                        self._logger.warning("pysheds catchment failed for %s: %s",
                                              row["station_id"], exc)
            except Exception as exc:
                self._logger.warning("pysheds pipeline failed (%s); using NumPy fall-back", exc)
                fdir = None

        if fdir is None:
            fdir, facc = self._delineate_numpy(self.dem)
            for _, row in meta.iterrows():
                i, j = self._lonlat_to_pixel(float(row["longitude"]),
                                              float(row["latitude"]))
                # Snap to highest accumulation within a 5-pixel window.
                if facc is not None:
                    r0, r1 = max(0, i - 5), min(facc.shape[0], i + 6)
                    c0, c1 = max(0, j - 5), min(facc.shape[1], j + 6)
                    sub = facc[r0:r1, c0:c1]
                    di, dj = np.unravel_index(int(np.argmax(sub)), sub.shape)
                    i, j = r0 + int(di), c0 + int(dj)
                snap_lon, snap_lat = self._pixel_to_lonlat(i, j)
                snapped_coords[str(row["station_id"])] = (snap_lon, snap_lat)
                catchment_masks[str(row["station_id"])] = self._fallback_catchment(
                    fdir, (i, j))

        self.flow_dir = fdir
        self.flow_acc = facc

        basins = self._masks_to_geodataframe(catchment_masks, meta)
        river_net = self._extract_river_network(facc)
        snapped = self._snapped_to_gdf(snapped_coords, meta)

        return {
            "basins": basins,
            "flow_dir": fdir,
            "flow_acc": facc,
            "river_network": river_net,
            "snapped_stations": snapped,
            "catchment_masks": catchment_masks,
        }

    # ------------------------------------------------------------------
    def _lonlat_to_pixel(self, lon: float, lat: float) -> Tuple[int, int]:
        """Convert WGS84 coordinates to DEM pixel indices."""
        lon_min, lat_min, lon_max, lat_max = self.config.bbox
        assert self.dem is not None
        rows, cols = self.dem.shape
        j = int(np.clip(round((lon - lon_min) / (lon_max - lon_min) * (cols - 1)), 0, cols - 1))
        i = int(np.clip(round((lat_max - lat) / (lat_max - lat_min) * (rows - 1)), 0, rows - 1))
        return i, j

    # ------------------------------------------------------------------
    def _pixel_to_lonlat(self, i: int, j: int) -> Tuple[float, float]:
        """Convert DEM pixel indices to WGS84 coordinates."""
        lon_min, lat_min, lon_max, lat_max = self.config.bbox
        assert self.dem is not None
        rows, cols = self.dem.shape
        lon = lon_min + (j / max(cols - 1, 1)) * (lon_max - lon_min)
        lat = lat_max - (i / max(rows - 1, 1)) * (lat_max - lat_min)
        return lon, lat

    # ------------------------------------------------------------------
    def _fallback_catchment(self, fdir: np.ndarray,
                            outlet: Tuple[int, int]) -> np.ndarray:
        """Very simple upstream-traversal catchment for the NumPy fall-back."""
        rows, cols = fdir.shape
        mask = np.zeros_like(fdir, dtype=bool)
        oi, oj = outlet
        if not (0 <= oi < rows and 0 <= oj < cols):
            return mask
        mask[oi, oj] = True

        # Map D8 code back to (dy, dx) of the downstream neighbour.
        code_to_delta = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
                         16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}
        # Iterative BFS upstream.
        stack = [outlet]
        while stack:
            i, j = stack.pop()
            for code, (dy, dx) in code_to_delta.items():
                ni, nj = i - dy, j - dx
                if not (0 <= ni < rows and 0 <= nj < cols):
                    continue
                if mask[ni, nj]:
                    continue
                if fdir[ni, nj] == code:
                    mask[ni, nj] = True
                    stack.append((ni, nj))
        return mask

    # ------------------------------------------------------------------
    def _masks_to_geodataframe(self, masks: Dict[str, np.ndarray],
                                meta: pd.DataFrame) -> Any:
        """Convert catchment raster masks into a basin GeoDataFrame."""
        if not _HAS_GEO or not _HAS_RASTERIO:
            rows = []
            for sid, mask in masks.items():
                area = float(mask.sum()) * self._pixel_area_km2()
                rows.append({"station_id": sid, "area_km2": area, "geometry": None})
            return pd.DataFrame(rows)

        assert self.profile is not None
        transform = self.profile.get("transform")
        records = []
        for sid, mask in masks.items():
            if mask.sum() == 0:
                poly = None
            else:
                shapes_iter = rio_features.shapes(
                    mask.astype(np.uint8), mask=mask, transform=transform)
                polys = [shape(g) for g, v in shapes_iter if v == 1]
                if not polys:
                    poly = None
                else:
                    try:
                        poly = unary_union(polys)
                    except Exception:
                        poly = polys[0]
            area_km2 = float(mask.sum()) * self._pixel_area_km2()
            river = meta.loc[meta["station_id"] == sid, "river"]
            records.append({
                "station_id": sid,
                "river": river.iloc[0] if not river.empty else "",
                "area_km2": area_km2,
                "geometry": poly,
            })
        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        return gdf

    # ------------------------------------------------------------------
    def _pixel_area_km2(self) -> float:
        """Approximate DEM pixel area in km^2 (equirectangular)."""
        assert self.dem is not None
        lon_min, lat_min, lon_max, lat_max = self.config.bbox
        rows, cols = self.dem.shape
        d_lat = (lat_max - lat_min) / max(rows, 1)
        d_lon = (lon_max - lon_min) / max(cols, 1)
        mean_lat = 0.5 * (lat_min + lat_max)
        km_per_deg = 111.32
        return abs(d_lat * km_per_deg) * abs(d_lon * km_per_deg * math.cos(math.radians(mean_lat)))

    # ------------------------------------------------------------------
    def _extract_river_network(self, facc: Optional[np.ndarray]) -> Any:
        """Trace the river network above a dynamic accumulation threshold."""
        if facc is None:
            return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry",
                                     crs="EPSG:4326") if _HAS_GEO else pd.DataFrame()
        threshold = max(500.0, float(facc.max()) * 0.002)
        river_mask = facc > threshold
        if not _HAS_GEO:
            return pd.DataFrame()
        rows, cols = river_mask.shape
        lines = []
        # Trace horizontal/vertical segments for visualisation only.
        visited = np.zeros_like(river_mask, dtype=bool)
        step = max(1, min(rows, cols) // 80)
        for i in range(0, rows, step):
            run_start = None
            for j in range(cols):
                if river_mask[i, j] and not visited[i, j]:
                    visited[i, j] = True
                    if run_start is None:
                        run_start = j
                else:
                    if run_start is not None and j - run_start >= 2:
                        lon_a, lat_a = self._pixel_to_lonlat(i, run_start)
                        lon_b, lat_b = self._pixel_to_lonlat(i, j - 1)
                        lines.append(LineString([(lon_a, lat_a), (lon_b, lat_b)]))
                    run_start = None
        return gpd.GeoDataFrame({"geometry": lines}, geometry="geometry",
                                 crs="EPSG:4326")

    # ------------------------------------------------------------------
    def _snapped_to_gdf(self, coords: Dict[str, Tuple[float, float]],
                        meta: pd.DataFrame) -> Any:
        """Build a GeoDataFrame from snapped station coordinates."""
        if not _HAS_GEO:
            return pd.DataFrame([
                {"station_id": sid, "lon": lon, "lat": lat}
                for sid, (lon, lat) in coords.items()
            ])
        geoms = [Point(lon, lat) for _, (lon, lat) in coords.items()]
        df = pd.DataFrame({
            "station_id": list(coords.keys()),
            "lon": [c[0] for c in coords.values()],
            "lat": [c[1] for c in coords.values()],
        })
        df = df.merge(meta, on="station_id", how="left")
        return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")

    # ------------------------------------------------------------------
    def get_basin_mean_precip(self, basins: Any, chirps_path: Path) -> pd.DataFrame:
        """Compute per-basin daily mean precipitation from a CHIRPS raster.

        Args:
            basins: Basin GeoDataFrame (one row per station).
            chirps_path: Path to a CHIRPS GeoTIFF or directory.

        Returns:
            DataFrame indexed by date with a column per station_id.
        """
        idx = pd.date_range(self.config.start_date, self.config.end_date, freq="D")
        if not (_HAS_RASTERIO and _HAS_GEO and Path(chirps_path).exists()):
            rng = np.random.default_rng(seed=20260421)
            data = {}
            for sid in basins["station_id"]:
                wet = np.isin(np.array([d.month for d in idx]), [5, 6, 7, 8, 9, 10])
                p = np.where(wet, rng.gamma(0.8, 12.0, len(idx)),
                             rng.gamma(0.3, 2.5, len(idx)))
                data[sid] = np.clip(p, 0.0, None)
            df = pd.DataFrame(data, index=idx)
            df.index.name = "date"
            return df
        # Real-data path would loop over CHIRPS files and clip each basin polygon.
        df = pd.DataFrame(index=idx)
        with rasterio.open(chirps_path) as src:
            for _, row in basins.iterrows():
                if row["geometry"] is None:
                    df[row["station_id"]] = 0.0
                    continue
                try:
                    out, _ = rio_mask.mask(src, [mapping(row["geometry"])], crop=True)
                    mean = float(np.ma.masked_invalid(out).mean())
                except Exception:
                    mean = 0.0
                df[row["station_id"]] = mean
        return df

    # ------------------------------------------------------------------
    def compute_upstream_area(self, station_id: str) -> float:
        """Upstream catchment area (km^2) from the flow-accumulation grid.

        Args:
            station_id: Station identifier.

        Returns:
            Area in square kilometres.
        """
        if self.flow_acc is None or self.dem is None:
            return 10_000.0
        meta = self._load_station_locations()
        row = meta.loc[meta["station_id"] == station_id]
        if row.empty:
            return 10_000.0
        i, j = self._lonlat_to_pixel(float(row["longitude"].iloc[0]),
                                      float(row["latitude"].iloc[0]))
        n_cells = float(self.flow_acc[i, j])
        return n_cells * self._pixel_area_km2()


# ===========================================================================
#                         5. HydrologicalModel
# ===========================================================================
class HydrologicalModel:
    """Per-basin GR4J rainfall-runoff model with calibration and forecasting."""

    def __init__(self, config: Config, basins: Any,
                 gauge_data: Dict[str, pd.DataFrame],
                 met_data: pd.DataFrame) -> None:
        """Bind model inputs.

        Args:
            config: Global Config.
            basins: Basin GeoDataFrame from WatershedDelineator.
            gauge_data: Observed / gap-filled discharge per station.
            met_data: Daily forcing DataFrame (precip, pet, temp).
        """
        self.config = config
        self.basins = basins
        self.gauge_data = gauge_data
        self.met_data = met_data
        self._logger = logging.getLogger(self.__class__.__name__)
        self.calibrated_params: Dict[str, Tuple[float, float, float, float]] = {}
        self.final_states: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    def _basin_area(self, station_id: str) -> float:
        """Return the basin area in km^2 for a station."""
        try:
            row = self.basins.loc[self.basins["station_id"] == station_id]
            if not row.empty:
                return float(row["area_km2"].iloc[0])
        except Exception:
            pass
        return 10_000.0

    # ------------------------------------------------------------------
    @staticmethod
    def _uh_ordinates(x4: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the S-curve unit-hydrograph ordinates UH1 and UH2.

        See Perrin et al. (2003) Eqns. 11-14. UH1 has ``ceil(x4)`` ordinates
        acting on the 90 % branch feeding the routing store; UH2 has
        ``ceil(2*x4)`` ordinates acting on the 10 % direct branch.
        """
        x4 = max(0.5, float(x4))
        n1 = int(math.ceil(x4))
        n2 = int(math.ceil(2.0 * x4))

        def sh1(j: float) -> float:
            if j <= 0:
                return 0.0
            if j < x4:
                return (j / x4) ** 2.5
            return 1.0

        def sh2(j: float) -> float:
            if j <= 0:
                return 0.0
            if j < x4:
                return 0.5 * (j / x4) ** 2.5
            if j < 2 * x4:
                return 1.0 - 0.5 * (2.0 - j / x4) ** 2.5
            return 1.0

        uh1 = np.array([sh1(i) - sh1(i - 1) for i in range(1, n1 + 1)], dtype=float)
        uh2 = np.array([sh2(i) - sh2(i - 1) for i in range(1, n2 + 1)], dtype=float)
        return uh1, uh2

    # ------------------------------------------------------------------
    def _gr4j_run(self, precip: np.ndarray, pet: np.ndarray,
                  x1: float, x2: float, x3: float, x4: float,
                  area_km2: float = 10_000.0,
                  init_state: Optional[Dict[str, Any]] = None,
                  return_state: bool = False
                  ) -> Any:
        """Vectorised GR4J daily simulation following Perrin et al. (2003).

        Args:
            precip: Daily precipitation (mm).
            pet:    Daily potential evapotranspiration (mm).
            x1, x2, x3, x4: GR4J parameters.
            area_km2: Basin area for mm -> m^3/s conversion.
            init_state: Optional warm-start state dict with keys
                ``S``, ``R``, ``uh1_buf``, ``uh2_buf``.
            return_state: If True also returns the final state.

        Returns:
            Discharge array (m^3/s) or ``(Q, state_dict)``.
        """
        n = len(precip)
        uh1, uh2 = self._uh_ordinates(x4)
        n1, n2 = len(uh1), len(uh2)

        if init_state is None:
            # Warm-start at half storage (standard GR4J practice).
            S = 0.5 * x1               # production store (Perrin Eq. 2)
            R = 0.5 * x3               # routing store    (Perrin Eq. 16)
            uh1_buf = np.zeros(n1)
            uh2_buf = np.zeros(n2)
        else:
            S = float(init_state.get("S", 0.5 * x1))
            R = float(init_state.get("R", 0.5 * x3))
            uh1_buf = np.array(init_state.get("uh1_buf", np.zeros(n1)), dtype=float)
            uh2_buf = np.array(init_state.get("uh2_buf", np.zeros(n2)), dtype=float)
            if len(uh1_buf) != n1:
                uh1_buf = np.resize(uh1_buf, n1)
            if len(uh2_buf) != n2:
                uh2_buf = np.resize(uh2_buf, n2)

        Q_mm = np.empty(n, dtype=float)
        inv_x1 = 1.0 / x1 if x1 > 0 else 0.0
        inv_x3 = 1.0 / x3 if x3 > 0 else 0.0

        for t in range(n):
            P = float(precip[t])
            E = float(pet[t])

            # Step 1: Interception (Perrin Eq. 1).
            if P >= E:
                Pn = P - E
                En = 0.0
            else:
                Pn = 0.0
                En = E - P

            # Step 2: Production store (Perrin Eqns. 3-6).
            if Pn > 0.0 and x1 > 0.0:
                t_pn = math.tanh(Pn * inv_x1)
                sx1 = S * inv_x1
                Ps = (x1 * (1.0 - sx1 * sx1) * t_pn) / (1.0 + sx1 * t_pn)
            else:
                Ps = 0.0
            if En > 0.0 and x1 > 0.0:
                t_en = math.tanh(En * inv_x1)
                sx1 = S * inv_x1
                Es = (S * (2.0 - sx1) * t_en) / (1.0 + (1.0 - sx1) * t_en)
            else:
                Es = 0.0
            S = S - Es + Ps
            if S < 0.0:
                S = 0.0

            # Percolation from production store (Perrin Eq. 7).
            perc = S * (1.0 - (1.0 + ((4.0 / 9.0) * S * inv_x1) ** 4) ** (-0.25)) if x1 > 0 else 0.0
            S = S - perc
            if S < 0.0:
                S = 0.0

            # Effective runoff (Perrin Eq. 8).
            Pr = Pn - Ps + perc

            # Step 3: 90/10 split into UH1 and UH2 (Perrin Eq. 9-10).
            # Push the new Pr into the circular buffers and convolve.
            uh1_buf = np.roll(uh1_buf, 1)
            uh1_buf[0] = Pr
            Q9 = 0.9 * float(np.dot(uh1, uh1_buf))

            uh2_buf = np.roll(uh2_buf, 1)
            uh2_buf[0] = Pr
            Q1 = 0.1 * float(np.dot(uh2, uh2_buf))

            # Step 5: Routing store update (spec formulation).
            Qr = R * (1.0 - (1.0 + (R * inv_x3) ** 4) ** (-0.25)) if x3 > 0 else 0.0
            R = R - Qr + Q9
            if R < 0.0:
                R = 0.0

            # Step 6: Groundwater exchange and direct discharge (Perrin Eq. 18-19).
            F = x2 * (R * inv_x3) ** 3.5 if (x3 > 0 and R > 0) else 0.0
            Qd = Q1 - F
            if Qd < 0.0:
                Qd = 0.0

            # Step 7: Total discharge in mm (Perrin Eq. 20).
            Q_mm[t] = Qr + Qd

        # Convert mm/day to m^3/s over the basin area.
        Q_m3s = Q_mm * (area_km2 * 1e6) / (1000.0 * 86400.0)

        if return_state:
            state = {"S": S, "R": R,
                     "uh1_buf": uh1_buf.copy(), "uh2_buf": uh2_buf.copy()}
            return Q_m3s, state
        return Q_m3s

    # ------------------------------------------------------------------
    @staticmethod
    def _kge(sim: np.ndarray, obs: np.ndarray) -> float:
        """Kling-Gupta Efficiency (2009 formulation) on finite pairs."""
        mask = np.isfinite(sim) & np.isfinite(obs)
        if mask.sum() < 10:
            return -9.0
        s = sim[mask]
        o = obs[mask]
        mu_s, mu_o = s.mean(), o.mean()
        sd_s, sd_o = s.std(), o.std()
        if sd_o == 0 or mu_o == 0:
            return -9.0
        r = np.corrcoef(s, o)[0, 1]
        if not np.isfinite(r):
            return -9.0
        alpha = sd_s / sd_o
        beta = mu_s / mu_o
        return float(1.0 - math.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    # ------------------------------------------------------------------
    @staticmethod
    def _nse(sim: np.ndarray, obs: np.ndarray) -> float:
        """Nash-Sutcliffe efficiency."""
        mask = np.isfinite(sim) & np.isfinite(obs)
        if mask.sum() < 10:
            return -9.0
        s = sim[mask]
        o = obs[mask]
        denom = np.sum((o - o.mean()) ** 2)
        if denom == 0:
            return -9.0
        return float(1.0 - np.sum((s - o) ** 2) / denom)

    # ------------------------------------------------------------------
    def calibrate(self, station_id: str) -> Tuple[float, float, float, float]:
        """Calibrate GR4J parameters by maximising KGE.

        Args:
            station_id: Station to calibrate.

        Returns:
            Tuple ``(x1, x2, x3, x4)``.
        """
        if station_id not in self.gauge_data:
            self._logger.info("Station %s absent - using default GR4J params", station_id)
            self.calibrated_params[station_id] = self.config.gr4j_default_params
            return self.config.gr4j_default_params

        obs_series = self.gauge_data[station_id]["discharge_m3s"].reindex(
            self.met_data.index)
        valid = obs_series.dropna()
        if len(valid) > 0:
            years = (valid.index.max() - valid.index.min()).days / 365.25
        else:
            years = 0.0
        if years < self.config.min_years_required:
            self._logger.info("Station %s below calibration threshold (%.2f yr) - "
                              "using defaults", station_id, years)
            self.calibrated_params[station_id] = self.config.gr4j_default_params
            return self.config.gr4j_default_params

        area = self._basin_area(station_id)
        P = self.met_data["precip_mm"].values
        E = self.met_data["pet_mm"].values
        Q_obs = obs_series.values
        warm = 365

        def objective(theta: np.ndarray) -> float:
            x1, x2, x3, x4 = theta
            Q_sim = self._gr4j_run(P, E, x1, x2, x3, x4, area_km2=area)
            score = self._kge(Q_sim[warm:], Q_obs[warm:])
            return -score  # minimise

        bounds = [(10.0, 2000.0), (-10.0, 10.0), (10.0, 500.0), (0.5, 10.0)]
        try:
            res = optimize.differential_evolution(
                objective, bounds, seed=42, maxiter=200, tol=0.005,
                workers=1, polish=True)
            params = tuple(float(v) for v in res.x)
        except Exception as exc:
            self._logger.warning("Calibration failed for %s (%s) - defaults used",
                                  station_id, exc)
            params = self.config.gr4j_default_params

        Q_check = self._gr4j_run(P, E, *params, area_km2=area)
        kge = self._kge(Q_check[warm:], Q_obs[warm:])
        nse = self._nse(Q_check[warm:], Q_obs[warm:])
        if _HAS_HYDROEVAL:
            try:
                _ = he.evaluator(he.kge, Q_check[warm:], Q_obs[warm:])
            except Exception:
                pass
        self._logger.info("Calibrated %s -> x1=%.1f x2=%.2f x3=%.1f x4=%.2f | KGE=%.3f NSE=%.3f",
                           station_id, *params, kge, nse)
        self.calibrated_params[station_id] = params  # type: ignore[assignment]
        return params  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def run_hindcast(self, station_id: str) -> pd.DataFrame:
        """Simulate the full historical window for a station.

        Args:
            station_id: Station identifier.

        Returns:
            DataFrame with columns ``Q_sim_m3s`` and ``Q_obs_m3s``.
        """
        params = self.calibrated_params.get(station_id, self.config.gr4j_default_params)
        area = self._basin_area(station_id)
        P = self.met_data["precip_mm"].values
        E = self.met_data["pet_mm"].values
        Q_sim, state = self._gr4j_run(P, E, *params, area_km2=area, return_state=True)
        self.final_states[station_id] = state

        if station_id in self.gauge_data:
            Q_obs = self.gauge_data[station_id]["discharge_m3s"].reindex(
                self.met_data.index).values
        else:
            Q_obs = np.full_like(Q_sim, np.nan)

        df = pd.DataFrame(
            {"Q_sim_m3s": Q_sim, "Q_obs_m3s": Q_obs},
            index=self.met_data.index)
        df.index.name = "date"
        return df

    # ------------------------------------------------------------------
    def run_forecast(self, station_id: str, forecast_precip: pd.Series,
                     forecast_pet: pd.Series) -> pd.DataFrame:
        """Run GR4J for ``forecast_horizon_days`` from the warm state.

        Args:
            station_id: Station identifier (must have a warm state).
            forecast_precip: Daily precipitation Series for the horizon.
            forecast_pet: Daily PET Series for the horizon.

        Returns:
            DataFrame indexed by forecast date with column
            ``Q_forecast_m3s``.
        """
        params = self.calibrated_params.get(station_id, self.config.gr4j_default_params)
        area = self._basin_area(station_id)
        state = self.final_states.get(station_id)
        horizon = min(self.config.forecast_horizon_days,
                      len(forecast_precip), len(forecast_pet))
        P = forecast_precip.values[:horizon]
        E = forecast_pet.values[:horizon]
        Q_sim, _ = self._gr4j_run(P, E, *params, area_km2=area,
                                    init_state=state, return_state=True)
        idx = forecast_precip.index[:horizon]
        df = pd.DataFrame({"Q_forecast_m3s": Q_sim}, index=idx)
        df.index.name = "date"
        return df

    # ------------------------------------------------------------------
    def compute_return_periods(self, station_id: str,
                                simulated_Q: pd.Series) -> pd.DataFrame:
        """Gumbel return-period analysis on annual maxima.

        Args:
            station_id: Station identifier (for logging only).
            simulated_Q: Daily discharge Series.

        Returns:
            DataFrame with columns ``return_period_yr`` and ``Q_m3s``.
        """
        if not isinstance(simulated_Q.index, pd.DatetimeIndex):
            simulated_Q = simulated_Q.copy()
            simulated_Q.index = pd.to_datetime(simulated_Q.index)
        annual_max = simulated_Q.resample("Y").max().dropna()

        rp_list = [2, 5, 10, 25, 50, 100]
        if len(annual_max) < 3:
            mean_q = float(simulated_Q.mean()) if len(simulated_Q) else 0.0
            q_values = [mean_q * (1.0 + 0.4 * math.log(rp)) for rp in rp_list]
            return pd.DataFrame({"return_period_yr": rp_list, "Q_m3s": q_values})
        try:
            loc, scale = stats.gumbel_r.fit(annual_max.values)
        except Exception:
            loc = float(annual_max.mean())
            scale = max(float(annual_max.std()), 1.0)
        q_values = [float(stats.gumbel_r.ppf(1.0 - 1.0 / rp, loc=loc, scale=scale))
                    for rp in rp_list]
        self._logger.info("Return periods for %s: %s",
                           station_id,
                           ", ".join(f"{rp}yr={q:.0f}" for rp, q in zip(rp_list, q_values)))
        return pd.DataFrame({"return_period_yr": rp_list, "Q_m3s": q_values})


# ===========================================================================
#                         6. FloodRouter
# ===========================================================================
class FloodRouter:
    """Muskingum routing through the river network plus RiverREM inundation."""

    def __init__(self, config: Config, watershed_data: Dict[str, Any],
                 Q_per_basin: Dict[str, pd.DataFrame]) -> None:
        """Bind routing inputs.

        Args:
            config: Global Config.
            watershed_data: Dict returned by ``WatershedDelineator.delineate``.
            Q_per_basin: Mapping ``{station_id: DataFrame}`` containing
                simulated discharge (col ``Q_sim_m3s``).
        """
        self.config = config
        self.watershed = watershed_data
        self.Q_per_basin = Q_per_basin
        self._logger = logging.getLogger(self.__class__.__name__)
        self._downstream: Dict[str, Optional[str]] = {}
        self._upstream: Dict[str, List[str]] = {}
        self._build_topology()

    # ------------------------------------------------------------------
    def _build_topology(self) -> None:
        """Derive a simple downstream relationship from station latitudes."""
        snapped = self.watershed.get("snapped_stations")
        if snapped is None or len(snapped) == 0:
            return
        df = snapped.copy()
        # Ensure we have columns lon / lat.
        if "lat" not in df.columns and "latitude" in df.columns:
            df["lat"] = df["latitude"]
        if "lon" not in df.columns and "longitude" in df.columns:
            df["lon"] = df["longitude"]
        # Sort by latitude (north -> south) to assume downstream is to the south.
        df = df.sort_values("lat", ascending=False).reset_index(drop=True)
        ids = df["station_id"].astype(str).tolist()
        self._downstream = {}
        self._upstream = {sid: [] for sid in ids}
        for i, sid in enumerate(ids):
            if i + 1 < len(ids):
                self._downstream[sid] = ids[i + 1]
                self._upstream[ids[i + 1]].append(sid)
            else:
                self._downstream[sid] = None

    # ------------------------------------------------------------------
    def _muskingum_route(self, Q_in: np.ndarray, K: float, X: float,
                         dt: float = 1.0) -> np.ndarray:
        """Classical Muskingum channel routing.

        Args:
            Q_in: Inflow hydrograph (m^3/s).
            K: Travel time (same units as ``dt``).
            X: Weighting factor in [0, 0.5].
            dt: Time step.

        Returns:
            Routed outflow array (m^3/s).

        Raises:
            None: invalid coefficient sums are logged as ERROR.
        """
        denom = K * (1.0 - X) + 0.5 * dt
        if denom == 0:
            self._logger.error("Muskingum denominator is zero")
            return np.asarray(Q_in, dtype=float).copy()
        C0 = (-K * X + 0.5 * dt) / denom
        C1 = (K * X + 0.5 * dt) / denom
        C2 = (K * (1.0 - X) - 0.5 * dt) / denom
        if abs(C0 + C1 + C2 - 1.0) > 1e-6:
            self._logger.error("Muskingum coefficients do not sum to 1: %.4f",
                                C0 + C1 + C2)
        Q = np.asarray(Q_in, dtype=float)
        n = len(Q)
        Q_out = np.empty(n, dtype=float)
        Q_out[0] = Q[0]
        for t in range(1, n):
            Q_out[t] = C0 * Q[t] + C1 * Q[t - 1] + C2 * Q_out[t - 1]
            if Q_out[t] < 0:
                Q_out[t] = 0.0
        return Q_out

    # ------------------------------------------------------------------
    def build_routing_order(self) -> List[str]:
        """Topological sort of the basin graph (Kahn's algorithm).

        Returns:
            List of station IDs from headwater to outlet.
        """
        indeg: Dict[str, int] = {sid: 0 for sid in self._upstream.keys()}
        for sid, ds in self._downstream.items():
            if ds is not None and ds in indeg:
                indeg[ds] += 0  # placeholder
        # in-degree = number of upstream basins
        for sid, ups in self._upstream.items():
            indeg[sid] = len(ups)

        queue = [sid for sid, d in indeg.items() if d == 0]
        order: List[str] = []
        while queue:
            sid = queue.pop(0)
            order.append(sid)
            ds = self._downstream.get(sid)
            if ds is not None:
                indeg[ds] -= 1
                if indeg[ds] == 0:
                    queue.append(ds)
        if len(order) != len(indeg):
            self._logger.warning("Routing order incomplete - cycle?")
            for sid in indeg:
                if sid not in order:
                    order.append(sid)
        return order

    # ------------------------------------------------------------------
    def route_network(self) -> Dict[str, pd.DataFrame]:
        """Route local+upstream inflow for every basin, in order.

        Returns:
            Mapping ``{station_id: DataFrame[Q_routed_m3s]}``.
        """
        order = self.build_routing_order()
        routed: Dict[str, pd.DataFrame] = {}
        for sid in order:
            local_df = self.Q_per_basin.get(sid)
            if local_df is None:
                continue
            q_col = "Q_sim_m3s" if "Q_sim_m3s" in local_df.columns else local_df.columns[0]
            Q_local = local_df[q_col].values.astype(float)
            Q_sum = np.zeros_like(Q_local)
            for up in self._upstream.get(sid, []):
                if up in routed:
                    Q_up = routed[up]["Q_routed_m3s"].reindex(local_df.index).fillna(0).values
                    Q_sum += Q_up
            Q_total = Q_local + Q_sum
            Q_routed = self._muskingum_route(
                Q_total, K=self.config.muskingum_k, X=self.config.muskingum_x)
            routed[sid] = pd.DataFrame({"Q_routed_m3s": Q_routed}, index=local_df.index)
        return routed

    # ------------------------------------------------------------------
    def compute_riverrem(self, dem_array: np.ndarray,
                         river_mask: np.ndarray,
                         dem_profile: Dict[str, Any]) -> np.ndarray:
        """Simplified River Relative Elevation Model.

        For every non-river pixel, the nearest-river elevation is looked up
        via a Euclidean distance transform. The REM is the DEM minus this
        interpolated river elevation, giving the height above the nearest
        river channel.

        Args:
            dem_array: Full DEM.
            river_mask: Boolean mask of river pixels.
            dem_profile: rasterio profile dict (unused but kept for API
                symmetry with future CRS-aware implementations).

        Returns:
            REM array of same shape as ``dem_array``.
        """
        if river_mask is None or river_mask.sum() == 0:
            return np.zeros_like(dem_array, dtype=np.float32)
        inv_mask = ~river_mask
        # distance_transform_edt returns nearest-feature indices when
        # return_indices=True; those are indices INTO the mask.
        _, indices = ndimage.distance_transform_edt(inv_mask, return_indices=True)
        river_elev = dem_array[indices[0], indices[1]]
        rem = dem_array - river_elev
        rem = np.where(river_mask, 0.0, rem)
        return rem.astype(np.float32)

    # ------------------------------------------------------------------
    def estimate_inundation(self, Q_routed: float, station_id: str,
                            rem: np.ndarray, catchment_mask: np.ndarray,
                            dem_profile: Dict[str, Any]) -> Any:
        """Manning-based flood polygon inside a catchment.

        Args:
            Q_routed: Routed discharge (m^3/s).
            station_id: Station whose basin is being flooded.
            rem: RiverREM array.
            catchment_mask: Boolean mask of the basin.
            dem_profile: rasterio profile including ``transform``.

        Returns:
            GeoDataFrame of flood polygons with columns
            ``geometry, depth_m, area_km2, station_id``.
        """
        if Q_routed <= 0 or rem is None:
            return self._empty_flood_gdf()
        n = 0.035
        width = max(20.0, math.sqrt(max(Q_routed, 1.0)) * 2.0)
        slope = 0.001  # fall-back slope
        try:
            if catchment_mask.sum() > 20:
                # Fit a simple slope from the DEM within the basin.
                slope = max(0.0005, float(np.std(rem[catchment_mask]) / 1000.0))
        except Exception:
            pass
        depth = ((Q_routed * n) / (width * slope ** 0.5)) ** 0.6
        depth = min(depth, 15.0)

        flood_binary = (rem < depth) & catchment_mask
        if flood_binary.sum() == 0:
            return self._empty_flood_gdf()

        if not (_HAS_RASTERIO and _HAS_GEO):
            return self._empty_flood_gdf()
        transform = dem_profile.get("transform")
        polys = []
        for geom, val in rio_features.shapes(flood_binary.astype(np.uint8),
                                               mask=flood_binary,
                                               transform=transform):
            if val == 1:
                try:
                    poly = shape(geom)
                    if poly.is_valid and poly.area > 0:
                        polys.append(poly)
                except Exception:
                    continue
        if not polys:
            return self._empty_flood_gdf()
        # Approximate polygon area in km^2 (equirectangular).
        rows, cols = rem.shape
        lon_min, lat_min, lon_max, lat_max = self.config.bbox
        mean_lat = 0.5 * (lat_min + lat_max)
        deg_km2 = (111.32 ** 2) * math.cos(math.radians(mean_lat))
        records = [{
            "geometry": p,
            "depth_m": float(depth),
            "area_km2": float(p.area * deg_km2),
            "station_id": station_id,
        } for p in polys]
        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    # ------------------------------------------------------------------
    def _empty_flood_gdf(self) -> Any:
        """Return an empty flood-polygon GeoDataFrame (or DataFrame)."""
        cols = ["geometry", "depth_m", "area_km2", "station_id"]
        if _HAS_GEO:
            return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")
        return pd.DataFrame(columns=cols)

    # ------------------------------------------------------------------
    def get_peak_flood_map(self, forecast_date: str) -> Any:
        """Concatenate per-basin inundation for a given date.

        Args:
            forecast_date: ISO date string.

        Returns:
            GeoDataFrame with all flood polygons for the date.
        """
        dem = self.watershed.get("dem") if isinstance(self.watershed, dict) else None
        flow_acc = self.watershed.get("flow_acc")
        masks = self.watershed.get("catchment_masks", {})
        profile = self.watershed.get("profile") or {}

        # If DEM was not stored in watershed dict, accept None.
        if dem is None:
            return self._empty_flood_gdf()
        if flow_acc is None or masks is None:
            return self._empty_flood_gdf()

        threshold = max(500.0, float(flow_acc.max()) * 0.002)
        river_mask = flow_acc > threshold
        rem = self.compute_riverrem(dem, river_mask, profile)

        target_ts = pd.to_datetime(forecast_date)
        routed = self.route_network()
        frames = []
        for sid, df in routed.items():
            if target_ts in df.index:
                Q = float(df.loc[target_ts, "Q_routed_m3s"])
            else:
                Q = float(df["Q_routed_m3s"].iloc[-1])
            mask = masks.get(sid)
            if mask is None or mask.sum() == 0:
                continue
            gdf = self.estimate_inundation(Q, sid, rem, mask, profile)
            if len(gdf) > 0:
                frames.append(gdf)
        if not frames:
            return self._empty_flood_gdf()
        if _HAS_GEO:
            return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                     geometry="geometry", crs="EPSG:4326")
        return pd.concat(frames, ignore_index=True)


# ===========================================================================
#                         7. AlertDashboard
# ===========================================================================
class AlertDashboard:
    """Interactive outputs: folium map, matplotlib plots, Streamlit UI."""

    def __init__(self, config: Config,
                 routed_Q: Dict[str, pd.DataFrame],
                 return_periods: Dict[str, pd.DataFrame],
                 flood_polygons: Any,
                 stations_gdf: Any,
                 hindcasts: Optional[Dict[str, pd.DataFrame]] = None,
                 forecasts: Optional[Dict[str, pd.DataFrame]] = None,
                 river_network: Any = None) -> None:
        """Bind outputs.

        Args:
            config: Global Config.
            routed_Q: Routed discharge per station.
            return_periods: Return-period tables per station.
            flood_polygons: Peak-flood GeoDataFrame.
            stations_gdf: Station metadata GeoDataFrame.
            hindcasts: Optional per-station hindcast DataFrames.
            forecasts: Optional per-station forecast DataFrames.
        """
        self.config = config
        self.routed_Q = routed_Q
        self.return_periods = return_periods
        self.flood_polygons = flood_polygons
        self.stations_gdf = stations_gdf
        self.hindcasts = hindcasts or {}
        self.forecasts = forecasts or {}
        self.river_network = river_network
        self._logger = logging.getLogger(self.__class__.__name__)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    # ------------------------------------------------------------------
    def _classify_alert(self, Q: float, Q_maf: float) -> str:
        """Classify a discharge into GREEN / AMBER / RED.

        Args:
            Q: Current discharge (m^3/s).
            Q_maf: Mean annual flood reference (m^3/s).

        Returns:
            Alert level string.
        """
        if Q_maf <= 0 or not np.isfinite(Q_maf):
            return "GREEN"
        ratio = Q / Q_maf
        if ratio >= self.config.alert_red:
            return "RED"
        if ratio >= self.config.alert_amber:
            return "AMBER"
        return "GREEN"

    # ------------------------------------------------------------------
    def _river_name(self, station_id: str) -> str:
        """Return the river name recorded for a station, or empty string."""
        try:
            row = self.stations_gdf.loc[
                self.stations_gdf["station_id"].astype(str) == str(station_id)]
            if not row.empty and "river" in row.columns:
                return str(row["river"].iloc[0])
        except Exception:
            pass
        return ""

    # ------------------------------------------------------------------
    def _station_name(self, station_id: str) -> str:
        """Return the human-readable station name."""
        try:
            row = self.stations_gdf.loc[
                self.stations_gdf["station_id"].astype(str) == str(station_id)]
            if not row.empty and "station_name" in row.columns:
                return str(row["station_name"].iloc[0])
        except Exception:
            pass
        return str(station_id)

    # ------------------------------------------------------------------
    def _rp_value(self, station_id: str, rp: int) -> float:
        """Return Q for a given return period, or NaN."""
        df = self.return_periods.get(station_id)
        if df is None or df.empty:
            return float("nan")
        row = df.loc[df["return_period_yr"] == rp]
        if row.empty:
            return float("nan")
        return float(row["Q_m3s"].iloc[0])

    # ------------------------------------------------------------------
    def _q_maf(self, station_id: str) -> float:
        """Mean annual flood from routed discharge."""
        df = self.routed_Q.get(station_id)
        if df is None or df.empty:
            return float("nan")
        try:
            ann_max = df["Q_routed_m3s"].resample("Y").max().dropna()
            if ann_max.empty:
                return float("nan")
            return float(ann_max.mean())
        except Exception:
            return float("nan")

    # ------------------------------------------------------------------
    def generate_folium_map(self, forecast_date: str) -> Any:
        """Build the interactive folium flood map.

        Args:
            forecast_date: ISO date for file-name and popup context.

        Returns:
            folium.Map instance (or None if folium is unavailable).
        """
        if not _HAS_FOLIUM:
            self._logger.warning("folium not available - map not generated")
            return None
        fmap = folium.Map(location=[9.0, 8.0], zoom_start=6,
                           tiles="CartoDB positron")

        # Flood polygons.
        try:
            if self.flood_polygons is not None and len(self.flood_polygons) > 0:
                def style_fn(feature):
                    d = feature["properties"].get("depth_m", 0.0)
                    if d < 0.5:
                        color = "#AED6F1"
                    elif d < 2.0:
                        color = "#2980B9"
                    else:
                        color = "#1A5276"
                    return {"fillColor": color, "color": color,
                            "weight": 1, "fillOpacity": 0.55}
                folium.GeoJson(
                    self.flood_polygons.__geo_interface__,
                    name="Flood extent",
                    style_function=style_fn,
                    tooltip=GeoJsonTooltip(
                        fields=["station_id", "depth_m", "area_km2"],
                        aliases=["Station", "Depth (m)", "Area (km^2)"]),
                ).add_to(fmap)
        except Exception as exc:
            self._logger.warning("Could not add flood polygons: %s", exc)

        # River network as a blue polyline layer.
        try:
            rivers = self.river_network
            if rivers is not None and len(rivers) > 0 and _HAS_GEO:
                for geom in rivers.geometry:
                    if geom is None or geom.is_empty:
                        continue
                    coords = list(geom.coords)
                    folium.PolyLine(
                        locations=[(lat, lon) for lon, lat in coords],
                        color="#1F618D", weight=1.5, opacity=0.8,
                    ).add_to(fmap)
        except Exception as exc:
            self._logger.warning("Could not add river network: %s", exc)

        # Station markers.
        try:
            for _, row in self.stations_gdf.iterrows():
                sid = str(row["station_id"])
                q_df = self.routed_Q.get(sid)
                current_q = float(q_df["Q_routed_m3s"].iloc[-1]) if q_df is not None and not q_df.empty else 0.0
                q2 = self._rp_value(sid, 2)
                q10 = self._rp_value(sid, 10)
                alert = self._classify_alert(current_q, self._q_maf(sid))
                color = {"RED": "#C0392B", "AMBER": "#F39C12",
                         "GREEN": "#27AE60"}[alert]
                popup_html = (
                    f"<b>{row.get('station_name', sid)}</b><br>"
                    f"River: {row.get('river', '')}<br>"
                    f"Current Q: {current_q:.1f} m^3/s<br>"
                    f"2yr RP: {q2:.1f} m^3/s<br>"
                    f"10yr RP: {q10:.1f} m^3/s<br>"
                    f"Alert: <b style='color:{color}'>{alert}</b>")
                folium.CircleMarker(
                    location=[float(row["latitude"]), float(row["longitude"])],
                    radius=8, color=color, fill=True, fill_color=color,
                    fill_opacity=0.9, weight=2,
                    popup=folium.Popup(popup_html, max_width=300),
                ).add_to(fmap)
        except Exception as exc:
            self._logger.warning("Could not add station markers: %s", exc)

        # Legend.
        try:
            legend_html = """
            {% macro html(this, kwargs) %}
            <div style="position: fixed; bottom: 40px; left: 40px; z-index: 9999;
                         background: white; padding: 10px 14px; border: 1px solid #888;
                         border-radius: 6px; font-family: Arial, sans-serif;
                         font-size: 13px; box-shadow: 0 2px 6px rgba(0,0,0,0.15)">
              <div style="font-weight: bold; margin-bottom: 6px">Flood alert</div>
              <div><span style="display:inline-block;width:14px;height:14px;
                         background:#27AE60;border-radius:50%;margin-right:6px"></span>Green (Q &lt; MAF)</div>
              <div><span style="display:inline-block;width:14px;height:14px;
                         background:#F39C12;border-radius:50%;margin-right:6px"></span>Amber (Q &ge; 1.5 MAF)</div>
              <div><span style="display:inline-block;width:14px;height:14px;
                         background:#C0392B;border-radius:50%;margin-right:6px"></span>Red (Q &ge; 2.5 MAF)</div>
              <div style="margin-top:8px;font-weight:bold">Flood depth</div>
              <div><span style="display:inline-block;width:14px;height:14px;
                         background:#AED6F1;margin-right:6px"></span>&lt; 0.5 m</div>
              <div><span style="display:inline-block;width:14px;height:14px;
                         background:#2980B9;margin-right:6px"></span>0.5 - 2.0 m</div>
              <div><span style="display:inline-block;width:14px;height:14px;
                         background:#1A5276;margin-right:6px"></span>&gt; 2.0 m</div>
            </div>
            {% endmacro %}
            """

            class Legend(MacroElement):
                def __init__(self):
                    super().__init__()
                    self._template = Template(legend_html)

            fmap.get_root().add_child(Legend())
        except Exception as exc:
            self._logger.warning("Legend creation failed: %s", exc)

        folium.LayerControl(collapsed=False).add_to(fmap)

        out_path = self.config.output_dir / f"flood_map_{forecast_date}.html"
        try:
            fmap.save(str(out_path))
            self._logger.info("Folium map saved to %s", out_path)
        except Exception as exc:
            self._logger.warning("Could not save folium map: %s", exc)
        return fmap

    # ------------------------------------------------------------------
    def plot_discharge_forecast(self, station_id: str,
                                 hindcast: pd.DataFrame,
                                 forecast: pd.DataFrame,
                                 return_periods: pd.DataFrame) -> Any:
        """Generate a discharge hindcast+forecast plot for a station.

        Args:
            station_id: Station identifier.
            hindcast: DataFrame with Q_sim_m3s and Q_obs_m3s.
            forecast: DataFrame with Q_forecast_m3s.
            return_periods: Return-period table.

        Returns:
            matplotlib Figure.
        """
        if not _HAS_MPL:
            self._logger.warning("matplotlib unavailable - skipping plot for %s", station_id)
            return None
        _ensure_mpl()
        fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
        last90 = hindcast.tail(90)
        ax.plot(last90.index, last90["Q_sim_m3s"],
                color="#2874A6", label="Simulated (hindcast)", linewidth=1.3)
        if "Q_obs_m3s" in last90.columns:
            ax.scatter(last90.index, last90["Q_obs_m3s"], color="black",
                        s=10, alpha=0.5, label="Observed")
        if forecast is not None and len(forecast) > 0:
            fc_idx = forecast.index
            ax.plot(fc_idx, forecast["Q_forecast_m3s"], color="#C0392B",
                    linestyle="--", linewidth=1.8, label="Forecast")
            ax.axvspan(fc_idx[0], fc_idx[-1], color="#F7DC6F", alpha=0.15)

        # Return-period lines.
        rp_colors = {2: "#7DCEA0", 10: "#F4D03F", 100: "#CB4335"}
        for rp in (2, 10, 100):
            try:
                q = float(return_periods.loc[return_periods["return_period_yr"] == rp,
                                              "Q_m3s"].iloc[0])
                ax.axhline(q, color=rp_colors[rp], linestyle=":",
                            linewidth=1.2, alpha=0.9,
                            label=f"{rp}yr RP ({q:.0f} m^3/s)")
            except Exception:
                continue

        ax.set_title(f"{station_id} - Discharge Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Discharge (m^3/s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, ncol=2)
        fig.autofmt_xdate()
        out_path = self.config.output_dir / f"{station_id}_forecast.png"
        try:
            fig.tight_layout()
            fig.savefig(out_path)
            self._logger.info("Saved forecast plot to %s", out_path)
        except Exception as exc:
            self._logger.warning("Could not save plot: %s", exc)
        return fig

    # ------------------------------------------------------------------
    def plot_all_stations(self) -> None:
        """Render per-station plots and a combined summary figure."""
        if not _HAS_MPL:
            return
        _ensure_mpl()
        station_ids = list(self.routed_Q.keys())
        for sid in station_ids:
            hind = self.hindcasts.get(sid, self.routed_Q[sid].rename(
                columns={"Q_routed_m3s": "Q_sim_m3s"}).assign(Q_obs_m3s=np.nan))
            fc = self.forecasts.get(sid, pd.DataFrame(columns=["Q_forecast_m3s"]))
            rp = self.return_periods.get(sid, pd.DataFrame(
                {"return_period_yr": [2, 10, 100], "Q_m3s": [np.nan, np.nan, np.nan]}))
            fig = self.plot_discharge_forecast(sid, hind, fc, rp)
            if fig is not None:
                plt.close(fig)

        n = len(station_ids)
        if n == 0:
            return
        cols = min(4, n)
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols,
                                  figsize=(5 * cols, 3.2 * rows), dpi=100,
                                  squeeze=False)
        for ax, sid in zip(axes.flat, station_ids):
            hind = self.hindcasts.get(sid, self.routed_Q[sid].rename(
                columns={"Q_routed_m3s": "Q_sim_m3s"}))
            ax.plot(hind.index[-180:], hind.iloc[-180:, 0], color="#2874A6")
            fc = self.forecasts.get(sid)
            if fc is not None and len(fc) > 0:
                ax.plot(fc.index, fc["Q_forecast_m3s"], color="#C0392B",
                        linestyle="--")
            ax.set_title(self._station_name(sid), fontsize=10)
            ax.grid(True, alpha=0.3)
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_fontsize(7)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle("Nigeria Flood Forecast - All Stations", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = self.config.output_dir / "all_stations_summary.png"
        try:
            fig.savefig(out)
            self._logger.info("Saved summary figure to %s", out)
        except Exception as exc:
            self._logger.warning("Could not save summary: %s", exc)
        finally:
            plt.close(fig)

    # ------------------------------------------------------------------
    def generate_alert_report(self, forecast_date: str) -> Dict[str, Dict[str, Any]]:
        """Produce a machine-readable per-station alert summary.

        Args:
            forecast_date: ISO date marking the start of the forecast horizon.

        Returns:
            ``{station_id: {alert_level, peak_Q_m3s, peak_date,
            Q_pct_of_2yr, river}}``.
        """
        report: Dict[str, Dict[str, Any]] = {}
        start = pd.to_datetime(forecast_date)
        horizon_end = start + pd.Timedelta(days=self.config.forecast_horizon_days)
        for sid, df in self.routed_Q.items():
            # Prefer the explicit forecast DataFrame if provided.
            fc = self.forecasts.get(sid)
            if fc is not None and len(fc) > 0:
                series = fc["Q_forecast_m3s"]
            else:
                mask = (df.index >= start) & (df.index <= horizon_end)
                series = df.loc[mask, "Q_routed_m3s"] if mask.any() else df["Q_routed_m3s"]
            if series.empty:
                continue
            peak_Q = float(series.max())
            peak_date = str(series.idxmax())
            q_maf = self._q_maf(sid)
            alert = self._classify_alert(peak_Q, q_maf if np.isfinite(q_maf) else peak_Q)
            q2 = self._rp_value(sid, 2)
            pct = float(peak_Q / q2 * 100.0) if np.isfinite(q2) and q2 > 0 else float("nan")
            report[sid] = {
                "alert_level": alert,
                "peak_Q_m3s": peak_Q,
                "peak_date": peak_date,
                "Q_pct_of_2yr": pct,
                "river": self._river_name(sid),
            }
        # Persist JSON for reproducibility.
        try:
            with open(self.config.output_dir / f"alert_report_{forecast_date}.json", "w") as fh:
                json.dump(report, fh, indent=2, default=str)
        except Exception as exc:
            self._logger.warning("Could not write alert JSON: %s", exc)
        return report

    # ------------------------------------------------------------------
    def build_streamlit_app(self) -> None:
        """Define and immediately invoke the Streamlit dashboard function."""
        config = self.config
        routed_Q = self.routed_Q
        forecasts = self.forecasts
        stations_gdf = self.stations_gdf
        report_fn = self.generate_alert_report
        map_fn = self.generate_folium_map

        def run_streamlit_dashboard() -> None:
            try:
                import streamlit as st
                import streamlit.components.v1 as components
            except Exception as exc:
                logging.getLogger("AlertDashboard").info(
                    "Streamlit not available: %s", exc)
                return

            st.set_page_config(page_title="Nigeria Flood Forecast", layout="wide")
            st.title("Nigeria Flood Forecast Dashboard")

            with st.sidebar:
                st.header("Controls")
                today = pd.Timestamp.today().normalize()
                try:
                    default_date = pd.to_datetime(config.end_date).date()
                except Exception:
                    default_date = today.date()
                forecast_date = st.date_input("Forecast date", default_date)
                station_ids = list(routed_Q.keys())
                selected = st.multiselect("Basins", station_ids, default=station_ids)
                alert_filter = st.selectbox(
                    "Alert level filter", ["ALL", "RED", "AMBER", "GREEN"])

            date_str = forecast_date.isoformat() if hasattr(forecast_date, "isoformat") else str(forecast_date)
            report = report_fn(date_str)
            if alert_filter != "ALL":
                report = {k: v for k, v in report.items() if v["alert_level"] == alert_filter}

            tabs = st.tabs(["Map", "Forecast plots", "Alert table", "Raw data"])

            with tabs[0]:
                st.subheader("Inundation map")
                map_path = config.output_dir / f"flood_map_{date_str}.html"
                if not map_path.exists():
                    try:
                        map_fn(date_str)
                    except Exception as exc:
                        st.error(f"Could not render map: {exc}")
                if map_path.exists():
                    with open(map_path, "r", encoding="utf-8") as fh:
                        components.html(fh.read(), height=600, scrolling=True)
                else:
                    st.info("Map not available.")

            with tabs[1]:
                st.subheader("Per-station forecast")
                for sid in selected:
                    img = config.output_dir / f"{sid}_forecast.png"
                    if img.exists():
                        st.image(str(img), caption=sid, use_column_width=True)
                    else:
                        st.write(f"No forecast plot for {sid}")

            with tabs[2]:
                st.subheader("Alert table")
                if not report:
                    st.write("No stations match the current filter.")
                else:
                    df = pd.DataFrame.from_dict(report, orient="index").reset_index()
                    df = df.rename(columns={"index": "station_id"})

                    def color_row(row):
                        c = {"RED": "#fadbd8", "AMBER": "#fdebd0",
                             "GREEN": "#d5f5e3"}.get(row["alert_level"], "white")
                        return [f"background-color: {c}"] * len(row)
                    try:
                        styled = df.style.apply(color_row, axis=1)
                        st.dataframe(styled, use_container_width=True)
                    except Exception:
                        st.dataframe(df, use_container_width=True)

            with tabs[3]:
                st.subheader("Routed discharge CSV export")
                sid = st.selectbox("Station", list(routed_Q.keys()),
                                    key="dl_station")
                df = routed_Q.get(sid)
                if df is not None:
                    csv = df.to_csv().encode("utf-8")
                    st.download_button(
                        label=f"Download {sid} CSV",
                        data=csv,
                        file_name=f"{sid}_routed_Q.csv",
                        mime="text/csv",
                    )
                    fc = forecasts.get(sid)
                    if fc is not None and len(fc) > 0:
                        st.write("Forecast tail:")
                        st.dataframe(fc)

        # Invoke immediately so that `streamlit run flood_forecast_nigeria.py`
        # renders the dashboard as soon as the script executes.
        run_streamlit_dashboard()


# ===========================================================================
#                         8. FloodForecastSystem  (orchestrator)
# ===========================================================================
class FloodForecastSystem:
    """Top-level orchestrator wiring every component together."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Instantiate the full pipeline.

        Args:
            config: Optional Config; default :class:`Config` is used if None.
        """
        self.config = config or Config()
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        self._logger = logging.getLogger(self.__class__.__name__)

        self.data_manager: Optional[DataManager] = None
        self.gap_filler: Optional[GapFiller] = None
        self.delineator: Optional[WatershedDelineator] = None
        self.hydro_model: Optional[HydrologicalModel] = None
        self.flood_router: Optional[FloodRouter] = None
        self.alert_dashboard: Optional[AlertDashboard] = None

        self.Q_per_basin: Dict[str, pd.DataFrame] = {}
        self.return_periods: Dict[str, pd.DataFrame] = {}
        self.hindcasts: Dict[str, pd.DataFrame] = {}
        self.routed_Q: Dict[str, pd.DataFrame] = {}
        self.watershed_data: Dict[str, Any] = {}
        self.met_data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    def run_historical(self) -> None:
        """Execute the full historical hindcast pipeline."""
        cfg = self.config

        # Step 1 - data ingestion.
        self._logger.info("Step 1/7 - loading gauge data and meteorology")
        dm = DataManager(cfg)
        dm.load_gauge_data()
        # Expose sparse list on Config for downstream consumers.
        setattr(cfg, "sparse_stations", list(dm.sparse_stations))
        chirps_df = dm.load_chirps_for_bbox(cfg.start_date, cfg.end_date)
        dm.download_era5(cfg.era5_variables, cfg.start_date, cfg.end_date)
        era5_df = dm.load_era5()
        met = dm.merge_met_data(chirps_df, era5_df)
        self.met_data = met
        self.data_manager = dm

        # Step 2 - gap filling.
        self._logger.info("Step 2/7 - gap-filling sparse stations")
        gf = GapFiller(cfg, dm.gauge_data, met)
        gf.fill_all()
        self.gap_filler = gf

        # Step 3 - watershed delineation and per-basin rainfall.
        self._logger.info("Step 3/7 - delineating watersheds")
        wd = WatershedDelineator(cfg)
        wd.load_dem()
        watershed = wd.delineate()
        watershed["dem"] = wd.dem
        watershed["profile"] = wd.profile
        basin_precip = wd.get_basin_mean_precip(
            watershed["basins"], cfg.data_dir / "chirps")
        watershed["basin_precip"] = basin_precip
        self.watershed_data = watershed
        self.delineator = wd

        # Step 4 - hydrological modelling.
        self._logger.info("Step 4/7 - calibrating and running GR4J")
        hm = HydrologicalModel(cfg, watershed["basins"], dm.gauge_data, met)
        for sid in dm.gauge_data.keys():
            hm.calibrate(sid)
        for sid in dm.gauge_data.keys():
            self.hindcasts[sid] = hm.run_hindcast(sid)
            self.Q_per_basin[sid] = self.hindcasts[sid][["Q_sim_m3s"]].copy()
            self.return_periods[sid] = hm.compute_return_periods(
                sid, self.hindcasts[sid]["Q_sim_m3s"])
        self.hydro_model = hm

        # Step 5 - routing and RiverREM.
        self._logger.info("Step 5/7 - Muskingum routing and RiverREM")
        fr = FloodRouter(cfg, watershed, self.Q_per_basin)
        routed = fr.route_network()
        if watershed.get("dem") is not None and watershed.get("flow_acc") is not None:
            facc = watershed["flow_acc"]
            threshold = max(500.0, float(np.asarray(facc).max()) * 0.002)
            river_mask = np.asarray(facc) > threshold
            rem = fr.compute_riverrem(
                np.asarray(watershed["dem"]), river_mask, watershed.get("profile") or {})
            watershed["rem"] = rem
        self.routed_Q = routed
        self.flood_router = fr

        # Step 6 - peak flood map.
        self._logger.info("Step 6/7 - building peak flood map")
        peak_flood = fr.get_peak_flood_map(cfg.end_date)

        # Step 7 - dashboard and alerts.
        self._logger.info("Step 7/7 - assembling dashboard and alert report")
        stations_gdf = dm.get_station_geodataframe()
        ad = AlertDashboard(cfg, routed, self.return_periods,
                             peak_flood, stations_gdf,
                             hindcasts=self.hindcasts, forecasts={},
                             river_network=watershed.get("river_network"))
        ad.plot_all_stations()
        ad.generate_folium_map(cfg.end_date)
        alerts = ad.generate_alert_report(cfg.end_date)
        self.alert_dashboard = ad

        red = [sid for sid, rec in alerts.items() if rec["alert_level"] == "RED"]
        if red:
            self._logger.warning("RED alerts at: %s", ", ".join(red))
        self._logger.info("Historical run complete.")

    # ------------------------------------------------------------------
    def run_forecast(self, nwp_precip: pd.DataFrame,
                      nwp_pet: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Run the operational forecast given NWP inputs.

        Args:
            nwp_precip: DataFrame indexed by date; columns either one per
                station or a single ``area_mean`` column.
            nwp_pet: Same shape as ``nwp_precip`` for PET.

        Returns:
            Alert-report dictionary for the first forecast day.
        """
        if self.hydro_model is None:
            raise RuntimeError("run_historical must be called before run_forecast")

        def series_for(sid: str, df: pd.DataFrame) -> pd.Series:
            if sid in df.columns:
                return df[sid]
            if "area_mean" in df.columns:
                return df["area_mean"]
            return df.iloc[:, 0]

        forecasts: Dict[str, pd.DataFrame] = {}
        for sid in self.Q_per_basin.keys():
            fp = series_for(sid, nwp_precip)
            fe = series_for(sid, nwp_pet)
            forecasts[sid] = self.hydro_model.run_forecast(sid, fp, fe)

        # Append forecast to historical Q for routing continuity.
        extended: Dict[str, pd.DataFrame] = {}
        for sid, hind in self.Q_per_basin.items():
            fc = forecasts[sid]
            merged = pd.concat([
                hind,
                fc.rename(columns={"Q_forecast_m3s": "Q_sim_m3s"})
            ])
            extended[sid] = merged

        fr = FloodRouter(self.config, self.watershed_data, extended)
        routed = fr.route_network()
        self.routed_Q = routed
        self.flood_router = fr

        forecast_start = list(forecasts.values())[0].index[0] if forecasts else pd.to_datetime(self.config.end_date)
        forecast_date = forecast_start.strftime("%Y-%m-%d")
        peak_flood = fr.get_peak_flood_map(forecast_date)

        assert self.data_manager is not None
        stations_gdf = self.data_manager.get_station_geodataframe()
        ad = AlertDashboard(self.config, routed, self.return_periods,
                             peak_flood, stations_gdf,
                             hindcasts=self.hindcasts, forecasts=forecasts,
                             river_network=self.watershed_data.get("river_network"))
        ad.plot_all_stations()
        ad.generate_folium_map(forecast_date)
        report = ad.generate_alert_report(forecast_date)
        self.alert_dashboard = ad
        return report

    # ------------------------------------------------------------------
    def _write_synthetic_gauge_csv(self) -> None:
        """Emit a small synthetic gauge CSV for demo mode."""
        cfg = self.config
        path = Path(cfg.gauge_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size > 0:
            return

        stations = [
            {"station_id": "NIG_LOK", "station_name": "Lokoja",
             "latitude": 7.80, "longitude": 6.74, "river": "Niger",
             "area_km2": 150_000.0, "mean_q": 6000.0, "record_years": 10},
            {"station_id": "BEN_MAK", "station_name": "Makurdi",
             "latitude": 7.73, "longitude": 8.54, "river": "Benue",
             "area_km2": 80_000.0, "mean_q": 2500.0, "record_years": 10},
            {"station_id": "KAD_KAD", "station_name": "Kaduna",
             "latitude": 10.52, "longitude": 7.43, "river": "Kaduna",
             "area_km2": 25_000.0, "mean_q": 300.0, "record_years": 1},
            {"station_id": "OSU_OSO", "station_name": "Oshogbo",
             "latitude": 7.76, "longitude": 4.56, "river": "Osun",
             "area_km2": 8_000.0, "mean_q": 120.0, "record_years": 10},
        ]

        rng = np.random.default_rng(seed=7)
        idx = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
        doy = np.array([d.timetuple().tm_yday for d in idx])
        seasonal = 1.0 + 1.4 * np.sin(2 * np.pi * (doy - 150) / 365.25)

        rows = []
        for s in stations:
            years = s["record_years"]
            mask = idx >= (idx.max() - pd.Timedelta(days=int(years * 365)))
            noise = rng.normal(1.0, 0.25, len(idx))
            q = np.clip(s["mean_q"] * seasonal * noise, 1.0, None)
            stage = 1.0 + 0.003 * q / s["mean_q"] * s["mean_q"]
            stage = 1.0 + np.log1p(q / max(s["mean_q"], 1.0)) * 1.5
            for d, Q, H, m in zip(idx, q, stage, mask):
                if not m:
                    continue
                rows.append({
                    "station_id": s["station_id"],
                    "station_name": s["station_name"],
                    "latitude": s["latitude"],
                    "longitude": s["longitude"],
                    "river": s["river"],
                    "date": d.strftime("%Y-%m-%d"),
                    "discharge_m3s": round(float(Q), 2),
                    "stage_m": round(float(H), 3),
                })
        pd.DataFrame(rows).to_csv(path, index=False)
        self._logger.info("Wrote synthetic gauge CSV to %s (%d rows)", path, len(rows))

    # ------------------------------------------------------------------
    def _synthesise_nwp(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a plausible ``forecast_horizon_days`` NWP forcing pair."""
        cfg = self.config
        start = pd.to_datetime(cfg.end_date) + pd.Timedelta(days=1)
        idx = pd.date_range(start, periods=cfg.forecast_horizon_days, freq="D")
        rng = np.random.default_rng(seed=99)
        precip = np.clip(rng.gamma(shape=1.0, scale=10.0, size=len(idx)), 0.0, None)
        pet = np.clip(rng.normal(4.5, 0.8, size=len(idx)), 0.5, None)
        # Add a strong rain pulse on day 2 to trigger noticeable forecasts.
        if len(precip) > 2:
            precip[2] += 40.0
            precip[3] += 30.0
        p_df = pd.DataFrame({"area_mean": precip}, index=idx)
        e_df = pd.DataFrame({"area_mean": pet}, index=idx)
        p_df.index.name = "date"
        e_df.index.name = "date"
        return p_df, e_df

    # ------------------------------------------------------------------
    def run_demo(self) -> None:
        """End-to-end demo using fully synthetic inputs."""
        self._logger.info("=== FloodForecastSystem demo starting ===")
        self._write_synthetic_gauge_csv()

        # Ensure the DEM is present (synthetic fallback writes it).
        WatershedDelineator(self.config).generate_synthetic_dem()

        # Historical hindcast.
        self.run_historical()

        # Synthetic 7-day forecast.
        nwp_p, nwp_e = self._synthesise_nwp()
        alerts = self.run_forecast(nwp_p, nwp_e)

        # Human-readable summary.
        print()
        print("=" * 72)
        print("  Nigeria Flood Forecast - Alert Report")
        print("=" * 72)
        print(f"  Horizon: next {self.config.forecast_horizon_days} days "
              f"(starting {nwp_p.index[0].date()})")
        print("-" * 72)
        if not alerts:
            print("  No alerts generated.")
        else:
            header = f"  {'Station':<10}{'River':<12}{'Level':<8}{'Peak Q':>12}{'%2yr':>10}  Peak date"
            print(header)
            print("-" * 72)
            for sid, rec in alerts.items():
                print(f"  {sid:<10}{rec['river']:<12}{rec['alert_level']:<8}"
                      f"{rec['peak_Q_m3s']:>10.1f}  "
                      f"{rec['Q_pct_of_2yr']:>8.1f}%  {rec['peak_date'][:10]}")
        print("=" * 72)
        print(f"  Outputs written to: {self.config.output_dir.resolve()}")
        print("=" * 72)
        self._logger.info("=== FloodForecastSystem demo complete ===")


# ===========================================================================
#                              Entry point
# ===========================================================================
if __name__ == "__main__":
    import sys
    if "--streamlit" in sys.argv or "streamlit" in sys.argv[0]:
        # Launched via `streamlit run flood_forecast_nigeria.py`.
        system = FloodForecastSystem()
        system.run_demo()
        system.alert_dashboard.build_streamlit_app()
    else:
        # Launched via `python flood_forecast_nigeria.py`.
        system = FloodForecastSystem()
        system.run_demo()
