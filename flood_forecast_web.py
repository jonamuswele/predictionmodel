
from __future__ import annotations

import io
import json
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import flood_forecast_nigeria as ffn

try:
    import cloud_storage
except Exception:  # pragma: no cover - optional dep
    cloud_storage = None  # type: ignore

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nigeria Flood Forecast",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .main > div { padding-top: 0.6rem; }
    .alert-red   { background:#fadbd8; padding:8px 14px; border-radius:6px;
                   border-left:6px solid #C0392B; margin-bottom:6px; }
    .alert-amber { background:#fdebd0; padding:8px 14px; border-radius:6px;
                   border-left:6px solid #F39C12; margin-bottom:6px; }
    .alert-green { background:#d5f5e3; padding:8px 14px; border-radius:6px;
                   border-left:6px solid #27AE60; margin-bottom:6px; }
    .stButton button { width: 100%; }
    .nav-title   { font-size: 1.05rem; font-weight: 600; margin-bottom: 4px; }
    .small-note  { color:#566573; font-size: 0.85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ===========================================================================
#                         Web-only helpers (no ffn changes)
# ===========================================================================
def _try_import_folium():
    try:
        import folium
        from folium import plugins  # noqa: F401
        return folium
    except Exception:
        return None


def _try_import_rasterio_features():
    try:
        from rasterio import features as rio_features
        from rasterio.transform import Affine  # noqa: F401
        return rio_features
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Nigeria geography (static fallbacks; OSM Overpass supersedes at runtime)
# ---------------------------------------------------------------------------
# (lon_min, lat_min, lon_max, lat_max)
NIGERIA_BBOX: Tuple[float, float, float, float] = (2.65, 4.20, 14.70, 13.90)

# Simplified Nigeria country boundary (~35 vertices). Good enough for a
# visual clip; not cartographically precise. CC0 approximation.
NIGERIA_BOUNDARY: List[Tuple[float, float]] = [
    (3.62, 13.75), (4.30, 13.80), (5.50, 13.85), (6.40, 13.60),
    (7.40, 13.70), (8.50, 13.20), (10.00, 13.00), (11.30, 13.40),
    (12.30, 13.10), (13.30, 13.60), (14.00, 13.10), (14.50, 12.90),
    (14.20, 12.40), (14.60, 12.00), (14.20, 11.50), (14.10, 10.50),
    (13.80, 10.00), (13.20, 9.60), (12.60, 9.00), (12.80, 8.50),
    (13.00, 7.80), (12.60, 7.40), (12.20, 7.00), (11.80, 6.60),
    (11.40, 6.50), (10.60, 7.00), (10.10, 6.90), (9.50, 6.80),
    (9.00, 6.50), (8.70, 5.60), (8.50, 4.80), (8.30, 4.55),
    (7.90, 4.30), (7.00, 4.40), (6.20, 4.30), (5.40, 5.50),
    (4.70, 6.10), (4.00, 6.10), (3.30, 6.30), (2.75, 6.37),
    (2.70, 6.80), (2.80, 7.80), (2.75, 8.60), (3.00, 9.10),
    (3.50, 9.90), (3.60, 10.70), (3.70, 11.40), (3.60, 12.30),
    (3.90, 12.80), (4.00, 13.30), (3.80, 13.70), (3.62, 13.75),
]

# Fallback rivers — approximate centerlines for 10 major Nigerian rivers.
# Coordinates are (lon, lat) ordered upstream -> downstream. These are used
# only if the OSM Overpass fetch fails (offline / rate-limited).
FALLBACK_RIVERS: List[Dict[str, Any]] = [
    {"name": "Niger", "coords": [
        (3.60, 11.90), (3.85, 11.55), (4.20, 11.10), (4.55, 10.55),
        (4.90, 10.00), (5.30, 9.50), (5.75, 9.10), (6.25, 8.45),
        (6.75, 7.80), (6.90, 7.20), (6.60, 6.60), (6.40, 5.95),
        (6.35, 5.35), (6.40, 4.80), (6.60, 4.50)]},
    {"name": "Benue", "coords": [
        (13.50, 7.90), (13.00, 8.40), (12.50, 9.20), (11.50, 8.80),
        (10.50, 8.40), (9.50, 8.10), (8.50, 7.75), (7.55, 7.75),
        (6.90, 7.80)]},
    {"name": "Kaduna", "coords": [
        (9.50, 9.80), (8.70, 10.15), (7.95, 10.45), (7.40, 10.50),
        (6.80, 10.10), (6.20, 9.60), (5.90, 9.10)]},
    {"name": "Sokoto", "coords": [
        (5.25, 13.05), (4.90, 12.60), (4.60, 12.20), (4.40, 11.85),
        (4.20, 11.60), (3.85, 11.55)]},
    {"name": "Gongola", "coords": [
        (10.60, 10.80), (10.90, 10.40), (11.10, 9.90), (11.30, 9.40),
        (11.50, 9.00), (11.50, 8.80)]},
    {"name": "Komadugu Yobe", "coords": [
        (11.00, 12.00), (11.60, 12.20), (12.20, 12.50), (12.80, 12.80),
        (13.20, 13.00), (13.40, 13.10)]},
    {"name": "Cross", "coords": [
        (9.10, 6.10), (8.80, 5.80), (8.55, 5.40), (8.30, 4.90),
        (8.25, 4.75)]},
    {"name": "Ogun", "coords": [
        (3.40, 8.50), (3.35, 8.00), (3.30, 7.40), (3.30, 6.90),
        (3.35, 6.55), (3.40, 6.40)]},
    {"name": "Osun", "coords": [
        (4.55, 7.80), (4.45, 7.40), (4.35, 7.00), (4.30, 6.60),
        (4.20, 6.30)]},
    {"name": "Anambra", "coords": [
        (7.20, 6.90), (7.00, 6.60), (6.85, 6.35), (6.75, 6.10)]},
]


def _fetch_osm_rivers_nigeria(timeout: float = 40.0
                              ) -> Optional[List[Dict[str, Any]]]:
    """Fetch Nigerian rivers from the OpenStreetMap Overpass API.

    Returns a list of ``{"name": str, "coords": [(lon, lat), ...]}`` dicts,
    or ``None`` on failure. No API key is required. Rate-limited by
    Overpass to ~2 requests per second per IP; this function is intended
    to be called at most once per container boot and its result cached.
    """
    try:
        import requests
    except Exception:
        return None
    query = (
        '[out:json][timeout:40];'
        'area["ISO3166-1"="NG"][admin_level=2]->.ng;'
        '(way["waterway"="river"](area.ng););'
        'out geom;'
    )
    for endpoint in ("https://overpass-api.de/api/interpreter",
                     "https://overpass.kumi.systems/api/interpreter"):
        try:
            resp = requests.post(endpoint, data={"data": query}, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            continue
        rivers: List[Dict[str, Any]] = []
        for elem in payload.get("elements", []):
            if elem.get("type") != "way":
                continue
            geom = elem.get("geometry") or []
            if len(geom) < 2:
                continue
            coords = [(float(p["lon"]), float(p["lat"])) for p in geom]
            name = (elem.get("tags") or {}).get("name") or "River"
            rivers.append({"name": name, "coords": coords})
        if rivers:
            return rivers
    return None


# ===========================================================================
#                          FloodForecastWebApp
# ===========================================================================
class FloodForecastWebApp:
    """Self-contained Streamlit front-end for FloodForecastSystem."""

    PAGES = ["🗺️  Map", "📤 Upload Data", "📊 Forecast & Alerts",
             "📈 Raw Data", "📚 Instructions", "ℹ️  About"]

    ALERT_HEX = {"RED": "#C0392B", "AMBER": "#F39C12", "GREEN": "#27AE60"}

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        defaults: Dict[str, Any] = {
            "page": self.PAGES[0],
            "system": None,
            "forecast_report": None,
            "forecast_date": None,
            "workdir": None,
            "run_log": [],
            "last_error": None,
            "is_demo": True,
            "horizon_days": 14,
            "forecast_start": None,
            "r2_urls": {},
            "r2_seeded": False,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        # Optional Cloudflare R2 persistence. Returns None if not configured.
        self.r2 = None
        if cloud_storage is not None:
            try:
                self.r2 = cloud_storage.get_r2_from_secrets(st.secrets)
            except Exception:
                self.r2 = None

    # ------------------------------------------------------------------
    # Filesystem
    # ------------------------------------------------------------------
    def _workdir(self) -> Path:
        if st.session_state.workdir is None:
            tmp = Path(tempfile.mkdtemp(prefix="flood_web_"))
            # NOTE: do NOT create data/chirps. The pipeline's
            # get_basin_mean_precip() falls back to a synthetic gamma-rain
            # series only when the chirps path does not exist; if we create
            # an empty directory, rasterio.open() is called on that directory
            # and raises RasterioIOError. Leaving it absent is the fix.
            for sub in ("data/gauges", "data/dem", "data/met", "outputs"):
                (tmp / sub).mkdir(parents=True, exist_ok=True)
            st.session_state.workdir = str(tmp)
        return Path(st.session_state.workdir)

    def _reset_workdir(self) -> None:
        if st.session_state.workdir and Path(st.session_state.workdir).exists():
            try:
                shutil.rmtree(st.session_state.workdir)
            except Exception:
                pass
        for k in ("workdir", "system", "forecast_report", "forecast_date",
                  "last_error"):
            st.session_state[k] = None
        st.session_state.run_log = []
        st.session_state.is_demo = True

    def _log(self, msg: str) -> None:
        st.session_state.run_log.append(msg)

    # ------------------------------------------------------------------
    # Cloudflare R2 (optional)
    # ------------------------------------------------------------------
    def _r2_enabled(self) -> bool:
        return self.r2 is not None

    def _r2_seed_inputs(self, wd: Path) -> None:
        """Pull any ``inputs/`` prefix from R2 into the local workdir.

        This is how persistent uploads survive a Streamlit Cloud restart:
        previously uploaded gauges/DEM/met/NWP live under the bucket's
        ``inputs/`` prefix and get downloaded into ``workdir/data/`` at the
        start of a run. Idempotent — only runs once per session.
        """
        if not self._r2_enabled() or st.session_state.r2_seeded:
            return
        try:
            n = self.r2.sync_prefix_to_local("inputs", wd / "data")
            if n:
                self._log(f"R2: seeded {n} file(s) from inputs/ into workdir.")
        except Exception as exc:
            self._log(f"R2 seed failed (continuing without cloud data): {exc}")
        st.session_state.r2_seeded = True

    def _r2_mirror_upload(self, local_path: Path, remote_key: str) -> None:
        """Mirror a user upload to R2 under ``inputs/`` for persistence."""
        if not self._r2_enabled():
            return
        try:
            url = self.r2.upload_file(local_path, f"inputs/{remote_key}")
            if url:
                self._log(f"R2: mirrored {local_path.name} -> {url}")
        except Exception as exc:
            self._log(f"R2 mirror failed: {exc}")

    def _r2_publish_outputs(self, cfg: ffn.Config, run_id: str) -> None:
        """Upload generated outputs (map, plots, CSVs, alerts) to R2."""
        if not self._r2_enabled():
            return
        out_dir = Path(cfg.output_dir)
        if not out_dir.exists():
            return
        try:
            urls = self.r2.sync_local_to_prefix(
                out_dir, f"runs/{run_id}",
                patterns=["*.html", "*.png", "*.csv", "*.json", "*.geojson"],
            )
            st.session_state.r2_urls = {Path(u).name: u for u in urls}
            self._log(f"R2: published {len(urls)} artefact(s) under runs/{run_id}/.")
        except Exception as exc:
            self._log(f"R2 publish failed: {exc}")

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------
    @staticmethod
    def _template_gauge_csv() -> bytes:
        idx = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        rng = np.random.default_rng(0)
        doy = np.array([d.timetuple().tm_yday for d in idx])
        stations = [
            ("NIG_LOK", "Lokoja", 7.80, 6.74, "Niger", 6000.0),
            ("BEN_MAK", "Makurdi", 7.73, 8.54, "Benue", 2500.0),
            ("OSU_OSO", "Oshogbo", 7.76, 4.56, "Osun", 120.0),
        ]
        seasonal = 1.0 + 1.3 * np.sin(2 * np.pi * (doy - 150) / 365.25)
        rows = []
        for sid, nm, lat, lon, riv, meanq in stations:
            q = np.clip(meanq * seasonal * rng.normal(1.0, 0.25, len(idx)), 1.0, None)
            h = 1.0 + np.log1p(q / meanq) * 1.5
            for d, Q, H in zip(idx, q, h):
                rows.append({
                    "station_id": sid, "station_name": nm,
                    "latitude": lat, "longitude": lon, "river": riv,
                    "date": d.strftime("%Y-%m-%d"),
                    "discharge_m3s": round(float(Q), 2),
                    "stage_m": round(float(H), 3),
                })
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    @staticmethod
    def _template_met_csv() -> bytes:
        idx = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        rng = np.random.default_rng(1)
        doy = np.array([d.timetuple().tm_yday for d in idx])
        wet = np.isin(np.array([d.month for d in idx]), [5, 6, 7, 8, 9, 10])
        precip = np.where(wet, rng.gamma(0.8, 12.0, len(idx)),
                          rng.gamma(0.3, 2.5, len(idx)))
        temp = 27.0 + 5.0 * np.sin(2 * np.pi * (doy - 90) / 365.25) + rng.normal(0, 1, len(idx))
        pet = np.clip(rng.normal(4.5, 0.8, len(idx)), 0.2, None)
        df = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"),
                           "precip_mm": np.clip(precip, 0, None).round(2),
                           "temp_c": temp.round(2),
                           "pet_mm": pet.round(2)})
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    @staticmethod
    def _template_nwp_csv(horizon_days: int) -> bytes:
        idx = pd.date_range(pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
                            periods=horizon_days, freq="D")
        rng = np.random.default_rng(2)
        p = np.clip(rng.gamma(1.0, 10.0, horizon_days), 0, None)
        df = pd.DataFrame({"area_mean": p}, index=idx)
        df.index.name = "date"
        buf = io.StringIO()
        df.to_csv(buf)
        return buf.getvalue().encode("utf-8")

    # ------------------------------------------------------------------
    # File ingest
    # ------------------------------------------------------------------
    def _save_uploaded_file(self, uploaded: Any, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(uploaded.getvalue())
        # Mirror to R2 so the upload persists across Streamlit Cloud restarts.
        wd = self._workdir()
        try:
            rel = dest.relative_to(wd / "data").as_posix()
        except ValueError:
            rel = dest.name
        self._r2_mirror_upload(dest, rel)
        return dest

    def _build_config(self, start_date: str, end_date: str,
                      horizon_days: int) -> ffn.Config:
        wd = self._workdir()
        return ffn.Config(
            data_dir=wd / "data",
            dem_path=wd / "data" / "dem" / "dem.tif",
            gauge_csv=wd / "data" / "gauges" / "gauge_stations.csv",
            output_dir=wd / "outputs",
            start_date=start_date,
            end_date=end_date,
            forecast_horizon_days=int(horizon_days),
        )

    def _ensure_gauge_csv(self, uploaded: Any, cfg: ffn.Config) -> Path:
        if uploaded is not None:
            path = self._save_uploaded_file(uploaded, Path(cfg.gauge_csv))
            self._log(f"Saved uploaded gauge CSV ({path.stat().st_size/1024:.1f} KB).")
            return path
        system = ffn.FloodForecastSystem(cfg)
        system._write_synthetic_gauge_csv()
        self._log("No gauge CSV - using synthetic 10-year demo record.")
        return Path(cfg.gauge_csv)

    def _ensure_dem(self, uploaded: Any, cfg: ffn.Config) -> Path:
        if uploaded is not None:
            path = self._save_uploaded_file(uploaded, Path(cfg.dem_path))
            self._log(f"Saved uploaded DEM ({path.stat().st_size/1024/1024:.1f} MB).")
            return path
        delin = ffn.WatershedDelineator(cfg)
        delin.generate_synthetic_dem()
        self._log("No DEM - generated synthetic Nigeria DEM.")
        return Path(cfg.dem_path)

    def _ensure_met(self, uploaded: Any) -> Optional[pd.DataFrame]:
        if uploaded is None:
            return None
        try:
            df = pd.read_csv(uploaded)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            required = {"precip_mm", "temp_c", "pet_mm"}
            if not required.issubset(df.columns):
                self._log(f"Met CSV missing {required - set(df.columns)} - ignoring.")
                return None
            self._log(f"Loaded user met CSV ({len(df):,} rows).")
            return df
        except Exception as exc:
            self._log(f"Could not parse met CSV: {exc}")
            return None

    def _build_nwp(self, uploaded: Any, horizon_days: int,
                   start: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
        idx = pd.date_range(start, periods=horizon_days, freq="D")
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                date_col = next((c for c in df.columns if c.lower() == "date"), None)
                if date_col is None:
                    raise ValueError("NWP CSV needs a 'date' column")
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()
                df = df.reindex(idx).ffill().bfill()
                precip_cols = [c for c in df.columns if c.lower() != "pet_mm"]
                p_df = df[precip_cols]
                if "pet_mm" in df.columns:
                    e_df = pd.DataFrame({"area_mean": df["pet_mm"]}, index=idx)
                else:
                    e_df = pd.DataFrame({"area_mean": np.full(horizon_days, 4.5)}, index=idx)
                self._log(f"Loaded uploaded NWP CSV for {horizon_days} days.")
                return p_df, e_df
            except Exception as exc:
                self._log(f"NWP parse failed ({exc}) - synthesising instead.")
        rng = np.random.default_rng(11)
        precip = np.clip(rng.gamma(1.0, 10.0, horizon_days), 0.0, None)
        if horizon_days >= 3:
            precip[2] += 35.0
            precip[min(3, horizon_days - 1)] += 20.0
        pet = np.clip(rng.normal(4.5, 0.8, horizon_days), 0.5, None)
        p_df = pd.DataFrame({"area_mean": precip}, index=idx)
        e_df = pd.DataFrame({"area_mean": pet}, index=idx)
        p_df.index.name = e_df.index.name = "date"
        return p_df, e_df

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------
    def _patch_chirps_download(self) -> None:
        """Short-circuit ``DataManager.download_chirps``.

        The default implementation hammers
        ``https://data.chc.ucsb.edu/products/CHIRPS-2.0/...`` three times per
        month of history with a 1-2-4 s back-off. On Streamlit Cloud those
        URLs return 404 (the public directory layout moved / IP-blocks
        cloud egress), so a 2-year run wastes ~90 s of retries before the
        downstream code falls back to synthetic met anyway. This patch makes
        the function return immediately with a non-existent path so the
        fallback triggers on attempt 0. Idempotent & class-level.
        """
        if getattr(ffn.DataManager, "_web_chirps_patched", False):
            return

        def _skip_download(self, year, month):  # type: ignore[no-redef]
            # Return a path that does not exist so load_chirps_for_bbox
            # treats CHIRPS as unavailable and uses generate_synthetic_met.
            return Path(self.config.data_dir) / "chirps" / (
                f"chirps-v2.0.{year}.{month:02d}.tif.gz"
            )

        ffn.DataManager.download_chirps = _skip_download
        ffn.DataManager._web_chirps_patched = True
        self._log("Patched download_chirps to skip network (uses synthetic met).")

    def _patch_basin_precip(self, system: ffn.FloodForecastSystem,
                            cfg: ffn.Config) -> None:
        """Class-level monkey-patch for WatershedDelineator.get_basin_mean_precip.

        The deployed flood_forecast_nigeria.py calls rasterio.open(chirps_path)
        unconditionally. When no real CHIRPS raster is available (the common
        web-demo case) this raises RasterioIOError. We replace the method with
        a synthetic gamma-distributed wet/dry-season rain generator that
        matches the fallback behaviour of the local reference implementation.
        Patch is applied once; idempotent.
        """
        if getattr(ffn.WatershedDelineator, "_web_precip_patched", False):
            return

        def _synthetic_basin_precip(self, basins, chirps_path):
            idx = pd.date_range(self.config.start_date,
                                self.config.end_date, freq="D")
            # Try real raster first if a valid file is provided.
            try:
                p = Path(chirps_path)
                if (p.exists() and p.is_file()
                        and hasattr(ffn, "rasterio")
                        and hasattr(ffn, "rio_mask")):
                    df = pd.DataFrame(index=idx)
                    with ffn.rasterio.open(p) as src:
                        for _, row in basins.iterrows():
                            if row["geometry"] is None:
                                df[row["station_id"]] = 0.0
                                continue
                            try:
                                out, _ = ffn.rio_mask.mask(
                                    src, [ffn.mapping(row["geometry"])],
                                    crop=True)
                                df[row["station_id"]] = float(
                                    np.ma.masked_invalid(out).mean())
                            except Exception:
                                df[row["station_id"]] = 0.0
                    return df
            except Exception:
                pass
            # Synthetic fallback: gamma-distributed wet-season rain.
            rng = np.random.default_rng(seed=20260421)
            months = np.array([d.month for d in idx])
            wet = np.isin(months, [5, 6, 7, 8, 9, 10])
            data = {}
            for sid in basins["station_id"]:
                p = np.where(wet,
                             rng.gamma(0.8, 12.0, len(idx)),
                             rng.gamma(0.3, 2.5, len(idx)))
                data[sid] = np.clip(p, 0.0, None)
            df = pd.DataFrame(data, index=idx)
            df.index.name = "date"
            return df

        ffn.WatershedDelineator.get_basin_mean_precip = _synthetic_basin_precip
        ffn.WatershedDelineator._web_precip_patched = True
        self._log("Patched get_basin_mean_precip with synthetic fallback.")

    def _patch_with_user_met(self, system: ffn.FloodForecastSystem,
                             cfg: ffn.Config, user_met: pd.DataFrame) -> None:
        """Instance-level monkey-patch: use the user met CSV verbatim."""
        user_met_clipped = user_met.reindex(
            pd.date_range(cfg.start_date, cfg.end_date, freq="D")
        ).interpolate(limit=14).ffill().bfill()

        def _patched():
            dm = ffn.DataManager(cfg)
            dm.load_gauge_data()
            setattr(cfg, "sparse_stations", list(dm.sparse_stations))
            merged = user_met_clipped.copy()
            for col in ("precip_mm", "temp_c", "pet_mm"):
                if col not in merged.columns:
                    merged[col] = 0.0
            system.met_data = merged
            system.data_manager = dm
            gf = ffn.GapFiller(cfg, dm.gauge_data, merged)
            gf.fill_all()
            system.gap_filler = gf
            wd = ffn.WatershedDelineator(cfg)
            wd.load_dem()
            watershed = wd.delineate()
            watershed["dem"] = wd.dem
            watershed["profile"] = wd.profile
            watershed["basin_precip"] = wd.get_basin_mean_precip(
                watershed["basins"], cfg.data_dir / "chirps")
            system.watershed_data = watershed
            system.delineator = wd
            hm = ffn.HydrologicalModel(cfg, watershed["basins"],
                                       dm.gauge_data, merged)
            for sid in dm.gauge_data.keys():
                hm.calibrate(sid)
            for sid in dm.gauge_data.keys():
                system.hindcasts[sid] = hm.run_hindcast(sid)
                system.Q_per_basin[sid] = system.hindcasts[sid][["Q_sim_m3s"]].copy()
                system.return_periods[sid] = hm.compute_return_periods(
                    sid, system.hindcasts[sid]["Q_sim_m3s"])
            system.hydro_model = hm
            fr = ffn.FloodRouter(cfg, watershed, system.Q_per_basin)
            system.routed_Q = fr.route_network()
            system.flood_router = fr
            peak = fr.get_peak_flood_map(cfg.end_date)
            stations_gdf = dm.get_station_geodataframe()
            ad = ffn.AlertDashboard(
                cfg, system.routed_Q, system.return_periods,
                peak, stations_gdf,
                hindcasts=system.hindcasts, forecasts={},
                river_network=watershed.get("river_network"))
            ad.plot_all_stations()
            ad.generate_folium_map(cfg.end_date)
            ad.generate_alert_report(cfg.end_date)
            system.alert_dashboard = ad

        _patched()

    def run_pipeline(self, uploaded_gauge: Any, uploaded_dem: Any,
                     uploaded_met: Any, uploaded_nwp: Any,
                     start_date: str, end_date: str,
                     forecast_start: pd.Timestamp, horizon_days: int,
                     is_demo: bool) -> None:
        """End-to-end run; stores the system + report in session state."""
        try:
            cfg = self._build_config(start_date, end_date, horizon_days)
            # Pull any persisted inputs from R2 before we start saving uploads.
            self._r2_seed_inputs(self._workdir())
            self._ensure_gauge_csv(uploaded_gauge, cfg)
            self._ensure_dem(uploaded_dem, cfg)
            user_met = self._ensure_met(uploaded_met)

            system = ffn.FloodForecastSystem(cfg)
            self._patch_chirps_download()
            self._patch_basin_precip(system, cfg)
            if user_met is not None:
                self._patch_with_user_met(system, cfg, user_met)
            else:
                system.run_historical()
            self._log("Hindcast complete.")

            system.config.forecast_horizon_days = int(horizon_days)
            p_df, e_df = self._build_nwp(uploaded_nwp, horizon_days, forecast_start)
            report = system.run_forecast(p_df, e_df)
            self._log(f"Forecast complete ({horizon_days} days from {forecast_start.date()}).")

            # Publish outputs to R2 under runs/<timestamp>/.
            run_id = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
            self._r2_publish_outputs(cfg, run_id)

            st.session_state.system = system
            st.session_state.forecast_report = report
            st.session_state.forecast_date = forecast_start.strftime("%Y-%m-%d")
            st.session_state.is_demo = is_demo
            st.session_state.horizon_days = horizon_days
            st.session_state.forecast_start = forecast_start
            st.session_state.last_error = None
        except Exception as exc:
            tb = traceback.format_exc()
            st.session_state.last_error = f"Pipeline failed: {exc}\n{tb}"
            self._log(f"ERROR: {exc}")

    def _ensure_demo_system(self) -> None:
        """Run a synthetic demo so the map has content on first load."""
        if st.session_state.system is not None:
            return
        today = pd.Timestamp.today().normalize()
        end = today - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=365 * 2)
        horizon = st.session_state.horizon_days or 14
        fc_start = today
        with st.spinner("Generating demo map (first visit only, ~30 s)..."):
            self.run_pipeline(
                uploaded_gauge=None, uploaded_dem=None,
                uploaded_met=None, uploaded_nwp=None,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                forecast_start=fc_start, horizon_days=horizon,
                is_demo=True,
            )

    # ------------------------------------------------------------------
    # Map rendering (rivers + watersheds + stations + flood overlay)
    # ------------------------------------------------------------------
    def _basins_to_geojson(self, system: ffn.FloodForecastSystem):
        """Convert the labelled-basin raster to polygon features."""
        ws = getattr(system, "watershed_data", None)
        if not ws:
            return None
        basins = ws.get("basins")
        profile = ws.get("profile") or {}
        if basins is None:
            return None
        rio_features = _try_import_rasterio_features()
        if rio_features is None:
            return None
        transform = profile.get("transform")
        if transform is None:
            return None
        try:
            mask = basins.astype("int32")
            shapes = rio_features.shapes(mask, mask=(mask > 0), transform=transform)
            feats = []
            for geom, val in shapes:
                feats.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"basin_id": int(val)},
                })
            return feats
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Real Nigerian rivers (cached OSM fetch with hardcoded fallback)
    # ------------------------------------------------------------------
    def _get_nigeria_rivers(self) -> List[Dict[str, Any]]:
        """Return a list of ``{name, coords}`` river records for Nigeria.

        Strategy (cache ordering):

        1. In-session cache (``st.session_state.rivers``).
        2. R2 cache (``cache/nigeria_rivers.json``) if R2 is configured.
        3. OSM Overpass API — live fetch, then persist to R2 + session.
        4. Hardcoded :data:`FALLBACK_RIVERS` so the map is never empty.
        """
        cached = st.session_state.get("rivers")
        if cached:
            return cached
        # R2 cache.
        if self._r2_enabled():
            try:
                wd = self._workdir()
                local = wd / "cache" / "nigeria_rivers.json"
                if self.r2.download_file("cache/nigeria_rivers.json", local):
                    data = json.loads(local.read_text(encoding="utf-8"))
                    if data:
                        st.session_state.rivers = data
                        self._log(f"Rivers: loaded {len(data)} from R2 cache.")
                        return data
            except Exception as exc:
                self._log(f"R2 river cache read failed: {exc}")
        # Live OSM fetch.
        osm = _fetch_osm_rivers_nigeria()
        if osm:
            st.session_state.rivers = osm
            self._log(f"Rivers: fetched {len(osm)} from OpenStreetMap.")
            if self._r2_enabled():
                try:
                    self.r2.upload_bytes(
                        json.dumps(osm).encode("utf-8"),
                        "cache/nigeria_rivers.json",
                        content_type="application/json",
                    )
                except Exception as exc:
                    self._log(f"R2 river cache write failed: {exc}")
            return osm
        # Last resort: hardcoded.
        self._log("Rivers: using hardcoded fallback (10 major rivers).")
        st.session_state.rivers = FALLBACK_RIVERS
        return FALLBACK_RIVERS

    def _river_alert_levels(self, recs: List[Dict[str, Any]]
                            ) -> Dict[str, str]:
        """Map lower-cased river name -> worst alert level across its gauges."""
        rank = {"GREEN": 0, "AMBER": 1, "RED": 2}
        worst: Dict[str, str] = {}
        for r in recs:
            river = (r.get("river") or "").strip().lower()
            if not river:
                continue
            level = r.get("alert_level", "GREEN")
            if river not in worst or rank[level] > rank[worst[river]]:
                worst[river] = level
        return worst

    def _station_records(self, system: ffn.FloodForecastSystem,
                         report: Optional[Dict[str, Dict[str, Any]]]):
        """Return station records for the map.

        Only returns records when the user has actually uploaded gauge data.
        In demo mode the map shows rivers and watersheds but no gauges, so
        the alert layer stays off until real data is supplied. Any gauge
        outside the Nigeria bounding box is dropped.
        """
        if st.session_state.get("is_demo", True):
            return []
        dm = getattr(system, "data_manager", None)
        if dm is None:
            return []
        gdf = dm.get_station_geodataframe()
        lon_min, lat_min, lon_max, lat_max = NIGERIA_BBOX
        recs = []
        for _, row in gdf.iterrows():
            sid = row.get("station_id")
            lat = float(row.get("latitude", np.nan))
            lon = float(row.get("longitude", np.nan))
            if not (np.isfinite(lat) and np.isfinite(lon)):
                continue
            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                # Gauge is outside Nigeria — skip (and only skip, do not alert).
                continue
            alert = (report or {}).get(sid, {})
            recs.append({
                "station_id": sid,
                "name": row.get("station_name", sid),
                "river": row.get("river", ""),
                "lat": lat, "lon": lon,
                "alert_level": alert.get("alert_level", "GREEN"),
                "peak_Q": alert.get("peak_Q_m3s", np.nan),
                "peak_date": alert.get("peak_date", ""),
                "pct_2yr": alert.get("Q_pct_of_2yr", np.nan),
            })
        return recs

    def _build_map_html(self, system: ffn.FloodForecastSystem,
                        report: Optional[Dict[str, Dict[str, Any]]],
                        forecast_date: Optional[str]) -> Optional[str]:
        folium = _try_import_folium()
        if folium is None:
            return None

        recs = self._station_records(system, report)
        # Lock the map viewport to Nigeria so nothing outside the country is
        # ever the primary view.
        lon_min, lat_min, lon_max, lat_max = NIGERIA_BBOX
        fmap = folium.Map(
            location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2],
            zoom_start=6, tiles="CartoDB positron", control_scale=True,
            min_lat=lat_min - 1, max_lat=lat_max + 1,
            min_lon=lon_min - 1, max_lon=lon_max + 1,
        )
        fmap.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])

        # --- Nigeria national boundary (always drawn first) ---
        fg_bnd = folium.FeatureGroup(name="Nigeria boundary", show=True)
        folium.PolyLine(
            [(lat, lon) for lon, lat in NIGERIA_BOUNDARY],
            color="#1B2631", weight=2.0, opacity=0.7, dash_array="6,4",
        ).add_to(fg_bnd)
        fg_bnd.add_to(fmap)

        # --- Watershed polygons, clipped to Nigeria bbox ---
        basin_feats = self._basins_to_geojson(system)
        if basin_feats:
            fg_b = folium.FeatureGroup(name="Watersheds", show=True)
            palette = ["#5DADE2", "#48C9B0", "#F5B041", "#AF7AC5",
                       "#EC7063", "#58D68D", "#F4D03F", "#5D6D7E"]
            for feat in basin_feats:
                geom = feat.get("geometry") or {}
                # Skip any basin whose first coordinate is outside Nigeria.
                coords = geom.get("coordinates") or []
                first = None
                if coords and isinstance(coords[0], list) and coords[0]:
                    first = coords[0][0] if isinstance(coords[0][0], list) else None
                if first and len(first) >= 2:
                    lon, lat = float(first[0]), float(first[1])
                    if not (lon_min <= lon <= lon_max
                            and lat_min <= lat <= lat_max):
                        continue
                bid = int(feat["properties"]["basin_id"])
                color = palette[bid % len(palette)]
                folium.GeoJson(
                    feat,
                    style_function=lambda _x, c=color: {
                        "fillColor": c, "color": "#2C3E50",
                        "weight": 1.0, "fillOpacity": 0.25,
                    },
                    tooltip=f"Watershed {bid}",
                ).add_to(fg_b)
            fg_b.add_to(fmap)

        # --- Real Nigerian river network, coloured by alert ---
        rivers = self._get_nigeria_rivers()
        alert_by_river = self._river_alert_levels(recs)
        fg_r = folium.FeatureGroup(name="Rivers", show=True)
        for riv in rivers:
            coords_lonlat = riv.get("coords") or []
            if len(coords_lonlat) < 2:
                continue
            # Clip each river to the Nigeria bbox (drop vertices outside).
            inside = [(lon, lat) for lon, lat in coords_lonlat
                      if lon_min <= lon <= lon_max
                      and lat_min <= lat <= lat_max]
            if len(inside) < 2:
                continue
            path_latlon = [(lat, lon) for lon, lat in inside]
            name = (riv.get("name") or "").strip()
            level = alert_by_river.get(name.lower(), None)
            if level == "RED":
                color, weight, opacity = "#C0392B", 4.0, 0.95
            elif level == "AMBER":
                color, weight, opacity = "#F39C12", 3.2, 0.90
            elif level == "GREEN":
                color, weight, opacity = "#27AE60", 2.8, 0.85
            else:
                # No gauge on this river -> baseline blue.
                color, weight, opacity = "#1F618D", 1.8, 0.75
            tooltip = f"{name or 'River'}"
            if level:
                tooltip += f" — {level}"
            folium.PolyLine(
                path_latlon, color=color, weight=weight, opacity=opacity,
                tooltip=tooltip,
            ).add_to(fg_r)
        fg_r.add_to(fmap)

        # --- Flood extent overlay (RiverREM based HAND) ---
        try:
            ad = getattr(system, "alert_dashboard", None)
            if ad is not None and forecast_date is not None:
                peak_map = system.flood_router.get_peak_flood_map(forecast_date) \
                    if hasattr(system, "flood_router") else None
                # AlertDashboard may have generated a dedicated HTML already;
                # overlay the raster if available in watershed_data.
                ws = getattr(system, "watershed_data", None) or {}
                dem = ws.get("dem")
                profile = ws.get("profile") or {}
                rio_features = _try_import_rasterio_features()
                if dem is not None and profile.get("transform") is not None \
                        and peak_map is not None and rio_features is not None:
                    flood = peak_map.get("inundation") if isinstance(peak_map, dict) else None
                    if flood is not None:
                        mask = (flood > 0).astype("uint8")
                        if mask.any():
                            fg_f = folium.FeatureGroup(name="Peak flood extent",
                                                       show=True)
                            shapes = rio_features.shapes(
                                mask, mask=mask.astype(bool),
                                transform=profile["transform"])
                            for geom, _v in shapes:
                                folium.GeoJson(
                                    geom,
                                    style_function=lambda _x: {
                                        "fillColor": "#1E3A8A",
                                        "color": "#1E3A8A",
                                        "weight": 0, "fillOpacity": 0.45,
                                    },
                                ).add_to(fg_f)
                            fg_f.add_to(fmap)
        except Exception:
            pass  # overlay is cosmetic; never block the map

        # --- Station markers ---
        if recs:
            fg_s = folium.FeatureGroup(name="Gauge stations", show=True)
            for r in recs:
                c = self.ALERT_HEX.get(r["alert_level"], "#2C3E50")
                pct = r["pct_2yr"]
                pct_s = f"{pct:.0f}%" if np.isfinite(pct) else "n/a"
                popup = (f"<b>{r['name']}</b> ({r['station_id']})<br>"
                         f"River: {r['river']}<br>"
                         f"Alert: <b>{r['alert_level']}</b><br>"
                         f"Peak Q: {r['peak_Q']:.1f} m³/s<br>"
                         f"% of 2-yr RP: {pct_s}<br>"
                         f"Peak date: {r['peak_date'][:10]}")
                folium.CircleMarker(
                    [r["lat"], r["lon"]],
                    radius=8, color="#212F3C", weight=1.5,
                    fill=True, fill_color=c, fill_opacity=0.9,
                    popup=folium.Popup(popup, max_width=280),
                    tooltip=f"{r['station_id']} — {r['alert_level']}",
                ).add_to(fg_s)
            fg_s.add_to(fmap)

        # Legend
        try:
            from branca.element import MacroElement, Template
            legend_html = """
            {% macro html(this, kwargs) %}
            <div style="position: fixed; bottom: 22px; left: 22px; z-index: 9999;
                        background: white; padding: 10px 14px; border-radius: 6px;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.25); font: 12px/1.4 sans-serif;">
              <b>Legend</b><br>
              <svg width="22" height="12" style="vertical-align:middle;">
                <path d="M1,10 Q7,2 12,8 T21,3" stroke="#C0392B" stroke-width="3" fill="none"/>
              </svg> River with RED alert<br>
              <svg width="22" height="12" style="vertical-align:middle;">
                <path d="M1,10 Q7,2 12,8 T21,3" stroke="#F39C12" stroke-width="3" fill="none"/>
              </svg> River with AMBER alert<br>
              <svg width="22" height="12" style="vertical-align:middle;">
                <path d="M1,10 Q7,2 12,8 T21,3" stroke="#27AE60" stroke-width="3" fill="none"/>
              </svg> River with GREEN alert<br>
              <svg width="22" height="12" style="vertical-align:middle;">
                <path d="M1,10 Q7,2 12,8 T21,3" stroke="#1F618D" stroke-width="2" fill="none"/>
              </svg> River (no gauge)<br>
              <span style='display:inline-block;width:14px;height:10px;background:rgba(93,173,226,0.4);border:1px solid #2C3E50;'></span> Watershed<br>
              <span style='display:inline-block;width:14px;height:10px;background:rgba(30,58,138,0.5);'></span> Peak flood extent<br>
              <span style='display:inline-block;width:14px;height:2px;border-top:2px dashed #1B2631;'></span> Nigeria boundary
            </div>
            {% endmacro %}
            """
            macro = MacroElement()
            macro._template = Template(legend_html)
            fmap.get_root().add_child(macro)
        except Exception:
            pass

        folium.LayerControl(collapsed=False).add_to(fmap)
        try:
            return fmap.get_root().render()
        except Exception:
            return fmap._repr_html_()

    # ------------------------------------------------------------------
    # Navigation sidebar
    # ------------------------------------------------------------------
    def render_nav(self) -> None:
        with st.sidebar:
            st.markdown("<div class='nav-title'>Navigation</div>",
                        unsafe_allow_html=True)
            st.session_state.page = st.radio(
                " ", self.PAGES,
                index=self.PAGES.index(st.session_state.page),
                key="nav_page_radio",
                label_visibility="collapsed",
            )
            st.divider()
            system = st.session_state.system
            if system is None:
                st.caption("No run yet.")
            else:
                mode = "Demo run" if st.session_state.is_demo else "User data run"
                n = len(system.routed_Q) if getattr(system, "routed_Q", None) else 0
                st.caption(f"**{mode}** — {n} basin(s)")
                st.caption(f"Horizon: {st.session_state.horizon_days} day(s)")
                if st.session_state.forecast_date:
                    st.caption(f"Forecast start: {st.session_state.forecast_date}")
            st.divider()
            if st.button("Reset session"):
                self._reset_workdir()
                st.rerun()

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    def page_map(self) -> None:
        st.title("Nigeria flood-forecast map")
        self._ensure_demo_system()
        system = st.session_state.system
        report = st.session_state.forecast_report
        forecast_date = st.session_state.forecast_date
        if system is None:
            st.error(st.session_state.last_error or "Map could not be generated.")
            return
        mode_badge = ("🟡 **Demo data** — upload your own on the Upload page "
                      "to refresh this map."
                      if st.session_state.is_demo
                      else "🟢 **Using uploaded data.**")
        st.info(mode_badge)
        html = self._build_map_html(system, report, forecast_date)
        if html is None:
            # Fall back to the ffn-produced map file if our builder failed.
            map_path = Path(system.config.output_dir) / f"flood_map_{forecast_date}.html"
            if map_path.exists():
                html = map_path.read_text(encoding="utf-8")
        if html:
            components.html(html, height=720, scrolling=False)
            st.download_button(
                "Download map HTML",
                data=html.encode("utf-8"),
                file_name="flood_map.html",
                mime="text/html",
            )
        else:
            st.warning("Folium is not installed; cannot render the interactive map.")

        # Summary metrics
        if report:
            red = sum(1 for r in report.values() if r["alert_level"] == "RED")
            amber = sum(1 for r in report.values() if r["alert_level"] == "AMBER")
            green = sum(1 for r in report.values() if r["alert_level"] == "GREEN")
            peak = max((r["peak_Q_m3s"] for r in report.values()), default=0.0)
            c = st.columns(4)
            c[0].metric("RED alerts", red)
            c[1].metric("AMBER alerts", amber)
            c[2].metric("GREEN alerts", green)
            c[3].metric("Peak Q (m³/s)", f"{peak:,.0f}")

    # ------------------------------------------------------------------
    def page_upload(self) -> None:
        st.title("Upload data & run forecast")
        st.markdown(
            "Upload whichever of the four files you have — anything missing is "
            "replaced with realistic synthetic data. The map auto-refreshes "
            "against your upload once you press **Run forecast**."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Historical period")
            start = st.date_input(
                "Start date",
                value=pd.to_datetime("2022-01-01").date(),
                min_value=pd.to_datetime("1990-01-01").date(),
                max_value=pd.to_datetime("2035-12-31").date(),
            )
            end = st.date_input(
                "End date",
                value=pd.to_datetime("2024-12-31").date(),
                min_value=start,
                max_value=pd.to_datetime("2035-12-31").date(),
            )
        with col_b:
            st.subheader("Forecast horizon")
            unit = st.radio("Unit", ["Days", "Weeks", "Months"],
                            index=1, horizontal=True,
                            key="horizon_unit_radio")
            if unit == "Days":
                v = st.slider("Length (days)", 1, 60, 14)
                horizon_days = v
            elif unit == "Weeks":
                v = st.slider("Length (weeks)", 1, 8, 2)
                horizon_days = v * 7
            else:
                v = st.slider("Length (months)", 1, 3, 1)
                horizon_days = v * 30
            st.caption(f"= **{horizon_days} day(s)**")
            default_fc = pd.to_datetime(end) + pd.Timedelta(days=1)
            fc_start = st.date_input(
                "Forecast start date",
                value=default_fc.date(),
                min_value=pd.to_datetime(end).date(),
            )

        st.subheader("Files")
        f1, f2 = st.columns(2)
        with f1:
            gauge_file = st.file_uploader(
                "Gauge stations CSV",
                type=["csv"],
                help="station_id, station_name, latitude, longitude, river, "
                     "date, discharge_m3s, stage_m",
            )
            dem_file = st.file_uploader(
                "DEM (GeoTIFF, optional)",
                type=["tif", "tiff"],
                help="Single-band elevation raster in WGS84.",
            )
        with f2:
            met_file = st.file_uploader(
                "Meteorology CSV (optional)",
                type=["csv"],
                help="date, precip_mm, temp_c, pet_mm",
            )
            nwp_file = st.file_uploader(
                "Forecast precipitation CSV (optional)",
                type=["csv"],
                help="date + area_mean (or one column per station_id); optional pet_mm.",
            )

        st.subheader("Templates")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.download_button("Gauge CSV template",
                               data=self._template_gauge_csv(),
                               file_name="gauge_template.csv",
                               mime="text/csv")
        with t2:
            st.download_button("Met CSV template",
                               data=self._template_met_csv(),
                               file_name="met_template.csv",
                               mime="text/csv")
        with t3:
            st.download_button("NWP CSV template",
                               data=self._template_nwp_csv(horizon_days),
                               file_name="nwp_template.csv",
                               mime="text/csv")

        st.divider()
        run = st.button("🚀  Run forecast with these inputs", type="primary")
        if run:
            if horizon_days < 1:
                st.error("Horizon must be ≥ 1 day.")
                return
            if horizon_days > 60:
                st.warning("Horizons > 60 days are extrapolative — proceeding.")
            is_demo = not any([gauge_file, dem_file, met_file, nwp_file])
            # Purge previous workdir so nothing stale leaks in.
            self._reset_workdir()
            with st.spinner("Calibrating GR4J per basin and routing the network..."):
                self.run_pipeline(
                    uploaded_gauge=gauge_file,
                    uploaded_dem=dem_file,
                    uploaded_met=met_file,
                    uploaded_nwp=nwp_file,
                    start_date=str(start),
                    end_date=str(end),
                    forecast_start=pd.to_datetime(fc_start),
                    horizon_days=int(horizon_days),
                    is_demo=is_demo,
                )
            if st.session_state.last_error:
                st.error(st.session_state.last_error)
            else:
                st.success("Forecast complete — switch to the Map tab.")
                st.session_state.page = self.PAGES[0]
                st.rerun()

    # ------------------------------------------------------------------
    def page_forecast(self) -> None:
        st.title("Forecast & alerts")
        system = st.session_state.system
        report = st.session_state.forecast_report
        if system is None or report is None:
            st.info("Run the pipeline first.")
            return
        red = sum(1 for r in report.values() if r["alert_level"] == "RED")
        amber = sum(1 for r in report.values() if r["alert_level"] == "AMBER")
        green = sum(1 for r in report.values() if r["alert_level"] == "GREEN")
        peak = max((r["peak_Q_m3s"] for r in report.values()), default=0.0)
        c = st.columns(4)
        c[0].metric("RED", red)
        c[1].metric("AMBER", amber)
        c[2].metric("GREEN", green)
        c[3].metric("Peak Q (m³/s)", f"{peak:,.0f}")

        df = pd.DataFrame.from_dict(report, orient="index").reset_index() \
            .rename(columns={"index": "station_id"})
        for col in df.columns:
            if df[col].dtype.kind in "fc":
                df[col] = df[col].astype(float).round(2)

        def color_row(row):
            c = {"RED": "#fadbd8", "AMBER": "#fdebd0",
                 "GREEN": "#d5f5e3"}.get(row["alert_level"], "white")
            return [f"background-color: {c}"] * len(row)

        try:
            st.dataframe(df.style.apply(color_row, axis=1),
                         use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download alerts JSON",
            data=json.dumps(report, indent=2, default=str).encode("utf-8"),
            file_name="alert_report.json",
            mime="application/json",
        )

        st.subheader("Per-station forecast plots")
        station_ids = list(system.routed_Q.keys())
        picks = st.multiselect("Stations", station_ids, default=station_ids)
        for sid in picks:
            img = Path(system.config.output_dir) / f"{sid}_forecast.png"
            if img.exists():
                st.image(str(img), caption=sid, use_container_width=True)
            else:
                st.caption(f"(no plot for {sid})")

    # ------------------------------------------------------------------
    def page_data(self) -> None:
        st.title("Raw data")
        system = st.session_state.system
        if system is None or not getattr(system, "routed_Q", None):
            st.info("Run the pipeline first.")
            return
        sid = st.selectbox("Station", list(system.routed_Q.keys()))
        t1, t2, t3 = st.tabs(["Routed discharge", "Forecast", "Return periods"])
        with t1:
            df = system.routed_Q[sid]
            st.line_chart(df["Q_routed_m3s"].tail(365), height=280)
            st.download_button(
                f"Download routed Q for {sid}",
                data=df.to_csv().encode("utf-8"),
                file_name=f"{sid}_routed_Q.csv",
                mime="text/csv",
            )
        with t2:
            fc = (system.alert_dashboard.forecasts.get(sid)
                  if system.alert_dashboard else None)
            if fc is not None and len(fc):
                st.dataframe(fc, use_container_width=True)
                st.download_button(
                    f"Download forecast for {sid}",
                    data=fc.to_csv().encode("utf-8"),
                    file_name=f"{sid}_forecast.csv",
                    mime="text/csv",
                )
            else:
                st.info("No forecast data.")
        with t3:
            rp = system.return_periods.get(sid)
            if rp is not None and len(rp):
                st.dataframe(rp, use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download return periods for {sid}",
                    data=rp.to_csv(index=False).encode("utf-8"),
                    file_name=f"{sid}_return_periods.csv",
                    mime="text/csv",
                )

    # ------------------------------------------------------------------
    def page_instructions(self) -> None:
        st.title("Instructions")
        st.markdown("""
### What you're looking at
The **Map** page is the landing view and always shows a live forecast —
a synthetic demo run on first load, or your uploaded data after you run the
pipeline.  It overlays:

- **Watershed polygons** — derived from the DEM via D8 flow-direction.
- **River network** — streams above the area-threshold drainage density.
- **Peak flood extent** — HAND / RiverREM-based inundation at the forecast horizon.
- **Gauge stations** — coloured by alert level (RED / AMBER / GREEN).

### How to use your own data
1. Open **Upload Data** in the left menu.
2. Pick a historical period, a forecast unit (**Days / Weeks / Months**) and a length.
   *Default is 2 weeks ahead.*
3. Upload any combination of:
   - Gauge CSV — `station_id, station_name, latitude, longitude, river, date, discharge_m3s, stage_m`
   - DEM GeoTIFF — single-band elevation, WGS84
   - Met CSV — `date, precip_mm, temp_c, pet_mm`
   - Forecast precipitation CSV — `date + area_mean` (or one column per station) and optional `pet_mm`
4. Press **Run forecast**. The Map refreshes automatically.

### Models used under the hood
- **Watershed delineation** — pysheds D8 (with a pure-NumPy fallback).
- **Gap filling** — IDW spatial interpolation + a windowed multivariate
  regression surrogate for short records.
- **Rainfall-runoff** — GR4J (Perrin et al. 2003) calibrated per basin with
  Kling-Gupta Efficiency via `scipy.optimize.differential_evolution`.
- **Channel routing** — Muskingum with C0+C1+C2 stability check.
- **Frequency analysis** — Gumbel EV-I return-period fit for 2 / 5 / 10 /
  25 / 50 / 100-yr flows.
- **Inundation** — RiverREM / HAND via `scipy.ndimage.distance_transform_edt`.
- **Alert triage** — station-level RED ≥ Q₂-yr, AMBER ≥ 0.7·Q₂-yr, else GREEN.

### Tips
- The app works with **no uploads at all** — the default demo exercises
  every model and is a good sanity check before committing your data.
- Use the **Reset session** button at the bottom of the sidebar if you
  want to wipe cached results without refreshing the browser.
- All outputs — routed Q, forecasts, return periods, alerts, the map HTML —
  are downloadable from the Forecast & Alerts and Raw Data pages.
        """)

    # ------------------------------------------------------------------
    def page_about(self) -> None:
        st.title("About")
        st.markdown("""
This dashboard is a thin Streamlit UI on top of
`flood_forecast_nigeria.py`, which implements the full eight-class
hydrological pipeline (Config, DataManager, GapFiller, WatershedDelineator,
HydrologicalModel, FloodRouter, AlertDashboard, FloodForecastSystem).

The web layer never modifies the pipeline — it only orchestrates uploads,
overrides the forecast horizon, and renders the outputs.

### Cloud storage (Cloudflare R2)
""")
        if self._r2_enabled():
            st.success(
                f"R2 is configured (bucket: `{self.r2.bucket}`). "
                "Uploads persist across restarts and every run publishes "
                "artefacts under `runs/<timestamp>/`."
            )
            urls = st.session_state.get("r2_urls") or {}
            if urls:
                st.markdown("**Latest run artefacts:**")
                for name, url in urls.items():
                    st.markdown(f"- [{name}]({url})")
            else:
                st.caption("No run artefacts published yet.")
        else:
            st.info(
                "R2 is not configured. Add an `[r2]` section to "
                "`.streamlit/secrets.toml` (or the Streamlit Cloud secrets UI) "
                "to enable persistence. Required keys: `endpoint_url`, "
                "`access_key`, `secret_key`, `bucket_name`."
            )

        st.markdown("### Run log")
        if st.session_state.run_log:
            st.code("\n".join(st.session_state.run_log), language="text")
        else:
            st.caption("Nothing logged yet.")
        if st.session_state.last_error:
            st.error(st.session_state.last_error)

    # ------------------------------------------------------------------
    def render(self) -> None:
        self.render_nav()
        page = st.session_state.page
        if page == self.PAGES[0]:
            self.page_map()
        elif page == self.PAGES[1]:
            self.page_upload()
        elif page == self.PAGES[2]:
            self.page_forecast()
        elif page == self.PAGES[3]:
            self.page_data()
        elif page == self.PAGES[4]:
            self.page_instructions()
        else:
            self.page_about()


# ===========================================================================
#                               Entry point
# ===========================================================================
def main() -> None:
    app = FloodForecastWebApp()
    app.render()


if __name__ == "__main__":
    main()
else:
    # Streamlit imports the script under `__mp_main__`, not `__main__`.
    main()
