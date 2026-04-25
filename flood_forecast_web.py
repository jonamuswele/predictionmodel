# flood_forecast_web.py - FANFAR-driven daily forecast + bias correction

from __future__ import annotations

import datetime as dt
import io
import json
import re
import shutil
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

try:
    import cloud_storage
except Exception:
    cloud_storage = None


# ===========================================================================
#                              CONSTANTS
# ===========================================================================

NIGERIA_BOUNDS = [[3.8, 2.3], [14.2, 15.0]]
NIGERIA_CENTER = [9.0, 8.1]

MAJOR_RIVER_NAMES = {
    "niger", "benue", "kaduna", "sokoto", "gongola", "komadugu", "komadugu yobe",
    "yobe", "cross", "ogun", "osun", "anambra", "hadejia", "katsina-ala",
    "katsina ala", "donga", "taraba", "imo", "kwa ibo", "qua iboe",
    "forcados", "escravos", "nun", "orashi",
}

ALERT_HEX = {
    "RED":     "#C0392B",
    "AMBER":   "#F39C12",
    "GREEN":   "#27AE60",
    "UNKNOWN": "#7F8C8D",
}

DEFAULT_FANFAR_URL = "https://api.fanfar.eu/v1"
FORECAST_HORIZON_DAYS = 7


# ===========================================================================
#                            FANFAR CLIENT
# ===========================================================================

class FANFARClient:
    """Minimal FANFAR REST client. Tolerant of misconfiguration."""

    def __init__(self, api_key: str = "", base_url: str = DEFAULT_FANFAR_URL,
                 timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.last_error: Optional[str] = None

    def _session(self):
        if not _HAS_REQUESTS:
            return None
        s = requests.Session()
        if self.api_key:
            s.headers["Authorization"] = f"Bearer {self.api_key}"
        s.headers["Accept"] = "application/json"
        return s

    def get_forecast(self, station_id: str,
                     days: int = FORECAST_HORIZON_DAYS) -> Optional[Dict]:
        s = self._session()
        if s is None:
            self.last_error = "requests library not installed"
            return None
        try:
            r = s.get(f"{self.base_url}/forecasts/daily",
                      params={"station": station_id, "days": days,
                              "format": "json"},
                      timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return {
                "station_id": station_id,
                "forecast_date": str(data.get("forecast_date")
                                     or dt.date.today().isoformat()),
                "discharge": [float(x) for x in data.get("discharge", [])],
                "water_level": [float(x) for x in data.get("water_level", [])],
                "units": "m³/s",
            }
        except Exception as e:
            self.last_error = f"{station_id}: {e}"
            return None

    def get_available_stations(self) -> List[Dict]:
        s = self._session()
        if s is None:
            return []
        try:
            r = s.get(f"{self.base_url}/stations", timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("stations", []) if isinstance(data, dict) else []
        except Exception as e:
            self.last_error = f"stations list: {e}"
            return []


# ===========================================================================
#                          R2-BACKED STORES
# ===========================================================================

class StationRegistry:
    """Per-station metadata stored at stations/registry.json in R2."""
    KEY = "stations/registry.json"

    def __init__(self, r2):
        self.r2 = r2
        self._stations: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        if not self.r2:
            return
        local = Path("./data/stations/registry.json")
        local.parent.mkdir(parents=True, exist_ok=True)
        if self.r2.exists(self.KEY) and self.r2.download_file(self.KEY, local):
            try:
                with open(local) as f:
                    self._stations = json.load(f) or {}
            except Exception:
                self._stations = {}

    def list(self) -> List[Dict]:
        return sorted(self._stations.values(),
                      key=lambda x: x.get("station_name", ""))

    def get(self, station_id: str) -> Optional[Dict]:
        return self._stations.get(station_id)

    def upsert(self, station: Dict) -> bool:
        sid = (station.get("station_id") or "").strip()
        if not sid:
            return False
        existing = self._stations.get(sid, {})
        existing.update(station)
        self._stations[sid] = existing
        return self._save()

    def remove(self, station_id: str) -> bool:
        self._stations.pop(station_id, None)
        return self._save()

    def _save(self) -> bool:
        if not self.r2:
            return False
        body = json.dumps(self._stations, indent=2).encode()
        return bool(self.r2.upload_bytes(body, self.KEY, "application/json"))


class HistoricalStore:
    """Per-station historical observations CSV at historical/<sid>.csv."""
    PREFIX = "historical"
    COLUMNS = ["date", "discharge_m3s", "stage_m"]

    def __init__(self, r2):
        self.r2 = r2
        self.local_dir = Path("./data/historical")
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, station_id: str) -> str:
        return f"{self.PREFIX}/{station_id}.csv"

    def load(self, station_id: str, refresh: bool = False) -> Optional[pd.DataFrame]:
        if not self.r2:
            return None
        local = self.local_dir / f"{station_id}.csv"
        if refresh or not local.exists():
            if not self.r2.exists(self._key(station_id)):
                return None
            if not self.r2.download_file(self._key(station_id), local):
                return None
        try:
            df = pd.read_csv(local)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"])
            return df
        except Exception:
            return None

    def save(self, station_id: str, df: pd.DataFrame) -> bool:
        if not self.r2 or df is None or df.empty:
            return False
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        local = self.local_dir / f"{station_id}.csv"
        df.to_csv(local, index=False)
        return bool(self.r2.upload_file(local, self._key(station_id)))

    def append(self, station_id: str, row: Dict) -> bool:
        existing = self.load(station_id, refresh=True)
        new_row = pd.DataFrame([row])
        new_row["date"] = pd.to_datetime(new_row["date"], errors="coerce")
        new_row = new_row.dropna(subset=["date"])
        if existing is not None and not existing.empty:
            combined = (pd.concat([existing, new_row], ignore_index=True)
                          .drop_duplicates(subset=["date"], keep="last")
                          .sort_values("date"))
        else:
            combined = new_row
        return self.save(station_id, combined)

    def stations_with_history(self) -> List[str]:
        if not self.r2:
            return []
        keys = self.r2.list_files(f"{self.PREFIX}/")
        return sorted({Path(k).stem for k in keys
                       if k.lower().endswith(".csv")})


class ForecastStore:
    """Latest daily forecast at forecasts/latest.json (+ dated archive)."""
    LATEST_KEY = "forecasts/latest.json"

    def __init__(self, r2):
        self.r2 = r2
        self.local = Path("./data/forecasts/latest.json")
        self.local.parent.mkdir(parents=True, exist_ok=True)

    def load_latest(self, refresh: bool = False) -> Optional[Dict]:
        if not self.r2:
            return None
        if refresh or not self.local.exists():
            if not self.r2.exists(self.LATEST_KEY):
                return None
            if not self.r2.download_file(self.LATEST_KEY, self.local):
                return None
        try:
            with open(self.local) as f:
                return json.load(f)
        except Exception:
            return None

    def save_latest(self, payload: Dict) -> bool:
        if not self.r2:
            return False
        body = json.dumps(payload, indent=2, default=str).encode()
        ok1 = self.r2.upload_bytes(body, self.LATEST_KEY, "application/json")
        date_str = payload.get("forecast_date") or dt.date.today().isoformat()
        ok2 = self.r2.upload_bytes(
            body, f"forecasts/{date_str}.json", "application/json"
        )
        # also keep a local copy
        try:
            with open(self.local, "wb") as f:
                f.write(body)
        except Exception:
            pass
        return bool(ok1 and ok2)


# ===========================================================================
#                          BIAS CORRECTION
# ===========================================================================

def fit_correction(history: Optional[pd.DataFrame],
                   fanfar_overlap: Optional[pd.DataFrame]) -> Dict[str, float]:
    """Median ratio + median bias from overlapping (obs, fanfar) pairs.

    history columns: date, discharge_m3s
    fanfar_overlap columns: date, discharge_m3s_fanfar
    """
    if (history is None or history.empty
            or fanfar_overlap is None or fanfar_overlap.empty):
        return {"factor": 1.0, "bias": 0.0, "samples": 0}
    merged = pd.merge(history[["date", "discharge_m3s"]],
                      fanfar_overlap, on="date", how="inner")
    valid = merged.dropna(subset=["discharge_m3s", "discharge_m3s_fanfar"])
    valid = valid[valid["discharge_m3s_fanfar"] > 0]
    if len(valid) < 5:
        return {"factor": 1.0, "bias": 0.0, "samples": int(len(valid))}
    factor = float((valid["discharge_m3s"] / valid["discharge_m3s_fanfar"]).median())
    bias = float((valid["discharge_m3s"] - valid["discharge_m3s_fanfar"]).median())
    return {"factor": factor, "bias": bias, "samples": int(len(valid))}


def apply_correction(values: List[float], correction: Dict[str, float]) -> List[float]:
    f = float(correction.get("factor", 1.0))
    b = float(correction.get("bias", 0.0))
    return [max(0.0, float(v) * f + b) for v in values]


def classify_alert(peak_q: float,
                   history: Optional[pd.DataFrame]) -> str:
    """Q95/Q99 of historical observations define AMBER/RED."""
    if (history is None or history.empty
            or "discharge_m3s" not in history.columns):
        return "UNKNOWN"
    obs = history["discharge_m3s"].dropna()
    if len(obs) < 30:
        return "UNKNOWN"
    q95 = float(obs.quantile(0.95))
    q99 = float(obs.quantile(0.99))
    if peak_q >= q99:
        return "RED"
    if peak_q >= q95:
        return "AMBER"
    return "GREEN"


# ===========================================================================
#                      SHAPEFILE / MAP DATA MANAGER
# ===========================================================================

class ProfessionalDataManager:
    """Mirrors geojson/ from R2 and loads boundary, watersheds, waterways."""

    BASIN_KEYWORDS = ("hybas", "basin", "watershed")
    MAJOR_RIV_KEYWORDS = ("ne_10m_rivers", "ne_50m_rivers",
                          "natural_earth", "rivers_lake_centerlines")
    MINOR_RIV_KEYWORDS = ("hotosm", "waterway", "osm_water", "water_lines",
                          "nga_water")

    def __init__(self, r2_storage=None):
        self.r2 = r2_storage
        self.data_dir = Path("./data/hydrological")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.watersheds = None
        self.rivers = None
        self.boundary = None
        if 'data_downloaded_from_r2' not in st.session_state:
            st.session_state.data_downloaded_from_r2 = False

    def _find_shp_by_keywords(self, keywords) -> Optional[Path]:
        for p in self.data_dir.rglob("*.shp"):
            if any(k in p.name.lower() for k in keywords):
                return p
        return None

    def _extract_zip(self, zip_path: Path) -> bool:
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(self.data_dir)
            return True
        except Exception:
            return False

    def ensure_data_loaded(self) -> bool:
        if self.watersheds is not None and self.rivers is not None:
            return True
        if not self.r2:
            return False
        if not st.session_state.data_downloaded_from_r2:
            if not self._sync_geojson_prefix():
                return False
            st.session_state.data_downloaded_from_r2 = True
        for z in self.data_dir.rglob("*.zip"):
            if any(self.data_dir.rglob(f"{z.stem}*.shp")):
                continue
            self._extract_zip(z)
        return self._load_from_local_cache()

    def _sync_geojson_prefix(self) -> bool:
        try:
            keys = self.r2.list_files("geojson/")
            if not keys:
                return False
            for key in keys:
                rel = key.split("geojson/", 1)[-1].lstrip("/")
                if not rel:
                    continue
                local = self.data_dir / rel
                local.parent.mkdir(parents=True, exist_ok=True)
                if local.exists() and local.stat().st_size > 0:
                    continue
                self.r2.download_file(key, local)
            return True
        except Exception:
            return False

    def _load_from_local_cache(self) -> bool:
        try:
            import geopandas as gpd
            from shapely.geometry import box, mapping

            boundary_path = self.data_dir / "nigeria_boundary.geojson"
            if boundary_path.exists():
                try:
                    with open(boundary_path, 'r') as f:
                        self.boundary = json.load(f)
                except Exception:
                    self.boundary = None

            nigeria_gdf = self._get_nigeria_boundary_gdf()

            basin_shp = self._find_shp_by_keywords(self.BASIN_KEYWORDS)
            if basin_shp and basin_shp.exists():
                try:
                    basins = gpd.read_file(basin_shp)
                    if basins.crs and basins.crs.to_epsg() != 4326:
                        basins = basins.to_crs("EPSG:4326")
                    nb = gpd.clip(basins, nigeria_gdf)
                    nb['geometry'] = nb['geometry'].buffer(0)
                    nb = nb[nb.is_valid]
                    self.watersheds = json.loads(nb.to_json())
                    self.watersheds['metadata'] = {
                        'source': basin_shp.name, 'count': len(nb)
                    }
                except Exception:
                    pass
            if self.watersheds is None:
                self._create_fallback_watersheds()

            all_features: List[Dict[str, Any]] = []
            minor_shp = self._find_shp_by_keywords(self.MINOR_RIV_KEYWORDS)
            major_shp = self._find_shp_by_keywords(self.MAJOR_RIV_KEYWORDS)
            if (major_shp and minor_shp
                    and major_shp.resolve() == minor_shp.resolve()):
                minor_shp = None

            if minor_shp and minor_shp.exists():
                try:
                    gdf = gpd.read_file(minor_shp)
                    if gdf.crs and gdf.crs.to_epsg() != 4326:
                        gdf = gdf.to_crs("EPSG:4326")
                    try:
                        clipped = gpd.clip(gdf, nigeria_gdf)
                    except Exception:
                        clipped = gdf[gdf.intersects(nigeria_gdf.unary_union)]
                    if len(clipped) == 0:
                        minx, miny, maxx, maxy = nigeria_gdf.total_bounds
                        bbox_poly = box(minx - 1, miny - 1, maxx + 1, maxy + 1)
                        clipped = gdf[gdf.intersects(bbox_poly)]
                    if 'waterway' in clipped.columns:
                        wanted = ['river', 'stream', 'canal', 'drain',
                                  'ditch', 'tidal_channel']
                        clipped = clipped[clipped['waterway'].isin(wanted)]
                    clipped = clipped[~clipped.geometry.is_empty]
                    clipped = clipped[clipped.geometry.is_valid]
                    if len(clipped) > 4000:
                        clipped = clipped.copy()
                        clipped['length_deg'] = clipped.geometry.length
                        clipped = clipped.nlargest(4000, 'length_deg')
                    for idx, row in clipped.iterrows():
                        try:
                            geom_json = mapping(row.geometry)
                        except Exception:
                            continue
                        name = None
                        for field in ('name', 'NAME', 'name_en'):
                            if field in row and row[field]:
                                name = str(row[field])
                                break
                        waterway = (row['waterway']
                                    if 'waterway' in row and row['waterway']
                                    else 'stream')
                        waterway = str(waterway).lower()
                        category = waterway
                        if waterway == 'river' and name:
                            ln = name.lower().strip()
                            if any(re.search(rf"\b{re.escape(m)}\b", ln)
                                   for m in MAJOR_RIVER_NAMES):
                                category = 'major_river'
                        if not name:
                            name = waterway.capitalize()
                        all_features.append({
                            "type": "Feature",
                            "geometry": geom_json,
                            "properties": {
                                "name": name,
                                "waterway": waterway,
                                "category": category,
                                "length_km": round(float(row.geometry.length) * 111.0, 1),
                            },
                        })
                except Exception:
                    pass

            if not any(f['properties']['category'] == 'major_river'
                       for f in all_features):
                if major_shp and major_shp.exists():
                    try:
                        gdf = gpd.read_file(major_shp)
                        if gdf.crs and gdf.crs.to_epsg() != 4326:
                            gdf = gdf.to_crs("EPSG:4326")
                        try:
                            clipped = gpd.clip(gdf, nigeria_gdf)
                        except Exception:
                            clipped = gdf[gdf.intersects(nigeria_gdf.unary_union)]
                        if len(clipped) == 0:
                            minx, miny, maxx, maxy = nigeria_gdf.total_bounds
                            bbox_poly = box(minx - 1, miny - 1, maxx + 1, maxy + 1)
                            clipped = gdf[gdf.intersects(bbox_poly)]
                        clipped = clipped[~clipped.geometry.is_empty]
                        clipped = clipped[clipped.geometry.is_valid]
                        for idx, row in clipped.iterrows():
                            try:
                                geom_json = mapping(row.geometry)
                            except Exception:
                                continue
                            name = None
                            for field in ('name', 'NAME', 'Name', 'name_en',
                                          'river', 'RIVER'):
                                if field in row and row[field]:
                                    name = str(row[field])
                                    break
                            if not name:
                                name = "Major River"
                            all_features.append({
                                "type": "Feature",
                                "geometry": geom_json,
                                "properties": {
                                    "name": name,
                                    "waterway": "river",
                                    "category": "major_river",
                                    "length_km": round(float(row.geometry.length) * 111.0, 1),
                                },
                            })
                    except Exception:
                        pass

            self.rivers = ({"type": "FeatureCollection",
                            "features": all_features,
                            "metadata": {"count": len(all_features)}}
                           if all_features else None)
            return True
        except Exception:
            return False

    def _get_nigeria_boundary_gdf(self):
        import geopandas as gpd
        from shapely.geometry import box
        boundary_path = self.data_dir / "nigeria_boundary.geojson"
        if boundary_path.exists():
            try:
                return gpd.read_file(boundary_path)
            except Exception:
                pass
        bbox = (2.5, 4.0, 14.7, 13.9)
        return gpd.GeoDataFrame(geometry=[box(*bbox)], crs='EPSG:4326')

    def _create_fallback_watersheds(self):
        watersheds_data = {
            "HA1": {"name": "Niger North",      "color": "#5DADE2", "bounds": (2.5, 9.5, 9.5, 13.9)},
            "HA2": {"name": "Niger Central",    "color": "#48C9B0", "bounds": (4.0, 7.5, 9.5, 11.0)},
            "HA3": {"name": "Lower Niger",      "color": "#F5B041", "bounds": (5.0, 4.5, 8.5, 8.0)},
            "HA4": {"name": "Upper Benue",      "color": "#AF7AC5", "bounds": (9.0, 7.5, 13.5, 10.5)},
            "HA5": {"name": "Lower Benue",      "color": "#EC7063", "bounds": (7.5, 6.0, 11.0, 8.5)},
            "HA6": {"name": "Cross River",      "color": "#58D68D", "bounds": (7.5, 4.5, 9.5, 6.5)},
            "HA7": {"name": "Western Littoral", "color": "#F4D03F", "bounds": (2.5, 6.0, 5.5, 8.0)},
            "HA8": {"name": "Lake Chad Basin",  "color": "#5D6D7E", "bounds": (9.5, 10.5, 14.7, 13.5)},
        }
        features = []
        for ha_id, info in watersheds_data.items():
            x0, y0, x1, y1 = info["bounds"]
            coords = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "id": ha_id, "name": info["name"], "color": info["color"],
                },
            })
        self.watersheds = {"type": "FeatureCollection",
                           "features": features,
                           "metadata": {"source": "Nigeria Hydrological Areas"}}


# ===========================================================================
#                                WEB APP
# ===========================================================================

class FloodForecastWebApp:

    PAGES = ["🗺️  Map", "🔮 Forecast", "📥 Historical Data",
             "🛰️ Stations", "ℹ️  About"]

    def __init__(self):
        defaults = {
            "page": self.PAGES[0],
            "r2_connected": False,
            "latest_forecast": None,
            "forecast_run_log": [],
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        # R2
        self.r2 = None
        if cloud_storage is not None:
            try:
                self.r2 = cloud_storage.get_r2_from_secrets(st.secrets)
                if self.r2:
                    st.session_state.r2_connected = True
            except Exception:
                pass

        # FANFAR config from secrets (with defaults)
        fanfar_url = DEFAULT_FANFAR_URL
        fanfar_key = ""
        try:
            section = st.secrets.get("fanfar") if hasattr(st.secrets, "get") else None
            if section:
                fanfar_url = section.get("base_url", fanfar_url)
                fanfar_key = section.get("api_key", "")
        except Exception:
            pass
        self.fanfar = FANFARClient(api_key=fanfar_key, base_url=fanfar_url)

        # Stores
        self.data_manager = ProfessionalDataManager(self.r2)
        self.stations = StationRegistry(self.r2)
        self.history = HistoricalStore(self.r2)
        self.forecasts = ForecastStore(self.r2)

        # Eager-load latest forecast for the map.
        if st.session_state.latest_forecast is None:
            st.session_state.latest_forecast = self.forecasts.load_latest()

    # ------------------------------------------------------------------
    # Forecast runner
    # ------------------------------------------------------------------
    def _run_forecast(self) -> Dict[str, Any]:
        run_log: List[str] = []
        stations = self.stations.list()
        if not stations:
            return {"ok": False, "error": "No stations registered. "
                    "Add stations on the Stations page first.",
                    "log": run_log}

        forecast_date = dt.date.today().isoformat()
        results: List[Dict[str, Any]] = []

        progress = st.progress(0.0, text="Starting forecast...")
        total = len(stations)

        for i, st_meta in enumerate(stations, start=1):
            sid = st_meta["station_id"]
            fanfar_id = st_meta.get("fanfar_station_id") or sid
            progress.progress(i / total,
                              text=f"Forecasting {st_meta.get('station_name', sid)} ({i}/{total})")

            forecast = self.fanfar.get_forecast(fanfar_id, FORECAST_HORIZON_DAYS)
            if not forecast or not forecast.get("discharge"):
                run_log.append(
                    f"FANFAR returned nothing for {sid} → "
                    f"{self.fanfar.last_error or 'no data'}"
                )
                results.append({
                    "station_id": sid,
                    "station_name": st_meta.get("station_name", sid),
                    "latitude": st_meta.get("latitude"),
                    "longitude": st_meta.get("longitude"),
                    "river": st_meta.get("river"),
                    "fanfar_station_id": fanfar_id,
                    "forecast_date": forecast_date,
                    "discharge_raw": [],
                    "discharge": [],
                    "alert_level": "UNKNOWN",
                    "peak_discharge": None,
                    "peak_day_offset": None,
                    "corrected": False,
                    "correction": {"factor": 1.0, "bias": 0.0, "samples": 0},
                    "error": self.fanfar.last_error or "FANFAR returned no data",
                })
                continue

            raw = forecast["discharge"]

            # Bias correction from history (if any)
            history = self.history.load(sid)
            correction = {"factor": 1.0, "bias": 0.0, "samples": 0}
            corrected = False

            if history is not None and not history.empty:
                # Pull a reforecast over the historical range to learn the bias.
                start = history["date"].min().strftime("%Y-%m-%d")
                end = history["date"].max().strftime("%Y-%m-%d")
                reforecast = None  # placeholder for a reforecast endpoint
                # Optimistic call; if the endpoint doesn't exist we just skip.
                try:
                    s = self.fanfar._session()
                    if s is not None:
                        r = s.get(f"{self.fanfar.base_url}/reforecast",
                                  params={"station": fanfar_id,
                                          "start_date": start,
                                          "end_date": end,
                                          "format": "json"},
                                  timeout=60)
                        if r.ok:
                            payload = r.json()
                            df = pd.DataFrame(payload.get("data", []))
                            if not df.empty and "date" in df and "discharge" in df:
                                df = df.rename(columns={"discharge":
                                                        "discharge_m3s_fanfar"})
                                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                                reforecast = df[["date", "discharge_m3s_fanfar"]]
                except Exception:
                    reforecast = None

                if reforecast is not None and not reforecast.empty:
                    correction = fit_correction(history, reforecast)
                    if correction["samples"] >= 5:
                        corrected = True

            adjusted = apply_correction(raw, correction) if corrected else raw
            peak = float(max(adjusted)) if adjusted else 0.0
            peak_idx = int(np.argmax(adjusted)) if adjusted else 0
            alert = classify_alert(peak, history)

            results.append({
                "station_id": sid,
                "station_name": st_meta.get("station_name", sid),
                "latitude": st_meta.get("latitude"),
                "longitude": st_meta.get("longitude"),
                "river": st_meta.get("river"),
                "fanfar_station_id": fanfar_id,
                "forecast_date": forecast_date,
                "discharge_raw": raw,
                "discharge": adjusted,
                "alert_level": alert,
                "peak_discharge": round(peak, 2),
                "peak_day_offset": peak_idx,
                "corrected": corrected,
                "correction": correction,
                "error": None,
            })

        progress.empty()

        payload = {
            "forecast_date": forecast_date,
            "run_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "horizon_days": FORECAST_HORIZON_DAYS,
            "stations": results,
            "summary": self._summarize(results),
        }
        ok = self.forecasts.save_latest(payload)
        st.session_state.latest_forecast = payload
        st.session_state.forecast_run_log = run_log

        return {"ok": ok, "log": run_log, "payload": payload}

    @staticmethod
    def _summarize(results: List[Dict]) -> Dict[str, Any]:
        counts = {"RED": 0, "AMBER": 0, "GREEN": 0, "UNKNOWN": 0}
        for r in results:
            counts[r.get("alert_level", "UNKNOWN")] = counts.get(
                r.get("alert_level", "UNKNOWN"), 0) + 1
        return {
            "total_stations": len(results),
            "alerts": counts,
            "stations_with_data": sum(1 for r in results if r["discharge"]),
            "stations_corrected": sum(1 for r in results if r["corrected"]),
        }

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    def _build_map_html(self) -> Optional[str]:
        try:
            import folium
            from folium.plugins import Fullscreen, MousePosition
        except ImportError:
            return None

        if not self.data_manager.ensure_data_loaded():
            return None

        fmap = folium.Map(
            location=NIGERIA_CENTER, zoom_start=6,
            min_zoom=5, max_zoom=14, max_bounds=True,
            tiles="CartoDB positron",
            control_scale=True, zoom_control=True,
        )
        fmap.fit_bounds(NIGERIA_BOUNDS)
        fmap.options['maxBounds'] = [
            [NIGERIA_BOUNDS[0][0] - 1, NIGERIA_BOUNDS[0][1] - 1],
            [NIGERIA_BOUNDS[1][0] + 1, NIGERIA_BOUNDS[1][1] + 1],
        ]
        Fullscreen(position='topleft').add_to(fmap)
        MousePosition(position='bottomright', separator=' | ',
                      num_digits=3, prefix='Lat/Lon:').add_to(fmap)

        # Boundary
        if self.data_manager.boundary:
            try:
                folium.GeoJson(
                    self.data_manager.boundary, name="Nigeria Boundary",
                    style_function=lambda x: {
                        "color": "#1B2631", "weight": 2,
                        "fillOpacity": 0.03, "fillColor": "#1B2631",
                    }, control=False,
                ).add_to(fmap)
            except Exception:
                pass

        # Watersheds
        watershed_fg = folium.FeatureGroup(name="Watersheds", show=True)
        if self.data_manager.watersheds:
            try:
                sample = self.data_manager.watersheds.get('features', [{}])[0]
                props = sample.get('properties', {})
                if 'HYBAS_ID' in props:
                    folium.GeoJson(
                        self.data_manager.watersheds,
                        style_function=lambda x: {
                            'fillColor': '#2C3E50', 'color': '#2C3E50',
                            'weight': 0.8, 'fillOpacity': 0.10,
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['HYBAS_ID', 'SUB_AREA', 'UP_AREA'],
                            aliases=['ID:', 'Area (km²):', 'Upstream Area (km²):'],
                            localize=True,
                        ),
                    ).add_to(watershed_fg)
                else:
                    folium.GeoJson(
                        self.data_manager.watersheds,
                        style_function=lambda x: {
                            'fillColor': x['properties'].get('color', '#2C3E50'),
                            'color': '#2C3E50',
                            'weight': 0.8, 'fillOpacity': 0.18,
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['id', 'name'],
                            aliases=['ID:', 'Name:'], localize=True,
                        ),
                    ).add_to(watershed_fg)
            except Exception:
                pass
        watershed_fg.add_to(fmap)

        # Rivers (toggleable)
        if self.data_manager.rivers and self.data_manager.rivers.get('features'):
            features = self.data_manager.rivers['features']

            def collect(predicate):
                sub = [f for f in features if predicate(f['properties'])]
                if not sub:
                    return None
                return {"type": "FeatureCollection", "features": sub}

            major = collect(lambda p: p.get('category') == 'major_river')
            rivers = collect(lambda p: p.get('category') == 'river'
                             and p.get('waterway') == 'river')
            streams = collect(lambda p: p.get('waterway') == 'stream')
            canals = collect(lambda p: p.get('waterway') == 'canal')
            drains = collect(lambda p: p.get('waterway')
                             in ('drain', 'ditch', 'tidal_channel'))

            def add_layer(fc, name, show, style):
                if not fc:
                    return
                fg = folium.FeatureGroup(name=name, show=show)
                folium.GeoJson(
                    fc, style_function=lambda x, s=style: s,
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'waterway', 'length_km'],
                        aliases=['Name:', 'Type:', 'Length (km):'],
                        localize=True, sticky=True,
                    ),
                ).add_to(fg)
                fg.add_to(fmap)

            add_layer(drains,  "Drains & ditches", False,
                      {"color": "#a5d8ff", "weight": 0.5, "opacity": 0.6, "fillOpacity": 0})
            add_layer(streams, "Streams", True,
                      {"color": "#74c0fc", "weight": 0.6, "opacity": 0.75, "fillOpacity": 0})
            add_layer(canals,  "Canals", False,
                      {"color": "#4dabf7", "weight": 0.7, "opacity": 0.85,
                       "fillOpacity": 0, "dashArray": "4,4"})
            add_layer(rivers,  "Rivers", True,
                      {"color": "#1a73e8", "weight": 1.6, "opacity": 0.9, "fillOpacity": 0})
            add_layer(major,   "Major rivers", True,
                      {"color": "#003366", "weight": 3.0, "opacity": 0.98, "fillOpacity": 0})

        # Gauge markers from latest forecast
        forecast = st.session_state.latest_forecast
        if forecast and forecast.get("stations"):
            gauges_fg = folium.FeatureGroup(name="Gauges", show=True)
            for r in forecast["stations"]:
                lat = r.get("latitude"); lon = r.get("longitude")
                if lat is None or lon is None:
                    continue
                alert = r.get("alert_level", "UNKNOWN")
                color = ALERT_HEX.get(alert, ALERT_HEX["UNKNOWN"])
                peak = r.get("peak_discharge")
                peak_offset = r.get("peak_day_offset")
                discharge_seq = r.get("discharge", [])
                # Build a small popup
                trend_html = ""
                if discharge_seq:
                    rows = "".join(
                        f"<tr><td>D+{i}</td><td>{q:.1f} m³/s</td></tr>"
                        for i, q in enumerate(discharge_seq)
                    )
                    trend_html = (
                        f"<table style='font-size:11px;border-collapse:collapse;"
                        f"margin-top:4px'>{rows}</table>"
                    )
                corrected_tag = (
                    "<span style='color:#27AE60'>bias-corrected</span>"
                    if r.get("corrected")
                    else "<span style='color:#7F8C8D'>raw FANFAR</span>"
                )
                popup_html = (
                    f"<b>{r.get('station_name', r['station_id'])}</b><br>"
                    f"<i>{r.get('river') or ''}</i><br>"
                    f"Alert: <b style='color:{color}'>{alert}</b><br>"
                    f"Peak: {peak if peak is not None else '—'} m³/s "
                    f"(D+{peak_offset if peak_offset is not None else '—'})<br>"
                    f"{corrected_tag}"
                    f"{trend_html}"
                )
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    color="#1B2631",
                    weight=1,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.9,
                    tooltip=f"{r.get('station_name', r['station_id'])} — {alert}",
                    popup=folium.Popup(popup_html, max_width=280),
                ).add_to(gauges_fg)
            gauges_fg.add_to(fmap)

        # Legend
        legend_html = """
        {% macro html(this, kwargs) %}
        <div style="position: fixed; bottom: 18px; left: 12px; z-index: 9999;
                    background: rgba(255,255,255,0.94); padding: 8px 12px;
                    border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.18);
                    font: 12px/1.45 -apple-system,Segoe UI,Roboto,Arial,sans-serif;">
            <div style="font-weight:600; margin-bottom:4px;">Legend</div>
            <div><span style="display:inline-block;width:14px;height:2px;background:#003366;vertical-align:middle;margin-right:6px;"></span>Major river</div>
            <div><span style="display:inline-block;width:14px;height:2px;background:#1a73e8;vertical-align:middle;margin-right:6px;"></span>River</div>
            <div><span style="display:inline-block;width:14px;height:2px;background:#74c0fc;vertical-align:middle;margin-right:6px;"></span>Stream</div>
            <div><span style="display:inline-block;width:14px;height:2px;background:#4dabf7;border-top:1px dashed #4dabf7;vertical-align:middle;margin-right:6px;"></span>Canal</div>
            <div style="margin-top:4px;"><span style="display:inline-block;width:10px;height:10px;background:#2C3E5044;border:1px solid #2C3E50;vertical-align:middle;margin-right:6px;"></span>Watershed</div>
            <div style="margin-top:4px;font-weight:600">Alert</div>
            <div><span style="color:#C0392B">●</span> RED  &nbsp;<span style="color:#F39C12">●</span> AMBER  &nbsp;<span style="color:#27AE60">●</span> GREEN  &nbsp;<span style="color:#7F8C8D">●</span> Unknown</div>
        </div>
        {% endmacro %}
        """
        from branca.element import MacroElement, Template
        legend = MacroElement()
        legend._template = Template(legend_html)
        fmap.get_root().add_child(legend)

        folium.LayerControl(collapsed=True, position='topright').add_to(fmap)
        return fmap.get_root().render()

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    def _render_map_page(self):
        st.markdown(
            """
            <style>
            #MainMenu, header, footer { visibility: hidden; }
            .block-container { padding: 0 !important; max-width: 100% !important; }
            section[data-testid="stSidebar"] > div { padding-top: 1rem; }
            .stApp > header { display: none; }
            .element-container iframe {
                width: 100% !important;
                height: calc(100vh - 10px) !important;
                border: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        forecast = st.session_state.latest_forecast
        if forecast:
            run_at = forecast.get("run_at", "")
            summary = forecast.get("summary", {})
            alerts = summary.get("alerts", {})
            badge = (f"Forecast {forecast.get('forecast_date')} · "
                     f"{summary.get('total_stations', 0)} stations · "
                     f"🔴 {alerts.get('RED', 0)} 🟠 {alerts.get('AMBER', 0)} "
                     f"🟢 {alerts.get('GREEN', 0)} ⚪ {alerts.get('UNKNOWN', 0)}")
            st.caption(badge)
        else:
            st.caption("No forecast yet. Run one from the Forecast page.")

        html = self._build_map_html()
        if html:
            components.html(html, height=1200, scrolling=False)
        else:
            st.error("Could not generate map. Check R2 contents.")

    def _render_forecast_page(self):
        st.title("Run today's forecast")

        forecast = st.session_state.latest_forecast
        c1, c2, c3, c4 = st.columns(4)
        if forecast:
            sm = forecast.get("summary", {})
            alerts = sm.get("alerts", {})
            c1.metric("Last run", forecast.get("forecast_date", "—"))
            c2.metric("Stations", sm.get("total_stations", 0))
            c3.metric("With data", sm.get("stations_with_data", 0))
            c4.metric("Bias-corrected", sm.get("stations_corrected", 0))
            cnt = (f"🔴 {alerts.get('RED', 0)}  "
                   f"🟠 {alerts.get('AMBER', 0)}  "
                   f"🟢 {alerts.get('GREEN', 0)}  "
                   f"⚪ {alerts.get('UNKNOWN', 0)}")
            st.markdown(f"**Alert breakdown:** {cnt}")
        else:
            c1.metric("Last run", "—")
            c2.metric("Stations", 0)
            c3.metric("With data", 0)
            c4.metric("Bias-corrected", 0)

        st.markdown("---")
        st.write(
            "Pulls a 7-day FANFAR forecast for every registered station, "
            "applies bias correction where historical data is available, "
            "classifies alerts against your historical Q95/Q99 thresholds, "
            "and writes the result to R2."
        )
        if st.button("▶ Run forecast now", type="primary"):
            with st.spinner("Calling FANFAR for every station..."):
                result = self._run_forecast()
            if result.get("ok"):
                payload = result["payload"]
                st.success(
                    f"Forecast complete — {payload['summary']['total_stations']} "
                    f"stations, {payload['summary']['stations_with_data']} returned data, "
                    f"{payload['summary']['stations_corrected']} bias-corrected."
                )
            else:
                st.error(result.get("error") or "Forecast run failed.")
            log = result.get("log") or []
            if log:
                with st.expander(f"Run log ({len(log)} messages)"):
                    for line in log:
                        st.code(line)

        if forecast:
            st.markdown("### Latest results")
            df = pd.DataFrame(forecast["stations"])
            if not df.empty:
                show_cols = ["station_id", "station_name", "river",
                             "alert_level", "peak_discharge",
                             "peak_day_offset", "corrected"]
                show_cols = [c for c in show_cols if c in df.columns]
                st.dataframe(df[show_cols], use_container_width=True,
                             hide_index=True)

    def _render_historical_page(self):
        st.title("Historical data")
        stations = self.stations.list()
        if not stations:
            st.info("Add a station on the Stations page first.")
            return

        sid_options = [(s["station_id"],
                        f"{s.get('station_name', s['station_id'])} ({s['station_id']})")
                       for s in stations]
        labels = [lbl for _, lbl in sid_options]
        ids = [sid for sid, _ in sid_options]

        tab1, tab2, tab3 = st.tabs(
            ["📤 Bulk CSV upload", "📝 Single observation", "📊 Existing data"]
        )

        with tab1:
            sel = st.selectbox("Station", labels, key="hist_csv_station")
            sid = ids[labels.index(sel)] if labels else None
            uploaded = st.file_uploader(
                "Upload CSV with columns: date, discharge_m3s, stage_m",
                type=["csv"], key="hist_csv_uploader",
            )
            mode = st.radio("Mode", ["Replace existing", "Merge with existing"],
                            horizontal=True, key="hist_csv_mode")
            if uploaded and sid and st.button("Save to R2",
                                              key="hist_csv_save"):
                try:
                    df = pd.read_csv(uploaded)
                    if "date" not in df.columns or "discharge_m3s" not in df.columns:
                        st.error("CSV must contain `date` and `discharge_m3s` columns.")
                    else:
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        df = df.dropna(subset=["date"])
                        if mode.startswith("Merge"):
                            existing = self.history.load(sid, refresh=True)
                            if existing is not None and not existing.empty:
                                df = (pd.concat([existing, df], ignore_index=True)
                                        .drop_duplicates(subset=["date"], keep="last")
                                        .sort_values("date"))
                        ok = self.history.save(sid, df)
                        if ok:
                            st.success(f"Saved {len(df)} rows for {sid}.")
                        else:
                            st.error("Save failed.")
                except Exception as e:
                    st.error(f"Could not parse CSV: {e}")

        with tab2:
            sel = st.selectbox("Station", labels, key="hist_obs_station")
            sid = ids[labels.index(sel)] if labels else None
            with st.form("single_obs_form", clear_on_submit=True):
                date = st.date_input("Date", value=dt.date.today())
                discharge = st.number_input("Discharge (m³/s)", min_value=0.0,
                                            step=1.0, format="%.2f")
                stage = st.number_input("Stage (m)", min_value=0.0,
                                        step=0.01, format="%.3f")
                submitted = st.form_submit_button("Append observation")
            if submitted and sid:
                row = {"date": date.isoformat(),
                       "discharge_m3s": float(discharge),
                       "stage_m": float(stage)}
                if self.history.append(sid, row):
                    st.success(f"Appended {date} to {sid}.")
                else:
                    st.error("Save failed.")

        with tab3:
            sids_with = self.history.stations_with_history()
            if not sids_with:
                st.info("No historical data uploaded yet.")
            else:
                sel = st.selectbox("Station", sids_with, key="hist_view_station")
                df = self.history.load(sel, refresh=True)
                if df is None or df.empty:
                    st.warning("Could not load data.")
                else:
                    st.write(f"**{len(df)}** observations from "
                             f"`{df['date'].min().date()}` to "
                             f"`{df['date'].max().date()}`")
                    if "discharge_m3s" in df.columns:
                        chart_df = df.set_index("date")[["discharge_m3s"]]
                        st.line_chart(chart_df)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Q50", f"{df['discharge_m3s'].median():.1f} m³/s")
                        c2.metric("Q95", f"{df['discharge_m3s'].quantile(0.95):.1f} m³/s")
                        c3.metric("Q99", f"{df['discharge_m3s'].quantile(0.99):.1f} m³/s")
                    st.dataframe(df.tail(50), use_container_width=True,
                                 hide_index=True)

    def _render_stations_page(self):
        st.title("Stations")

        stations = self.stations.list()
        sids_with_history = set(self.history.stations_with_history())
        forecast = st.session_state.latest_forecast or {}
        last_results = {r["station_id"]: r
                        for r in forecast.get("stations", [])}

        if stations:
            rows = []
            for s in stations:
                sid = s["station_id"]
                last = last_results.get(sid)
                rows.append({
                    "Station ID": sid,
                    "Name": s.get("station_name", ""),
                    "River": s.get("river", ""),
                    "Lat": s.get("latitude"),
                    "Lon": s.get("longitude"),
                    "FANFAR ID": s.get("fanfar_station_id", sid),
                    "History": "✅" if sid in sids_with_history else "—",
                    "Last alert": last.get("alert_level") if last else "—",
                    "Last peak (m³/s)": last.get("peak_discharge") if last else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)
        else:
            st.info("No stations registered. Add one below.")

        st.markdown("### Add / update a station")
        with st.form("add_station_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                sid = st.text_input("Station ID *", help="Your internal ID")
                name = st.text_input("Station name *")
                river = st.text_input("River")
            with c2:
                lat = st.number_input("Latitude *", min_value=-90.0,
                                      max_value=90.0, value=9.0, format="%.5f")
                lon = st.number_input("Longitude *", min_value=-180.0,
                                      max_value=180.0, value=8.0, format="%.5f")
                fanfar_id = st.text_input(
                    "FANFAR station ID",
                    help="Leave blank to reuse Station ID"
                )
            submitted = st.form_submit_button("Save station")
        if submitted:
            if not sid.strip() or not name.strip():
                st.error("Station ID and Name are required.")
            else:
                ok = self.stations.upsert({
                    "station_id": sid.strip(),
                    "station_name": name.strip(),
                    "river": river.strip(),
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "fanfar_station_id": (fanfar_id.strip() or sid.strip()),
                })
                if ok:
                    st.success(f"Saved station {sid}.")
                    st.rerun()
                else:
                    st.error("Save failed.")

        if stations:
            st.markdown("### Remove a station")
            del_sel = st.selectbox(
                "Pick a station to delete",
                ["—"] + [s["station_id"] for s in stations]
            )
            if del_sel != "—" and st.button("Delete", type="secondary"):
                if self.stations.remove(del_sel):
                    st.success(f"Deleted {del_sel}.")
                    st.rerun()
                else:
                    st.error("Delete failed.")

    def _render_about_page(self):
        st.title("About")
        st.markdown(
            """
            ### Nigeria Flood Forecast (FANFAR-driven)

            **How forecasts work**

            1. Each registered station is queried against the **FANFAR** API
               for a 7-day daily-discharge forecast.
            2. If you have uploaded historical observations for that station,
               the system fits a per-station bias correction (median ratio +
               offset against FANFAR's reforecast over the overlapping period)
               and applies it to the live forecast.
            3. The forecast peak is classified against your historical Q95
               (AMBER) and Q99 (RED) thresholds. With no history, the alert
               level is `UNKNOWN` (gray marker).

            **Data flow**

            - `stations/registry.json` — your station list
            - `historical/<station_id>.csv` — uploaded observations
            - `forecasts/latest.json` + `forecasts/YYYY-MM-DD.json` — outputs
            - `geojson/...` — boundaries, watersheds, waterways

            **FANFAR endpoint**
            """
        )
        st.code(self.fanfar.base_url)
        if self.fanfar.last_error:
            st.warning(f"Last FANFAR error: {self.fanfar.last_error}")
        st.markdown(
            "Configure FANFAR in `secrets.toml`:\n"
            "```toml\n[fanfar]\nbase_url = \"https://api.fanfar.eu/v1\"\n"
            "api_key = \"...\"\n```"
        )
        st.markdown("### R2 status")
        st.write("Connected" if st.session_state.r2_connected
                 else "Not configured")

    # ------------------------------------------------------------------
    # Top-level render
    # ------------------------------------------------------------------
    def render(self):
        # Top-of-page tabs (unless we're on the Map page, which is full-screen).
        page = st.session_state.page

        if page != self.PAGES[0]:
            cols = st.columns(len(self.PAGES))
            for i, pg in enumerate(self.PAGES):
                with cols[i]:
                    is_active = (pg == page)
                    if st.button(pg, key=f"nav_{i}",
                                 use_container_width=True,
                                 type=("primary" if is_active else "secondary")):
                        st.session_state.page = pg
                        st.rerun()
            st.markdown("---")
        else:
            # Map page gets a slim floating tab bar via columns at top
            cols = st.columns(len(self.PAGES))
            for i, pg in enumerate(self.PAGES):
                with cols[i]:
                    is_active = (pg == page)
                    if st.button(pg, key=f"nav_{i}",
                                 use_container_width=True,
                                 type=("primary" if is_active else "secondary")):
                        st.session_state.page = pg
                        st.rerun()

        if page == self.PAGES[0]:
            self._render_map_page()
        elif page == self.PAGES[1]:
            self._render_forecast_page()
        elif page == self.PAGES[2]:
            self._render_historical_page()
        elif page == self.PAGES[3]:
            self._render_stations_page()
        elif page == self.PAGES[4]:
            self._render_about_page()


# ===========================================================================
#                                MAIN
# ===========================================================================

def main():
    st.set_page_config(
        page_title="Nigeria Flood Forecast",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    app = FloodForecastWebApp()
    app.render()


if __name__ == "__main__":
    main()
