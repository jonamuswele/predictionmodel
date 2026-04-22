# flood_forecast_nigeria.py - Complete rewrite with FANFAR integration and Virtual Gauges

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
# Optional third-party imports
# --------------------------------------------------------------------------
try:
    import rasterio
    from rasterio import features as rio_features
    from rasterio import mask as rio_mask
    from rasterio.transform import from_bounds as _rio_from_bounds
    _HAS_RASTERIO = True
except Exception:
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
except Exception:
    gpd = None
    LineString = MultiPolygon = Point = Polygon = shape = None
    mapping = unary_union = None
    _HAS_GEO = False

try:
    from pysheds.grid import Grid as _PyShedsGrid
    _HAS_PYSHEDS = True
except Exception:
    _PyShedsGrid = None
    _HAS_PYSHEDS = False

try:
    import hydroeval as he
    _HAS_HYDROEVAL = True
except Exception:
    he = None
    _HAS_HYDROEVAL = False

try:
    import pastas as ps
    _HAS_PASTAS = True
except Exception:
    ps = None
    _HAS_PASTAS = False

try:
    import requests
    _HAS_REQUESTS = True
except Exception:
    requests = None
    _HAS_REQUESTS = False

try:
    import matplotlib as _mpl
    from matplotlib.figure import Figure
    _HAS_MPL = True
    plt = None
except Exception:
    _mpl = None
    plt = None
    Figure = Any
    _HAS_MPL = False

try:
    import folium
    from folium.features import GeoJson, GeoJsonTooltip
    from branca.element import MacroElement, Template
    _HAS_FOLIUM = True
except Exception:
    folium = None
    GeoJson = GeoJsonTooltip = None
    MacroElement = Template = None
    _HAS_FOLIUM = False


def _ensure_mpl():
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


# ===========================================================================
#                          1. Config
# ===========================================================================
@dataclass
class Config:
    data_dir: Path = Path("./data")
    dem_path: Path = Path("./data/dem/nigeria_srtm30.tif")
    gauge_csv: Path = Path("./data/gauges/gauge_stations.csv")
    output_dir: Path = Path("./outputs")
    
    # Nigeria bounding box (from Nigeria polygon, expanded slightly)
    bbox: Tuple[float, float, float, float] = (2.5, 4.0, 14.7, 13.9)
    
    start_date: str = "2015-01-01"
    end_date: str = "2026-04-22"  # Updated to current date
    forecast_horizon_days: int = 7
    
    # FANFAR API configuration
    fanfar_api_url: str = "https://api.fanfar.eu/v1"
    fanfar_api_key: str = ""
    
    # GR4J defaults (fallback)
    gr4j_default_params: Tuple[float, float, float, float] = (350.0, 0.0, 90.0, 1.7)
    
    muskingum_k: float = 1.0
    muskingum_x: float = 0.2
    
    alert_green: float = 1.0
    alert_amber: float = 1.5
    alert_red: float = 2.5
    
    min_years_required: int = 3
    log_level: str = "INFO"
    
    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.dem_path = Path(self.dem_path)
        self.gauge_csv = Path(self.gauge_csv)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "gauges").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "dem").mkdir(parents=True, exist_ok=True)


# ===========================================================================
#                      2. FANFAR API Client
# ===========================================================================
class FANFARClient:
    """Client for FANFAR API - uses station IDs (not coordinates)."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.fanfar.eu/v1"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or os.environ.get("FANFAR_API_KEY", "")
        self.session = requests.Session() if _HAS_REQUESTS else None
        if self.session and self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def get_station_forecast(self, station_id: str, days: int = 10) -> Optional[Dict]:
        """Get daily discharge forecast for a station using its ID."""
        if not self.session:
            return None
        try:
            endpoint = f"{self.base_url}/forecasts/daily"
            params = {"station": station_id, "days": days, "format": "json"}
            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return {
                "station_id": station_id,
                "forecast_date": data.get("forecast_date"),
                "discharge": data.get("discharge", []),
                "water_level": data.get("water_level", []),
                "units": "m³/s",
                "source": "FANFAR"
            }
        except Exception as e:
            self._logger.warning(f"FANFAR forecast failed for {station_id}: {e}")
            return None
    
    def get_historical_reforecast(self, station_id: str, start_date: str, end_date: str) -> Optional[Dict]:
        """Get historical reforecast for bias correction."""
        if not self.session:
            return None
        try:
            endpoint = f"{self.base_url}/reforecast"
            params = {"station": station_id, "start_date": start_date, "end_date": end_date, "format": "json"}
            response = self.session.get(endpoint, params=params, timeout=60)
            response.raise_for_status()
            return {"station_id": station_id, "data": response.json(), "source": "FANFAR"}
        except Exception as e:
            self._logger.warning(f"FANFAR reforecast failed for {station_id}: {e}")
            return None
    
    def get_available_stations(self) -> List[Dict]:
        """Get list of stations available in FANFAR system."""
        if not self.session:
            return []
        try:
            endpoint = f"{self.base_url}/stations"
            response = self.session.get(endpoint, timeout=30)
            response.raise_for_status()
            return response.json().get("stations", [])
        except Exception as e:
            self._logger.warning(f"Failed to get FANFAR stations: {e}")
            return []


# ===========================================================================
#                      3. Correction Engine
# ===========================================================================
class FloodCorrectionEngine:
    """Apply bias corrections to FANFAR forecasts using historical gauge data."""
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        self.historical_data = historical_data
        self.correction_factors: Dict[str, Dict[str, float]] = {}
        if historical_data is not None:
            self._calculate_correction_factors()
    
    def _calculate_correction_factors(self):
        for station_id, group in self.historical_data.groupby('station_id'):
            valid = group.dropna(subset=['discharge_observed', 'discharge_fanfar'])
            if len(valid) < 10:
                self.correction_factors[station_id] = {'factor': 1.0, 'bias': 0.0}
                continue
            ratios = valid['discharge_observed'] / valid['discharge_fanfar']
            factor = ratios.median()
            bias = (valid['discharge_observed'] - valid['discharge_fanfar']).median()
            self.correction_factors[station_id] = {
                'factor': float(factor), 'bias': float(bias),
                'samples': len(valid),
                'correlation': valid['discharge_observed'].corr(valid['discharge_fanfar'])
            }
    
    def apply_correction(self, station_id: str, fanfar_discharge: float) -> float:
        correction = self.correction_factors.get(station_id, {'factor': 1.0, 'bias': 0.0})
        corrected = (fanfar_discharge * correction['factor']) + correction['bias']
        return max(0.0, corrected)
    
    def get_correction_summary(self) -> pd.DataFrame:
        if not self.correction_factors:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self.correction_factors, orient='index')


# ===========================================================================
#                      4. Virtual Gauge Generator
# ===========================================================================
class VirtualGaugeGenerator:
    """Intelligently places virtual gauge stations at strategic locations."""
    
    def __init__(self, config: Config, watershed_data: Dict[str, Any]):
        self.config = config
        self.watershed_data = watershed_data
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def generate_virtual_gauges(self) -> pd.DataFrame:
        """Generate virtual gauges at watershed centroids, confluences, and cities."""
        basins = self.watershed_data.get("basins")
        gauges = []
        
        # Strategy 1: One gauge per watershed (at centroid)
        if basins is not None and len(basins) > 0:
            for idx, basin in basins.iterrows():
                if basin.geometry is None:
                    continue
                centroid = basin.geometry.centroid
                area_km2 = basin.get('area_km2', 10000)
                river_name = basin.get('river', f"Basin_{idx}")
                gauges.append({
                    'station_id': f"VIRTUAL_{idx:04d}",
                    'station_name': f"Virtual - {river_name}",
                    'latitude': centroid.y,
                    'longitude': centroid.x,
                    'river': river_name,
                    'area_km2': area_km2,
                    'type': 'virtual',
                    'priority': 1
                })
        
        # Strategy 2: Major river confluences
        confluences = [
            {'name': 'Niger-Benue', 'lat': 7.80, 'lon': 6.74, 'river': 'Niger', 'area_km2': 230000},
            {'name': 'Kaduna-Niger', 'lat': 8.85, 'lon': 5.85, 'river': 'Kaduna', 'area_km2': 65000},
            {'name': 'Sokoto-Niger', 'lat': 10.32, 'lon': 4.78, 'river': 'Sokoto', 'area_km2': 45000},
            {'name': 'Gongola-Benue', 'lat': 9.30, 'lon': 12.00, 'river': 'Gongola', 'area_km2': 56000},
        ]
        for conf in confluences:
            gauges.append({
                'station_id': f"VIRTUAL_CONF_{conf['name'].replace(' ', '_')}",
                'station_name': f"Virtual - {conf['name']} Confluence",
                'latitude': conf['lat'],
                'longitude': conf['lon'],
                'river': conf['river'],
                'area_km2': conf['area_km2'],
                'type': 'virtual',
                'priority': 2
            })
        
        # Strategy 3: Major cities (flood risk areas)
        cities = [
            {'name': 'Lagos', 'lat': 6.45, 'lon': 3.40, 'river': 'Lagos Lagoon', 'area_km2': 500},
            {'name': 'Kano', 'lat': 12.00, 'lon': 8.52, 'river': 'Jakara', 'area_km2': 300},
            {'name': 'Ibadan', 'lat': 7.38, 'lon': 3.90, 'river': 'Ogunpa', 'area_km2': 200},
            {'name': 'Kaduna', 'lat': 10.53, 'lon': 7.44, 'river': 'Kaduna', 'area_km2': 400},
            {'name': 'Port Harcourt', 'lat': 4.82, 'lon': 7.00, 'river': 'Bonny', 'area_km2': 350},
            {'name': 'Maiduguri', 'lat': 11.83, 'lon': 13.15, 'river': 'Ngadda', 'area_km2': 200},
            {'name': 'Onitsha', 'lat': 6.15, 'lon': 6.78, 'river': 'Niger', 'area_km2': 300},
        ]
        for city in cities:
            gauges.append({
                'station_id': f"VIRTUAL_CITY_{city['name'].replace(' ', '_')}",
                'station_name': f"Virtual - {city['name']}",
                'latitude': city['lat'],
                'longitude': city['lon'],
                'river': city['river'],
                'area_km2': city['area_km2'],
                'type': 'virtual',
                'priority': 3
            })
        
        gauges_df = pd.DataFrame(gauges)
        self._save_virtual_gauges(gauges_df)
        self._logger.info(f"Generated {len(gauges_df)} virtual gauge stations")
        return gauges_df
    
    def _save_virtual_gauges(self, gauges_df: pd.DataFrame):
        """Save virtual gauges with synthetic historical data."""
        path = Path(self.config.gauge_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        rows = []
        idx = pd.date_range(self.config.start_date, self.config.end_date, freq="D")
        
        for _, gauge in gauges_df.iterrows():
            base_q = gauge['area_km2'] * 0.05
            rng = np.random.default_rng(seed=hash(gauge['station_id']) % 2**32)
            doy = np.array([d.timetuple().tm_yday for d in idx])
            seasonal = 1.0 + 0.8 * np.sin(2 * np.pi * (doy - 150) / 365.25)
            noise = rng.normal(1.0, 0.3, len(idx))
            q = np.clip(base_q * seasonal * noise, base_q * 0.3, base_q * 2.5)
            stage = 1.0 + np.log1p(q / max(base_q, 1.0)) * 1.5
            
            for d, Q, H in zip(idx, q, stage):
                rows.append({
                    "station_id": gauge['station_id'],
                    "station_name": gauge['station_name'],
                    "latitude": gauge['latitude'],
                    "longitude": gauge['longitude'],
                    "river": gauge['river'],
                    "date": d.strftime("%Y-%m-%d"),
                    "discharge_m3s": round(float(Q), 2),
                    "stage_m": round(float(H), 3),
                    "type": "virtual"
                })
        
        pd.DataFrame(rows).to_csv(path, index=False)
        self._logger.info(f"Saved {len(rows)} rows for {len(gauges_df)} virtual gauges")


# ===========================================================================
#                      5. FANFAR Hydrological Model
# ===========================================================================
class FANFARHydrologicalModel:
    """Wrapper that uses FANFAR API instead of running GR4J locally."""
    
    def __init__(self, config: Config, fanfar_client: FANFARClient,
                 correction_engine: Optional[FloodCorrectionEngine] = None):
        self.config = config
        self.fanfar_client = fanfar_client
        self.correction_engine = correction_engine
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def run_forecast(self, station_id: str, fanfar_station_id: Optional[str] = None) -> pd.DataFrame:
        """Get forecast from FANFAR API and apply corrections."""
        target_id = fanfar_station_id or station_id
        forecast_data = self.fanfar_client.get_station_forecast(target_id, self.config.forecast_horizon_days)
        
        if not forecast_data or not forecast_data.get('discharge'):
            return pd.DataFrame()
        
        fanfar_discharge = forecast_data['discharge']
        
        if self.correction_engine:
            corrected_discharge = [self.correction_engine.apply_correction(station_id, q) for q in fanfar_discharge]
        else:
            corrected_discharge = fanfar_discharge
        
        start_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        forecast_index = pd.date_range(start_date, periods=len(corrected_discharge), freq='D')
        
        return pd.DataFrame({
            'Q_forecast_m3s': corrected_discharge,
            'Q_raw_m3s': fanfar_discharge,
            'source': 'FANFAR'
        }, index=forecast_index)


# ===========================================================================
#                      6. DataManager (Simplified)
# ===========================================================================
class DataManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self.gauge_data: Dict[str, pd.DataFrame] = {}
        self.sparse_stations: List[str] = []
        self._station_meta: Optional[pd.DataFrame] = None
    
    def load_gauge_data(self) -> Dict[str, pd.DataFrame]:
        path = Path(self.config.gauge_csv)
        if not path.exists():
            raise FileNotFoundError(f"Gauge CSV not found at {path}")
        
        raw = pd.read_csv(path)
        required = {"station_id", "station_name", "latitude", "longitude", "river", "date", "discharge_m3s", "stage_m"}
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
            valid = df["discharge_m3s"].dropna()
            if len(valid) > 0:
                years = (valid.index.max() - valid.index.min()).days / 365.25
            else:
                years = 0.0
            if years < self.config.min_years_required:
                self.sparse_stations.append(str(sid))
            self.gauge_data[str(sid)] = df
        
        self._logger.info(f"Loaded {len(self.gauge_data)} gauge stations ({len(self.sparse_stations)} sparse)")
        return self.gauge_data
    
    def get_station_geodataframe(self) -> "gpd.GeoDataFrame":
        if self._station_meta is None:
            self.load_gauge_data()
        meta = self._station_meta.copy() if self._station_meta is not None else pd.DataFrame()
        if _HAS_GEO:
            geom = gpd.points_from_xy(meta["longitude"], meta["latitude"])
            return gpd.GeoDataFrame(meta, geometry=geom, crs="EPSG:4326")
        return meta


# ===========================================================================
#                      7. WatershedDelineator (Simplified)
# ===========================================================================
class WatershedDelineator:
    def __init__(self, config: Config) -> None:
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)
        self.dem: Optional[np.ndarray] = None
        self.profile: Optional[Dict[str, Any]] = None
        self.flow_dir: Optional[np.ndarray] = None
        self.flow_acc: Optional[np.ndarray] = None
    
    def load_dem(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        path = Path(self.config.dem_path)
        if not path.exists():
            self.generate_synthetic_dem()
        if _HAS_RASTERIO and path.exists():
            with rasterio.open(path) as src:
                self.dem = src.read(1).astype(float)
                self.profile = dict(src.profile)
        else:
            self.dem, self.profile = self._build_synthetic_dem_array()
        return self.dem, self.profile
    
    def _build_synthetic_dem_array(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        lon_min, lat_min, lon_max, lat_max = self.config.bbox
        res = 0.05
        ncols = max(32, int(round((lon_max - lon_min) / res)))
        nrows = max(32, int(round((lat_max - lat_min) / res)))
        rng = np.random.default_rng(seed=20260421)
        dem = 50.0 + 1200.0 * np.linspace(1, 0, nrows)[:, None] + 400.0 * rng.random((nrows, ncols))
        transform = None
        if _HAS_RASTERIO:
            transform = _rio_from_bounds(lon_min, lat_min, lon_max, lat_max, ncols, nrows)
        profile = {"driver": "GTiff", "dtype": "float32", "width": ncols, "height": nrows, "count": 1,
                   "crs": "EPSG:4326", "transform": transform, "nodata": -9999.0}
        return dem.astype(np.float32), profile
    
    def generate_synthetic_dem(self) -> Path:
        dem, profile = self._build_synthetic_dem_array()
        path = Path(self.config.dem_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_RASTERIO:
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(dem, 1)
        self.dem = dem
        self.profile = profile
        return path
    
    def delineate(self) -> Dict[str, Any]:
        if self.dem is None:
            self.load_dem()
        
        # Create synthetic basins (one per virtual station)
        from shapely.geometry import box, Polygon
        basins = []
        meta = pd.read_csv(self.config.gauge_csv).drop_duplicates("station_id") if Path(self.config.gauge_csv).exists() else pd.DataFrame()
        
        for idx, row in meta.iterrows():
            # Create a simple bounding box around each station
            lon, lat = row['longitude'], row['latitude']
            buffer = 0.5
            poly = box(lon - buffer, lat - buffer, lon + buffer, lat + buffer)
            basins.append({
                "station_id": row['station_id'],
                "river": row.get('river', ''),
                "area_km2": 10000,
                "geometry": poly
            })
        
        if _HAS_GEO:
            basins_gdf = gpd.GeoDataFrame(basins, geometry="geometry", crs="EPSG:4326")
        else:
            basins_gdf = pd.DataFrame(basins)
        
        return {"basins": basins_gdf, "flow_acc": None, "catchment_masks": {}, "river_network": None, "snapped_stations": None}


# ===========================================================================
#                      8. FloodRouter (Simplified)
# ===========================================================================
class FloodRouter:
    def __init__(self, config: Config, watershed_data: Dict[str, Any], Q_per_basin: Dict[str, pd.DataFrame]) -> None:
        self.config = config
        self.Q_per_basin = Q_per_basin
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def route_network(self) -> Dict[str, pd.DataFrame]:
        # Simple routing - just return the input as routed
        routed = {}
        for sid, df in self.Q_per_basin.items():
            routed[sid] = pd.DataFrame({"Q_routed_m3s": df.iloc[:, 0].values}, index=df.index)
        return routed
    
    def get_peak_flood_map(self, forecast_date: str) -> Any:
        if _HAS_GEO:
            return gpd.GeoDataFrame(columns=["geometry", "depth_m", "area_km2", "station_id"], geometry="geometry", crs="EPSG:4326")
        return pd.DataFrame()


# ===========================================================================
#                      9. AlertDashboard (Simplified for Web)
# ===========================================================================
class AlertDashboard:
    def __init__(self, config: Config, routed_Q: Dict[str, pd.DataFrame],
                 return_periods: Dict[str, pd.DataFrame], flood_polygons: Any,
                 stations_gdf: Any, hindcasts: Optional[Dict[str, pd.DataFrame]] = None,
                 forecasts: Optional[Dict[str, pd.DataFrame]] = None,
                 river_network: Any = None) -> None:
        self.config = config
        self.routed_Q = routed_Q
        self.return_periods = return_periods
        self.stations_gdf = stations_gdf
        self.forecasts = forecasts or {}
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def _classify_alert(self, Q: float, Q_maf: float) -> str:
        if Q_maf <= 0:
            return "GREEN"
        ratio = Q / Q_maf
        if ratio >= self.config.alert_red:
            return "RED"
        if ratio >= self.config.alert_amber:
            return "AMBER"
        return "GREEN"
    
    def _q_maf(self, station_id: str) -> float:
        df = self.routed_Q.get(station_id)
        if df is None or df.empty:
            return 0.0
        try:
            ann_max = df["Q_routed_m3s"].resample("Y").max().dropna()
            return float(ann_max.mean()) if not ann_max.empty else 0.0
        except Exception:
            return 0.0
    
    def generate_alert_report(self, forecast_date: str) -> Dict[str, Dict[str, Any]]:
        report = {}
        start = pd.to_datetime(forecast_date)
        for sid, df in self.routed_Q.items():
            series = df["Q_routed_m3s"]
            if series.empty:
                continue
            peak_Q = float(series.max())
            peak_date = str(series.idxmax())
            q_maf = self._q_maf(sid)
            alert = self._classify_alert(peak_Q, q_maf if q_maf > 0 else peak_Q)
            report[sid] = {"alert_level": alert, "peak_Q_m3s": peak_Q, "peak_date": peak_date, "river": ""}
        return report


# ===========================================================================
#                      10. FloodForecastSystem (Orchestrator)
# ===========================================================================
class FloodForecastSystem:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        logging.basicConfig(level=getattr(logging, self.config.log_level.upper(), logging.INFO))
        self._logger = logging.getLogger(self.__class__.__name__)
        
        self.data_manager: Optional[DataManager] = None
        self.delineator: Optional[WatershedDelineator] = None
        self.flood_router: Optional[FloodRouter] = None
        self.alert_dashboard: Optional[AlertDashboard] = None
        self.fanfar_client: Optional[FANFARClient] = None
        self.correction_engine: Optional[FloodCorrectionEngine] = None
        self.fanfar_model: Optional[FANFARHydrologicalModel] = None
        
        self.return_periods: Dict[str, pd.DataFrame] = {}
        self.routed_Q: Dict[str, pd.DataFrame] = {}
        self.watershed_data: Dict[str, Any] = {}
        self.fanfar_station_mapping: Dict[str, str] = {}
    
    def initialize_with_virtual_gauges(self) -> None:
        """Initialize system with virtual gauges (no real data needed)."""
        self._logger.info("Initializing with virtual gauges...")
        
        # Delineate watersheds
        wd = WatershedDelineator(self.config)
        wd.load_dem()
        self.watershed_data = wd.delineate()
        self.delineator = wd
        
        # Generate virtual gauges
        vg = VirtualGaugeGenerator(self.config, self.watershed_data)
        vg.generate_virtual_gauges()
        
        # Load data manager
        self.data_manager = DataManager(self.config)
        self.data_manager.load_gauge_data()
        
        # Create synthetic return periods
        for sid in self.data_manager.gauge_data.keys():
            self.return_periods[sid] = pd.DataFrame({
                "return_period_yr": [2, 5, 10, 25, 50, 100],
                "Q_m3s": [1000, 2000, 3000, 5000, 7000, 10000]
            })
        
        self._logger.info(f"Initialized with {len(self.data_manager.gauge_data)} virtual stations")
    
    def initialize_fanfar(self, api_key: Optional[str] = None) -> None:
        """Initialize FANFAR client and try to map stations."""
        self.fanfar_client = FANFARClient(api_key=api_key or self.config.fanfar_api_key)
        self.fanfar_model = FANFARHydrologicalModel(self.config, self.fanfar_client, self.correction_engine)
        
        # Try to get available stations and create mapping
        fanfar_stations = self.fanfar_client.get_available_stations()
        if fanfar_stations and self.data_manager:
            for fs in fanfar_stations:
                fs_id = fs.get('id')
                fs_lat, fs_lon = fs.get('latitude'), fs.get('longitude')
                if fs_id and fs_lat and fs_lon:
                    # Find closest virtual station
                    for sid in self.data_manager.gauge_data.keys():
                        meta = self.data_manager._station_meta
                        if meta is not None:
                            row = meta[meta['station_id'] == sid]
                            if not row.empty:
                                dist = ((row['latitude'].iloc[0] - fs_lat) ** 2 + (row['longitude'].iloc[0] - fs_lon) ** 2) ** 0.5
                                if dist < 0.5:
                                    self.fanfar_station_mapping[sid] = fs_id
                                    self._logger.info(f"Mapped {sid} → FANFAR {fs_id}")
                                    break
    
    def run_forecast_with_fanfar(self) -> Dict[str, Dict[str, Any]]:
        """Run forecast using FANFAR for mapped stations, synthetic for others."""
        if self.fanfar_model is None:
            self._logger.warning("FANFAR not initialized")
            return {}
        
        forecasts = {}
        for station_id in self.data_manager.gauge_data.keys():
            fanfar_id = self.fanfar_station_mapping.get(station_id)
            if fanfar_id:
                self._logger.info(f"Using FANFAR for {station_id} → {fanfar_id}")
                fc_df = self.fanfar_model.run_forecast(station_id, fanfar_id)
            else:
                # Generate synthetic forecast
                area = 10000
                base_q = area * 0.05
                rng = np.random.default_rng(seed=hash(station_id) % 2**32)
                q_values = base_q * (1.0 + 0.3 * np.sin(np.linspace(0, np.pi, self.config.forecast_horizon_days)))
                q_values = q_values * rng.normal(1.0, 0.1, self.config.forecast_horizon_days)
                if len(q_values) > 3:
                    q_values[2] *= 1.8
                start_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
                fc_df = pd.DataFrame({'Q_forecast_m3s': q_values, 'Q_raw_m3s': q_values, 'source': 'SYNTHETIC'},
                                      index=pd.date_range(start_date, periods=len(q_values), freq='D'))
            
            if not fc_df.empty:
                forecasts[station_id] = fc_df
        
        # Route forecasts
        Q_per_basin = {sid: fc[['Q_forecast_m3s']].rename(columns={'Q_forecast_m3s': 'Q_sim_m3s'})
                       for sid, fc in forecasts.items()}
        
        if Q_per_basin:
            fr = FloodRouter(self.config, self.watershed_data, Q_per_basin)
            self.routed_Q = fr.route_network()
            self.flood_router = fr
        
        # Generate alerts
        stations_gdf = self.data_manager.get_station_geodataframe()
        ad = AlertDashboard(self.config, self.routed_Q, self.return_periods, pd.DataFrame(),
                            stations_gdf, forecasts=forecasts)
        
        report = ad.generate_alert_report(pd.Timestamp.now().strftime("%Y-%m-%d"))
        self.alert_dashboard = ad
        return report
    
    def upgrade_virtual_to_real(self, station_id: str, real_data: pd.DataFrame) -> None:
        """Upgrade virtual station to real with historical data."""
        self._logger.info(f"Upgrading station {station_id} from virtual to real")
        real_data['date'] = pd.to_datetime(real_data['date'])
        real_data = real_data.set_index('date').sort_index()
        
        if station_id in self.data_manager.gauge_data:
            self.data_manager.gauge_data[station_id] = real_data
        
        # Update CSV
        csv_path = Path(self.config.gauge_csv)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.loc[df['station_id'] == station_id, 'type'] = 'real'
            df.to_csv(csv_path, index=False)
        
        # Recalculate correction if FANFAR available
        if self.fanfar_client and station_id in self.fanfar_station_mapping:
            fanfar_id = self.fanfar_station_mapping[station_id]
            start = real_data.index.min().strftime('%Y-%m-%d')
            end = real_data.index.max().strftime('%Y-%m-%d')
            reforecast = self.fanfar_client.get_historical_reforecast(fanfar_id, start, end)
            if reforecast and reforecast.get('data'):
                fanfar_vals = pd.Series(reforecast['data'])
                fanfar_vals.index = pd.date_range(start, end, periods=len(fanfar_vals))
                aligned = pd.DataFrame({'observed': real_data['discharge_m3s'], 'fanfar': fanfar_vals}).dropna()
                if len(aligned) > 10:
                    if self.correction_engine is None:
                        self.correction_engine = FloodCorrectionEngine()
                    factor = (aligned['observed'] / aligned['fanfar']).median()
                    bias = (aligned['observed'] - aligned['fanfar']).median()
                    self.correction_engine.correction_factors[station_id] = {'factor': float(factor), 'bias': float(bias), 'samples': len(aligned)}
                    self.fanfar_model = FANFARHydrologicalModel(self.config, self.fanfar_client, self.correction_engine)
                    self._logger.info(f"Calculated correction for {station_id}: factor={factor:.3f}, bias={bias:.1f}")
        
        self._logger.info(f"Station {station_id} upgraded to real")
    
    def run_demo(self) -> None:
        """Demo mode using virtual gauges and synthetic forecasts."""
        self._logger.info("=== FloodForecastSystem demo starting ===")
        self.initialize_with_virtual_gauges()
        print("\n" + "=" * 72)
        print("  Nigeria Flood Forecast System - Virtual Mode")
        print("=" * 72)
        print(f"  Generated {len(self.data_manager.gauge_data)} virtual gauge stations")
        print("  Use FANFAR API key to get real forecasts")
        print("=" * 72)


# ===========================================================================
#                              Entry point
# ===========================================================================
if __name__ == "__main__":
    system = FloodForecastSystem()
    system.run_demo()
