from __future__ import annotations

import json
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import flood_forecast_nigeria as ffn

try:
    import cloud_storage
except Exception:
    cloud_storage = None

# ===========================================================================
#              PROFESSIONAL DATA MANAGER (Downloads from R2)
# ===========================================================================

class ProfessionalDataManager:
    """
    Downloads professional hydrological data from R2 bucket.
    You upload the files to R2 once, this downloads them automatically.
    """

    # Candidate shapefile base names we might see after unzipping.
    RIVER_SHP_CANDIDATES = (
        "ne_10m_rivers_lake_centerlines",
        "ne_10m_rivers_europe",
        "ne_10m_rivers_north_america",
        "ne_10m_rivers",
        "rivers_lake_centerlines",
        "rivers",
        "hotosm_nga_waterways_lines_shp",
        "hotosm_nga_waterways_lines",
    )
    BASIN_SHP_CANDIDATES = (
        "hybas_af_lev06_v1c",
        "hybas_af_lev07_v1c",
        "hybas_af_lev05_v1c",
        "hybas_af_lev08_v1c",
    )

    def __init__(self, r2_storage=None):
        self.r2 = r2_storage
        self.data_dir = Path("./data/hydrological")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.watersheds = None
        self.rivers = None
        self.water_bodies = None
        self.boundary = None

        # Track download status (per-session)
        if 'data_downloaded_from_r2' not in st.session_state:
            st.session_state.data_downloaded_from_r2 = False

    # ------------------------------------------------------------------
    # Shapefile discovery helpers
    # ------------------------------------------------------------------
    def _find_shp(self, candidates) -> Optional[Path]:
        """Recursively search data_dir for any of the candidate .shp names."""
        for name in candidates:
            hits = list(self.data_dir.rglob(f"{name}.shp"))
            if hits:
                return hits[0]
        return None

    def _find_any_shp(self, keyword: str) -> Optional[Path]:
        """Fallback: any .shp whose filename contains the keyword."""
        for p in self.data_dir.rglob("*.shp"):
            if keyword.lower() in p.name.lower():
                return p
        return None

    def _extract_zip(self, zip_path: Path) -> bool:
        """Extract a zip into data_dir. Safe to call multiple times."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(self.data_dir)
            return True
        except Exception as e:
            st.warning(f"Could not extract {zip_path.name}: {e}")
            return False

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    def ensure_data_loaded(self) -> bool:
        """Download (once) and load data from R2 / local cache."""
        if self.watersheds is not None and (self.rivers is not None or self._find_shp(self.RIVER_SHP_CANDIDATES) is None):
            # Already fully loaded in this instance.
            return True

        if not self.r2:
            st.error("R2 storage not configured. Add R2 credentials to secrets.toml")
            return False

        # Download from R2 if not already done this session.
        if not st.session_state.data_downloaded_from_r2:
            with st.spinner("Downloading hydrological data from R2 (first time)..."):
                success = self._download_all_from_r2()
                if success:
                    st.session_state.data_downloaded_from_r2 = True
                else:
                    st.error("Failed to download data from R2")
                    return False

        # Make sure every zip we have on disk is unzipped (cheap, idempotent).
        for z in self.data_dir.rglob("*.zip"):
            # Only extract if we don't yet see a .shp with matching stem.
            if not any(self.data_dir.rglob(f"{z.stem}*.shp")):
                self._extract_zip(z)

        return self._load_from_local_cache()

    # ------------------------------------------------------------------
    # R2 download
    # ------------------------------------------------------------------
    def _download_all_from_r2(self) -> bool:
        try:
            items = [
                ("geojson/hybas_af_lev06_v1c.zip", self.data_dir / "hybas_af_lev06_v1c.zip", True),
                ("geojson/ne_10m_rivers.zip",      self.data_dir / "ne_10m_rivers.zip",      True),
                ("geojson/nigeria_boundary.geojson", self.data_dir / "nigeria_boundary.geojson", False),
                ("geojson/hotosm_nga_waterways_lines_shp.shp", self.data_dir / "hotosm_nga_waterways_lines_shp.shp", False),
                ("geojson/hotosm_nga_waterways_lines_shp.shx", self.data_dir / "hotosm_nga_waterways_lines_shp.shx", False),
                ("geojson/hotosm_nga_waterways_lines_shp.dbf", self.data_dir / "hotosm_nga_waterways_lines_shp.dbf", False),
                ("geojson/hotosm_nga_waterways_lines_shp.prj", self.data_dir / "hotosm_nga_waterways_lines_shp.prj", False),
                ("geojson/hotosm_nga_waterways_lines_shp.cpg", self.data_dir / "hotosm_nga_waterways_lines_shp.cpg", False),
            ]
            for remote, local, is_zip in items:
                if not self.r2.exists(remote):
                    st.warning(f"Not in R2: {remote}")
                    continue
                if local.exists() and local.stat().st_size > 0:
                    st.info(f"Using cached {local.name}")
                else:
                    st.info(f"Downloading {remote} ...")
                    if not self.r2.download_file(remote, local):
                        st.warning(f"Download failed: {remote}")
                        continue
                    st.success(f"Downloaded {local.name}")
                if is_zip:
                    self._extract_zip(local)
            return True
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.code(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Load cached data into memory
    # ------------------------------------------------------------------
    def _load_from_local_cache(self) -> bool:
        try:
            import geopandas as gpd
            from shapely.geometry import box, mapping
    
            # Diagnostic: show what's actually on disk
            all_shps = list(self.data_dir.rglob("*.shp"))
            if all_shps:
                st.caption(f"Shapefiles found: {[str(p.relative_to(self.data_dir)) for p in all_shps]}")
            else:
                st.warning("No .shp files found under data_dir after extraction.")
    
            # ---------- Boundary ----------
            boundary_path = self.data_dir / "nigeria_boundary.geojson"
            if boundary_path.exists():
                try:
                    with open(boundary_path, 'r') as f:
                        self.boundary = json.load(f)
                except Exception as e:
                    st.warning(f"Boundary load failed: {e}")
                    self.boundary = None
    
            nigeria_gdf = self._get_nigeria_boundary_gdf()
    
            # ---------- Watersheds (HydroBASINS) ----------
            shp_path = self._find_shp(self.BASIN_SHP_CANDIDATES) or self._find_any_shp("hybas")
            if shp_path and shp_path.exists():
                try:
                    basins = gpd.read_file(shp_path)
                    if basins.crs and basins.crs.to_epsg() != 4326:
                        basins = basins.to_crs("EPSG:4326")
                    nigeria_basins = gpd.clip(basins, nigeria_gdf)
                    nigeria_basins['geometry'] = nigeria_basins['geometry'].buffer(0)
                    nigeria_basins = nigeria_basins[nigeria_basins.is_valid]
                    self.watersheds = json.loads(nigeria_basins.to_json())
                    self.watersheds['metadata'] = {
                        'source': 'HydroBASINS v1c',
                        'count': len(nigeria_basins),
                    }
                    st.success(f"Loaded {len(nigeria_basins)} watershed basins")
                except Exception as e:
                    st.warning(f"HydroBASINS load failed: {e}")
    
            if self.watersheds is None:
                self._create_fallback_watersheds()
    
            # ---------- RIVERS: Load BOTH datasets (Major + Smaller streams) ----------
            all_river_features = []
            
            # First, look for HOTOSM smaller rivers/streams (PRIORITY - more detailed)
            hotosm_shp = self.data_dir / "hotosm_nga_waterways_lines_shp.shp"
            if not hotosm_shp.exists():
                hotosm_shp = self._find_any_shp("hotosm_nga_waterways")
            
            if hotosm_shp and hotosm_shp.exists():
                st.info("🌊 Loading HOTOSM waterways (smaller rivers & streams)...")
                try:
                    hotosm_gdf = gpd.read_file(hotosm_shp)
                    st.caption(f"HOTOSM source: {hotosm_shp.name} ({len(hotosm_gdf)} raw features)")
                    
                    if hotosm_gdf.crs and hotosm_gdf.crs.to_epsg() != 4326:
                        hotosm_gdf = hotosm_gdf.to_crs("EPSG:4326")
                    
                    # Filter to Nigeria
                    hotosm_clipped = gpd.clip(hotosm_gdf, nigeria_gdf)
                    
                    # Filter by waterway type (river, stream, canal - exclude drains)
                    if 'waterway' in hotosm_clipped.columns:
                        valid_types = ['river', 'stream', 'canal']
                        hotosm_clipped = hotosm_clipped[hotosm_clipped['waterway'].isin(valid_types)]
                        st.caption(f"  After filtering by waterway type: {len(hotosm_clipped)} features")
                    
                    # Limit to 2000 longest features for performance
                    if len(hotosm_clipped) > 2000:
                        hotosm_clipped = hotosm_clipped.copy()
                        hotosm_clipped['length_km'] = hotosm_clipped.geometry.length * 111
                        hotosm_clipped = hotosm_clipped.nlargest(2000, 'length_km')
                        st.caption(f"  Limited to 2000 longest features for performance")
                    
                    # Fix geometries
                    hotosm_clipped = hotosm_clipped[~hotosm_clipped.geometry.is_empty]
                    hotosm_clipped = hotosm_clipped[hotosm_clipped.geometry.is_valid]
                    
                    # Convert to features
                    for idx, row in hotosm_clipped.iterrows():
                        try:
                            geom_json = mapping(row.geometry)
                        except Exception:
                            continue
                        
                        # Get river name (try different field names)
                        name = None
                        for field in ['name', 'NAME', 'name_en', 'name:en']:
                            val = row.get(field) if hasattr(row, 'get') else None
                            if val and str(val).strip():
                                name = str(val)
                                break
                        
                        if not name:
                            # Use waterway type as name if no name exists
                            waterway_type = row.get('waterway', 'stream') if hasattr(row, 'get') else 'stream'
                            name = f"{waterway_type.capitalize()} (HOTOSM)"
                        
                        # Get waterway type for styling
                        waterway = row.get('waterway', 'stream') if hasattr(row, 'get') else 'stream'
                        
                        all_river_features.append({
                            "type": "Feature",
                            "geometry": geom_json,
                            "properties": {
                                "name": name,
                                "waterway": waterway,
                                "length_km": round(float(row.geometry.length) * 111.0, 1),
                                "source": "HOTOSM",
                            },
                        })
                    
                    st.success(f"  ✅ Loaded {len([f for f in all_river_features if f['properties']['source'] == 'HOTOSM'])} HOTOSM river/stream segments")
                    
                except Exception as e:
                    st.error(f"HOTOSM loading failed: {e}")
                    st.code(traceback.format_exc())
            
            # Second, load major rivers from Natural Earth
            rivers_shp = self._find_shp(self.RIVER_SHP_CANDIDATES) or self._find_any_shp("river")
            if rivers_shp and rivers_shp.exists() and "hotosm" not in str(rivers_shp).lower():
                try:
                    rivers_gdf = gpd.read_file(rivers_shp)
                    st.caption(f"Major rivers source: {rivers_shp.name} ({len(rivers_gdf)} raw features)")
                    
                    if rivers_gdf.crs and rivers_gdf.crs.to_epsg() != 4326:
                        rivers_gdf = rivers_gdf.to_crs("EPSG:4326")
                    
                    # Clip to Nigeria
                    try:
                        clipped = gpd.clip(rivers_gdf, nigeria_gdf)
                    except Exception:
                        nigeria_geom = nigeria_gdf.unary_union
                        clipped = rivers_gdf[rivers_gdf.intersects(nigeria_geom)]
                    
                    if len(clipped) == 0:
                        nigeria_geom = nigeria_gdf.unary_union
                        minx, miny, maxx, maxy = nigeria_geom.bounds
                        bbox_poly = box(minx - 1, miny - 1, maxx + 1, maxy + 1)
                        clipped = rivers_gdf[rivers_gdf.intersects(bbox_poly)]
                    
                    # Fix geometries
                    clipped = clipped[~clipped.geometry.is_empty]
                    clipped = clipped[clipped.geometry.is_valid]
                    
                    # Convert to features
                    major_river_count = 0
                    for idx, row in clipped.iterrows():
                        try:
                            geom_json = mapping(row.geometry)
                        except Exception:
                            continue
                        
                        name = None
                        for field in ('name', 'NAME', 'Name', 'name_en', 'river', 'RIVER', 'featurecla'):
                            val = row.get(field) if hasattr(row, 'get') else None
                            if val and str(val).strip():
                                name = str(val)
                                break
                        if not name:
                            name = f"Major River"
                        
                        all_river_features.append({
                            "type": "Feature",
                            "geometry": geom_json,
                            "properties": {
                                "name": name,
                                "waterway": "major_river",
                                "length_km": round(float(row.geometry.length) * 111.0, 1),
                                "source": "NaturalEarth",
                            },
                        })
                        major_river_count += 1
                    
                    st.success(f"  ✅ Loaded {major_river_count} major river segments from Natural Earth")
                    
                except Exception as e:
                    st.error(f"Major rivers loading failed: {e}")
                    st.code(traceback.format_exc())
            
            # Combine all river features
            if all_river_features:
                self.rivers = {
                    "type": "FeatureCollection",
                    "features": all_river_features,
                    "metadata": {
                        "source": "HOTOSM + Natural Earth",
                        "count": len(all_river_features),
                    },
                }
                st.success(f"✅ TOTAL rivers loaded: {len(all_river_features)} features")
            else:
                st.warning("No river features loaded from any source")
                self.rivers = None
    
            return True
    
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.code(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Fallbacks
    # ------------------------------------------------------------------
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
        st.info("HydroBASINS not available, using Nigeria Hydrological Areas")
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
            min_lon, min_lat, max_lon, max_lat = info["bounds"]
            coords = [(min_lon, min_lat), (max_lon, min_lat),
                      (max_lon, max_lat), (min_lon, max_lat),
                      (min_lon, min_lat)]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "id": ha_id, "name": info["name"],
                    "color": info["color"],
                    "source": "NIHSA Hydrological Areas",
                },
            })
        self.watersheds = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"source": "Nigeria Hydrological Areas"},
        }


# ===========================================================================
#                         MAIN WEB APP
# ===========================================================================

class FloodForecastWebApp:
    """Main web application."""

    PAGES = ["🗺️  Map", "📤 Upload Data", "📊 Forecast & Alerts",
             "📈 Raw Data", "📚 Instructions", "ℹ️  About"]

    ALERT_HEX = {"RED": "#C0392B", "AMBER": "#F39C12", "GREEN": "#27AE60"}

    def __init__(self):
        defaults = {
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
            "r2_connected": False,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

        self.r2 = None
        if cloud_storage is not None:
            try:
                self.r2 = cloud_storage.get_r2_from_secrets(st.secrets)
                if self.r2:
                    st.session_state.r2_connected = True
            except Exception:
                pass

        self.data_manager = ProfessionalDataManager(self.r2)

    def _build_map_html(self, system, report, forecast_date) -> Optional[str]:
        try:
            import folium
        except ImportError:
            st.warning("Folium not installed. Run: pip install folium")
            return None

        if not self.data_manager.ensure_data_loaded():
            st.error("Could not load hydrological data")
            return None

        fmap = folium.Map(location=[9.0, 8.0], zoom_start=6,
                          tiles="CartoDB positron")

        # Boundary
        if self.data_manager.boundary:
            try:
                folium.GeoJson(
                    self.data_manager.boundary,
                    name="Nigeria Boundary",
                    style_function=lambda x: {
                        "color": "#1B2631", "weight": 2, "fillOpacity": 0.05
                    },
                ).add_to(fmap)
            except Exception as e:
                st.warning(f"Could not add boundary: {e}")

        # Watersheds
        if self.data_manager.watersheds:
            try:
                sample = self.data_manager.watersheds.get('features', [{}])[0]
                props = sample.get('properties', {})
                if 'HYBAS_ID' in props:
                    folium.GeoJson(
                        self.data_manager.watersheds,
                        name="Watersheds",
                        style_function=lambda x: {
                            'fillColor': '#2C3E50', 'color': '#2C3E50',
                            'weight': 1.0, 'fillOpacity': 0.12,
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['HYBAS_ID', 'SUB_AREA', 'UP_AREA'],
                            aliases=['ID:', 'Area (km²):', 'Upstream Area (km²):'],
                            localize=True,
                        ),
                    ).add_to(fmap)
                else:
                    folium.GeoJson(
                        self.data_manager.watersheds,
                        name="Watersheds",
                        style_function=lambda x: {
                            'fillColor': x['properties'].get('color', '#2C3E50'),
                            'color': '#2C3E50',
                            'weight': 1.0, 'fillOpacity': 0.18,
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['id', 'name'],
                            aliases=['ID:', 'Name:'],
                            localize=True,
                        ),
                    ).add_to(fmap)
            except Exception as e:
                st.warning(f"Could not add watersheds: {e}")

        # Rivers - rendered LAST so they sit on top
        if self.data_manager.rivers and self.data_manager.rivers.get('features'):
            try:
                def style_river_by_type(feature):
                    props = feature.get('properties', {})
                    waterway = props.get('waterway', 'stream')
                    source = props.get('source', '')
                    
                    # Major rivers (Natural Earth) - thicker, darker blue
                    if waterway == 'major_river' or source == 'NaturalEarth':
                        return {
                            "color": "#003366",
                            "weight": 3.0,
                            "opacity": 0.95,
                            "fillOpacity": 0,
                        }
                    # HOTOSM rivers - medium thickness, medium blue
                    elif waterway == 'river':
                        return {
                            "color": "#1a73e8",
                            "weight": 2.0,
                            "opacity": 0.85,
                            "fillOpacity": 0,
                        }
                    # HOTOSM streams - thinner, lighter blue
                    elif waterway == 'stream':
                        return {
                            "color": "#4dabf7",
                            "weight": 1.2,
                            "opacity": 0.7,
                            "fillOpacity": 0,
                        }
                    # Canals - dashed style
                    elif waterway == 'canal':
                        return {
                            "color": "#74c0fc",
                            "weight": 1.5,
                            "opacity": 0.8,
                            "fillOpacity": 0,
                            "dashArray": "5,5",
                        }
                    # Default
                    else:
                        return {
                            "color": "#3399ff",
                            "weight": 1.5,
                            "opacity": 0.8,
                            "fillOpacity": 0,
                        }
                
                folium.GeoJson(
                    self.data_manager.rivers,
                    name="Rivers & Streams",
                    style_function=style_river_by_type,
                    highlight_function=lambda x: {
                        "weight": 4, "color": "#FF3333",
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'waterway', 'length_km'],
                        aliases=['Name:', 'Type:', 'Length (km):'],
                        localize=True, sticky=True,
                    ),
                ).add_to(fmap)
                
                total_rivers = len(self.data_manager.rivers.get('features', []))
                st.caption(f"📊 Showing {total_rivers} river features on map")
                
            except Exception as e:
                st.warning(f"Could not add rivers: {e}")
        else:
            st.info("Rivers layer is empty — nothing to draw. Check the "
                    "shapefile upload in R2 under geojson/ne_10m_rivers.zip.")

        # Legend
        legend_html = """
        {% macro html(this, kwargs) %}
        <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                    background: white; padding: 10px 12px; border-radius: 6px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.25);
                    font: 12px/1.4 Arial,sans-serif;">
            <b>Legend</b><br>
            <span style="color:#1B2631">■</span> Nigeria Boundary<br>
            <span style="color:#2C3E50">▯</span> Watersheds<br>
            <span style="color:#0066CC">━</span> Rivers<br>
            <span style="color:#C0392B">●</span> RED Alert<br>
            <span style="color:#F39C12">●</span> AMBER Alert<br>
            <span style="color:#27AE60">●</span> GREEN Alert
        </div>
        {% endmacro %}
        """
        from branca.element import MacroElement, Template
        legend = MacroElement()
        legend._template = Template(legend_html)
        fmap.get_root().add_child(legend)

        folium.LayerControl(collapsed=False).add_to(fmap)
        return fmap.get_root().render()

    def render(self):
        if st.session_state.page == self.PAGES[0] and st.session_state.r2_connected:
            st.success("Connected to R2")

        with st.sidebar:
            st.session_state.page = st.radio(
                "Navigation", self.PAGES, key="nav_page_radio"
            )

        if st.session_state.page == self.PAGES[0]:
            st.title("Nigeria Flood Forecast Map")

            if st.session_state.system is None:
                with st.spinner("Initializing forecast system..."):
                    from flood_forecast_nigeria import Config, FloodForecastSystem
                    cfg = Config()
                    system = FloodForecastSystem(cfg)
                    system._write_synthetic_gauge_csv()
                    st.session_state.system = system

            html = self._build_map_html(
                st.session_state.system,
                st.session_state.forecast_report,
                st.session_state.forecast_date,
            )

            if html:
                components.html(html, height=700, scrolling=False)
            else:
                st.error("Could not generate map. Check R2 contents.")

        elif st.session_state.page == self.PAGES[1]:
            st.title("Upload Data")
            st.info(
                "### Required files in R2 bucket\n"
                "- `geojson/hybas_af_lev06_v1c.zip` — HydroBASINS watershed data\n"
                "- `geojson/nigeria_boundary.geojson` — Nigeria boundary\n"
                "- `geojson/ne_10m_rivers.zip` — Natural Earth rivers "
                "(must contain `.shp/.shx/.dbf/.prj` inside)\n"
            )
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Check R2 Connection"):
                    if self.r2:
                        files = self.r2.list_files("geojson/")
                        st.write(f"Files in R2 bucket: {files}")
                    else:
                        st.error("R2 not configured")
            with col2:
                if st.button("Force re-download from R2"):
                    st.session_state.data_downloaded_from_r2 = False
                    # Nuke local cache so extraction re-runs too.
                    try:
                        shutil.rmtree(self.data_manager.data_dir,
                                      ignore_errors=True)
                        self.data_manager.data_dir.mkdir(
                            parents=True, exist_ok=True
                        )
                        self.data_manager.watersheds = None
                        self.data_manager.rivers = None
                        self.data_manager.boundary = None
                        st.success("Cache cleared — reload the Map page.")
                    except Exception as e:
                        st.error(f"Cache clear failed: {e}")


# ===========================================================================
#                               MAIN
# ===========================================================================

def main():
    app = FloodForecastWebApp()
    app.render()


if __name__ == "__main__":
    main()
