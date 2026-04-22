# flood_forecast_web.py - Top navigation using Streamlit tabs

from __future__ import annotations

import json
import re
import shutil
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
#                          CONSTANTS / CONFIG
# ===========================================================================

# Nigeria bounding box for fit_bounds (lat_min, lon_min, lat_max, lon_max)
NIGERIA_BOUNDS = [[4.0, 2.5], [14.0, 14.7]]
NIGERIA_CENTER = [9.5, 8.0]

# Named rivers we want to promote to "major" styling
MAJOR_RIVER_NAMES = {
    "niger", "benue", "kaduna", "sokoto", "gongola", "komadugu", "komadugu yobe",
    "yobe", "cross", "ogun", "osun", "anambra", "hadejia", "katsina-ala",
    "katsina ala", "donga", "taraba", "shari", "chari", "imo", "kwa ibo",
    "kwa iboe", "qua iboe", "forcados", "escravos", "nun", "orashi",
    "ogun river", "kaduna river", "benue river", "niger river", "sokoto river",
}


# ===========================================================================
#                     PROFESSIONAL DATA MANAGER (R2)
# ===========================================================================

class ProfessionalDataManager:
    """Downloads professional hydrological data from R2 bucket."""

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
            name = p.name.lower()
            if any(k in name for k in keywords):
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

            # Watersheds
            basin_shp = self._find_shp_by_keywords(self.BASIN_KEYWORDS)
            if basin_shp and basin_shp.exists():
                try:
                    basins = gpd.read_file(basin_shp)
                    if basins.crs and basins.crs.to_epsg() != 4326:
                        basins = basins.to_crs("EPSG:4326")
                    nigeria_basins = gpd.clip(basins, nigeria_gdf)
                    nigeria_basins['geometry'] = nigeria_basins['geometry'].buffer(0)
                    nigeria_basins = nigeria_basins[nigeria_basins.is_valid]
                    self.watersheds = json.loads(nigeria_basins.to_json())
                except Exception:
                    pass

            if self.watersheds is None:
                self._create_fallback_watersheds()

            # Rivers
            all_features: List[Dict[str, Any]] = []

            minor_shp = self._find_shp_by_keywords(self.MINOR_RIV_KEYWORDS)
            major_shp = self._find_shp_by_keywords(self.MAJOR_RIV_KEYWORDS)
            if major_shp and minor_shp and major_shp.resolve() == minor_shp.resolve():
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
                            lname = name.lower().strip()
                            if any(re.search(rf"\b{re.escape(m)}\b", lname)
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

            if all_features:
                self.rivers = {
                    "type": "FeatureCollection",
                    "features": all_features,
                    "metadata": {"count": len(all_features)},
                }
            else:
                self.rivers = None

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
                },
            })
        self.watersheds = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"source": "Nigeria Hydrological Areas"},
        }


# ===========================================================================
#                            WEB APP
# ===========================================================================

class FloodForecastWebApp:

    def __init__(self):
        defaults = {
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
            location=NIGERIA_CENTER,
            zoom_start=6.5,
            min_zoom=5.5,
            max_zoom=12,
            max_bounds=True,
            tiles="CartoDB positron",
            control_scale=True,
            zoom_control=True,
        )
        fmap.fit_bounds(NIGERIA_BOUNDS)
        
        fmap.options['maxBounds'] = [
            [NIGERIA_BOUNDS[0][0] - 2, NIGERIA_BOUNDS[0][1] - 2],
            [NIGERIA_BOUNDS[1][0] + 2, NIGERIA_BOUNDS[1][1] + 2],
        ]

        Fullscreen(position='topleft',
                   title='Fullscreen',
                   title_cancel='Exit fullscreen',
                   force_separate_button=True).add_to(fmap)
        MousePosition(position='bottomright',
                      separator=' | ',
                      num_digits=3,
                      prefix='Lat/Lon:').add_to(fmap)

        # Boundary
        if self.data_manager.boundary:
            try:
                folium.GeoJson(
                    self.data_manager.boundary,
                    name="Nigeria Boundary",
                    style_function=lambda x: {
                        "color": "#1B2631", "weight": 2,
                        "fillOpacity": 0.03, "fillColor": "#1B2631",
                    },
                    control=False,
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
                            aliases=['ID:', 'Name:'],
                            localize=True,
                        ),
                    ).add_to(watershed_fg)
            except Exception:
                pass
        watershed_fg.add_to(fmap)

        # Rivers
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
                    fc,
                    style_function=lambda x, s=style: s,
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'waterway', 'length_km'],
                        aliases=['Name:', 'Type:', 'Length (km):'],
                        localize=True, sticky=True,
                    ),
                ).add_to(fg)
                fg.add_to(fmap)

            add_layer(drains, "Drains & ditches", False,
                      {"color": "#a5d8ff", "weight": 0.5,
                       "opacity": 0.6, "fillOpacity": 0})
            add_layer(streams, "Streams", True,
                      {"color": "#74c0fc", "weight": 0.6,
                       "opacity": 0.75, "fillOpacity": 0})
            add_layer(canals, "Canals", False,
                      {"color": "#4dabf7", "weight": 0.7,
                       "opacity": 0.85, "fillOpacity": 0,
                       "dashArray": "4,4"})
            add_layer(rivers, "Rivers", True,
                      {"color": "#1a73e8", "weight": 1.6,
                       "opacity": 0.9, "fillOpacity": 0})
            add_layer(major, "Major rivers", True,
                      {"color": "#003366", "weight": 3.0,
                       "opacity": 0.98, "fillOpacity": 0})

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
            <div><span style="color:#C0392B">●</span> RED &nbsp;<span style="color:#F39C12">●</span> AMBER &nbsp;<span style="color:#27AE60">●</span> GREEN</div>
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
        # Make map take full width
        st.markdown(
            """
            <style>
            .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
            .element-container iframe {
                width: 100% !important;
                height: calc(100vh - 80px) !important;
                border: 0 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.system is None:
            with st.spinner("Initializing forecast system..."):
                from flood_forecast_nigeria import Config, FloodForecastSystem
                cfg = Config()
                system = FloodForecastSystem(cfg)
                system._write_synthetic_gauge_csv()
                st.session_state.system = system

        html = self._build_map_html()
        if html:
            components.html(html, height=800, scrolling=False)
        else:
            st.error("Could not generate map. Check R2 contents.")

    def _render_upload_page(self):
        st.title("📤 Upload Data")
        st.info(
            "### Expected files under `geojson/` in R2\n"
            "- `hybas_af_lev06_v1c.zip` — HydroBASINS watersheds\n"
            "- `nigeria_boundary.geojson` — Nigeria boundary\n"
            "- `ne_10m_rivers.zip` — Natural Earth major rivers (fallback)\n"
            "- HOTOSM / OSM waterways shapefile\n\n"
            "The app mirrors everything under `geojson/`, extracts zips, and discovers shapefiles by keyword."
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 List R2 contents"):
                if self.r2:
                    files = self.r2.list_files("geojson/")
                    if files:
                        st.write(files)
                    else:
                        st.info("No files found in geojson/")
                else:
                    st.error("R2 not configured")
        with col2:
            if st.button("🔄 Force re-sync from R2"):
                st.session_state.data_downloaded_from_r2 = False
                try:
                    shutil.rmtree(self.data_manager.data_dir, ignore_errors=True)
                    self.data_manager.data_dir.mkdir(parents=True, exist_ok=True)
                    self.data_manager.watersheds = None
                    self.data_manager.rivers = None
                    self.data_manager.boundary = None
                    st.success("Cache cleared — reload the Map page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Cache clear failed: {e}")

    def _render_alerts_page(self):
        st.title("📊 Forecast & Alerts")
        if st.session_state.forecast_report:
            st.json(st.session_state.forecast_report)
        else:
            st.info("No forecast data available. Run the forecast system first.")

    def _render_data_page(self):
        st.title("📈 Raw Data")
        if st.session_state.system:
            st.write("System outputs are available in the output directory.")
            st.info("Use the Upload page to sync data from R2.")
        else:
            st.info("No data available. Initialize the system first.")

    def _render_info_page(self):
        st.title("📚 Instructions")
        st.markdown("""
        ### How to Use This App
        
        **Map Navigation:**
        - Use mouse/touch to pan and zoom
        - Click the layer control (top-right) to toggle different waterway types
        - Use fullscreen button (top-left) for better viewing
        
        **Data Layers:**
        - **Major rivers** (thick dark blue) - Niger, Benue, Kaduna, etc.
        - **Rivers** (medium blue) - Smaller named rivers
        - **Streams** (thin light blue) - Minor waterways
        - **Canals** (dashed blue) - Man-made channels
        - **Drains** (very thin) - Drainage ditches
        - **Watersheds** (semi-transparent) - Hydrological basins
        
        **Data Sources:**
        - Watersheds: HydroBASINS v1c
        - Waterways: HOTOSM OpenStreetMap
        - Boundary: Natural Earth
        """)

    def _render_about_page(self):
        st.title("ℹ️ About")
        st.markdown("""
        ### Nigeria Flood Forecast System
        
        This application displays hydrological data for Nigeria:
        
        - **139 watershed basins** from HydroBASINS (Level 6)
        - **Thousands of waterways** from HOTOSM OpenStreetMap
        - **Nigeria boundary** from Natural Earth
        
        **Technology Stack:**
        - Streamlit for the web interface
        - Folium for interactive maps
        - GeoPandas for geospatial processing
        - Cloudflare R2 for data storage
        
        **Data is automatically downloaded from R2 on first use and cached locally.**
        """)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------
    def render(self):
        # Create tabs at the top for navigation
        tabs = st.tabs(["🗺️ Map", "📤 Upload", "📊 Alerts", "📈 Data", "📚 Info", "ℹ️ About"])
        
        with tabs[0]:
            self._render_map_page()
        with tabs[1]:
            self._render_upload_page()
        with tabs[2]:
            self._render_alerts_page()
        with tabs[3]:
            self._render_data_page()
        with tabs[4]:
            self._render_info_page()
        with tabs[5]:
            self._render_about_page()


# ===========================================================================
#                               MAIN
# ===========================================================================

def main():
    st.set_page_config(
        page_title="Nigeria Flood Forecast",
        page_icon="🌊",
        layout="wide",
    )
    app = FloodForecastWebApp()
    app.render()


if __name__ == "__main__":
    main()
