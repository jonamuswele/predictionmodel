# flood_forecast_web.py - FIXED VERSION

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
    
    def __init__(self, r2_storage=None):
        self.r2 = r2_storage
        self.data_dir = Path("./data/hydrological")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.watersheds = None
        self.rivers = None
        self.water_bodies = None
        self.boundary = None
        
        # Track download status
        if 'data_downloaded_from_r2' not in st.session_state:
            st.session_state.data_downloaded_from_r2 = False
    
    def ensure_data_loaded(self) -> bool:
        """Download and load data from R2."""
        if self.watersheds is not None:
            return True
        
        if not self.r2:
            st.error("R2 storage not configured. Please add R2 credentials to secrets.toml")
            return False
        
        # Download from R2 if not already done this session
        if not st.session_state.data_downloaded_from_r2:
            with st.spinner("📥 Downloading hydrological data from R2 (first time, caching for future runs)..."):
                success = self._download_all_from_r2()
                if success:
                    st.session_state.data_downloaded_from_r2 = True
                    st.success("✅ Hydrological data downloaded and cached")
                else:
                    st.error("Failed to download data from R2")
                    return False
        
        # Load the downloaded data
        return self._load_from_local_cache()
    
    def _download_all_from_r2(self) -> bool:
        """Download all hydrological files from R2."""
        try:
            # List all files in R2 to debug
            st.info("Checking files in R2 bucket...")
            
            # Use list_files with empty string to get all files
            all_files = self.r2.list_files("")
            st.write(f"Files found in R2: {all_files}")
            
            # 1. Download HydroBASINS watersheds
            hydrobasins_remote = None
            # Check all possible paths
            possible_hydro_paths = [
                "hybas_af_lev06_v1c.zip",
                "geojson/hybas_af_lev06_v1c.zip",
                "/hybas_af_lev06_v1c.zip",
                "/geojson/hybas_af_lev06_v1c.zip"
            ]
            
            for path in possible_hydro_paths:
                if self.r2.exists(path):
                    hydrobasins_remote = path
                    st.write(f"Found HydroBASINS at: {path}")
                    break
            
            if hydrobasins_remote:
                st.info("Downloading HydroBASINS watershed data (523 MB - please wait)...")
                zip_path = self.data_dir / "hybas_af_lev06_v1c.zip"
                
                if self.r2.download_file(hydrobasins_remote, zip_path):
                    # Extract the zip file
                    with zipfile.ZipFile(zip_path, 'r') as z:
                        z.extractall(self.data_dir)
                    st.success("✓ HydroBASINS downloaded and extracted")
                else:
                    st.warning("Could not download HydroBASINS, will use fallback")
            else:
                st.warning("HydroBASINS file not found in R2 bucket")
            
            # 2. Download Nigeria boundary
            boundary_remote = None
            possible_boundary_paths = [
                "nigeria_boundary.geojson",
                "geojson/nigeria_boundary.geojson",
                "/nigeria_boundary.geojson",
                "/geojson/nigeria_boundary.geojson"
            ]
            
            for path in possible_boundary_paths:
                if self.r2.exists(path):
                    boundary_remote = path
                    st.write(f"Found boundary at: {path}")
                    break
            
            if boundary_remote:
                boundary_path = self.data_dir / "nigeria_boundary.geojson"
                if self.r2.download_file(boundary_remote, boundary_path):
                    st.success("✓ Nigeria boundary downloaded")
            
            # 3. Download rivers (Natural Earth)
            rivers_remote = None
            possible_river_paths = [
                "ne_10m_rivers.zip",
                "ne_10m_rivers_lake_centerlines.zip",
                "geojson/ne_10m_rivers.zip",
                "geojson/ne_10m_rivers_lake_centerlines.zip",
                "/ne_10m_rivers.zip",
                "/geojson/ne_10m_rivers.zip"
            ]
            
            for path in possible_river_paths:
                if self.r2.exists(path):
                    rivers_remote = path
                    st.write(f"Found rivers at: {path}")
                    break
            
            if rivers_remote:
                st.info("Downloading river network data...")
                rivers_zip = self.data_dir / "ne_10m_rivers_lake_centerlines.zip"
                if self.r2.download_file(rivers_remote, rivers_zip):
                    with zipfile.ZipFile(rivers_zip, 'r') as z:
                        z.extractall(self.data_dir)
                    st.success("✓ River network downloaded")
            
            return True
            
        except Exception as e:
            st.error(f"Download failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def _load_from_local_cache(self) -> bool:
        """Load the downloaded data into memory."""
        try:
            import geopandas as gpd
            from shapely.geometry import box
            
            # First, check what files we have locally
            st.write("Local files in data_dir:", list(self.data_dir.glob("*")))
            
            # Load watersheds from HydroBASINS
            shp_path = self.data_dir / "hybas_af_lev06_v1c.shp"
            if not shp_path.exists():
                # Try looking in subdirectories
                for subdir in self.data_dir.glob("*/"):
                    potential_shp = subdir / "hybas_af_lev06_v1c.shp"
                    if potential_shp.exists():
                        shp_path = potential_shp
                        break
            
            if shp_path.exists():
                st.info("Loading watershed boundaries...")
                basins = gpd.read_file(shp_path)
                st.write(f"Loaded basins shapefile with {len(basins)} features")
                
                # Get Nigeria boundary
                nigeria = self._get_nigeria_boundary_gdf()
                
                # Clip to Nigeria
                nigeria_basins = gpd.clip(basins, nigeria)
                
                # Convert to GeoJSON
                self.watersheds = json.loads(nigeria_basins.to_json())
                self.watersheds['metadata'] = {
                    'source': 'HydroBASINS v1c',
                    'level': 6,
                    'count': len(nigeria_basins)
                }
                st.success(f"✅ Loaded {len(nigeria_basins)} watershed basins")
            else:
                st.warning(f"Shapefile not found at {shp_path}")
            
            # Load rivers
            rivers_shp = self.data_dir / "ne_10m_rivers_lake_centerlines.shp"
            if not rivers_shp.exists():
                rivers_shp = self.data_dir / "ne_10m_rivers.shp"
            
            if rivers_shp.exists():
                st.info("Loading river network...")
                rivers = gpd.read_file(rivers_shp)
                st.write(f"Loaded rivers shapefile with {len(rivers)} features")
                
                # Filter to Nigeria
                nigeria = self._get_nigeria_boundary_gdf()
                nigeria_rivers = rivers[rivers.within(nigeria.unary_union.buffer(0.5))]
                
                self.rivers = json.loads(nigeria_rivers.to_json())
                self.rivers['metadata'] = {
                    'source': 'Natural Earth (1:10m)',
                    'count': len(nigeria_rivers)
                }
                st.success(f"✅ Loaded {len(nigeria_rivers)} river segments")
            
            # Load boundary
            boundary_path = self.data_dir / "nigeria_boundary.geojson"
            if boundary_path.exists():
                with open(boundary_path, 'r') as f:
                    self.boundary = json.load(f)
                st.success("✅ Loaded Nigeria boundary")
            
            # If any data is missing, create fallback
            if self.watersheds is None:
                self._create_fallback_watersheds()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def _get_nigeria_boundary_gdf(self):
        """Get Nigeria boundary as GeoDataFrame."""
        import geopandas as gpd
        from shapely.geometry import box
        
        boundary_path = self.data_dir / "nigeria_boundary.geojson"
        if boundary_path.exists():
            return gpd.read_file(boundary_path)
        
        # Fallback bounding box
        bbox = (2.5, 4.0, 14.7, 13.9)
        return gpd.GeoDataFrame(geometry=[box(*bbox)], crs='EPSG:4326')
    
    def _create_fallback_watersheds(self):
        """Create fallback watersheds if HydroBASINS not available."""
        st.warning("HydroBASINS not available, using Nigeria Hydrological Areas")
        
        watersheds_data = {
            "HA1": {"name": "Niger North", "color": "#5DADE2", "bounds": (2.5, 9.5, 9.5, 13.9)},
            "HA2": {"name": "Niger Central", "color": "#48C9B0", "bounds": (4.0, 7.5, 9.5, 11.0)},
            "HA3": {"name": "Lower Niger", "color": "#F5B041", "bounds": (5.0, 4.5, 8.5, 8.0)},
            "HA4": {"name": "Upper Benue", "color": "#AF7AC5", "bounds": (9.0, 7.5, 13.5, 10.5)},
            "HA5": {"name": "Lower Benue", "color": "#EC7063", "bounds": (7.5, 6.0, 11.0, 8.5)},
            "HA6": {"name": "Cross River", "color": "#58D68D", "bounds": (7.5, 4.5, 9.5, 6.5)},
            "HA7": {"name": "Western Littoral", "color": "#F4D03F", "bounds": (2.5, 6.0, 5.5, 8.0)},
            "HA8": {"name": "Lake Chad Basin", "color": "#5D6D7E", "bounds": (9.5, 10.5, 14.7, 13.5)}
        }
        
        features = []
        for ha_id, info in watersheds_data.items():
            min_lon, min_lat, max_lon, max_lat = info["bounds"]
            
            # Create polygon from bounds
            coords = [
                (min_lon, min_lat), (max_lon, min_lat),
                (max_lon, max_lat), (min_lon, max_lat),
                (min_lon, min_lat)
            ]
            
            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {
                    "id": ha_id,
                    "name": info["name"],
                    "color": info["color"],
                    "source": "NIHSA Hydrological Areas"
                }
            }
            features.append(feature)
        
        self.watersheds = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": {"source": "Nigeria Hydrological Areas"}
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
        # Initialize session state
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
            "r2_connected": False,  # Track R2 connection status
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v
        
        # Initialize R2 storage (don't show messages here)
        self.r2 = None
        if cloud_storage is not None:
            try:
                self.r2 = cloud_storage.get_r2_from_secrets(st.secrets)
                if self.r2:
                    st.session_state.r2_connected = True
            except Exception:
                pass
        
        # Initialize data manager
        self.data_manager = ProfessionalDataManager(self.r2)
    
    def _build_map_html(self, system, report, forecast_date) -> Optional[str]:
        """Build map using data from R2."""
        try:
            import folium
        except ImportError:
            st.warning("Folium not installed. Run: pip install folium")
            return None
        
        # Ensure data is loaded from R2
        if not self.data_manager.ensure_data_loaded():
            st.error("Could not load hydrological data")
            return None
        
        # Initialize map
        fmap = folium.Map(location=[9.0, 8.0], zoom_start=6, tiles="CartoDB positron")
        
        # Add Nigeria boundary
        if self.data_manager.boundary:
            folium.GeoJson(
                self.data_manager.boundary,
                name="Nigeria Boundary",
                style_function=lambda x: {"color": "#1B2631", "weight": 2, "fillOpacity": 0.1}
            ).add_to(fmap)
        
        # Add watersheds (from HydroBASINS or fallback)
        if self.data_manager.watersheds:
            def style_watershed(feature):
                props = feature.get('properties', {})
                color = props.get('color', '#2C3E50')
                return {
                    'fillColor': color,
                    'color': '#2C3E50',
                    'weight': 1.3,
                    'fillOpacity': 0.22
                }
            
            folium.GeoJson(
                self.data_manager.watersheds,
                name="Watersheds",
                style_function=style_watershed,
                tooltip=folium.GeoJsonTooltip(fields=['name', 'id'], aliases=['Name:', 'ID:'])
            ).add_to(fmap)
        
        # Add rivers (from Natural Earth)
        if self.data_manager.rivers:
            folium.GeoJson(
                self.data_manager.rivers,
                name="Rivers",
                style_function=lambda x: {"color": "#3498DB", "weight": 1.5, "opacity": 0.7}
            ).add_to(fmap)
        
        # Add legend
        legend_html = """
        <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                    background: white; padding: 10px; border-radius: 5px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.25);">
            <b>Legend</b><br>
            <span style="color:#1B2631">■</span> Nigeria Boundary<br>
            <span style="color:#2C3E50;background:#2C3E50;">▯</span> Watersheds<br>
            <span style="color:#3498DB">━</span> Rivers<br>
            <span style="color:#C0392B">●</span> RED Alert<br>
            <span style="color:#F39C12">●</span> AMBER Alert<br>
            <span style="color:#27AE60">●</span> GREEN Alert
        </div>
        """
        
        from branca.element import MacroElement, Template
        legend = MacroElement()
        legend._template = Template(legend_html)
        fmap.get_root().add_child(legend)
        
        folium.LayerControl().add_to(fmap)
        
        return fmap.get_root().render()
    
    def render(self):
        """Render the app."""
        # Show R2 connection status only when on the map page
        if st.session_state.page == self.PAGES[0] and st.session_state.r2_connected:
            st.success(f"✅ Connected to R2 bucket: nigeriahydro")
        
        with st.sidebar:
            st.session_state.page = st.radio("Navigation", self.PAGES)
        
        if st.session_state.page == self.PAGES[0]:
            st.title("Nigeria Flood Forecast Map")
            
            # Initialize demo if needed
            if st.session_state.system is None:
                with st.spinner("Initializing forecast system..."):
                    from flood_forecast_nigeria import Config, FloodForecastSystem
                    cfg = Config()
                    system = FloodForecastSystem(cfg)
                    system._write_synthetic_gauge_csv()
                    st.session_state.system = system
            
            # Build and display map
            html = self._build_map_html(
                st.session_state.system,
                st.session_state.forecast_report,
                st.session_state.forecast_date
            )
            
            if html:
                components.html(html, height=700, scrolling=False)
            else:
                st.error("Could not generate map. Check that R2 contains the required files.")
        
        elif st.session_state.page == self.PAGES[1]:
            st.title("Upload Data")
            st.info("""
            ### Required files in R2 bucket:
            - `geojson/hybas_af_lev06_v1c.zip` - HydroBASINS watershed data
            - `geojson/nigeria_boundary.geojson` - Nigeria boundary
            - `geojson/ne_10m_rivers.zip` - River network
            
            Upload these files to your R2 bucket first.
            """)
            
            if st.button("Check R2 Connection"):
                if self.r2:
                    files = self.r2.list_files("geojson/")
                    st.write(f"Files in R2 bucket: {files}")
                else:
                    st.error("R2 not configured")


# ===========================================================================
#                               MAIN
# ===========================================================================

def main():
    app = FloodForecastWebApp()
    app.render()


if __name__ == "__main__":
    main()
