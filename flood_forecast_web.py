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
            st.info("🔍 Checking files in R2 bucket...")
            all_files = self.r2.list_files("")
            st.write(f"Files found in geojson/: {all_files}")
            
            # 1. Download HydroBASINS watersheds
            hydrobasins_remote = "geojson/hybas_af_lev06_v1c.zip"
            if self.r2.exists(hydrobasins_remote):
                st.info("📥 Downloading HydroBASINS watershed data (549 MB - please wait)...")
                zip_path = self.data_dir / "hybas_af_lev06_v1c.zip"
                
                if self.r2.download_file(hydrobasins_remote, zip_path):
                    st.success("✓ HydroBASINS downloaded")
                else:
                    st.warning("Could not download HydroBASINS, will use fallback")
            else:
                st.warning(f"HydroBASINS file not found at {hydrobasins_remote}")
            
            # 2. Download rivers
            rivers_remote = "geojson/ne_10m_rivers.zip"
            if self.r2.exists(rivers_remote):
                st.info("📥 Downloading river network data (2 MB)...")
                rivers_zip = self.data_dir / "ne_10m_rivers.zip"
                if self.r2.download_file(rivers_remote, rivers_zip):
                    st.success("✓ River network downloaded")
                    # Extract the zip file immediately
                    st.info("Extracting river data...")
                    with zipfile.ZipFile(rivers_zip, 'r') as z:
                        z.extractall(self.data_dir)
                        st.write(f"Extracted river files: {z.namelist()[:10]}")
            else:
                st.warning(f"Rivers file not found at {rivers_remote}")
            
            # 3. Download Nigeria boundary
            boundary_remote = "geojson/nigeria_boundary.geojson"
            if self.r2.exists(boundary_remote):
                st.info("📥 Downloading Nigeria boundary...")
                boundary_path = self.data_dir / "nigeria_boundary.geojson"
                if self.r2.download_file(boundary_remote, boundary_path):
                    st.success("✓ Nigeria boundary downloaded")
            else:
                st.warning(f"Boundary file not found at {boundary_remote}")
            
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
            from shapely.geometry import box, mapping
            import json
            
            # First, check what files we have locally
            st.write("📁 Local files in data_dir:", list(self.data_dir.glob("*")))
            
            # ============================================================
            # Load watersheds from HydroBASINS (keep as is - working)
            # ============================================================
            shp_path = self.data_dir / "hybas_af_lev06_v1c.shp"
            
            if shp_path.exists():
                st.info("🗺️ Loading watershed boundaries...")
                basins = gpd.read_file(shp_path)
                st.write(f"Loaded basins shapefile with {len(basins)} features")
                
                # Get Nigeria boundary
                nigeria = self._get_nigeria_boundary_gdf()
                
                # Clip to Nigeria
                nigeria_basins = gpd.clip(basins, nigeria)
                st.write(f"After clipping to Nigeria: {len(nigeria_basins)} basins")
                
                # Fix any invalid geometries
                nigeria_basins['geometry'] = nigeria_basins['geometry'].buffer(0)
                nigeria_basins = nigeria_basins[nigeria_basins.is_valid]
                st.write(f"Valid geometries: {len(nigeria_basins)} basins")
                
                # Convert to GeoJSON
                geojson_str = nigeria_basins.to_json()
                self.watersheds = json.loads(geojson_str)
                self.watersheds['metadata'] = {
                    'source': 'HydroBASINS v1c',
                    'level': 6,
                    'count': len(nigeria_basins)
                }
                st.success(f"✅ Loaded {len(nigeria_basins)} watershed basins")
            else:
                st.warning(f"No HydroBASINS shapefile found")
            
            # ============================================================
            # Load rivers - SIMPLIFIED AND GUARANTEED TO WORK
            # ============================================================
            rivers_shp = self.data_dir / "ne_10m_rivers_lake_centerlines.shp"
            
            if rivers_shp and rivers_shp.exists():
                st.info("🌊 Loading river network...")
                try:
                    # Read the shapefile
                    rivers_gdf = gpd.read_file(rivers_shp)
                    st.write(f"Raw rivers loaded: {len(rivers_gdf)} features")
                    st.write(f"CRS: {rivers_gdf.crs}")
                    
                    # Get Nigeria boundary
                    nigeria_gdf = self._get_nigeria_boundary_gdf()
                    st.write(f"Nigeria boundary CRS: {nigeria_gdf.crs}")
                    
                    # Ensure both are in the same CRS (WGS84 - EPSG:4326)
                    if rivers_gdf.crs != "EPSG:4326" and rivers_gdf.crs is not None:
                        st.info("Reprojecting rivers to EPSG:4326...")
                        rivers_gdf = rivers_gdf.to_crs("EPSG:4326")
                    
                    # Get Nigeria geometry as a single polygon
                    nigeria_geom = nigeria_gdf.unary_union
                    
                    # Method 1: Clip rivers to Nigeria (more aggressive)
                    st.info("Clipping rivers to Nigeria boundary...")
                    try:
                        # Clip the rivers to Nigeria
                        clipped_rivers = gpd.clip(rivers_gdf, nigeria_gdf)
                        st.write(f"After clipping: {len(clipped_rivers)} rivers")
                    except:
                        # If clip fails, use intersection
                        st.info("Clip failed, using intersection...")
                        clipped_rivers = rivers_gdf[rivers_gdf.intersects(nigeria_geom)]
                        st.write(f"After intersection: {len(clipped_rivers)} rivers")
                    
                    # If still no rivers, take all rivers that have any overlap with Nigeria
                    if len(clipped_rivers) == 0:
                        st.warning("No rivers found with clip, taking all rivers within bounding box...")
                        minx, miny, maxx, maxy = nigeria_geom.bounds
                        # Expand bounds slightly
                        minx -= 2
                        maxx += 2
                        miny -= 2
                        maxy += 2
                        # Create a bounding box polygon
                        bbox_poly = box(minx, miny, maxx, maxy)
                        # Get rivers that intersect the bounding box
                        clipped_rivers = rivers_gdf[rivers_gdf.intersects(bbox_poly)]
                        st.write(f"After bounding box filter: {len(clipped_rivers)} rivers")
                    
                    # Take only the top 200 longest rivers for performance
                    if len(clipped_rivers) > 200:
                        st.info(f"Limiting to 200 longest rivers (from {len(clipped_rivers)})...")
                        clipped_rivers['length_km'] = clipped_rivers.geometry.length * 111  # Approx km
                        clipped_rivers = clipped_rivers.nlargest(200, 'length_km')
                        st.write(f"After length filter: {len(clipped_rivers)} rivers")
                    
                    # Fix invalid geometries
                    if len(clipped_rivers) > 0:
                        clipped_rivers['geometry'] = clipped_rivers['geometry'].buffer(0)
                        clipped_rivers = clipped_rivers[clipped_rivers.is_valid]
                        st.write(f"After geometry fix: {len(clipped_rivers)} valid rivers")
                    
                    # Create a clean GeoJSON with just the rivers
                    if len(clipped_rivers) > 0:
                        # Convert to GeoJSON with explicit structure
                        features = []
                        for idx, row in clipped_rivers.iterrows():
                            try:
                                # Get geometry as GeoJSON
                                geom_json = mapping(row.geometry)
                                
                                # Get river name (try different field names)
                                name = None
                                for field in ['name', 'NAME', 'Name', 'river', 'RIVER']:
                                    if field in row and row[field]:
                                        name = str(row[field])
                                        break
                                if not name:
                                    name = f"River {idx}"
                                
                                features.append({
                                    "type": "Feature",
                                    "geometry": geom_json,
                                    "properties": {
                                        "name": name,
                                        "length_km": float(row.geometry.length * 111) if hasattr(row, 'length_km') else 0
                                    }
                                })
                            except Exception as e:
                                st.write(f"Could not add river {idx}: {e}")
                        
                        self.rivers = {
                            "type": "FeatureCollection",
                            "features": features,
                            "metadata": {
                                'source': 'Natural Earth (1:10m)',
                                'count': len(features)
                            }
                        }
                        st.success(f"✅ Loaded {len(features)} river segments into GeoJSON")
                        
                        # Display sample river info
                        if len(features) > 0:
                            st.write(f"Sample river: {features[0]['properties']['name']}")
                            st.write(f"Sample geometry type: {features[0]['geometry']['type']}")
                    else:
                        st.warning("No valid river geometries found")
                        self.rivers = None
                        
                except Exception as e:
                    st.error(f"River loading failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    self.rivers = None
            
            # ============================================================
            # Load boundary
            # ============================================================
            boundary_path = self.data_dir / "nigeria_boundary.geojson"
            if boundary_path.exists():
                try:
                    with open(boundary_path, 'r') as f:
                        self.boundary = json.load(f)
                    st.success("✅ Loaded Nigeria boundary")
                except Exception as e:
                    st.error(f"Boundary loading failed: {e}")
                    self.boundary = None
            
            # If any data is missing, create fallback
            if self.watersheds is None:
                st.warning("No watershed data found, using fallback")
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
            try:
                folium.GeoJson(
                    self.data_manager.boundary,
                    name="Nigeria Boundary",
                    style_function=lambda x: {"color": "#1B2631", "weight": 2, "fillOpacity": 0.1}
                ).add_to(fmap)
            except Exception as e:
                st.warning(f"Could not add boundary: {e}")
        
        # Add watersheds (from HydroBASINS or fallback)
        if self.data_manager.watersheds:
            try:
                # Check if this is fallback data (has 'id' and 'name' properties) or HydroBASINS (has 'HYBAS_ID')
                sample_feature = self.data_manager.watersheds.get('features', [{}])[0]
                props = sample_feature.get('properties', {})
                
                # Determine which fields are available
                if 'HYBAS_ID' in props:
                    # HydroBASINS data - use different tooltip fields
                    folium.GeoJson(
                        self.data_manager.watersheds,
                        name="Watersheds",
                        style_function=lambda x: {
                            'fillColor': '#2C3E50',
                            'color': '#2C3E50',
                            'weight': 1.3,
                            'fillOpacity': 0.22
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['HYBAS_ID', 'SUB_AREA', 'UP_AREA'],
                            aliases=['ID:', 'Area (km²):', 'Upstream Area (km²):'],
                            localize=True
                        )
                    ).add_to(fmap)
                else:
                    # Fallback data - use id and name
                    folium.GeoJson(
                        self.data_manager.watersheds,
                        name="Watersheds",
                        style_function=lambda x: {
                            'fillColor': x['properties'].get('color', '#2C3E50'),
                            'color': '#2C3E50',
                            'weight': 1.3,
                            'fillOpacity': 0.22
                        },
                        tooltip=folium.GeoJsonTooltip(
                            fields=['id', 'name'],
                            aliases=['ID:', 'Name:'],
                            localize=True
                        )
                    ).add_to(fmap)
            except Exception as e:
                st.warning(f"Could not add watersheds: {e}")
        
        # Add rivers (from Natural Earth)
        if self.data_manager.rivers:
            try:
                num_features = len(self.data_manager.rivers.get('features', []))
                st.write(f"🎯 Rendering {num_features} river features on map")
                
                # Add rivers with very visible styling
                folium.GeoJson(
                    self.data_manager.rivers,
                    name="🌊 Rivers",
                    style_function=lambda x: {
                        "color": "#0066CC",      # Bright blue
                        "weight": 3,              # Thicker line
                        "opacity": 1.0,
                        "fillOpacity": 0
                    },
                    highlight_function=lambda x: {
                        "weight": 5,
                        "color": "#FF0000"
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'length_km'],
                        aliases=['River:', 'Length (km):'],
                        localize=True,
                        sticky=True
                    )
                ).add_to(fmap)
                
                st.success(f"✅ Added {num_features} rivers to map")
            except Exception as e:
                st.warning(f"Could not add rivers: {e}")
        
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
