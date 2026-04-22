# flood_forecast_web.py - robust R2 sync + flexible shapefile discovery

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
    """Downloads professional hydrological data from R2 bucket."""

    # Keywords used to classify shapefiles found on disk.
    BASIN_KEYWORDS  = ("hybas", "basin", "watershed")
    MAJOR_RIV_KEYWORDS = ("ne_10m_rivers", "ne_50m_rivers",
                          "natural_earth", "rivers_lake_centerlines")
    # HOTOSM / OSM detailed waterways — the "smaller rivers"
    MINOR_RIV_KEYWORDS = ("hotosm", "waterway", "osm_water", "water_lines",
                          "nga_water")

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
    def _find_shp_by_keywords(self, keywords) -> Optional[Path]:
        """Return the first .shp whose filename matches any keyword."""
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
        except Exception as e:
            st.warning(f"Could not extract {zip_path.name}: {e}")
            return False

    # ------------------------------------------------------------------
    # Load pipeline
    # ------------------------------------------------------------------
    def ensure_data_loaded(self) -> bool:
        if self.watersheds is not None and self.rivers is not None:
            return True

        if not self.r2:
            st.error("R2 storage not configured. Add R2 credentials to secrets.toml")
            return False

        if not st.session_state.data_downloaded_from_r2:
            with st.spinner("Syncing hydrological data from R2 ..."):
                if not self._sync_geojson_prefix():
                    st.error("Failed to sync data from R2")
                    return False
                st.session_state.data_downloaded_from_r2 = True

        # Extract every zip we now have on disk (idempotent).
        for z in self.data_dir.rglob("*.zip"):
            # Skip if there's already an extracted shapefile whose stem
            # starts with the zip stem.
            if any(self.data_dir.rglob(f"{z.stem}*.shp")):
                continue
            self._extract_zip(z)

        return self._load_from_local_cache()

    # ------------------------------------------------------------------
    # R2 sync — mirror everything under geojson/
    # ------------------------------------------------------------------
    def _sync_geojson_prefix(self) -> bool:
        """Download every object under ``geojson/`` that isn't already
        cached locally with the same size.
        """
        try:
            keys = self.r2.list_files("geojson/")
            if not keys:
                st.warning("No files found under geojson/ in R2.")
                return False

            st.caption(f"R2 geojson/ contains {len(keys)} objects")
            for key in keys:
                # Strip the "geojson/" prefix to get the local relative path.
                rel = key.split("geojson/", 1)[-1].lstrip("/")
                if not rel:
                    continue
                local = self.data_dir / rel
                local.parent.mkdir(parents=True, exist_ok=True)

                if local.exists() and local.stat().st_size > 0:
                    continue

                if self.r2.download_file(key, local):
                    st.caption(f"↓ {rel}")
                else:
                    st.warning(f"Download failed: {key}")
            return True
        except Exception as e:
            st.error(f"R2 sync failed: {e}")
            st.code(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Load cached data into memory
    # ------------------------------------------------------------------
    def _load_from_local_cache(self) -> bool:
        try:
            import geopandas as gpd
            from shapely.geometry import box, mapping

            all_shps = list(self.data_dir.rglob("*.shp"))
            if all_shps:
                st.caption(
                    "Shapefiles on disk: "
                    + ", ".join(str(p.relative_to(self.data_dir)) for p in all_shps)
                )
            else:
                st.warning("No .shp files on disk after sync.")

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

            # ---------- Watersheds ----------
            shp_path = self._find_shp_by_keywords(self.BASIN_KEYWORDS)
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
                        'source': shp_path.name,
                        'count': len(nigeria_basins),
                    }
                    st.success(f"Loaded {len(nigeria_basins)} watershed basins from {shp_path.name}")
                except Exception as e:
                    st.warning(f"HydroBASINS load failed ({shp_path.name}): {e}")

            if self.watersheds is None:
                self._create_fallback_watersheds()

            # ---------- Rivers: major + minor ----------
            all_features: List[Dict[str, Any]] = []

            major_shp = self._find_shp_by_keywords(self.MAJOR_RIV_KEYWORDS)
            minor_shp = self._find_shp_by_keywords(self.MINOR_RIV_KEYWORDS)

            # Defensive: the same file shouldn't be loaded twice.
            if major_shp and minor_shp and major_shp.resolve() == minor_shp.resolve():
                minor_shp = None

            # ---- Minor (HOTOSM / OSM waterways) ----
            if minor_shp and minor_shp.exists():
                st.info(f"Loading smaller waterways from {minor_shp.name} ...")
                try:
                    gdf = gpd.read_file(minor_shp)
                    st.caption(
                        f"  raw features: {len(gdf)} | columns: "
                        + ", ".join(gdf.columns[:15])
                    )
                    if gdf.crs and gdf.crs.to_epsg() != 4326:
                        gdf = gdf.to_crs("EPSG:4326")

                    # Clip to Nigeria (with bbox fallback).
                    try:
                        clipped = gpd.clip(gdf, nigeria_gdf)
                    except Exception:
                        clipped = gdf[gdf.intersects(nigeria_gdf.unary_union)]

                    if len(clipped) == 0:
                        minx, miny, maxx, maxy = nigeria_gdf.total_bounds
                        bbox_poly = box(minx - 1, miny - 1, maxx + 1, maxy + 1)
                        clipped = gdf[gdf.intersects(bbox_poly)]

                    # Filter to line waterways if a `waterway` column exists.
                    # (HOTOSM uses this tag.) Keep river/stream/canal/drain.
                    if 'waterway' in clipped.columns:
                        wanted = ['river', 'stream', 'canal', 'drain',
                                  'ditch', 'tidal_channel']
                        before = len(clipped)
                        clipped = clipped[clipped['waterway'].isin(wanted)]
                        st.caption(
                            f"  waterway filter: {before} → {len(clipped)}"
                        )

                    clipped = clipped[~clipped.geometry.is_empty]
                    clipped = clipped[clipped.geometry.is_valid]

                    # Cap at 3000 longest to keep render fast.
                    if len(clipped) > 3000:
                        clipped = clipped.copy()
                        clipped['length_deg'] = clipped.geometry.length
                        clipped = clipped.nlargest(3000, 'length_deg')
                        st.caption("  capped to 3000 longest features")

                    minor_count = 0
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
                        if not name:
                            name = f"{str(waterway).capitalize()}"

                        all_features.append({
                            "type": "Feature",
                            "geometry": geom_json,
                            "properties": {
                                "name": name,
                                "waterway": str(waterway),
                                "length_km": round(float(row.geometry.length) * 111.0, 1),
                                "source": "OSM",
                            },
                        })
                        minor_count += 1

                    if minor_count:
                        st.success(f"Loaded {minor_count} OSM waterway features")
                    else:
                        st.warning(
                            "OSM shapefile loaded but 0 features remained "
                            "after filtering — check the `waterway` column."
                        )
                except Exception as e:
                    st.error(f"OSM waterways load failed: {e}")
                    st.code(traceback.format_exc())
            else:
                st.info(
                    "No OSM/HOTOSM waterway shapefile found on disk. "
                    "Upload the HOTOSM export (all sidecars or the original "
                    "zip) under `geojson/` in R2."
                )

            # ---- Major (Natural Earth) ----
            if major_shp and major_shp.exists():
                try:
                    gdf = gpd.read_file(major_shp)
                    st.caption(
                        f"Major rivers source: {major_shp.name} "
                        f"({len(gdf)} raw features)"
                    )
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

                    major_count = 0
                    for idx, row in clipped.iterrows():
                        try:
                            geom_json = mapping(row.geometry)
                        except Exception:
                            continue
                        name = None
                        for field in ('name', 'NAME', 'Name', 'name_en',
                                      'river', 'RIVER', 'featurecla'):
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
                                "waterway": "major_river",
                                "length_km": round(float(row.geometry.length) * 111.0, 1),
                                "source": "NaturalEarth",
                            },
                        })
                        major_count += 1
                    st.success(f"Loaded {major_count} major river segments from {major_shp.name}")
                except Exception as e:
                    st.error(f"Major rivers load failed: {e}")
                    st.code(traceback.format_exc())

            if all_features:
                self.rivers = {
                    "type": "FeatureCollection",
                    "features": all_features,
                    "metadata": {
                        "source": "OSM + Natural Earth",
                        "count": len(all_features),
                    },
                }
                st.success(f"TOTAL river features on map: {len(all_features)}")
            else:
                st.warning("No river features loaded from any source.")
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
        st.info("HydroBASINS not available — using Nigeria Hydrological Areas")
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

        # Rivers — draw smaller first so major rivers sit on top.
        if self.data_manager.rivers and self.data_manager.rivers.get('features'):
            try:
                def style_river(feature):
                    props = feature.get('properties', {})
                    waterway = props.get('waterway', 'stream')
                    if waterway == 'major_river':
                        return {"color": "#003366", "weight": 3.0,
                                "opacity": 0.95, "fillOpacity": 0}
                    if waterway == 'river':
                        return {"color": "#1a73e8", "weight": 2.0,
                                "opacity": 0.9, "fillOpacity": 0}
                    if waterway == 'stream':
                        return {"color": "#4dabf7", "weight": 1.1,
                                "opacity": 0.75, "fillOpacity": 0}
                    if waterway == 'canal':
                        return {"color": "#74c0fc", "weight": 1.4,
                                "opacity": 0.85, "fillOpacity": 0,
                                "dashArray": "5,5"}
                    if waterway in ('drain', 'ditch', 'tidal_channel'):
                        return {"color": "#a5d8ff", "weight": 0.9,
                                "opacity": 0.7, "fillOpacity": 0}
                    return {"color": "#3399ff", "weight": 1.3,
                            "opacity": 0.8, "fillOpacity": 0}

                folium.GeoJson(
                    self.data_manager.rivers,
                    name="Rivers & Streams",
                    style_function=style_river,
                    highlight_function=lambda x: {
                        "weight": 4, "color": "#FF3333",
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'waterway', 'length_km'],
                        aliases=['Name:', 'Type:', 'Length (km):'],
                        localize=True, sticky=True,
                    ),
                ).add_to(fmap)

                n = len(self.data_manager.rivers.get('features', []))
                st.caption(f"Showing {n} river/waterway features on map")
            except Exception as e:
                st.warning(f"Could not add rivers: {e}")
        else:
            st.info("Rivers layer is empty — upload waterway shapefiles to R2 under `geojson/`.")

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
            <span style="color:#003366">━</span> Major Rivers<br>
            <span style="color:#1a73e8">━</span> Rivers<br>
            <span style="color:#4dabf7">━</span> Streams<br>
            <span style="color:#74c0fc">╌</span> Canals<br>
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
                "### Expected files under `geojson/` in R2\n"
                "- `hybas_af_lev06_v1c.zip` — HydroBASINS watersheds\n"
                "- `nigeria_boundary.geojson` — Nigeria boundary\n"
                "- `ne_10m_rivers.zip` — Natural Earth major rivers\n"
                "- **Any HOTOSM / OSM waterways shapefile** (e.g. "
                "`hotosm_nga_waterways_lines.zip` or the loose "
                "`.shp/.shx/.dbf/.prj/.cpg`) — the app matches on the "
                "`hotosm`/`waterway`/`water_lines`/`nga_water` keywords.\n\n"
                "The app mirrors **everything** under `geojson/` — so any "
                "name that contains those keywords is picked up."
            )
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("List R2 contents"):
                    if self.r2:
                        files = self.r2.list_files("geojson/")
                        st.write(files or "(empty)")
                    else:
                        st.error("R2 not configured")
            with col2:
                if st.button("Show local files"):
                    try:
                        items = sorted(
                            str(p.relative_to(self.data_manager.data_dir))
                            for p in self.data_manager.data_dir.rglob("*")
                            if p.is_file()
                        )
                        st.write(items or "(empty)")
                    except Exception as e:
                        st.error(str(e))
            with col3:
                if st.button("Force re-sync from R2"):
                    st.session_state.data_downloaded_from_r2 = False
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
