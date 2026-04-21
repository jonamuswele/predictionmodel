
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Import the full pipeline from the existing module. We do NOT modify it.
import flood_forecast_nigeria as ffn


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nigeria Flood Forecast",
    page_icon="[water]",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
    .main > div { padding-top: 1rem; }
    .alert-red    { background:#fadbd8; padding:8px 14px; border-radius:6px;
                     border-left:6px solid #C0392B; margin-bottom:6px; }
    .alert-amber  { background:#fdebd0; padding:8px 14px; border-radius:6px;
                     border-left:6px solid #F39C12; margin-bottom:6px; }
    .alert-green  { background:#d5f5e3; padding:8px 14px; border-radius:6px;
                     border-left:6px solid #27AE60; margin-bottom:6px; }
    .metric-card  { background:#F8F9F9; padding:14px 16px; border-radius:8px;
                     box-shadow:0 1px 3px rgba(0,0,0,0.08); }
    .stButton button { width: 100%; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ===========================================================================
#                         WebApp orchestration class
# ===========================================================================
class FloodForecastWebApp:
    """Self-contained Streamlit front-end for FloodForecastSystem."""

    ALERT_COLORS = {"RED": "#C0392B", "AMBER": "#F39C12", "GREEN": "#27AE60"}

    # ------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialise session state keys used across reruns."""
        defaults: Dict[str, Any] = {
            "pipeline_ready": False,
            "system": None,
            "workdir": None,
            "gauge_path": None,
            "dem_path": None,
            "met_path": None,
            "forecast_report": None,
            "forecast_date": None,
            "forecast_df": None,
            "last_error": None,
            "run_log": [],
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _workdir(self) -> Path:
        """Return (and lazily create) a per-session working directory."""
        if st.session_state.workdir is None:
            tmp = Path(tempfile.mkdtemp(prefix="flood_web_"))
            (tmp / "data" / "gauges").mkdir(parents=True, exist_ok=True)
            (tmp / "data" / "dem").mkdir(parents=True, exist_ok=True)
            (tmp / "data" / "met").mkdir(parents=True, exist_ok=True)
            (tmp / "data" / "chirps").mkdir(parents=True, exist_ok=True)
            (tmp / "outputs").mkdir(parents=True, exist_ok=True)
            st.session_state.workdir = str(tmp)
        return Path(st.session_state.workdir)

    def _save_to_r2(self, system) -> None:
        """Save results to R2 if cloud storage is configured"""
        if not system.config.use_cloud_storage:
            return
        
        try:
            from cloud_storage import R2Storage
            
            storage = R2Storage(
                endpoint_url=system.config.r2_endpoint_url,
                access_key=system.config.r2_access_key,
                secret_key=system.config.r2_secret_key,
                bucket_name=system.config.r2_bucket_name
            )
            
            session_id = st.session_state.get("session_id", "latest")
            output_dir = Path(system.config.output_dir)
            
            # Upload flood map
            map_file = output_dir / f"flood_map_{st.session_state.forecast_date}.html"
            if map_file.exists():
                url = storage.upload_file(map_file, f"{session_id}/flood_map.html")
                st.success(f"✅ Map saved to cloud: {url}")
            
            # Upload alert report
            alert_file = output_dir / f"alert_report_{st.session_state.forecast_date}.json"
            if alert_file.exists():
                storage.upload_file(alert_file, f"{session_id}/alert_report.json")
                st.success(f"✅ Alert report saved to cloud")
            
            # Upload all station forecast plots
            for png in output_dir.glob("*_forecast.png"):
                storage.upload_file(png, f"{session_id}/plots/{png.name}")
            
            # Upload routed discharge CSVs
            for sid, df in system.routed_Q.items():
                csv_path = output_dir / f"{sid}_routed.csv"
                df.to_csv(csv_path)
                storage.upload_file(csv_path, f"{session_id}/data/{sid}_routed.csv")
                
        except Exception as e:
            st.warning(f"Cloud save failed: {e}")

    # ------------------------------------------------------------------
    def _reset_workdir(self) -> None:
        """Clear any cached pipeline state and remove the workdir tree."""
        if st.session_state.workdir and Path(st.session_state.workdir).exists():
            try:
                shutil.rmtree(st.session_state.workdir)
            except Exception:
                pass
        st.session_state.workdir = None
        st.session_state.pipeline_ready = False
        st.session_state.system = None
        st.session_state.gauge_path = None
        st.session_state.dem_path = None
        st.session_state.met_path = None
        st.session_state.forecast_report = None
        st.session_state.forecast_df = None
        st.session_state.last_error = None
        st.session_state.run_log = []

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        """Append a user-visible log line for the next rerun."""
        st.session_state.run_log.append(msg)

    # ------------------------------------------------------------------
    @staticmethod
    def _template_gauge_csv() -> bytes:
        """Return an example gauge-station CSV (bytes) for the user to download."""
        idx = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        rng = np.random.default_rng(0)
        doy = np.array([d.timetuple().tm_yday for d in idx])
        rows = []
        sample_stations = [
            ("NIG_LOK", "Lokoja", 7.80, 6.74, "Niger", 6000.0),
            ("BEN_MAK", "Makurdi", 7.73, 8.54, "Benue", 2500.0),
            ("OSU_OSO", "Oshogbo", 7.76, 4.56, "Osun", 120.0),
        ]
        seasonal = 1.0 + 1.3 * np.sin(2 * np.pi * (doy - 150) / 365.25)
        for sid, name, lat, lon, river, mean_q in sample_stations:
            q = np.clip(mean_q * seasonal * rng.normal(1.0, 0.25, len(idx)), 1.0, None)
            stage = 1.0 + np.log1p(q / mean_q) * 1.5
            for d, Q, H in zip(idx, q, stage):
                rows.append({
                    "station_id": sid, "station_name": name,
                    "latitude": lat, "longitude": lon, "river": river,
                    "date": d.strftime("%Y-%m-%d"),
                    "discharge_m3s": round(float(Q), 2),
                    "stage_m": round(float(H), 3),
                })
        buf = io.StringIO()
        pd.DataFrame(rows).to_csv(buf, index=False)
        return buf.getvalue().encode("utf-8")

    # ------------------------------------------------------------------
    @staticmethod
    def _template_met_csv() -> bytes:
        """Return an example merged meteorology CSV."""
        idx = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        rng = np.random.default_rng(1)
        doy = np.array([d.timetuple().tm_yday for d in idx])
        wet = np.isin(np.array([d.month for d in idx]), [5, 6, 7, 8, 9, 10])
        precip = np.where(wet,
                           rng.gamma(0.8, 12.0, len(idx)),
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

    # ------------------------------------------------------------------
    @staticmethod
    def _template_nwp_csv(horizon_days: int, station_ids: Optional[List[str]] = None) -> bytes:
        """Return an example NWP precipitation CSV for the given horizon."""
        idx = pd.date_range(pd.Timestamp.today().normalize() + pd.Timedelta(days=1),
                             periods=horizon_days, freq="D")
        rng = np.random.default_rng(2)
        p = np.clip(rng.gamma(1.0, 10.0, horizon_days), 0, None)
        if station_ids:
            data = {sid: np.clip(p + rng.normal(0, 2, horizon_days), 0, None)
                    for sid in station_ids}
        else:
            data = {"area_mean": p}
        df = pd.DataFrame(data, index=idx)
        df.index.name = "date"
        buf = io.StringIO()
        df.to_csv(buf)
        return buf.getvalue().encode("utf-8")

    # ------------------------------------------------------------------
    def _save_uploaded_file(self, uploaded: Any, dest: Path) -> Path:
        """Persist a Streamlit UploadedFile to disk and return the path."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(uploaded.getvalue())
        return dest

    # ------------------------------------------------------------------
    def _build_config(self, start_date: str, end_date: str,
                       forecast_horizon_days: int) -> ffn.Config:
        """Construct a Config whose paths live inside the per-session workdir."""
        wd = self._workdir()
        cfg = ffn.Config(
            data_dir=wd / "data",
            dem_path=wd / "data" / "dem" / "dem.tif",
            gauge_csv=wd / "data" / "gauges" / "gauge_stations.csv",
            output_dir=wd / "outputs",
            start_date=start_date,
            end_date=end_date,
            forecast_horizon_days=int(forecast_horizon_days),
        )
        return cfg

    # ------------------------------------------------------------------
    def _ensure_gauge_csv(self, uploaded_gauge: Any, cfg: ffn.Config) -> Path:
        """Place the user-uploaded or synthetic gauge CSV at cfg.gauge_csv."""
        if uploaded_gauge is not None:
            path = self._save_uploaded_file(uploaded_gauge, Path(cfg.gauge_csv))
            self._log(f"Saved uploaded gauge CSV ({path.stat().st_size/1024:.1f} KB).")
            return path
        # Synthetic fallback via FloodForecastSystem helper (reused, not duplicated).
        system = ffn.FloodForecastSystem(cfg)
        system._write_synthetic_gauge_csv()
        self._log("No gauge CSV uploaded - using synthetic 10-year demo record.")
        return Path(cfg.gauge_csv)

    # ------------------------------------------------------------------
    def _ensure_dem(self, uploaded_dem: Any, cfg: ffn.Config) -> Path:
        """Place the user-uploaded DEM GeoTIFF or build the synthetic one."""
        if uploaded_dem is not None:
            path = self._save_uploaded_file(uploaded_dem, Path(cfg.dem_path))
            self._log(f"Saved uploaded DEM ({path.stat().st_size/1024/1024:.1f} MB).")
            return path
        delin = ffn.WatershedDelineator(cfg)
        delin.generate_synthetic_dem()
        self._log("No DEM uploaded - generated synthetic Nigeria DEM.")
        return Path(cfg.dem_path)

    # ------------------------------------------------------------------
    def _ensure_met(self, uploaded_met: Any, cfg: ffn.Config) -> Optional[pd.DataFrame]:
        """Return a merged met DataFrame from upload, or None to let the pipeline synthesise."""
        if uploaded_met is None:
            return None
        try:
            df = pd.read_csv(uploaded_met)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            required = {"precip_mm", "temp_c", "pet_mm"}
            if not required.issubset(df.columns):
                self._log(f"Met CSV missing columns {required - set(df.columns)} - ignoring.")
                return None
            self._log(f"Loaded user met CSV with {len(df):,} rows.")
            return df
        except Exception as exc:
            self._log(f"Could not parse met CSV: {exc}")
            return None

    # ------------------------------------------------------------------
    def _build_nwp(self, uploaded_nwp: Any, horizon_days: int, start: pd.Timestamp,
                    station_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (precip_df, pet_df) for the forecast horizon."""
        idx = pd.date_range(start, periods=horizon_days, freq="D")
        if uploaded_nwp is not None:
            try:
                df = pd.read_csv(uploaded_nwp)
                date_col = next((c for c in df.columns if c.lower() == "date"), None)
                if date_col is None:
                    raise ValueError("NWP CSV needs a 'date' column")
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()
                df = df.reindex(idx).ffill().bfill()
                # Accept either per-station columns or a single area_mean column.
                precip_cols = [c for c in df.columns if c.lower() != "pet_mm"]
                p_df = df[precip_cols]
                # PET column optional; synthesise if missing.
                if "pet_mm" in df.columns:
                    e_df = pd.DataFrame({"area_mean": df["pet_mm"]}, index=idx)
                else:
                    e_df = pd.DataFrame(
                        {"area_mean": np.full(horizon_days, 4.5)}, index=idx)
                self._log(f"Loaded uploaded NWP CSV for {horizon_days} days.")
                return p_df, e_df
            except Exception as exc:
                self._log(f"NWP CSV parse failed ({exc}) - synthesising instead.")

        rng = np.random.default_rng(11)
        precip = np.clip(rng.gamma(1.0, 10.0, horizon_days), 0.0, None)
        # Inject a modest pulse so forecasts are visually non-trivial.
        if horizon_days >= 3:
            precip[2] += 35.0
            precip[min(3, horizon_days - 1)] += 20.0
        pet = np.clip(rng.normal(4.5, 0.8, horizon_days), 0.5, None)
        p_df = pd.DataFrame({"area_mean": precip}, index=idx)
        e_df = pd.DataFrame({"area_mean": pet}, index=idx)
        p_df.index.name = e_df.index.name = "date"
        return p_df, e_df

    # ------------------------------------------------------------------
    # Pipeline runners
    # ------------------------------------------------------------------
    def run_historical(self, uploaded_gauge: Any, uploaded_dem: Any,
                        uploaded_met: Any, start_date: str, end_date: str,
                        horizon_days: int) -> Optional[ffn.FloodForecastSystem]:
        """Execute hindcast through the full FloodForecastSystem pipeline."""
        cfg = self._build_config(start_date, end_date, horizon_days)
        if hasattr(st, 'secrets') and 'r2' in st.secrets:
            cfg.use_cloud_storage = True
            cfg.r2_endpoint_url = st.secrets["r2"]["endpoint_url"]
            cfg.r2_access_key = st.secrets["r2"]["access_key_id"]
            cfg.r2_secret_key = st.secrets["r2"]["secret_access_key"]
            cfg.r2_bucket_name = st.secrets["r2"]["bucket_name"]
        self._ensure_gauge_csv(uploaded_gauge, cfg)
        self._ensure_dem(uploaded_dem, cfg)
        user_met = self._ensure_met(uploaded_met, cfg)

        system = ffn.FloodForecastSystem(cfg)

        # If the user supplied met data, monkey-patch the DataManager loader
        # for this run so the pipeline uses it verbatim. We do NOT modify the
        # original class - just instance-level attribute assignment below.
        if user_met is not None:
            user_met_clipped = user_met.reindex(
                pd.date_range(start_date, end_date, freq="D")
            ).interpolate(limit=14).ffill().bfill()

            original_run_historical = system.run_historical

            def _patched_run_historical():
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

            try:
                _patched_run_historical()
            except Exception as exc:  # pragma: no cover
                st.session_state.last_error = f"Hindcast failed: {exc}\n{traceback.format_exc()}"
                return None
        else:
            try:
                system.run_historical()
            except Exception as exc:  # pragma: no cover
                st.session_state.last_error = f"Hindcast failed: {exc}\n{traceback.format_exc()}"
                return None

        self._log("Hindcast complete - ready to forecast.")
        return system

    # ------------------------------------------------------------------
    def run_forecast(self, system: ffn.FloodForecastSystem,
                      uploaded_nwp: Any, forecast_start: pd.Timestamp,
                      horizon_days: int) -> Optional[Dict[str, Dict[str, Any]]]:
        """Run the forecast stage; returns the alert report dict."""
        # Override horizon for this run (Config is mutable).
        system.config.forecast_horizon_days = int(horizon_days)
        station_ids = list(system.Q_per_basin.keys())
        p_df, e_df = self._build_nwp(uploaded_nwp, horizon_days, forecast_start, station_ids)
        try:
            report = system.run_forecast(p_df, e_df)
        except Exception as exc:  # pragma: no cover
            st.session_state.last_error = f"Forecast failed: {exc}\n{traceback.format_exc()}"
            return None
        self._log(f"Forecast complete for {horizon_days} days starting {forecast_start.date()}.")
        self._save_to_r2(system)

        return report

    # ------------------------------------------------------------------
    # UI sections
    # ------------------------------------------------------------------
    def render_sidebar(self) -> Dict[str, Any]:
        """Render the control sidebar and return collected parameters."""
        with st.sidebar:
            st.header("1. Historical period")
            start = st.date_input("Start date",
                                    value=pd.to_datetime("2020-01-01").date(),
                                    min_value=pd.to_datetime("1990-01-01").date(),
                                    max_value=pd.to_datetime("2030-12-31").date())
            end = st.date_input("End date",
                                  value=pd.to_datetime("2024-12-31").date(),
                                  min_value=start,
                                  max_value=pd.to_datetime("2030-12-31").date())

            st.header("2. Forecast horizon")
            unit = st.radio("Unit", ["Days", "Weeks", "Months"], index=1,
                             horizontal=True)
            if unit == "Days":
                horizon_val = st.slider("Forecast length (days)", 1, 60, 14)
                horizon_days = int(horizon_val)
            elif unit == "Weeks":
                horizon_val = st.slider("Forecast length (weeks)", 1, 8, 2)
                horizon_days = int(horizon_val * 7)
            else:
                horizon_val = st.slider("Forecast length (months)", 1, 3, 1)
                horizon_days = int(horizon_val * 30)
            st.caption(f"Horizon = {horizon_days} day(s)")

            st.header("3. Forecast start")
            default_start = pd.to_datetime(end) + pd.Timedelta(days=1)
            fc_start = st.date_input("Forecast start date",
                                       value=default_start.date(),
                                       min_value=pd.to_datetime(end).date())

            st.header("4. Data uploads")
            gauge_file = st.file_uploader(
                "Gauge stations CSV",
                type=["csv"],
                help="Columns: station_id, station_name, latitude, longitude, "
                     "river, date, discharge_m3s, stage_m")
            dem_file = st.file_uploader(
                "DEM (GeoTIFF, optional)",
                type=["tif", "tiff"],
                help="Single-band elevation raster in WGS84. If omitted, a "
                     "synthetic DEM is used.")
            met_file = st.file_uploader(
                "Meteorology CSV (optional)",
                type=["csv"],
                help="Columns: date, precip_mm, temp_c, pet_mm. If omitted, "
                     "CHIRPS/ERA5 or synthetic data are used.")
            nwp_file = st.file_uploader(
                "Forecast precipitation CSV (optional)",
                type=["csv"],
                help="Columns: date plus either 'area_mean' or one column per "
                     "station_id. A 'pet_mm' column is optional.")

            st.header("5. Templates")
            st.download_button(
                "Gauge CSV template",
                data=self._template_gauge_csv(),
                file_name="gauge_template.csv",
                mime="text/csv")
            st.download_button(
                "Met CSV template",
                data=self._template_met_csv(),
                file_name="met_template.csv",
                mime="text/csv")
            st.download_button(
                "NWP CSV template",
                data=self._template_nwp_csv(horizon_days),
                file_name="nwp_template.csv",
                mime="text/csv")

            st.header("6. Actions")
            run_clicked = st.button("Run pipeline", type="primary")
            reset_clicked = st.button("Reset session")

            if reset_clicked:
                self._reset_workdir()
                st.rerun()

        return {
            "start_date": str(start),
            "end_date": str(end),
            "horizon_days": horizon_days,
            "forecast_start": pd.to_datetime(fc_start),
            "gauge_file": gauge_file,
            "dem_file": dem_file,
            "met_file": met_file,
            "nwp_file": nwp_file,
            "run_clicked": run_clicked,
        }

    # ------------------------------------------------------------------
    def render_header(self) -> None:
        """Title + brief description banner."""
        st.title("Nigeria Flood Forecast Dashboard")
        st.markdown(
            "Upload your own gauge records, DEM and meteorology, pick a "
            "forecast horizon (day / week / month) and get routed-discharge "
            "forecasts, inundation maps and colour-coded alerts across every "
            "sub-basin.")

    # ------------------------------------------------------------------
    def render_metrics(self, report: Dict[str, Dict[str, Any]]) -> None:
        """Top metric row showing alert counts and peak discharge."""
        if not report:
            return
        red = sum(1 for r in report.values() if r["alert_level"] == "RED")
        amber = sum(1 for r in report.values() if r["alert_level"] == "AMBER")
        green = sum(1 for r in report.values() if r["alert_level"] == "GREEN")
        peak = max(r["peak_Q_m3s"] for r in report.values())
        cols = st.columns(4)
        cols[0].metric("RED alerts", red)
        cols[1].metric("AMBER alerts", amber)
        cols[2].metric("GREEN alerts", green)
        cols[3].metric("Peak forecast Q (m^3/s)", f"{peak:,.0f}")

    # ------------------------------------------------------------------
    def render_map_tab(self, system: ffn.FloodForecastSystem,
                        forecast_date: str) -> None:
        """Render the folium map tab."""
        st.subheader("Inundation & station map")
        map_path = Path(system.config.output_dir) / f"flood_map_{forecast_date}.html"
        if not map_path.exists() and system.alert_dashboard is not None:
            try:
                system.alert_dashboard.generate_folium_map(forecast_date)
            except Exception as exc:
                st.warning(f"Could not regenerate map: {exc}")
        if map_path.exists():
            try:
                with open(map_path, "r", encoding="utf-8") as fh:
                    html = fh.read()
                components.html(html, height=650, scrolling=True)
                st.download_button(
                    "Download map HTML",
                    data=html.encode("utf-8"),
                    file_name=map_path.name,
                    mime="text/html")
            except Exception as exc:
                st.error(f"Failed to load map: {exc}")
        else:
            st.info("Map not yet generated.")

    # ------------------------------------------------------------------
    def render_plot_tab(self, system: ffn.FloodForecastSystem,
                         selected_stations: List[str]) -> None:
        """Render per-station forecast PNGs."""
        st.subheader("Per-station discharge forecasts")
        if not selected_stations:
            st.info("Select one or more basins in the sidebar filter below.")
            return
        for sid in selected_stations:
            img = Path(system.config.output_dir) / f"{sid}_forecast.png"
            if img.exists():
                st.image(str(img), caption=sid, use_column_width=True)
            else:
                st.write(f"No forecast plot produced for {sid}")

    # ------------------------------------------------------------------
    def render_alert_tab(self, report: Dict[str, Dict[str, Any]]) -> None:
        """Coloured alert table + individual alert banners."""
        st.subheader("Alert table")
        if not report:
            st.info("No alerts yet.")
            return
        df = pd.DataFrame.from_dict(report, orient="index").reset_index()
        df = df.rename(columns={"index": "station_id"})
        numeric_cols = [c for c in df.columns
                         if df[c].dtype.kind in "fc" and c not in ("station_id",)]
        for c in numeric_cols:
            df[c] = df[c].astype(float).round(2)

        def color_row(row):
            c = {"RED": "#fadbd8", "AMBER": "#fdebd0",
                 "GREEN": "#d5f5e3"}.get(row["alert_level"], "white")
            return [f"background-color: {c}"] * len(row)

        try:
            styled = df.style.apply(color_row, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download alerts JSON",
            data=json.dumps(report, indent=2, default=str).encode("utf-8"),
            file_name="alert_report.json",
            mime="application/json")

        st.markdown("#### Individual alerts")
        for sid, rec in sorted(report.items(),
                                 key=lambda kv: ["RED", "AMBER", "GREEN"].index(kv[1]["alert_level"])):
            lvl = rec["alert_level"]
            cls = {"RED": "alert-red", "AMBER": "alert-amber", "GREEN": "alert-green"}[lvl]
            pct = rec.get("Q_pct_of_2yr", float("nan"))
            pct_str = f"{pct:.1f}%" if np.isfinite(pct) else "n/a"
            st.markdown(
                f'<div class="{cls}"><b>{sid}</b> ({rec["river"]}) - '
                f'<b>{lvl}</b>, peak {rec["peak_Q_m3s"]:.1f} m^3/s on '
                f'{rec["peak_date"][:10]} ({pct_str} of 2-yr RP)</div>',
                unsafe_allow_html=True)

    # ------------------------------------------------------------------
    def render_data_tab(self, system: ffn.FloodForecastSystem) -> None:
        """Raw-data exports and return-period tables."""
        st.subheader("Raw data export")
        if not system.routed_Q:
            st.info("Run the pipeline first.")
            return
        station = st.selectbox("Station", list(system.routed_Q.keys()),
                                 key="export_station")
        tabs = st.tabs(["Routed discharge", "Forecast", "Return periods"])
        with tabs[0]:
            df = system.routed_Q[station]
            st.line_chart(df["Q_routed_m3s"].tail(365), height=260)
            st.download_button(
                f"Download routed Q for {station} (CSV)",
                data=df.to_csv().encode("utf-8"),
                file_name=f"{station}_routed_Q.csv",
                mime="text/csv")
        with tabs[1]:
            fc = (system.alert_dashboard.forecasts.get(station)
                   if system.alert_dashboard else None)
            if fc is not None and len(fc) > 0:
                st.dataframe(fc, use_container_width=True)
                st.download_button(
                    f"Download forecast for {station} (CSV)",
                    data=fc.to_csv().encode("utf-8"),
                    file_name=f"{station}_forecast.csv",
                    mime="text/csv")
            else:
                st.info("No forecast data for this station.")
        with tabs[2]:
            rp = system.return_periods.get(station)
            if rp is not None and len(rp) > 0:
                st.dataframe(rp, use_container_width=True, hide_index=True)
                st.download_button(
                    f"Download return periods for {station} (CSV)",
                    data=rp.to_csv(index=False).encode("utf-8"),
                    file_name=f"{station}_return_periods.csv",
                    mime="text/csv")

    # ------------------------------------------------------------------
    def render_log_tab(self) -> None:
        """Show the per-run log and any error traceback."""
        st.subheader("Run log")
        if st.session_state.run_log:
            st.code("\n".join(st.session_state.run_log), language="text")
        else:
            st.info("Nothing logged yet.")
        if st.session_state.last_error:
            st.error(st.session_state.last_error)

    # ------------------------------------------------------------------
    def render_about_tab(self) -> None:
        """Static "About / Help" tab."""
        st.markdown("""
        ### How it works

        1. **Upload** your gauge CSV, optional DEM and optional met CSV in the sidebar.
        2. **Pick a forecast horizon** (any number of days, weeks or months, up to
           60 days) and a start date.
        3. **Click "Run pipeline"**. The app will, in order:
           - load and QC your gauge data,
           - fill any stations with < 3 years of record,
           - delineate watersheds from the DEM,
           - calibrate a GR4J model per basin against your observations (using
             `scipy.optimize.differential_evolution` and Kling-Gupta Efficiency),
           - route forecast discharges with Muskingum channel routing,
           - build a RiverREM-based inundation map,
           - classify each station as RED / AMBER / GREEN against its mean
             annual flood.
        4. **Inspect results** in the four tabs above.

        ### Expected CSV formats
        - **Gauge CSV** — `station_id, station_name, latitude, longitude, river,
          date, discharge_m3s, stage_m`
        - **Meteorology CSV** — `date, precip_mm, temp_c, pet_mm`
        - **NWP CSV** — `date` plus either `area_mean` or one column per
          `station_id`; an optional `pet_mm` column is used if present.

        ### Notes
        - Without a Copernicus CDS API key, the app falls back to a bundled
          synthetic meteorology model (seasonal gamma rainfall, sinusoidal
          temperature, Hargreaves PET).
        - CHIRPS downloads are attempted when no met CSV is supplied, but
          silently fall back to synthetic data if the server is unreachable.
        - All computation is delegated to the eight classes in
          `flood_forecast_nigeria.py` - this web app is pure UI + I/O.
        """)

    # ------------------------------------------------------------------
    def render(self) -> None:
        """Main entry: wire up sidebar + main panel for a single rerun."""
        self.render_header()
        params = self.render_sidebar()

        if params["run_clicked"]:
            # Validate horizon against start date to avoid empty forecasts.
            horizon_days = int(params["horizon_days"])
            if horizon_days < 1:
                st.error("Forecast horizon must be at least 1 day.")
                return
            if horizon_days > 60:
                st.warning("Horizons > 60 days are extrapolative and unreliable "
                            "given the underlying GR4J state - proceeding anyway.")

            with st.spinner("Running hindcast and calibrating GR4J... this may take a minute."):
                system = self.run_historical(
                    params["gauge_file"], params["dem_file"], params["met_file"],
                    params["start_date"], params["end_date"], horizon_days)
            if system is None:
                st.error(st.session_state.last_error or "Hindcast failed.")
                return

            with st.spinner(f"Running {horizon_days}-day forecast..."):
                report = self.run_forecast(system, params["nwp_file"],
                                             params["forecast_start"], horizon_days)
            if report is None:
                st.error(st.session_state.last_error or "Forecast failed.")
                return

            st.session_state.system = system
            st.session_state.forecast_report = report
            st.session_state.forecast_date = params["forecast_start"].strftime("%Y-%m-%d")
            st.session_state.pipeline_ready = True

        if not st.session_state.pipeline_ready:
            st.info("Configure the sidebar and press **Run pipeline** to get started. "
                     "You can run a full demo with zero uploads.")
            self.render_about_tab()
            return

        system: ffn.FloodForecastSystem = st.session_state.system
        report: Dict[str, Dict[str, Any]] = st.session_state.forecast_report
        forecast_date: str = st.session_state.forecast_date

        self.render_metrics(report)

        all_stations = list(system.routed_Q.keys())
        level_filter = st.multiselect(
            "Filter alert levels",
            ["RED", "AMBER", "GREEN"], default=["RED", "AMBER", "GREEN"])
        visible = [sid for sid, rec in report.items()
                    if rec["alert_level"] in level_filter]
        station_filter = st.multiselect(
            "Stations", all_stations, default=visible or all_stations)

        filtered_report = {sid: rec for sid, rec in report.items()
                            if sid in station_filter and rec["alert_level"] in level_filter}

        tabs = st.tabs(["Map", "Forecast plots", "Alert table", "Raw data", "Log", "About"])
        with tabs[0]:
            self.render_map_tab(system, forecast_date)
        with tabs[1]:
            self.render_plot_tab(system, station_filter)
        with tabs[2]:
            self.render_alert_tab(filtered_report)
        with tabs[3]:
            self.render_data_tab(system)
        with tabs[4]:
            self.render_log_tab()
        with tabs[5]:
            self.render_about_tab()


# ===========================================================================
#                              Entry point
# ===========================================================================
def main() -> None:
    """Streamlit scripts rerun top-to-bottom; instantiate and render each time."""
    app = FloodForecastWebApp()
    app.render()


if __name__ == "__main__":
    main()
else:
    # Streamlit imports the script under `__mp_main__`, not `__main__`, so we
    # trigger the render on module load as well.
    main()
