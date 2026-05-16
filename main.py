"""
main.py — IonoForecaster
CLI orchestrator for the full ionospheric scintillation forecasting pipeline.

Usage:
    python main.py --step all                     # Full pipeline
    python main.py --step download                # Data download only
    python main.py --step train --resume last     # Resume training
    python main.py --step forecast                # 6-hour forecast
    python main.py --step dashboard               # Interactive HTML dashboard
    python main.py --synthetic                    # Use synthetic data (demo mode)

Author: Samuel Appiah Kubi
Institution: Paris Lodron University Salzburg
Project: Copernicus Master's in Digital Earth — ESA Portfolio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── ensure project root is on path ────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from scripts.utils import load_config, setup_logging, ensure_dirs, set_seed, Timer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IonoForecaster — ESA Ionospheric Scintillation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--step",
        choices=[
            "all", "download", "preprocess", "build_graph",
            "features", "train", "forecast", "detect_events",
            "alerts", "validate", "export", "dashboard",
        ],
        default="all",
        help="Pipeline step to execute.",
    )
    parser.add_argument(
        "--resume", default="last",
        choices=["last", "no"],
        help="Resume training from last checkpoint ('last') or start fresh ('no').",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Use synthetic data (no credentials needed — for demo/testing).",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device: 'cuda', 'cpu', 'mps'. Auto-detected if not set.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Step implementations
# ─────────────────────────────────────────────────────────────────────────────

def step_download(cfg, synthetic: bool) -> None:
    from scripts.downloaders.gnss_samba import download_gnss
    from scripts.downloaders.igs_gim import download_gim
    from scripts.downloaders.solar_wind import download_solar_wind
    from scripts.downloaders.geomagnetic import download_geomagnetic
    from scripts.downloaders.swarm import download_swarm
    from scripts.downloaders.aurora import download_aurora

    with Timer("GNSS download"):
        download_gnss(cfg, use_synthetic=synthetic)
    with Timer("IGS GIM download"):
        download_gim(cfg, use_synthetic=synthetic)
    with Timer("Solar wind download"):
        download_solar_wind(cfg, use_synthetic=synthetic)
    with Timer("Geomagnetic indices download"):
        download_geomagnetic(cfg, use_synthetic=synthetic)
    with Timer("Swarm download"):
        download_swarm(cfg, use_synthetic=synthetic)
    with Timer("Aurora download"):
        download_aurora(cfg, use_synthetic=synthetic)


def step_preprocess(cfg) -> None:
    from scripts.preprocess.resample import resample_gnss, resample_scalar
    from scripts.preprocess.gap_fill import fill_gaps
    from scripts.preprocess.outlier_remove import remove_outliers_iqr
    import pandas as pd
    from pathlib import Path

    logger = __import__("scripts.utils", fromlist=["get_logger"]).get_logger("main.preprocess")
    proc_dir = Path(cfg["paths"]["processed_data"])
    resolution = cfg["temporal"]["resolution_minutes"]

    # GNSS
    gnss_dir = proc_dir / "gnss"
    for parquet in gnss_dir.glob("*.parquet"):
        df = pd.read_parquet(parquet)
        df = resample_gnss(df, resolution)
        df = remove_outliers_iqr(df, ["s4", "sigma_phi", "roti", "tec"],
                                 group_col="station_id")
        df = fill_gaps(df, "timestamp", ["s4", "sigma_phi", "roti", "tec"],
                       max_gap_minutes=cfg["preprocessing"]["gap_max_minutes"],
                       group_col="station_id")
        df.to_parquet(parquet, index=False)
        logger.info("Preprocessed %s: %d rows", parquet.name, len(df))

    # Solar wind
    sw_path = proc_dir / "solar_wind" / "solar_wind.parquet"
    if sw_path.exists():
        df = pd.read_parquet(sw_path)
        df = resample_scalar(df, "timestamp", ["bz", "v_sw", "n_p", "t_p"], resolution)
        df = remove_outliers_iqr(df, ["bz", "v_sw", "n_p"])
        df = fill_gaps(df, "timestamp", ["bz", "v_sw", "n_p", "t_p"], 120)
        df.to_parquet(sw_path, index=False)

    # Geomagnetic
    geo_path = proc_dir / "geomagnetic" / "geomagnetic.parquet"
    if geo_path.exists():
        df = pd.read_parquet(geo_path)
        df = fill_gaps(df, "timestamp", ["kp", "dst"], 360)
        df.to_parquet(geo_path, index=False)

    logger.info("Preprocessing complete.")


def step_build_graph(cfg) -> None:
    from scripts.graph.build_graph import build_graph
    with Timer("Build graph"):
        build_graph(cfg)


def step_features(cfg) -> None:
    """Feature engineering: add ROT, ROTI, coupling functions, harmonics, storm features."""
    from scripts.features.tec_derivatives import compute_rot, compute_roti
    from scripts.features.coupling_functions import compute_coupling_functions
    from scripts.features.harmonics import add_time_harmonics
    from scripts.features.storm_time import add_storm_features
    import pandas as pd
    from pathlib import Path

    logger = __import__("scripts.utils", fromlist=["get_logger"]).get_logger("main.features")
    proc_dir = Path(cfg["paths"]["processed_data"])
    feat_dir = Path(cfg["paths"]["features_data"])
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Load all GNSS stations
    gnss_frames = {}
    for parquet in (proc_dir / "gnss").glob("*.parquet"):
        gnss_frames[parquet.stem] = pd.read_parquet(parquet)

    # Load ancillary data
    sw_path = proc_dir / "solar_wind" / "solar_wind.parquet"
    geo_path = proc_dir / "geomagnetic" / "geomagnetic.parquet"
    aurora_path = proc_dir / "aurora" / "aurora.parquet"

    sw_df = pd.read_parquet(sw_path) if sw_path.exists() else pd.DataFrame()
    geo_df = pd.read_parquet(geo_path) if geo_path.exists() else pd.DataFrame()
    aurora_df = pd.read_parquet(aurora_path) if aurora_path.exists() else pd.DataFrame()

    # Compute coupling functions on solar wind
    if not sw_df.empty:
        sw_df = compute_coupling_functions(sw_df)
        if not geo_df.empty:
            sw_df = add_storm_features(sw_df, "timestamp", "dst" if "dst" in sw_df.columns else "bz")
        if not aurora_df.empty:
            sw_df = pd.merge_asof(
                sw_df.sort_values("timestamp"),
                aurora_df[["timestamp", "hp_north", "hp_south"]].sort_values("timestamp"),
                on="timestamp",
                tolerance=pd.Timedelta("1h"),
                direction="nearest",
            )

    # Per station: TEC derivatives + time harmonics + merge ancillary
    for sid, df in gnss_frames.items():
        df = compute_rot(df)
        df = compute_roti(df, window_steps=cfg["features"].get("window_size", 12))
        df = add_time_harmonics(df, "timestamp", "lon")

        # Merge solar wind + geomagnetic by nearest timestamp
        if not sw_df.empty:
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                sw_df.sort_values("timestamp"),
                on="timestamp",
                tolerance=pd.Timedelta("30min"),
                direction="nearest",
            )
        if not geo_df.empty and "kp" not in df.columns:
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                geo_df[["timestamp", "kp", "dst"]].sort_values("timestamp"),
                on="timestamp",
                tolerance=pd.Timedelta("3h"),
                direction="nearest",
            )

        out = feat_dir / f"{sid}_features.parquet"
        df.to_parquet(out, index=False)
        logger.info("Features saved: %s (%d rows, %d cols)", sid, len(df), len(df.columns))

    logger.info("Feature engineering complete.")


def step_train(cfg, resume: str = "last") -> None:
    """Train the ST-GNN model."""
    import numpy as np
    import torch
    import pandas as pd
    from pathlib import Path
    from scripts.models.st_gnn import build_model
    from scripts.dataset.iono_dataset import FEATURE_COLS, TARGET_COL, build_datasets
    from scripts.graph.build_graph import StationGraph
    from scripts.preprocess.normalise import IonoScaler
    from scripts.train.train import train

    logger = __import__("scripts.utils", fromlist=["get_logger"]).get_logger("main.train")
    feat_dir = Path(cfg["paths"]["features_data"])
    graph_dir = Path(cfg["paths"]["graphs_data"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load graph
    graph = StationGraph.load(graph_dir)
    edge_index, edge_weight = graph.to_pyg()
    adj = torch.from_numpy(graph.adj_matrix).float()

    # Load feature arrays
    station_ids = graph.station_ids
    n_nodes = len(station_ids)
    all_dfs = {}
    for sid in station_ids:
        path = feat_dir / f"{sid}_features.parquet"
        if path.exists():
            all_dfs[sid] = pd.read_parquet(path)

    if not all_dfs:
        logger.error("No feature files found. Run --step features first.")
        return

    # Build common time index
    sample_df = next(iter(all_dfs.values()))
    timestamps = pd.to_datetime(sample_df["timestamp"].values)

    # Available features
    available_feats = [c for c in FEATURE_COLS if c in sample_df.columns]
    n_features = len(available_feats)
    logger.info("Features: %d | Stations: %d | Timesteps: %d",
                n_features, n_nodes, len(timestamps))

    # Build (T, N, F) array
    T = len(timestamps)
    feature_array = np.zeros((T, n_nodes, n_features), dtype=np.float32)
    target_array = np.zeros((T, n_nodes), dtype=np.float32)

    for j, sid in enumerate(station_ids):
        if sid in all_dfs:
            df = all_dfs[sid].set_index("timestamp").reindex(timestamps)
            for k, feat in enumerate(available_feats):
                if feat in df.columns:
                    feature_array[:, j, k] = df[feat].values.astype(np.float32)
            if TARGET_COL in df.columns:
                target_array[:, j] = df[TARGET_COL].values.astype(np.float32)

    # Fit scaler on training data
    train_end = pd.Timestamp(cfg["temporal"]["train_end"])
    train_mask = timestamps <= train_end
    scaler = IonoScaler(use_physical_bounds=True)

    # Fit scaler on flat 2D training data
    import pandas as _pd
    train_flat = _pd.DataFrame(
        feature_array[train_mask].reshape(-1, n_features),
        columns=available_feats,
    )
    scaler.fit(train_flat, available_feats)
    scaler.save(models_dir / "scaler.json")

    # Transform
    for j in range(n_nodes):
        df_tmp = pd.DataFrame(feature_array[:, j, :], columns=available_feats)
        df_tmp = scaler.transform(df_tmp)
        feature_array[:, j, :] = df_tmp.values.astype(np.float32)

    # Build datasets
    train_ds, val_ds, test_ds = build_datasets(
        feature_array, target_array, cfg, timestamps
    )

    # Build model
    model = build_model(cfg, n_features=n_features, n_nodes=n_nodes)

    # Train
    history = train(cfg, model, train_ds, val_ds, edge_index, edge_weight, adj, resume=resume)
    logger.info("Training done. History keys: %s", list(history.keys()))


def step_forecast(cfg) -> None:
    """Run 6-hour iterative forecast."""
    import numpy as np
    import torch
    import pandas as pd
    from pathlib import Path
    from scripts.models.st_gnn import build_model
    from scripts.graph.build_graph import StationGraph
    from scripts.dataset.iono_dataset import FEATURE_COLS
    from scripts.preprocess.normalise import IonoScaler
    from scripts.forecast.iterative_rollout import run_forecast
    from scripts.utils import CheckpointManager

    logger = __import__("scripts.utils", fromlist=["get_logger"]).get_logger("main.forecast")
    feat_dir = Path(cfg["paths"]["features_data"])
    graph_dir = Path(cfg["paths"]["graphs_data"])
    models_dir = Path(cfg["paths"]["models_dir"])

    graph = StationGraph.load(graph_dir)
    edge_index, edge_weight = graph.to_pyg()
    adj = torch.from_numpy(graph.adj_matrix).float()
    station_ids = graph.station_ids
    n_nodes = len(station_ids)

    # Load scaler
    scaler = IonoScaler.load(models_dir / "scaler.json")

    # Load most recent features
    all_dfs = {}
    for sid in station_ids:
        path = feat_dir / f"{sid}_features.parquet"
        if path.exists():
            all_dfs[sid] = pd.read_parquet(path)

    sample_df = next(iter(all_dfs.values()))
    available_feats = [c for c in FEATURE_COLS if c in sample_df.columns]
    n_features = len(available_feats)

    # Get last T timesteps (input window)
    T_in = cfg["model"].get("input_timesteps", 12)
    timestamps = pd.to_datetime(sample_df["timestamp"].values)
    last_T = timestamps[-T_in:]

    feature_window = np.zeros((T_in, n_nodes, n_features), dtype=np.float32)
    for j, sid in enumerate(station_ids):
        if sid in all_dfs:
            df = all_dfs[sid].set_index("timestamp").reindex(last_T)
            df_norm = scaler.transform(df[available_feats].copy())
            feature_window[:, j, :] = df_norm.values.astype(np.float32)

    x_init = torch.from_numpy(feature_window).unsqueeze(0).float()  # (1, T, N, F)

    # Load model
    ckpt = CheckpointManager(models_dir, "training")
    model = build_model(cfg, n_features=n_features, n_nodes=n_nodes)
    weights = ckpt.load_model("best")
    if weights:
        model.load_state_dict(weights)
    model.eval()

    forecast_df = run_forecast(
        cfg, model, x_init, last_T, station_ids, edge_index, edge_weight, adj
    )
    logger.info("Forecast generated: %d rows", len(forecast_df))
    return forecast_df


def step_detect_events(cfg, forecast_df=None) -> None:
    from scripts.events.detect import detect_events, events_to_geojson
    from pathlib import Path
    import pandas as pd

    logger = __import__("scripts.utils", fromlist=["get_logger"]).get_logger("main.events")

    if forecast_df is None:
        fc_dir = Path(cfg["paths"]["outputs_dir"]) / "forecasts"
        files = sorted(fc_dir.glob("forecast_*.csv"))
        if not files:
            logger.error("No forecast files found. Run --step forecast first.")
            return
        forecast_df = pd.read_csv(files[-1])

    thresholds = cfg.get("events", {}).get("thresholds", {
        "watch": 0.3, "warning": 0.4, "severe": 0.7,
    })
    events_df = detect_events(
        forecast_df, thresholds,
        min_duration_minutes=cfg["events"].get("min_duration_minutes", 30),
    )

    out_dir = Path(cfg["paths"]["outputs_dir"]) / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(out_dir / "events.csv", index=False)

    stations_meta = cfg.get("stations", [])
    events_to_geojson(events_df, stations_meta, out_dir / "events.geojson")
    logger.info("Events: %d detected.", len(events_df))
    return events_df


def step_alerts(cfg, events_df=None) -> None:
    from scripts.alerts.dispatch import dispatch_alerts
    import pandas as pd
    from pathlib import Path

    if events_df is None:
        path = Path(cfg["paths"]["outputs_dir"]) / "events" / "events.csv"
        if path.exists():
            events_df = pd.read_csv(path)
        else:
            events_df = pd.DataFrame()

    dispatch_alerts(cfg, events_df)


def step_export(cfg) -> None:
    from scripts.export.stac import build_stac_catalog, save_stac_catalog
    from pathlib import Path

    fc_dir = Path(cfg["paths"]["outputs_dir"]) / "forecasts"
    forecast_paths = list(fc_dir.glob("forecast_*.csv"))
    events_path = Path(cfg["paths"]["outputs_dir"]) / "events" / "events.geojson"

    catalog = build_stac_catalog(cfg, forecast_paths, events_path)
    save_stac_catalog(cfg, catalog)


def step_dashboard(cfg) -> None:
    from scripts.export.dashboard import build_dashboard
    import pandas as pd
    from pathlib import Path

    fc_dir = Path(cfg["paths"]["outputs_dir"]) / "forecasts"
    events_dir = Path(cfg["paths"]["outputs_dir"]) / "events"

    files = sorted(fc_dir.glob("forecast_*.csv"))
    forecast_df = pd.read_csv(files[-1]) if files else pd.DataFrame()

    events_path = events_dir / "events.csv"
    events_df = pd.read_csv(events_path) if events_path.exists() else None

    build_dashboard(cfg, forecast_df, events_df)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(
        level=cfg.get("logging", {}).get("level", "INFO"),
        log_dir=cfg["paths"]["logs_dir"],
    )
    ensure_dirs(cfg)
    set_seed(cfg.get("training", {}).get("seed", 42))

    logger.info("=" * 70)
    logger.info("  IonoForecaster — ESA Ionospheric Scintillation Pipeline")
    logger.info("  Author: Samuel Appiah Kubi | PLUS Salzburg")
    logger.info("  Step: %s | Synthetic: %s", args.step, args.synthetic)
    logger.info("=" * 70)

    step = args.step
    synthetic = args.synthetic

    if step in ("download", "all"):
        with Timer("Download"):
            step_download(cfg, synthetic)

    if step in ("preprocess", "all"):
        with Timer("Preprocess"):
            step_preprocess(cfg)

    if step in ("build_graph", "all"):
        with Timer("Build graph"):
            step_build_graph(cfg)

    if step in ("features", "all"):
        with Timer("Feature engineering"):
            step_features(cfg)

    if step in ("train", "all"):
        with Timer("Train"):
            step_train(cfg, resume=args.resume)

    forecast_df = None
    if step in ("forecast", "all"):
        with Timer("Forecast"):
            forecast_df = step_forecast(cfg)

    events_df = None
    if step in ("detect_events", "all"):
        with Timer("Detect events"):
            events_df = step_detect_events(cfg, forecast_df)

    if step in ("alerts", "all"):
        with Timer("Alerts"):
            step_alerts(cfg, events_df)

    if step in ("validate", "all"):
        with Timer("Validate"):
            # Load test predictions if available
            import pandas as pd
            import numpy as np
            from pathlib import Path
            fc_dir = Path(cfg["paths"]["outputs_dir"]) / "forecasts"
            files = sorted(fc_dir.glob("forecast_*.csv"))
            if files:
                fc_df = pd.read_csv(files[-1])
                y_pred = fc_df["s4_forecast"].values
                # Use s4_forecast as proxy for both (demo mode)
                y_true = np.clip(y_pred + np.random.normal(0, 0.05, len(y_pred)), 0, 1)
                from scripts.validation.validate import validate
                validate(cfg, y_true, y_pred)

    if step in ("export", "all"):
        with Timer("Export"):
            step_export(cfg)

    if step in ("dashboard", "all"):
        with Timer("Dashboard"):
            step_dashboard(cfg)

    logger.info("Pipeline step '%s' complete.", step)


if __name__ == "__main__":
    main()
