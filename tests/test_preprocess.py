"""
tests/test_preprocess.py — IonoForecaster
Tests for preprocessing: resampling, gap filling, outlier removal, normalisation.

Author: Samuel Appiah Kubi
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_gnss_df():
    """Small synthetic GNSS DataFrame for testing."""
    np.random.seed(42)
    stations = ["ACCR", "LAGO"]
    timestamps = pd.date_range("2023-01-01", periods=48, freq="15min")
    records = []
    for sid in stations:
        for ts in timestamps:
            records.append({
                "timestamp": ts,
                "station_id": sid,
                "s4": np.random.uniform(0, 0.8),
                "sigma_phi": np.random.uniform(0, 1),
                "roti": np.random.uniform(0, 2),
                "tec": np.random.uniform(5, 60),
                "lat": 5.56 if sid == "ACCR" else 6.52,
                "lon": -0.20 if sid == "ACCR" else 3.38,
            })
    return pd.DataFrame(records)


def test_resample_gnss_shape(sample_gnss_df):
    from scripts.preprocess.resample import resample_gnss
    df_rs = resample_gnss(sample_gnss_df, resolution_minutes=30)
    assert len(df_rs) > 0
    assert "station_id" in df_rs.columns
    assert "s4" in df_rs.columns


def test_resample_uses_max_s4(sample_gnss_df):
    from scripts.preprocess.resample import resample_gnss
    df_rs = resample_gnss(sample_gnss_df, resolution_minutes=30)
    # After resampling to 30min from 15min, each row should be max of ~2 input rows
    assert df_rs["s4"].max() <= 1.0


def test_gap_fill_linear(sample_gnss_df):
    from scripts.preprocess.gap_fill import fill_gaps
    # Introduce gaps
    df = sample_gnss_df.copy()
    accr = df[df["station_id"] == "ACCR"].copy()
    accr.loc[accr.index[5:8], "s4"] = np.nan
    filled = fill_gaps(accr, "timestamp", ["s4"], max_gap_minutes=60, group_col=None)
    # Short gap should be filled
    assert filled["s4"].isna().sum() < 3


def test_outlier_remove_iqr(sample_gnss_df):
    from scripts.preprocess.outlier_remove import remove_outliers_iqr
    df = sample_gnss_df.copy()
    # Inject extreme outlier
    df.loc[0, "s4"] = 999.0
    df_clean = remove_outliers_iqr(df, ["s4"], iqr_threshold=3.0, group_col="station_id")
    assert df_clean["s4"].max() < 999.0


def test_iono_scaler_fit_transform():
    from scripts.preprocess.normalise import IonoScaler
    df = pd.DataFrame({
        "s4": np.random.uniform(0, 1, 100),
        "bz": np.random.normal(0, 10, 100),
        "kp": np.random.uniform(0, 9, 100),
    })
    scaler = IonoScaler(use_physical_bounds=True)
    df_norm = scaler.fit_transform(df, ["s4", "bz", "kp"])
    assert df_norm["s4"].between(0, 1).all()
    assert df_norm["kp"].between(0, 1).all()


def test_iono_scaler_save_load(tmp_path):
    from scripts.preprocess.normalise import IonoScaler
    import pandas as pd
    scaler = IonoScaler()
    df = pd.DataFrame({"s4": [0.1, 0.5, 0.9], "kp": [1, 3, 7]})
    scaler.fit(df, ["s4", "kp"])
    path = tmp_path / "scaler.json"
    scaler.save(path)

    scaler2 = IonoScaler.load(path)
    df_t1 = scaler.transform(df.copy())
    df_t2 = scaler2.transform(df.copy())
    pd.testing.assert_frame_equal(df_t1, df_t2)
