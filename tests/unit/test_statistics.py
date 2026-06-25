"""Unit tests for the statistics module."""

import xarray as xr

from mllam_verification.operations.statistics import compute_pipeline_statistic, rmse


class TestComputePipelineStatistic:
    """Unit tests for the compute_pipeline_statistic function."""

    def test_compute_pipeline_statistic(self, da_prediction_2d_utc: xr.DataArray):
        """Test computing a pipeline statistic."""
        da_stat = compute_pipeline_statistic(
            [da_prediction_2d_utc],
            stats_op="mean",
            stats_op_kwargs={"dim": ["x", "y"]},
        )
        assert isinstance(da_stat, xr.DataArray)


class TestRmse:
    """Unit tests for the rmse function."""

    def test_rmse(
        self, da_prediction_2d_utc: xr.DataArray, da_reference_2d_utc: xr.DataArray
    ):
        """Test computing the root mean squared error."""
        da_rmse = rmse(da_prediction_2d_utc, da_reference_2d_utc, reduce_dims=["x", "y"])
        assert isinstance(da_rmse, xr.DataArray)
        assert "cell_methods" in da_rmse.attrs


class TestCrps:
    """Tests for the crps() function."""

    def test_crps_returns_dataarray(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """crps() should return a DataArray."""
        from mllam_verification.operations.statistics import crps

        result = crps(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            reduce_dims=["x", "y"],
        )
        assert isinstance(result, xr.DataArray)

    def test_crps_ensemble_dim_collapsed(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """Ensemble member dimension must not appear in output."""
        from mllam_verification.operations.statistics import crps

        result = crps(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            reduce_dims=["x", "y"],
        )
        assert "ensemble_member" not in result.dims

    def test_crps_has_cell_methods(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """Output must have cell_methods attribute."""
        from mllam_verification.operations.statistics import crps

        result = crps(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            reduce_dims=["x", "y"],
        )
        assert "cell_methods" in result.attrs

    def test_crps_non_negative(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """CRPS must be non-negative."""
        from mllam_verification.operations.statistics import crps

        result = crps(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            reduce_dims=["x", "y"],
        )
        assert float(result.min()) >= 0

    def test_crps_method(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """crps() should support method='fair' and method='ecdf'."""
        from mllam_verification.operations.statistics import crps

        result_fair = crps(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            method="fair",
            reduce_dims=["x", "y"],
        )
        result_ecdf = crps(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            method="ecdf",
            reduce_dims=["x", "y"],
        )
        assert result_fair.attrs["cell_methods"].endswith("crps(method=fair)")
        assert result_ecdf.attrs["cell_methods"].endswith("crps(method=ecdf)")
        assert not result_fair.equals(result_ecdf)


class TestSpreadSkillRatio:
    """Tests for the spread_skill_ratio() function."""

    def test_ssr_returns_dataarray(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """spread_skill_ratio() should return a DataArray."""
        from mllam_verification.operations.statistics import spread_skill_ratio

        result = spread_skill_ratio(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            reduce_dims=["x", "y"],
        )
        assert isinstance(result, xr.DataArray)

    def test_ssr_positive(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """SSR must be positive."""
        from mllam_verification.operations.statistics import spread_skill_ratio

        result = spread_skill_ratio(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            reduce_dims=["x", "y"],
        )
        assert float(result.min()) > 0

    def test_ssr_perfect_ensemble_near_one(self):
        """A perfectly calibrated ensemble should have SSR near 1.0."""
        import numpy as np

        from mllam_verification.operations.statistics import spread_skill_ratio

        rng = np.random.default_rng(seed=0)
        # Truth
        truth = xr.DataArray(
            rng.normal(0, 1, (100,)),
            dims=["time"],
        )
        # Ensemble: sample from same distribution as truth
        members = [
            xr.DataArray(
                rng.normal(0, 1, (100,)),
                dims=["time"],
            ).assign_coords(ensemble_member=i)
            for i in range(50)
        ]
        ensemble = xr.concat(members, dim="ensemble_member")
        result = spread_skill_ratio(
            truth,
            ensemble,
            ensemble_member_dim="ensemble_member",
            reduce_dims=["time"],
        )
        # For a large well-calibrated ensemble SSR should be near 1.0
        assert 0.5 < float(result) < 2.0


class TestBrierScore:
    """Tests for the brier_score() function."""

    def test_brier_score_returns_dataarray(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """brier_score() should return a DataArray."""
        from mllam_verification.operations.statistics import brier_score

        result = brier_score(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            thresholds=0.5,
            reduce_dims=["x", "y"],
        )
        assert isinstance(result, xr.DataArray)

    def test_brier_score_ensemble_dim_collapsed(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """Ensemble member dimension must not appear in output."""
        from mllam_verification.operations.statistics import brier_score

        result = brier_score(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            thresholds=0.5,
            reduce_dims=["x", "y"],
        )
        assert "ensemble_member" not in result.dims

    def test_brier_score_has_cell_methods(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """Output must have cell_methods attribute."""
        from mllam_verification.operations.statistics import brier_score

        result = brier_score(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            thresholds=0.5,
            reduce_dims=["x", "y"],
        )
        assert "cell_methods" in result.attrs

    def test_brier_score_range(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """Brier Score must be in range [0, 1]."""
        from mllam_verification.operations.statistics import brier_score

        result = brier_score(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            thresholds=0.5,
            reduce_dims=["x", "y"],
        )
        assert float(result.min()) >= 0
        assert float(result.max()) <= 1

    def test_brier_score_multiple_thresholds(
        self,
        da_ensemble_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """Brier Score should work with multiple thresholds."""
        from mllam_verification.operations.statistics import brier_score

        result = brier_score(
            da_reference_2d_utc,
            da_ensemble_prediction_2d_utc,
            ensemble_member_dim="ensemble_member",
            thresholds=[0.1, 0.5, 0.9],
            reduce_dims=["x", "y"],
        )
        assert isinstance(result, xr.DataArray)
        assert "threshold" in result.dims


class TestEquitableThreatScore:
    """Tests for the equitable_threat_score() function."""

    def test_ets_returns_dataarray(
        self,
        da_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """equitable_threat_score() should return a DataArray."""
        from mllam_verification.operations.statistics import equitable_threat_score

        result = equitable_threat_score(
            da_reference_2d_utc,
            da_prediction_2d_utc,
            threshold=0.5,
            reduce_dims="all",
        )
        assert isinstance(result, xr.DataArray)

    def test_ets_has_cell_methods(
        self,
        da_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """Output must have cell_methods attribute."""
        from mllam_verification.operations.statistics import equitable_threat_score

        result = equitable_threat_score(
            da_reference_2d_utc,
            da_prediction_2d_utc,
            threshold=0.5,
            reduce_dims="all",
        )
        assert "cell_methods" in result.attrs

    def test_ets_range(
        self,
        da_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """ETS must be in range [-1/3, 1]."""
        from mllam_verification.operations.statistics import equitable_threat_score

        result = equitable_threat_score(
            da_reference_2d_utc,
            da_prediction_2d_utc,
            threshold=0.5,
            reduce_dims="all",
        )
        assert float(result) >= -1 / 3
        assert float(result) <= 1

    def test_ets_perfect_forecast_is_one(self):
        """A perfect forecast should have ETS = 1.0."""
        import numpy as np

        from mllam_verification.operations.statistics import equitable_threat_score

        # Create identical observation and forecast
        data = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        obs = xr.DataArray(data, dims=["grid_index"])
        fcst = xr.DataArray(data, dims=["grid_index"])

        result = equitable_threat_score(
            obs,
            fcst,
            threshold=0.5,
            reduce_dims="all",
        )
        assert float(result) == 1.0

    def test_ets_no_skill_near_zero(self):
        """A random forecast should have ETS near 0."""
        import numpy as np

        from mllam_verification.operations.statistics import equitable_threat_score

        rng = np.random.default_rng(seed=42)
        obs = xr.DataArray(rng.choice([0.0, 1.0], size=1000), dims=["grid_index"])
        fcst = xr.DataArray(rng.choice([0.0, 1.0], size=1000), dims=["grid_index"])

        result = equitable_threat_score(
            obs,
            fcst,
            threshold=0.5,
            reduce_dims="all",
        )
        # Random forecast should have ETS near 0 (within ±0.15)
        assert -0.15 < float(result) < 0.15


class TestFractionsSkillScore:
    """Tests for the fractions_skill_score() function."""

    def test_fss_returns_dataarray(
        self,
        da_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """fractions_skill_score() should return a DataArray."""
        from mllam_verification.operations.statistics import fractions_skill_score

        result = fractions_skill_score(
            da_reference_2d_utc,
            da_prediction_2d_utc,
            threshold=0.5,
            window_size=5,
            spatial_dims=["x", "y"],
        )
        assert isinstance(result, xr.DataArray)

    def test_fss_has_cell_methods(
        self,
        da_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """Output must have cell_methods attribute."""
        from mllam_verification.operations.statistics import fractions_skill_score

        result = fractions_skill_score(
            da_reference_2d_utc,
            da_prediction_2d_utc,
            threshold=0.5,
            window_size=5,
            spatial_dims=["x", "y"],
        )
        assert "cell_methods" in result.attrs

    def test_fss_range(
        self,
        da_prediction_2d_utc: xr.DataArray,
        da_reference_2d_utc: xr.DataArray,
    ):
        """FSS must be in range [0, 1]."""
        from mllam_verification.operations.statistics import fractions_skill_score

        result = fractions_skill_score(
            da_reference_2d_utc,
            da_prediction_2d_utc,
            threshold=0.5,
            window_size=5,
            spatial_dims=["x", "y"],
        )
        assert float(result.min()) >= 0
        assert float(result.max()) <= 1

    def test_fss_perfect_forecast_is_one(self):
        """A perfect forecast should have FSS = 1.0."""
        import numpy as np

        from mllam_verification.operations.statistics import fractions_skill_score

        data = np.random.rand(20, 20)
        obs = xr.DataArray(data, dims=["x", "y"])
        fcst = xr.DataArray(data, dims=["x", "y"])

        result = fractions_skill_score(
            obs, fcst, threshold=0.5, window_size=3, spatial_dims=["x", "y"]
        )
        assert float(result) == 1.0

    def test_fss_increases_with_window_size(self):
        """FSS should generally increase with larger window sizes."""
        import numpy as np

        from mllam_verification.operations.statistics import fractions_skill_score

        rng = np.random.default_rng(seed=42)
        obs = xr.DataArray(rng.random((50, 50)), dims=["x", "y"])
        # Spatially shifted forecast
        fcst = xr.DataArray(np.roll(obs.values, 3, axis=0), dims=["x", "y"])

        fss_small = fractions_skill_score(
            obs, fcst, threshold=0.5, window_size=3, spatial_dims=["x", "y"]
        )
        fss_large = fractions_skill_score(
            obs, fcst, threshold=0.5, window_size=11, spatial_dims=["x", "y"]
        )
        assert float(fss_large) >= float(fss_small)

    def test_fss_rejects_even_window(self):
        """Even window_size should raise ValueError."""
        import numpy as np
        import pytest

        from mllam_verification.operations.statistics import fractions_skill_score

        data = np.random.rand(10, 10)
        obs = xr.DataArray(data, dims=["x", "y"])
        fcst = xr.DataArray(data, dims=["x", "y"])

        with pytest.raises(ValueError, match="window_size must be odd"):
            fractions_skill_score(
                obs, fcst, threshold=0.5, window_size=4, spatial_dims=["x", "y"]
            )

    def test_fss_rejects_wrong_spatial_dims(self):
        """spatial_dims with != 2 elements should raise ValueError."""
        import numpy as np
        import pytest

        from mllam_verification.operations.statistics import fractions_skill_score

        data = np.random.rand(10, 10)
        obs = xr.DataArray(data, dims=["x", "y"])
        fcst = xr.DataArray(data, dims=["x", "y"])

        with pytest.raises(ValueError, match="spatial_dims must have exactly 2"):
            fractions_skill_score(
                obs, fcst, threshold=0.5, window_size=3, spatial_dims=["x"]
            )
