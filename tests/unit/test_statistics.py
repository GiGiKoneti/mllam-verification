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


class TestEnergyScore:
    """Tests for energy_score().

    Mathematical reference:
        ES = E||X-y||₂ - 0.5·E||X-X'||₂
        Perfect ensemble (all members = obs): ES = 0
        Random ensemble: ES > 0
    """

    def test_energy_score_returns_correct_type(self):
        """energy_score returns same type as input."""
        import numpy as np

        from mllam_verification.operations.statistics import energy_score

        # DataArray input → DataArray output
        obs = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=["space"],
        )
        members = xr.DataArray(
            np.random.randn(5, 3),
            dims=["ensemble_member", "space"],
        )
        result = energy_score(members, obs)
        assert isinstance(result, xr.DataArray)

    def test_energy_score_ensemble_dim_collapsed(self):
        """ensemble_member dim must not appear in output."""
        import numpy as np

        from mllam_verification.operations.statistics import energy_score

        obs = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=["space"],
        )
        members = xr.DataArray(
            np.random.randn(5, 3),
            dims=["ensemble_member", "space"],
        )
        result = energy_score(members, obs)
        assert "ensemble_member" not in result.dims

    def test_energy_score_perfect_ensemble_is_zero(self):
        """
        Perfect ensemble: all M members equal observation.
        ES = E||X-y||₂ - 0.5·E||X-X'||₂
           = 0        - 0.5·0
           = 0
        """
        import numpy as np
        import pytest

        from mllam_verification.operations.statistics import energy_score

        obs = xr.DataArray(
            np.array([1.0, 2.0, 3.0]),
            dims=["space"],
        )
        # All 5 members identical to obs
        members = xr.DataArray(
            np.stack([obs.values] * 5, axis=0),
            dims=["ensemble_member", "space"],
        )
        result = energy_score(members, obs)
        assert float(result.mean()) == pytest.approx(0.0, abs=1e-6)

    def test_energy_score_non_negative(self):
        """
        Energy Score is non-negative by definition.
        ES >= 0 for any ensemble and observation.
        """
        import numpy as np

        from mllam_verification.operations.statistics import energy_score

        rng = np.random.default_rng(42)
        obs = xr.DataArray(
            rng.standard_normal(10),
            dims=["space"],
        )
        members = xr.DataArray(
            rng.standard_normal((8, 10)),
            dims=["ensemble_member", "space"],
        )
        result = energy_score(members, obs)
        assert float(result.min()) >= 0.0
