from types import FunctionType
from typing import List, Mapping, Optional, Tuple, Union

import numpy as np
import scores.categorical as scc_cat
import scores.continuous as scc_cont
import scores.probability as scc_prob
import xarray as xr

xr.set_options(keep_attrs=True)


def compute_pipeline_statistic(  # noqa: C901
    datasets: List[xr.Dataset | xr.DataArray],
    stats_op: Optional[str | FunctionType] = None,
    stats_op_kwargs: Optional[Mapping] = None,
    diff_dim: Optional[str] = None,
    n_diff_steps: Optional[int] = 1,
    groupby: Optional[str] = None,
) -> xr.Dataset | xr.DataArray:
    """Apply a series of operations to compute a specific compound statistic.

    The operations applied in order are:
    1. (If diff_dim != None) Apply diff over the `diff_dim` dimension
        (default to 1 step diff)
    2. (If groupby != None) Apply grouping of dataset/dataarray according
        to the `groupby` index
    3. Apply the stats_op to the dataarray with the stats_op_kwargs.

    Parameters
    ----------
    datasets : List[xr.Dataset | xr.DataArray]
        Datasets/dataarrays to compute the statistic on
    stats_op : str | FunctionType, optional
        Statistic operation to apply. If a string, it must be a valid xarray
        operation. If a FunctionType, it must be a function that takes in
        the datasets/dataarray and returns one new dataset/dataarray, by default None
    stats_op_kwargs : Mapping, optional
        Keyword arguments to pass to the stats_op function, by default None
    diff_dim : str, optional
        Dimension to apply diff over, by default None
    n_diff_steps : int, optional
        Number of steps to compute the diff over, by default 1
    groupby : str, optional
        Index to group over, by default None
    """
    # Build up CF compliant cell-method attribute so that people know what
    # operations were applied
    cell_methods = []
    new_datasets = []
    for ds in datasets:
        if diff_dim:
            # Only keep variables that have the diff_dim as a dimension
            vars_to_keep = [v for v in ds.data_vars if diff_dim in ds[v].dims]
            if not vars_to_keep:
                raise ValueError(f"No variables found with dimension {diff_dim}")

            # Apply the diff operation
            ds = ds[vars_to_keep].diff(dim=diff_dim, n=n_diff_steps)
            # Get unit of the diff'ed array
            diff_unit_array: xr.DataArray = ds[diff_dim][1] - ds[diff_dim][0]
            diff_unit = diff_unit_array.values
            if diff_dim == "elapsed_forecast_duration":
                # Convert the diff unit to hours
                diff_unit = diff_unit.astype("timedelta64[h]")
            else:
                raise NotImplementedError(
                    f"diff_dim of type {type(diff_dim)} not supported"
                )

            # Update the cell_methods with the operation applied
            cell_methods.append(f"{diff_dim}: diff (interval: {diff_unit})")
        new_datasets.append(ds)
    datasets = new_datasets

    if stats_op:
        ds_stat: (
            xr.Dataset
            | xr.DataArray
            | xr.core.groupby.DataArrayGroupBy
            | xr.core.groupby.DataSetGroupBy
        )
        if isinstance(stats_op, FunctionType):
            ds_stat = stats_op(
                *datasets, **stats_op_kwargs if stats_op_kwargs is not None else {}
            )
        elif isinstance(stats_op, str):
            if len(datasets) == 1:
                # Assume that the stats_op is an xarray operation
                ds_stat = getattr(datasets[0], stats_op)(
                    **stats_op_kwargs if stats_op_kwargs is not None else {}
                )
            else:
                raise ValueError(
                    "stats_op as a string is only supported for a single "
                    "dataset/dataarray"
                )
        else:
            raise NotImplementedError(f"stats_op {stats_op} not supported")

        # Add stats_op cell_methods attribute to all variables in addition to
        # existing cell_methods
        if isinstance(ds_stat, xr.DataArray):
            update_cell_methods(ds_stat, cell_methods)
        elif isinstance(ds_stat, xr.Dataset):
            for var in ds_stat.data_vars:
                update_cell_methods(ds_stat[var], cell_methods)

        if groupby:
            # Apply the groupby operation
            ds_stat = ds_stat.groupby(groupby)

        return ds_stat

    # Update cell_methods attributes in case stats_op is None
    new_datasets = []
    for ds in datasets:
        if groupby:
            # Apply the groupby operation
            ds = ds.groupby(groupby)
            # Update the cell_methods with the operation applied
            cell_methods.append(f"{groupby}: groupby, mean")

        if isinstance(ds, xr.DataArray):
            update_cell_methods(ds, cell_methods)
        elif isinstance(ds, xr.Dataset):
            for var in ds.data_vars:
                update_cell_methods(ds[var], cell_methods)
        new_datasets.append(ds)
    return new_datasets


def rmse(
    ds_prediction: xr.Dataset | xr.DataArray,
    ds_reference: xr.Dataset | xr.DataArray,
    groupby: Optional[str] = None,
    **stats_op_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Compute the root mean squared error across grid_index for all variables.

    Args:
        ds_prediction (xr.Dataset | xr.DataArray): Prediction dataset
        ds_reference (xr.Dataset | xr.DataArray): Reference dataset
        reduce_dims (List[str]): Dimensions to reduce over
    Returns:
        xr.Dataset | xr.DataArray: Dataset with the computed statistical variables
    """
    ds_rmse = compute_pipeline_statistic(
        datasets=[ds_reference, ds_prediction],
        stats_op=scc_cont.rmse,
        groupby=groupby,
        stats_op_kwargs=stats_op_kwargs,
    )
    ds_rmse.name = ds_prediction.name

    # Get difference in dimensions betwee input datasets and ds_rmse.
    # Input dataset are assumed to have the same dimensions.
    reduce_dims = list(set(ds_reference.dims) - set(ds_rmse.dims))
    new_cell_methods = [",".join(reduce_dims) + ": root_mean_square"]

    # Update cell_methods attributes
    if isinstance(ds_rmse, xr.DataArray):
        update_cell_methods(ds_rmse, new_cell_methods)
    elif isinstance(ds_rmse, xr.Dataset):
        for _, da_var in ds_rmse.items():
            update_cell_methods(da_var, new_cell_methods)
    # else:
    #     raise ValueError("ds_rmse must be an xr.Dataset or xr.DataArray")

    return ds_rmse


def mae(
    ds_reference: xr.Dataset | xr.DataArray,
    ds_prediction: xr.Dataset | xr.DataArray,
    groupby: Optional[str] = None,
    **stats_op_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Compute the mean absolute error across specified dimensions.

    Args:
        ds_prediction (xr.Dataset | xr.DataArray): Prediction dataset
        ds_reference (xr.Dataset | xr.DataArray): Reference dataset
        reduce_dims (List[str]): Dimensions to reduce over
    Returns:
        xr.Dataset | xr.DataArray: Dataset with the mean absolute error computed
    """
    ds_mae = compute_pipeline_statistic(
        datasets=[ds_reference, ds_prediction],
        stats_op=scc_cont.mae,
        groupby=groupby,
        stats_op_kwargs=stats_op_kwargs,
    )

    # Get difference in dimensions betwee input datasets and ds_rmse.
    # Input dataset are assumed to have the same dimensions.
    reduce_dims = list(set(ds_reference.dims) - set(ds_mae.dims))
    new_cell_methods = [",".join(reduce_dims) + ": mean_absolute_error"]

    # Update cell_methods attributes
    if isinstance(ds_mae, xr.DataArray):
        update_cell_methods(ds_mae, new_cell_methods)
    elif isinstance(ds_mae, xr.Dataset):
        for _, da_var in ds_mae.items():
            update_cell_methods(da_var, new_cell_methods)
    # else:
    #     raise ValueError("ds_mae must be an xr.Dataset or xr.DataArray")

    return ds_mae


def crps(
    ds_reference: xr.Dataset | xr.DataArray,
    ds_prediction: xr.Dataset | xr.DataArray,
    ensemble_member_dim: str = "ensemble_member",
    **stats_op_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Compute the Continuous Ranked Probability Score (CRPS).

    Wraps `scores.probability.crps_for_ensemble` via
    `compute_pipeline_statistic`. Uses the fair (unbiased) estimator
    which correctly accounts for finite ensemble size.

    A perfectly calibrated ensemble achieves minimum CRPS. Lower is better.
    Unlike `crps_gauss`, this function makes no distributional assumptions
    and accepts raw ensemble member trajectories.

    Args:
        ds_reference: Reference (observation) dataset or data array.
            Must NOT contain `ensemble_member_dim`.
        ds_prediction: Ensemble forecast dataset or data array.
            Must contain `ensemble_member_dim` as a dimension.
        ensemble_member_dim: Name of the ensemble member dimension
            in `ds_prediction`. Defaults to ``"ensemble_member"``.
        **stats_op_kwargs: Additional keyword arguments forwarded to
            `scores.probability.crps_for_ensemble`, such as
            ``reduce_dims`` or ``preserve_dims``.

    Returns:
        Dataset or DataArray with CRPS values. The ensemble member
        dimension is collapsed. The ``cell_methods`` attribute records
        which dimensions were reduced.

    References:
        Zamo, M. & Naveau, P. (2018). Estimation of the Continuous
        Ranked Probability Score with Limited Information and
        Applications to Ensemble Weather Forecasts.
        https://doi.org/10.1007/s11004-017-9709-7

    Example:
        >>> da_crps = crps(
        ...     da_reference,
        ...     da_ensemble_prediction,
        ...     ensemble_member_dim="ensemble_member",
        ...     reduce_dims=["x", "y"],
        ... )
    """
    groupby = stats_op_kwargs.pop("groupby", None)
    stats_op_kwargs["ensemble_member_dim"] = ensemble_member_dim
    ds_crps = compute_pipeline_statistic(
        datasets=[ds_prediction, ds_reference],
        stats_op=scc_prob.crps_for_ensemble,
        stats_op_kwargs=stats_op_kwargs,
        groupby=groupby,
    )

    if isinstance(ds_crps, (xr.DataArray, xr.Dataset)):
        ds_crps.name = getattr(ds_prediction, "name", "crps")
        reduce_dims = list(set(ds_reference.dims) - set(ds_crps.dims))
        new_cell_methods = [",".join(reduce_dims) + ": crps"]
        if isinstance(ds_crps, xr.DataArray):
            update_cell_methods(ds_crps, new_cell_methods)
        elif isinstance(ds_crps, xr.Dataset):
            for _, da_var in ds_crps.items():
                update_cell_methods(da_var, new_cell_methods)
    return ds_crps


def spread_skill_ratio(
    ds_reference: xr.Dataset | xr.DataArray,
    ds_prediction: xr.Dataset | xr.DataArray,
    ensemble_member_dim: str = "ensemble_member",
    **stats_op_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Compute the Spread-Skill Ratio (SSR) for ensemble calibration.

    SSR = ensemble_spread / RMSE_of_ensemble_mean

    A perfectly calibrated ensemble has SSR = 1.0. Values below 1.0
    indicate underdispersion (ensemble too confident). Values above 1.0
    indicate overdispersion (ensemble too uncertain).

    Ensemble spread is the mean standard deviation across members.
    Skill is the RMSE of the ensemble mean against the reference.

    Args:
        ds_reference: Reference (observation) dataset or data array.
        ds_prediction: Ensemble forecast dataset or data array.
            Must contain `ensemble_member_dim` as a dimension.
        ensemble_member_dim: Name of the ensemble member dimension
            in `ds_prediction`. Defaults to ``"ensemble_member"``.
        **stats_op_kwargs: Additional keyword arguments forwarded
            to xarray reduction operations, such as ``reduce_dims``.

    Returns:
        Dataset or DataArray with SSR values. Values near 1.0 indicate
        good ensemble calibration. The ``cell_methods`` attribute
        records which dimensions were reduced.

    References:
        Fortin, V. et al. (2014). Why Should Ensemble Spread Match
        the RMSE of the Ensemble Mean?
        https://doi.org/10.1175/MWR-D-14-00037.1
    """
    groupby = stats_op_kwargs.pop("groupby", None)
    preserve_dims = stats_op_kwargs.pop("preserve_dims", None)
    reduce_dims = stats_op_kwargs.get("reduce_dims", None)

    # Derive reduce_dims from preserve_dims if not explicitly provided
    if reduce_dims is None and preserve_dims is not None:
        all_dims = [d for d in ds_prediction.dims if d != ensemble_member_dim]
        reduce_dims = [d for d in all_dims if d not in preserve_dims]

    # Ensemble spread: mean standard deviation across members
    spread = ds_prediction.std(dim=ensemble_member_dim)
    if reduce_dims:
        spread = spread.mean(dim=reduce_dims)

    # Skill: RMSE of ensemble mean
    ensemble_mean = ds_prediction.mean(dim=ensemble_member_dim)
    squared_error = (ensemble_mean - ds_reference) ** 2
    if reduce_dims:
        skill = squared_error.mean(dim=reduce_dims) ** 0.5
    else:
        skill = squared_error.mean() ** 0.5

    ds_ssr = spread / skill
    ds_ssr.name = getattr(ds_prediction, "name", "spread_skill_ratio")

    if groupby:
        ds_ssr = ds_ssr.groupby(groupby)

    reduce_dims_list = reduce_dims if reduce_dims else []
    new_cell_methods = [",".join(reduce_dims_list) + ": spread_skill_ratio"]
    if isinstance(ds_ssr, xr.DataArray):
        update_cell_methods(ds_ssr, new_cell_methods)
    elif isinstance(ds_ssr, xr.Dataset):
        for _, da_var in ds_ssr.items():
            update_cell_methods(da_var, new_cell_methods)

    return ds_ssr


def brier_score(
    ds_reference: xr.Dataset | xr.DataArray,
    ds_prediction: xr.Dataset | xr.DataArray,
    ensemble_member_dim: str = "ensemble_member",
    thresholds: Union[float, List[float]] = 0.5,
    **stats_op_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Compute the Brier Score for ensemble forecasts.

    Evaluates the accuracy of probabilistic predictions for binary
    events defined by threshold exceedance. The ensemble forecast is
    converted to a probability (fraction of members exceeding the
    threshold) and compared against the observed binary outcome.

    A Brier Score of 0 indicates a perfect forecast; 1 indicates the
    worst possible forecast. Lower is better.

    Args:
        ds_reference: Reference (observation) dataset or data array.
            Must NOT contain ``ensemble_member_dim``.
        ds_prediction: Ensemble forecast dataset or data array.
            Must contain ``ensemble_member_dim`` as a dimension.
        ensemble_member_dim: Name of the ensemble member dimension
            in ``ds_prediction``. Defaults to ``"ensemble_member"``.
        thresholds: Event threshold(s). A single float or a list of
            floats defining the exceedance events to evaluate.
            Defaults to ``0.5``.
        **stats_op_kwargs: Additional keyword arguments forwarded to
            ``scores.probability.brier_score_for_ensemble``.

    Returns:
        Dataset or DataArray with Brier Score values.

    References:
        Ferro, C. A. T. (2013). Fair scores for ensemble forecasts.
        Quarterly Journal of the Royal Meteorological Society,
        140(683), 1917-1923. https://doi.org/10.1002/qj.2270

    Example:
        >>> da_bs = brier_score(
        ...     da_reference,
        ...     da_ensemble_prediction,
        ...     ensemble_member_dim="ensemble_member",
        ...     thresholds=[0.5, 1.0, 2.0],
        ... )
    """
    groupby = stats_op_kwargs.pop("groupby", None)
    stats_op_kwargs["ensemble_member_dim"] = ensemble_member_dim
    stats_op_kwargs["event_thresholds"] = thresholds

    ds_bs = compute_pipeline_statistic(
        datasets=[ds_prediction, ds_reference],
        stats_op=scc_prob.brier_score_for_ensemble,
        stats_op_kwargs=stats_op_kwargs,
        groupby=groupby,
    )

    if isinstance(ds_bs, (xr.DataArray, xr.Dataset)):
        ds_bs.name = getattr(ds_prediction, "name", "brier_score")
        reduce_dims = list(set(ds_reference.dims) - set(ds_bs.dims))
        new_cell_methods = [",".join(reduce_dims) + ": brier_score"]
        if isinstance(ds_bs, xr.DataArray):
            update_cell_methods(ds_bs, new_cell_methods)
        elif isinstance(ds_bs, xr.Dataset):
            for _, da_var in ds_bs.items():
                update_cell_methods(da_var, new_cell_methods)
    return ds_bs


def equitable_threat_score(
    ds_reference: xr.Dataset | xr.DataArray,
    ds_prediction: xr.Dataset | xr.DataArray,
    threshold: float = 0.5,
    reduce_dims: Optional[Union[str, List[str]]] = "all",
    **stats_op_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Compute the Equitable Threat Score (ETS) for categorical forecasts.

    Also known as Gilbert Skill Score. Evaluates how well the forecast
    "yes" events correspond to the observed "yes" events, accounting
    for hits due to chance.

    Binary events are defined by exceedance of ``threshold``:
    a grid point is considered a "hit" if both forecast and observation
    exceed the threshold.

    Args:
        ds_reference: Reference (observation) dataset or data array.
        ds_prediction: Deterministic forecast dataset or data array.
        threshold: Event threshold. Grid points where the value exceeds
            this threshold are classified as events. Defaults to ``0.5``.
        reduce_dims: Dimensions to reduce when computing the
            contingency table. Defaults to ``"all"`` (scalar output).
            Pass a list of dimension names to preserve other dimensions.
        **stats_op_kwargs: Additional keyword arguments.

    Returns:
        Dataset or DataArray with ETS values. Range: -1/3 to 1.
        0 indicates no skill; 1 indicates a perfect score.

    References:
        Gilbert, G.K., 1884. Finley's tornado predictions.
        American Meteorological Journal, 1(5), pp.166-172.

        Hogan, R.J. et al., 2010. Equitability revisited: Why the
        "equitable threat score" is not equitable. Weather and
        Forecasting, 25(2), pp.710-726.
        https://doi.org/10.1175/2009WAF2222350.1

    Example:
        >>> da_ets = equitable_threat_score(
        ...     da_reference,
        ...     da_prediction,
        ...     threshold=0.5,
        ...     reduce_dims=["x", "y"],
        ... )
    """
    groupby = stats_op_kwargs.pop("groupby", None)

    # Create binary events based on threshold exceedance
    obs_events = ds_reference >= threshold
    fcst_events = ds_prediction >= threshold

    # Build contingency table using scores library
    bcm = scc_cat.BinaryContingencyManager(
        fcst_events=fcst_events,
        obs_events=obs_events,
    )

    # Reduce dimensions to compute the contingency counts
    basic_cm = bcm.transform(reduce_dims=reduce_dims)

    # Calculate ETS from the contingency table
    ds_ets = basic_cm.equitable_threat_score()
    ds_ets.name = getattr(ds_prediction, "name", "equitable_threat_score")

    if groupby:
        ds_ets = ds_ets.groupby(groupby)

    new_cell_methods = [f"threshold({threshold}): equitable_threat_score"]
    if isinstance(ds_ets, xr.DataArray):
        update_cell_methods(ds_ets, new_cell_methods)
    elif isinstance(ds_ets, xr.Dataset):
        for _, da_var in ds_ets.items():
            update_cell_methods(da_var, new_cell_methods)

    return ds_ets


def fractions_skill_score(
    ds_reference: xr.Dataset | xr.DataArray,
    ds_prediction: xr.Dataset | xr.DataArray,
    threshold: float,
    window_size: int,
    spatial_dims: Optional[List[str]] = None,
    **stats_op_kwargs,
) -> xr.Dataset | xr.DataArray:
    """Compute the Fractions Skill Score (FSS) for spatial forecasts.

    FSS evaluates the spatial accuracy of a forecast by comparing
    fractional coverage of events (values exceeding a threshold) within
    spatial neighborhoods of a given size. It addresses the "double
    penalty" problem inherent in point-wise verification of
    high-resolution models.

    FSS = 1 - MSE / MSE_ref

    where MSE is the mean squared difference of fractional coverages,
    and MSE_ref is the worst-case reference (no skill).

    Args:
        ds_reference: Reference (observation) dataset or data array.
            Must contain spatial dimensions.
        ds_prediction: Forecast dataset or data array. Must contain
            spatial dimensions matching ``ds_reference``.
        threshold: Event threshold. Grid points where the value
            exceeds this threshold are classified as events.
        window_size: Size of the spatial neighborhood window (must
            be an odd integer). The window is applied as a square
            ``window_size x window_size`` rolling mean over the
            spatial dimensions.
        spatial_dims: Names of the two spatial dimensions.
            Defaults to ``["x", "y"]``. Compatible with
            ``mlwp-data-specs`` variants such as ``["xc", "yc"]``
            or ``["longitude", "latitude"]``.
        **stats_op_kwargs: Additional keyword arguments.

    Returns:
        Dataset or DataArray with FSS value(s). Range: 0 to 1.
        FSS = 1 indicates a perfect forecast; FSS = 0 indicates no
        skill. FSS > 0.5 is generally considered "useful" at the
        given spatial scale.

    References:
        Roberts, N.M. and Lean, H.W. (2008). Scale-Selective
        Verification of Rainfall Accumulations from High-Resolution
        Forecasts of Convective Events. Monthly Weather Review,
        136(1), pp.78-97. https://doi.org/10.1175/2007MWR2123.1

    Example:
        >>> da_fss = fractions_skill_score(
        ...     da_reference,
        ...     da_prediction,
        ...     threshold=1.0,
        ...     window_size=11,
        ...     spatial_dims=["x", "y"],
        ... )
    """
    if spatial_dims is None:
        spatial_dims = ["x", "y"]

    if len(spatial_dims) != 2:
        raise ValueError(
            f"spatial_dims must have exactly 2 elements, got {len(spatial_dims)}"
        )

    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    groupby = stats_op_kwargs.pop("groupby", None)

    # Convert to binary fields based on threshold exceedance
    obs_binary = (ds_reference >= threshold).astype(float)
    fcst_binary = (ds_prediction >= threshold).astype(float)

    # Compute fractional coverage using rolling mean over spatial dims
    dim_x, dim_y = spatial_dims
    obs_frac = (
        obs_binary.rolling({dim_x: window_size, dim_y: window_size}, center=True)
        .mean()
        .dropna(dim=dim_x)
        .dropna(dim=dim_y)
    )
    fcst_frac = (
        fcst_binary.rolling({dim_x: window_size, dim_y: window_size}, center=True)
        .mean()
        .dropna(dim=dim_x)
        .dropna(dim=dim_y)
    )

    # MSE of fractional coverages
    mse = ((obs_frac - fcst_frac) ** 2).mean(dim=spatial_dims)

    # Worst-case MSE reference (no overlap between obs and fcst fractions)
    mse_ref = (obs_frac**2 + fcst_frac**2).mean(dim=spatial_dims)

    # FSS = 1 - MSE / MSE_ref
    # When mse_ref == 0, both fields have zero fractional coverage
    # (or are identical), so the forecast is trivially perfect → FSS = 1.0
    ds_fss = xr.where(mse_ref > 0, 1.0 - mse / mse_ref, 1.0)
    ds_fss.name = getattr(ds_prediction, "name", "fractions_skill_score")

    if groupby:
        ds_fss = ds_fss.groupby(groupby)

    new_cell_methods = [
        f"{','.join(spatial_dims)}: fractions_skill_score"
        f" (threshold: {threshold}, window: {window_size})"
    ]
    if isinstance(ds_fss, xr.DataArray):
        update_cell_methods(ds_fss, new_cell_methods)
    elif isinstance(ds_fss, xr.Dataset):
        for _, da_var in ds_fss.items():
            update_cell_methods(da_var, new_cell_methods)

    return ds_fss


def mean(ds: xr.Dataset | xr.DataArray, **stats_op_kwargs) -> xr.Dataset | xr.DataArray:
    """Compute the mean across specified dimensions.

    Args:
        ds (xr.Dataset | xr.Dataarray): Input dataset/dataarray
    Returns:
        xr.Dataset | xr.Dataarray: Dataset/dataarray with the mean computed
    """
    ds_mean = compute_pipeline_statistic(
        datasets=[ds],
        stats_op="mean",
        stats_op_kwargs=stats_op_kwargs,
    )

    # Get difference in dimensions betwee input datasets and ds_rmse.
    # Input dataset are assumed to have the same dimensions.
    reduce_dims = list(set(ds.dims) - set(ds_mean.dims))
    new_cell_methods = [",".join(reduce_dims) + ": mean"]

    # Update cell_methods attributes
    if isinstance(ds_mean, xr.DataArray):
        update_cell_methods(ds_mean, new_cell_methods)
    elif isinstance(ds_mean, xr.Dataset):
        for _, da_var in ds_mean.items():
            update_cell_methods(da_var, new_cell_methods)
    else:
        raise ValueError("ds_mean must be an xr.Dataset or xr.DataArray")

    return ds_mean


def difference(
    ds_prediction: xr.Dataset | xr.DataArray,
    ds_reference: xr.Dataset | xr.DataArray,
    groupby: Optional[str] = None,
) -> xr.Dataset | xr.DataArray:
    """Compute the root mean squared error across grid_index for all variables.

    Args:
        ds_prediction (xr.Dataset | xr.DataArray): Prediction dataset
        ds_reference (xr.Dataset | xr.DataArray): Reference dataset
        reduce_dims (List[str]): Dimensions to reduce over
    Returns:
        xr.Dataset | xr.DataArray: Dataset with the computed statistical variables
    """
    ds_difference = ds_reference - ds_prediction
    diff_dims = ["x", "y"]
    new_cell_methods = [",".join(diff_dims) + ": difference"]

    # Update cell_methods attributes
    if isinstance(ds_difference, xr.DataArray):
        update_cell_methods(ds_difference, new_cell_methods)
    elif isinstance(ds_difference, xr.Dataset):
        for _, da_var in ds_difference.items():
            update_cell_methods(da_var, new_cell_methods)
    else:
        raise ValueError("ds_difference must be an xr.Dataset or xr.DataArray")

    if groupby is not None:
        ds_difference = compute_pipeline_statistic(
            datasets=[ds_difference],
            groupby=groupby,
        )[0]

    return ds_difference


def update_cell_methods(da: xr.DataArray, cell_methods: List[str]):
    """Update the cell_methods attribute of a DataArray with new cell_methods.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to update cell_methods attribute
    cell_methods : List[str]
        List of cell_methods to add to the existing cell_methods attribute
    """
    existing_cell_methods = da.attrs.get("cell_methods", "")
    if existing_cell_methods:
        cell_methods.insert(0, existing_cell_methods)
    da.attrs["cell_methods"] = " ".join(cell_methods)


def add_persistence_to_dataarray(
    da_reference: xr.DataArray, da_prediction: xr.DataArray
) -> Tuple[xr.DataArray, xr.DataArray]:
    """Add persistence datasource to dataarrays.

    Parameters
    ----------
    da_reference : xr.DataArray
        Reference dataarray.
    da_prediction : xr.DataArray
        Prediction dataarray.

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        Reference and prediction dataarrays with persistence datasource added.
    """
    # Set persistence prediction as the reference shifted by 1
    ds_persistence_prediction = da_reference.shift(elapsed_forecast_duration=1)
    # Save original datasources before concatenating
    reference_datasources = da_reference["datasource"].values
    # Concatenate the dataarrays
    da_reference = xr.concat([da_reference, da_reference], dim="datasource")
    da_prediction = xr.concat(
        [da_prediction, ds_persistence_prediction], dim="datasource"
    )
    # Update datasource coordinates
    da_reference["datasource"] = da_prediction["datasource"] = np.append(
        reference_datasources, "persistence"
    )

    return da_reference, da_prediction
