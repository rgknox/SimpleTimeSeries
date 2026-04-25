import argparse
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import math
import numpy as np
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_grid_dims(n):
    """Return (rows, cols) for a grid of n subplots."""
    if n == 1: return 1, 1
    if n == 2: return 1, 2
    cols = 2
    rows = math.ceil(n / cols)
    return rows, cols


def symbolic_eval(ds, expression):
    """Evaluate a math expression string against variables in an xarray Dataset."""
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expression)
    missing = [t for t in tokens if t not in ds.variables]
    if missing:
        raise KeyError(f"Variables missing from dataset: {missing}")
    local_dict = {t: ds[t] for t in tokens}
    try:
        return eval(expression, {"__builtins__": None}, local_dict)
    except Exception as e:
        raise ValueError(f"Error evaluating expression '{expression}': {e}")


def apply_smoother(data, config):
    """Apply rolling mean smoother if nyrs_smoother is specified."""
    if "nyrs_smoother" not in config:
        return data
    dt_obj = data.time.values[1] - data.time.values[0]
    days_between = dt_obj.days + dt_obj.seconds / 86400.0
    window = max(1, int(config["nyrs_smoother"] * 365 / days_between))
    return data.rolling(time=window, center=True).mean()


def get_time_axis(time_index):
    """
    Return a numeric array of decimal years from a CFTimeIndex or DatetimeIndex.
    Works even when years are out of pandas nanosecond range (e.g. year 1).
    """
    try:
        # Works for standard calendar within pandas range
        return time_index.to_datetimeindex().year.astype(float)
    except Exception:
        pass
    # Fall back: extract year + fractional month from cftime objects
    vals = time_index.values
    return np.array([v.year + (v.month - 1) / 12.0 for v in vals])


def find_coord_index(ds, lat, lon, tol=1.0):
    """Find the lndgrid index closest to the given lat/lon within tolerance."""
    if 'lat' not in ds and 'lon' not in ds:
        return None
    lats = ds['lat'].values
    lons = ds['lon'].values
    dists = np.sqrt((lats - lat)**2 + (lons - lon)**2)
    idx = int(np.argmin(dists))
    if dists[idx] > tol:
        print(f"Warning: no coordinate within {tol} deg of ({lat},{lon}), closest is {dists[idx]:.2f} deg away")
    return idx


def build_series(datasets, labels, coord_filter):
    """
    Build a flat list of (label_str, ds_single_coord) tuples.
    datasets: list of xr.Dataset
    labels:   list of str (one per file)
    coord_filter: list of {lat, lon} dicts or None entries, or None for all
    """
    series = []
    for ds, file_label in zip(datasets, labels):
        n_coords = ds.sizes.get('lndgrid', 1)

        # Determine which lndgrid indices to use
        if coord_filter:
            indices = []
            for cf in coord_filter:
                if cf is None:
                    continue
                idx = find_coord_index(ds, cf['lat'], cf['lon'])
                if idx is not None:
                    indices.append(idx)
            if not indices:
                indices = list(range(n_coords))
        else:
            indices = list(range(n_coords))

        for rank, idx in enumerate(indices, start=1):
            lbl = file_label if len(indices) == 1 else f"{file_label}{rank}"
            if 'lndgrid' in ds.dims:
                series.append((lbl, ds.isel(lndgrid=idx)))
            else:
                series.append((lbl, ds))
    return series


# ---------------------------------------------------------------------------
# 1-D time series plotting
# ---------------------------------------------------------------------------

def plot_1d_group(group_name, var_configs, series, savefigs, nametag):
    plt.rcParams.update({
        'axes.titlesize': 14, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'font.family': 'sans-serif'
    })

    # Tol "muted" colorblind-safe palette (10 colors)
    CB_COLORS = [
        '#CC6677', '#332288', '#DDCC77', '#117733',
        '#88CCEE', '#882255', '#44AA99', '#999933',
        '#AA4499', '#DDDDDD',
    ]

    n = len(var_configs)
    rows, cols = get_grid_dims(n)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.0 * rows),
                             squeeze=False, sharex=True,
                             gridspec_kw={'hspace': 0})
    axes_flat = axes.flatten()

    for i, (expression, config) in enumerate(var_configs.items()):
        ax = axes_flat[i]
        # 'expr' overrides the key as the actual variable expression to evaluate
        eval_expr = config.get('expr', expression)

        for series_idx, (lbl, ds) in enumerate(series):
            try:
                data = symbolic_eval(ds, eval_expr) * config.get('mult', 1)
            except (KeyError, ValueError) as e:
                print(f"  Skipping {expression} for {lbl}: {e}")
                continue

            # Handle extra_dim reduction to collapse to 1D
            if 'extra_dim' in config and 'reduce' in config:
                dim = config['extra_dim']
                reduce = config['reduce']
                if dim in data.dims:
                    if reduce == 'sum':
                        data = data.sum(dim=dim)
                    elif reduce == 'mean':
                        data = data.mean(dim=dim)
                    elif isinstance(reduce, dict) and 'index' in reduce:
                        data = data.isel({dim: reduce['index'] - 1})  # 1-based

            data = apply_smoother(data, config)

            logscale = config.get('logscale', 'no') == 'yes'
            if logscale:
                ax.set_yscale('log')

            # Convert cftime to decimal years for matplotlib
            times = get_time_axis(data.indexes['time'])
            color = CB_COLORS[series_idx % len(CB_COLORS)]
            ax.plot(times, data.values.squeeze(), label=lbl, color=color)

        # Y range cap
        if 'vrange' in config:
            ax.set_ylim(config['vrange'])

        # Obs reference
        if 'obs' in config:
            obs = config['obs']
            if hasattr(obs, '__len__') and len(obs) == 2:
                ax.axhspan(obs[0], obs[1], color='gray', alpha=0.3)
                ax.axhline(np.mean(obs), color='gray', linestyle='--', lw=1, alpha=0.6)
            else:
                ax.axhline(obs, color='gray', linestyle='--', lw=1, alpha=0.6)

        smooth_str = f" (sm={config['nyrs_smoother']} yr)" if 'nyrs_smoother' in config else ""
        units = config.get('units', '')
        ylabel = f"{expression}{smooth_str}\n{'ln(' + units + ')' if config.get('logscale') == 'yes' else units}"
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, linestyle=':')
        if i == 0:
            ax.legend()
        # Label x-axis only on the bottom row
        if i >= n - cols:
            ax.set_xlabel("Year")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout()
    _save_or_show(fig, savefigs, nametag, group_name)


# ---------------------------------------------------------------------------
# 2-D heatmap plotting
# ---------------------------------------------------------------------------

def plot_2d_group(group_name, var_configs, series, savefigs, nametag):
    """
    For each 2D variable in the group, produce one figure:
    a grid of heatmaps (one per series entry), shared colorbar, shared axes.
    """
    plt.rcParams.update({
        'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'font.family': 'sans-serif'
    })

    for expression, config in var_configs.items():
        eval_expr = config.get('expr', expression)
        n = len(series)
        cols = min(n, 4)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(4.0 * cols, 3.5 * rows),
                                 squeeze=False,
                                 sharex=True, sharey=True,
                                 gridspec_kw={'hspace': 0, 'wspace': 0},
                                 constrained_layout=True)
        axes_flat = axes.flatten()

        dim = config.get('extra_dim', None)

        # 1-based dim_range -> 0-based slice
        dim_range = config.get('dim_range', None)
        dim_slice = slice(dim_range[0] - 1, dim_range[1]) if dim_range else None

        # First pass: compute data and collect value range
        all_data = []
        computed = []
        for lbl, ds in series:
            try:
                data = symbolic_eval(ds, eval_expr) * config.get('mult', 1)
                if dim and dim_slice is not None and dim in data.dims:
                    data = data.isel({dim: dim_slice})
                data = apply_smoother(data, config)
                all_data.append(float(data.min()))
                all_data.append(float(data.max()))
                computed.append((lbl, data, ds))
            except (KeyError, ValueError) as e:
                print(f"  Skipping {expression} for {lbl}: {e}")
                computed.append((lbl, None, ds))

        vmin = config['vrange'][0] if 'vrange' in config else (min(all_data) if all_data else 0)
        vmax = config['vrange'][1] if 'vrange' in config else (max(all_data) if all_data else 1)
        logscale = config.get('logscale', 'no') == 'yes'
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=vmax) if logscale else mcolors.Normalize(vmin=vmin, vmax=vmax)

        im = None
        for i, (lbl, data, ds) in enumerate(computed):
            ax = axes_flat[i]
            ax.set_title(lbl)
            if data is None:
                ax.axis('off')
                continue

            if not (dim and dim in data.dims):
                print(f"  Warning: extra_dim '{dim}' not found in {expression}, skipping heatmap")
                ax.axis('off')
                continue

            # Ensure (time, extra_dim) order
            plot_data = data.transpose('time', dim)

            # Convert cftime to decimal years for matplotlib
            try:
                time_vals = get_time_axis(plot_data.indexes['time'])
            except Exception:
                time_vals = np.arange(plot_data.sizes['time'])

            # Use coordinate values if available, otherwise fall back to 1-based indices
            dim_invert = config.get('dim_invert', 'no') == 'yes'
            dim_label = config.get('dim_label', None)
            if dim in ds.coords:
                coord_vals = ds[dim].values
                if dim_slice is not None:
                    coord_vals = coord_vals[dim_slice]
                y_vals = -coord_vals if dim_invert else coord_vals
                ylabel = dim_label if dim_label else dim
            else:
                y_start = dim_range[0] if dim_range else 1
                y_vals = np.arange(y_start, y_start + plot_data.sizes[dim])
                ylabel = dim_label if dim_label else dim

            im = ax.pcolormesh(
                time_vals,
                y_vals,
                plot_data.values.T,
                norm=norm, cmap='viridis', shading='auto'
            )
            if i % cols == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlabel("Year")
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

        for j in range(len(computed), len(axes_flat)):
            axes_flat[j].axis('off')

        if im is not None:
            fig.colorbar(im, ax=axes_flat[:len(computed)], location='right', shrink=0.8, label=config.get('units', ''))

        fig.suptitle(f"{group_name} — {expression}")
        _save_or_show(fig, savefigs, nametag, f"{group_name}_{expression}_heatmap")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_or_show(fig, savefigs, nametag, name):
    if savefigs:
        dirname = f"fates_plots_{nametag}"
        Path(dirname).mkdir(parents=True, exist_ok=True)
        path = f"{dirname}/{name}.png"
        fig.savefig(path, bbox_inches='tight')
        print(f"Saved {path}")
    else:
        plt.show()
    plt.close(fig)


def load_datasets(files):
    return [xr.open_dataset(f, decode_times=True, engine='netcdf4') for f in files]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot timeseries from FATES NetCDF files.")
    parser.add_argument("files", nargs='+', help="NetCDF files to process")
    parser.add_argument("--labels", nargs='+', help="Labels for each file (default: A, B, C...)")
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument("--savefigs", action="store_true", help="Save figures to disk")
    parser.add_argument("--nametag", type=str, default="out", help="Tag for output directory name")
    args = parser.parse_args()

    # Build file labels
    default_labels = [chr(ord('A') + i) for i in range(len(args.files))]
    labels = args.labels if args.labels else default_labels
    if len(labels) < len(args.files):
        labels += default_labels[len(labels):]

    # Load JSON config
    with open(args.config) as f:
        cfg = json.load(f)

    coord_filter = cfg.get('coordinates', None)
    variable_groups = cfg.get('variable_groups', {})

    # Load datasets
    datasets = load_datasets(args.files)

    # Build flat series list
    series = build_series(datasets, labels, coord_filter)

    # Plot each group
    for group_name, var_configs in variable_groups.items():
        # Separate 1D and 2D (heatmap) variables
        heatmap_vars = {k: v for k, v in var_configs.items()
                        if v.get('plot_type') == 'heatmap'}
        line_vars = {k: v for k, v in var_configs.items()
                     if v.get('plot_type') != 'heatmap'}

        if line_vars:
            plot_1d_group(group_name, line_vars, series, args.savefigs, args.nametag)
        if heatmap_vars:
            plot_2d_group(group_name, heatmap_vars, series, args.savefigs, args.nametag)

    for ds in datasets:
        ds.close()


if __name__ == "__main__":
    main()
