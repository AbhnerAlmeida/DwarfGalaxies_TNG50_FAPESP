#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:45:34 2026

@author: abhner
"""

import os
import io
import sys
import shutil
import logging
from typing import Optional, Sequence, Tuple

import h5py
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from sphviewer.tools import QuickView

sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/analyzes")
sys.path.append(os.getenv("HOME")+"/PROJECTS/2026/DwarfGalaxies_TNG50_FAPESP/src")

import MATH
import TNGFunctions as TNG
import PlotFunctions as plot


logger = logging.getLogger(__name__)

# cosmological parameters
Omegam0 = 0.3089
h = 0.6774

# =========================================================
# API CONFIG
# =========================================================

TNG_API_KEY = "caaddaa4f3de0daa0e043058306cf52d"
TNG_BASE = "https://www.tng-project.org/api/"


dfTime = pd.read_csv(os.getenv("HOME")+'/TNG_Analyzes/SubhaloHistory/SNAPS_TIME.csv')



def _api_headers():
    if not TNG_API_KEY:
        raise RuntimeError(
            "TNG_API_KEY is not set. Define os.environ['TNG_API_KEY'] or export it in the shell."
        )
    return {"api-key": TNG_API_KEY}


def _api_get(url, params=None, stream=False, timeout=180):
    r = requests.get(
        url,
        params=params,
        headers=_api_headers(),
        stream=stream,
        timeout=timeout,
    )
    r.raise_for_status()
    return r


def _sim_api_name(sim: str) -> str:
    # para TNG50-1, TNG100-1, etc.
    return sim


# =========================================================
# BASIC HELPERS
# =========================================================

def FixPeriodic(dx, sim='TNG50-1'):
    """
    Handle periodic boundary conditions
    Arguments:
        dx: difference in positions (in ckpc/h)
        sim: simulation (default "TNG50-1")

    Returns: dx corrected for periodic box (in ckpc/h)
    """
    if sim == 'TNG300-1':
        L = 205000.0
    elif sim == 'TNG100-1' or sim == 'TNG100':
        L = 75000.0
    elif sim == 'TNG50-1' or sim == 'TNG50':
        L = 35000.0
    
    dx = np.where(dx > L/2, dx - L, dx)
    dx = np.where(dx < -L/2, dx + L, dx)
    return dx

def _safe_str_contains(text: Optional[str], pattern: str) -> bool:
    return isinstance(text, str) and (pattern in text)


def _validate_axis(axis: str) -> None:
    if axis not in {"x", "z"}:
        raise ValueError(f"axis must be 'x' or 'z', got {axis!r}")


def _validate_plot_type(plot_type: str) -> None:
    if plot_type not in {"Mosaic", "Evolution"}:
        raise ValueError(f"Type must be 'Mosaic' or 'Evolution', got {plot_type!r}")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _get_redshift_from_snap(snap: int) -> float:
    zvals = dfTime.loc[dfTime.Snap == snap, "z"].values
    if len(zvals) == 0:
        raise ValueError(f"Snapshot {snap} not found in dfTime")
    return float(zvals[0])


def _get_age_from_snap(snap: int) -> float:
    avals = dfTime.loc[dfTime.Snap == snap, "Age"].values
    if len(avals) == 0:
        raise ValueError(f"Snapshot {snap} not found in dfTime")
    return float(avals[0])


def _snap_from_age_window(target_snap: int, delta_gyr: float, mode: str = "min") -> int:
    target_age = _get_age_from_snap(target_snap)
    mask = np.abs(dfTime.Age.values - target_age) < delta_gyr
    snaps = dfTime.loc[mask, "Snap"].values

    if len(snaps) == 0:
        return int(target_snap)

    if mode == "min":
        return int(np.nanmin(snaps))
    if mode == "max":
        return int(np.nanmax(snaps))

    raise ValueError(f"Invalid mode {mode!r}; use 'min' or 'max'")


def _format_zlabel(snap: int) -> str:
    z = dfTime.z.iloc[99 - int(snap)]
    frac = z - int(z)
    return f"z = {z:.0f}" if frac < 0.005 else f"z = {z:.1f}"


def _panel_display_settings(param: Optional[str], part_type: str,
                            default_cmap: str,
                            default_vmin: Optional[float],
                            default_vmax: Optional[float]) -> Tuple[str, Optional[float], Optional[float], bool]:
    cmap = default_cmap
    vmin = default_vmin
    vmax = default_vmax

    do_log = (param in ['Mass', 'T', 'MassSF', 'SFR']) or _safe_str_contains(param, 'Mass')

    if _safe_str_contains(param, 'Mass') and part_type == 'PartType0':
        cmap = 'magma'
    elif _safe_str_contains(param, 'Mass') and part_type == 'PartType4':
        cmap = 'bone'

    if param in ['Jnorm', 'Jpar_norm']:
        vmin, vmax = -1.0, 1.0

    return cmap, vmin, vmax, do_log


def _prepare_names_array(names: Optional[Sequence], total_size: int) -> np.ndarray:
    if names is None:
        return np.array(["None"] * total_size, dtype=object)

    names = np.array(names, dtype=object)

    if len(names) == 0:
        return np.array(["None"] * total_size, dtype=object)

    if len(names) < total_size:
        fill_value = names[0]
        names = np.append(names, [fill_value] * (total_size - len(names)))
    elif len(names) > total_size:
        names = names[:total_size]

    return names


def _prepare_ids_array(ids: Sequence, total_size: int) -> np.ndarray:
    ids = np.array(ids, dtype=float)

    if len(ids) < total_size:
        ids = np.append(ids, [np.nan] * (total_size - len(ids)))
    elif len(ids) > total_size:
        ids = ids[:total_size]

    return ids


def _safe_log10_image(img: np.ndarray) -> np.ndarray:
    out = np.full_like(img, np.nan, dtype=float)
    mask = np.isfinite(img) & (img > 0)
    out[mask] = np.log10(img[mask])
    return out


def _mass_weighted_center_and_velocity(pos_ref: np.ndarray, vel_ref: np.ndarray, mass_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cen = np.array([
        MATH.weighted_median(pos_ref[:, 0], mass_ref),
        MATH.weighted_median(pos_ref[:, 1], mass_ref),
        MATH.weighted_median(pos_ref[:, 2], mass_ref)
    ])

    vmean = np.array([
        MATH.weighted_median(vel_ref[:, 0], mass_ref),
        MATH.weighted_median(vel_ref[:, 1], mass_ref),
        MATH.weighted_median(vel_ref[:, 2], mass_ref)
    ])
    return cen, vmean


def _read_particle_block(file: h5py.File, part_type: str, snap: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if part_type not in file:
        raise KeyError(f"{part_type} not found in file")

    zsnap = _get_redshift_from_snap(snap)

    pos = file[part_type]['Coordinates'][:] / (1.0 + zsnap) / h
    mass = file[part_type]['Masses'][:] * 1e10 / h
    vel = file[part_type]['Velocities'][:] * np.sqrt(1.0 / (1.0 + zsnap))

    return pos, mass, vel


def _try_read_particle_block(file: h5py.File, part_type: str, snap: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        return _read_particle_block(file, part_type, snap)
    except (KeyError, ValueError):
        return np.array([[0.0, 0.0, 0.0]]), np.array([0.0]), np.array([[0.0, 0.0, 0.0]])


def _get_projection_setup(axis: str, vel: np.ndarray, pos: np.ndarray, vel_plot: bool):
    if axis == 'x':
        prad = 0
        t = 90
        velx = vely = posx = posy = None
        if vel_plot:
            velx = vel[:, 0]
            vely = vel[:, 2]
            posx = pos[:, 0]
            posy = pos[:, 2]

    elif axis == 'z':
        prad = 0
        t = 0
        velx = vely = posx = posy = None
        if vel_plot:
            velx = vel[:, 0]
            vely = vel[:, 1]
            posx = pos[:, 0]
            posy = pos[:, 1]
    else:
        raise ValueError(f"Invalid axis: {axis}")

    return prad, t, velx, vely, posx, posy


def _make_streamplot(ax, posx, posy, velx, vely, step: int = 100, grid_size: int = 30):
    if posx is None or posy is None or velx is None or vely is None:
        return
    if len(posx) == 0:
        return

    posx = posx[::step]
    posy = posy[::step]
    velx = velx[::step]
    vely = vely[::step]

    if len(posx) < 5:
        return

    x_bins = np.linspace(np.nanmin(posx), np.nanmax(posx), grid_size)
    y_bins = np.linspace(np.nanmin(posy), np.nanmax(posy), grid_size)
    x_grid, y_grid = np.meshgrid(x_bins, y_bins)

    vx_grid = np.zeros_like(x_grid, dtype=float)
    vy_grid = np.zeros_like(y_grid, dtype=float)
    counts = np.zeros_like(x_grid, dtype=float)

    for k in range(len(velx)):
        xi = np.digitize(posx[k], x_bins) - 1
        yi = np.digitize(posy[k], y_bins) - 1
        if 0 <= xi < grid_size - 1 and 0 <= yi < grid_size - 1:
            vx_grid[yi, xi] += velx[k]
            vy_grid[yi, xi] += vely[k]
            counts[yi, xi] += 1

    mask = counts > 0
    vx_grid[mask] /= counts[mask]
    vy_grid[mask] /= counts[mask]

    if np.any(mask):
        ax.streamplot(
            x_grid, y_grid, vx_grid, vy_grid,
            color='cyan', linewidth=1, density=1.2, arrowstyle='->'
        )


# =========================================================
# API BACKEND
# =========================================================

def backend_load_main_progenitor_branch(ID_z0: int, sim: str = 'TNG50-1') -> pd.DataFrame:
    """
    Resolve a main progenitor branch diretamente da API.
    """
    sim_name = _sim_api_name(sim)

    subhalo_url = f"{TNG_BASE}{sim_name}/snapshots/99/subhalos/{int(ID_z0)}/"
    sub = _api_get(subhalo_url).json()

    trees = sub.get("trees", {})
    mpb_url = trees.get("sublink_mpb")

    if mpb_url is None:
        raise KeyError(f"'trees.sublink_mpb' not found for z=0 subhalo {ID_z0}")

    r = _api_get(mpb_url, stream=True)

    with h5py.File(io.BytesIO(r.content), "r") as f:
        snapnum = f["SnapNum"][:]

        if "SubfindID" in f:
            subfind = f["SubfindID"][:]
        elif "SubhaloNumber" in f:
            subfind = f["SubhaloNumber"][:]
        else:
            raise KeyError("MPB file does not contain 'SubfindID' or 'SubhaloNumber'.")

        data = {
            "SnapNum": snapnum.astype(int),
            "SubfindID": subfind.astype(int),
        }

        if "SubhaloHalfmassRadType" in f:
            rhalf = f["SubhaloHalfmassRadType"][:]
            if rhalf.ndim == 2 and rhalf.shape[1] >= 5:
                data["SubhaloHalfmassRadType0"] = rhalf[:, 0]
                data["SubhaloHalfmassRadType4"] = rhalf[:, 4]

    df = pd.DataFrame(data).sort_values("SnapNum", ascending=False).reset_index(drop=True)
    return df


def backend_load_subhalo_catalog_entry(snap: int, subfind_id: int, sim: str = 'TNG50-1') -> dict:
    """
    Carrega propriedades do subhalo diretamente do catálogo da API.
    """
    sim_name = _sim_api_name(sim)
    subhalo_url = f"{TNG_BASE}{sim_name}/snapshots/{int(snap)}/subhalos/{int(subfind_id)}/"
    sub = _api_get(subhalo_url).json()

    out = {}

    if "halfmassrad_type" in sub:
        out["SubhaloHalfmassRadType"] = sub["halfmassrad_type"]
    elif "SubhaloHalfmassRadType" in sub:
        out["SubhaloHalfmassRadType"] = sub["SubhaloHalfmassRadType"]
    else:
        out["SubhaloHalfmassRadType"] = [np.nan] * 6

    return out


def backend_download_cutout_by_subfindid(
    snap: int,
    subfind_id: int,
    outpath: str,
    sim: str = 'TNG50-1',
    fields: Optional[dict] = None
) -> str:
    """
    Baixa cutout direto da API oficial da TNG.
    """
    sim_name = _sim_api_name(sim)

    if fields is None:
        fields = {
            "gas": "Coordinates,Velocities,Masses,InternalEnergy,ElectronAbundance,StarFormationRate,ParticleIDs",
            "stars": "Coordinates,Velocities,Masses,ParticleIDs",
            "dm": "Coordinates,Velocities,ParticleIDs",
        }

    cutout_url = f"{TNG_BASE}{sim_name}/snapshots/{int(snap)}/subhalos/{int(subfind_id)}/cutout.hdf5"

    r = _api_get(cutout_url, params=fields, stream=True)

    _ensure_dir(os.path.dirname(outpath))
    with open(outpath, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    if not os.path.exists(outpath):
        raise FileNotFoundError(f"Cutout download failed: {outpath}")

    return outpath


# =========================================================
# TREE / HISTORY RESOLUTION
# =========================================================

def _tree_cache_file(tree_cache_dir: str, ID_z0: int) -> str:
    return os.path.join(tree_cache_dir, f"{int(ID_z0)}_main_progenitor_branch.csv")


def load_or_build_tree_for_z0(ID_z0: int,
                              tree_cache_dir: str,
                              sim: str = 'TNG50-1',
                              overwrite: bool = False) -> pd.DataFrame:
    ID_z0 = int(ID_z0)
    _ensure_dir(tree_cache_dir)
    cache_file = _tree_cache_file(tree_cache_dir, ID_z0)

    if (not overwrite) and os.path.exists(cache_file):
        tree = pd.read_csv(cache_file)
    else:
        tree = backend_load_main_progenitor_branch(ID_z0=ID_z0, sim=sim)
        tree.to_csv(cache_file, index=False)

    required_cols = {'SnapNum', 'SubfindID'}
    missing = required_cols - set(tree.columns)
    if missing:
        raise ValueError(f"Tree for ID_z0={ID_z0} is missing required columns: {missing}")

    tree = tree.copy()
    tree['SnapNum'] = tree['SnapNum'].astype(int)
    tree['SubfindID'] = tree['SubfindID'].astype(int)

    return tree.sort_values('SnapNum', ascending=False).reset_index(drop=True)


def resolve_subfind_at_snap_from_z0(ID_z0: int,
                                    snap: int,
                                    dfSubfindID: Optional[pd.DataFrame] = None,
                                    tree_cache_dir: Optional[str] = None,
                                    sim: str = 'TNG50-1') -> Tuple[int, str]:
    ID_z0 = int(ID_z0)
    snap = int(snap)

    if dfSubfindID is not None and str(ID_z0) in dfSubfindID.columns:
        val = dfSubfindID[str(ID_z0)].iloc[99 - snap]
        if pd.isna(val):
            raise ValueError(f"ID_z0={ID_z0} exists in local DF but has NaN at snap={snap}")
        return int(val), 'local_df'

    if tree_cache_dir is None:
        raise ValueError("tree_cache_dir must be provided when local DF does not contain the ID.")

    tree = load_or_build_tree_for_z0(ID_z0, tree_cache_dir=tree_cache_dir, sim=sim)
    row = tree.loc[tree['SnapNum'] == snap]

    if len(row) == 0:
        raise ValueError(f"No progenitor found in tree for ID_z0={ID_z0} at snap={snap}")

    return int(row['SubfindID'].iloc[0]), 'tree_fallback'


def get_halfmass_radius_for_z0(ID_z0: int,
                               snap: int,
                               ptype: int,
                               df_local: Optional[pd.DataFrame],
                               dfSubfindID: Optional[pd.DataFrame],
                               tree_cache_dir: str,
                               sim: str = 'TNG50-1') -> float:
    ID_z0 = int(ID_z0)
    snap = int(snap)

    if df_local is not None and str(ID_z0) in df_local.columns:
        val = df_local[str(ID_z0)].iloc[99 - snap]
        if pd.notna(val):
            return 10 ** float(val)

    subfind_id, _ = resolve_subfind_at_snap_from_z0(
        ID_z0, snap, dfSubfindID=dfSubfindID, tree_cache_dir=tree_cache_dir, sim=sim
    )

    tree = load_or_build_tree_for_z0(ID_z0, tree_cache_dir=tree_cache_dir, sim=sim)
    row = tree.loc[tree['SnapNum'] == snap]

    colname = f'SubhaloHalfmassRadType{ptype}'
    if colname in tree.columns and len(row) > 0 and pd.notna(row[colname].iloc[0]):
        return float(row[colname].iloc[0] / (1+dfTime.z.loc[dfTime.Snap == snap].values[0]) / h )

    cat = backend_load_subhalo_catalog_entry(snap=snap, subfind_id=subfind_id, sim=sim)
    if 'SubhaloHalfmassRadType' not in cat:
        raise KeyError("Catalog entry does not contain 'SubhaloHalfmassRadType'")

    return float(cat['SubhaloHalfmassRadType'][ptype] / (1+dfTime.z.loc[dfTime.Snap == snap].values[0]) / h  )


def get_subhalo_state_from_z0(ID_z0: int,
                              snap: int,
                              tree_cache_dir: str,
                              sim: str = 'TNG50-1') -> dict:
    dfSubfindID = TNG.extractDF('SubfindID')
    dfHalf0 = TNG.extractDF('SubhaloHalfmassRadType0')
    dfHalf4 = TNG.extractDF('SubhaloHalfmassRadType4')

    subfind_id, source = resolve_subfind_at_snap_from_z0(
        ID_z0, snap, dfSubfindID=dfSubfindID, tree_cache_dir=tree_cache_dir, sim=sim
    )

    rhalf0 = get_halfmass_radius_for_z0(
        ID_z0, snap, ptype=0, df_local=dfHalf0, dfSubfindID=dfSubfindID,
        tree_cache_dir=tree_cache_dir, sim=sim
    )

    rhalf4 = get_halfmass_radius_for_z0(
        ID_z0, snap, ptype=4, df_local=dfHalf4, dfSubfindID=dfSubfindID,
        tree_cache_dir=tree_cache_dir, sim=sim
    )

    return {
        'ID_z0': int(ID_z0),
        'snap': int(snap),
        'subfind_id': int(subfind_id),
        'rhalf0': float(rhalf0) if np.isfinite(rhalf0) else np.nan,
        'rhalf4': float(rhalf4) if np.isfinite(rhalf4) else np.nan,
        'source': source
    }


# =========================================================
# CUTOUT RESOLUTION
# =========================================================

def _local_cutout_path_from_subfind(HOME: str, SIMS: str, snap: int, subfind_id: int) -> str:
    return os.path.join(
        HOME, 'SIMS', 'TNG', SIMS,
        'snapshots', str(snap),
        'subhalos', str(int(subfind_id)),
        f'cutout_{SIMS}_{snap}_{int(subfind_id)}.hdf5'
    )


def ensure_cutout_for_z0_id(
    ID_z0: int,
    snap: int,
    PATH: str,
    SIMS: str = 'TNG50-1',
    SIMTNG: str = 'TNG50',
    HOME: Optional[str] = None,
    TREE_CACHE_DIR: Optional[str] = None,
    force_download: bool = False
) -> Tuple[str, dict]:
    if HOME is None:
        HOME = os.getenv("HOME") + '/'

    if TREE_CACHE_DIR is None:
        TREE_CACHE_DIR = os.path.join(PATH, SIMTNG, 'Trees')

    state = get_subhalo_state_from_z0(
        ID_z0=ID_z0,
        snap=snap,
        tree_cache_dir=TREE_CACHE_DIR,
        sim=SIMS
    )

    subfind_id_at_snap = state['subfind_id']

    target_dir = os.path.join(PATH, SIMTNG, 'Particles', str(int(ID_z0)))
    _ensure_dir(target_dir)
    target_file = os.path.join(target_dir, f'{int(snap)}Rotate.hdf5')

    if (not force_download) and os.path.exists(target_file):
        return target_file, state

    local_origin = _local_cutout_path_from_subfind(HOME, SIMS, snap, subfind_id_at_snap)

    if (not force_download) and os.path.exists(local_origin):
        shutil.copyfile(local_origin, target_file)
        return target_file, state

    backend_download_cutout_by_subfindid(
        snap=snap,
        subfind_id=subfind_id_at_snap,
        outpath=target_file,
        sim=SIMS
    )

    if not os.path.exists(target_file):
        raise FileNotFoundError(
            f"Backend download did not create expected file: {target_file}"
        )

    return target_file, state


# =========================================================
# PHYSICS HELPERS
# =========================================================

def galaxy_Lhat_and_jspec(pos, vel, mass, R_ap):
    r = np.linalg.norm(pos, axis=1)
    sel = (r <= R_ap) & np.isfinite(r) & np.isfinite(mass)

    if np.sum(sel) < 10:
        return None, np.nan

    r_sel = pos[sel]
    v_sel = vel[sel]
    m_sel = mass[sel]

    J_vec = np.sum(m_sel[:, None] * np.cross(r_sel, v_sel), axis=0)
    J_norm = np.linalg.norm(J_vec)

    if (not np.isfinite(J_norm)) or (J_norm == 0):
        return None, np.nan

    Lhat = J_vec / J_norm
    Mtot = np.sum(m_sel)
    j_spec = J_norm / Mtot

    return Lhat, j_spec


def mass_weighted_map(pos, mass, value, r, t, prad, width):
    qv_m = QuickView(
        pos, mass=mass, plot=False, r=r, t=t, x=0, y=0, z=0, p=prad,
        extent=[-width, width, -width, width], logscale=False
    )
    mass_field = qv_m.get_image()
    extent = qv_m.get_extent()

    qv_mv = QuickView(
        pos, mass=mass * value, plot=False, r=r, t=t, x=0, y=0, z=0, p=prad,
        extent=[-width, width, -width, width], logscale=False
    )
    mv_field = qv_mv.get_image()

    out = np.full_like(mass_field, np.nan, dtype=float)
    mask = mass_field > 0
    out[mask] = mv_field[mask] / mass_field[mask]

    return out, extent


# =========================================================
# CORE MAP BUILDER
# =========================================================

def MakeMap(
    ID,
    snap,
    PartType='PartType0',
    axis='x',
    r='infinity',
    Param=None,
    VelPlot=False,
    seed=160401,
    lenLim=2500000,
    width=25,
    cmap_dm='GnBu',
    PATH=os.path.join(os.getenv("HOME"), 'TNG_Analyzes', 'SubhaloHistory'),
    SIMS='TNG50-1',
    SIMTNG='TNG50',
    HOME=os.getenv("HOME") + '/',
    TREE_CACHE_DIR=None,
    FORCE_CUTOUT_DOWNLOAD=False
):
    _validate_axis(axis)

    ID = int(ID)
    snap = int(snap)

    if TREE_CACHE_DIR is None:
        TREE_CACHE_DIR = os.path.join(PATH, SIMTNG, 'Trees')

    target_file, state = ensure_cutout_for_z0_id(
        ID_z0=ID,
        snap=snap,
        PATH=PATH,
        SIMS=SIMS,
        SIMTNG=SIMTNG,
        HOME=HOME,
        TREE_CACHE_DIR=TREE_CACHE_DIR,
        force_download=FORCE_CUTOUT_DOWNLOAD
    )
    

    rhalf0 = state['rhalf0']
    rhalf4 = state['rhalf4']

    try:
        file = h5py.File(target_file, 'r')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Cutout file could not be opened for ID_z0={ID}, snap={snap}: {target_file}"
        ) from e

    needs_rotation = True
    try:
        flag = file.attrs.get("RotatedToAngularMomentumFrame", None)
        if flag is not None:
            needs_rotation = False
    except Exception:
        pass

    if needs_rotation:
        file.close()
        file = h5py.File(target_file, 'r+')
        zsnap = _get_redshift_from_snap(snap)

        if 'PartType0' in file.keys() and np.isfinite(rhalf0) and rhalf0 > 0:
            file = MATH.Rotate(file, rhalf0, z=zsnap, TNG=True)
            try:
                file.attrs["RotatedToAngularMomentumFrame"] = True
            except Exception:
                pass
        elif 'PartType4' in file.keys() and np.isfinite(rhalf4) and rhalf4 > 0:
            file = MATH.Rotate(file, rhalf4, z=zsnap, TNG=True)
            try:
                file.attrs["RotatedToAngularMomentumFrame"] = True
            except Exception:
                pass

    np.random.seed(seed)

    pos_gas, mass_gas, vel_gas = _try_read_particle_block(file, 'PartType0', snap)
    pos_star, mass_star, vel_star = _try_read_particle_block(file, 'PartType4', snap)

    if len(mass_star) > 1 and mass_star[0] != 0:
        Cen, VelMean = _mass_weighted_center_and_velocity(pos_star, vel_star, mass_star)
    else:
        Cen, VelMean = _mass_weighted_center_and_velocity(pos_gas, vel_gas, mass_gas)

    if PartType not in file.keys():
        file.close()
        if VelPlot:
            return None, None, None, None, None, None
        return None, None

    pos, mass, vel = _read_particle_block(file, PartType, snap)
    pos = FixPeriodic(pos - Cen)
    vel = vel - VelMean

    ncoords = len(file[PartType]['Coordinates'])
    if ncoords > lenLim:
        args = np.random.choice(np.arange(ncoords), size=lenLim, replace=False)
        pos = pos[args]
        mass = mass[args]
        vel = vel[args]

    prad, t, VelArrayX, VelArrayY, PosArrayX, PosArrayY = _get_projection_setup(axis, vel, pos, VelPlot)

    img = None
    extent = None

    if Param == 'Mass':
        qv = QuickView(
            pos, mass=mass, plot=False, r=r, t=t, x=0, y=0, z=0, p=prad,
            extent=[-width, width, -width, width], logscale=False
        )
        img = qv.get_image()
        extent = qv.get_extent()

    elif Param == 'T':
        if 'PartType0' not in file:
            img, extent = None, None
        else:
            u = file['PartType0']['InternalEnergy'][:]
            xe = file['PartType0']['ElectronAbundance'][:]

            Xh = 0.76
            gamma = 5.0 / 3.0
            mp = 1.673e-24
            kb = 1.380658e-16

            mu = 4.0 / (1.0 + 3.0 * Xh + 4.0 * Xh * xe) * mp
            T = (gamma - 1.0) * (u / kb) * mu
            T = T * 1e10

            qv = QuickView(
                pos, mass=T, plot=False, r=r, t=t, x=0, y=0, z=0, p=prad,
                extent=[-width, width, -width, width], logscale=False
            )
            img = qv.get_image()
            extent = qv.get_extent()

    elif Param == 'SFR':
        if 'PartType0' not in file:
            img, extent = None, None
        else:
            sfr = file['PartType0']['StarFormationRate'][:]

            qv = QuickView(
                pos, mass=mass, r=r, t=t, x=0, y=0, z=0, p=prad, plot=False,
                extent=[-width, width, -width, width], logscale=False
            )
            density_field = qv.get_image()

            qv = QuickView(
                pos, mass=sfr, plot=False, r=r, t=t, x=0, y=0, z=0, p=prad,
                extent=[-width, width, -width, width], logscale=False
            )
            img = qv.get_image()

            valid = density_field > 0
            out = np.full_like(img, np.nan, dtype=float)
            out[valid] = img[valid] / density_field[valid]
            img = out
            extent = qv.get_extent()

    elif Param == 'MassSF':
        if 'PartType0' not in file:
            img, extent = None, None
        else:
            sfr = file['PartType0']['StarFormationRate'][:]
            mask = sfr > 0

            if np.sum(mask) == 0:
                img, extent = None, None
            else:
                qv = QuickView(
                    pos[mask], mass=mass[mask], plot=False, r=r, t=t, x=0, y=0, z=0, p=prad,
                    extent=[-width, width, -width, width], logscale=False
                )
                img = qv.get_image()
                extent = qv.get_extent()

    elif _safe_str_contains(Param, 'Situ'):
        ids = file[PartType]['ParticleIDs'][:]
        df = pd.read_csv(os.path.join(PATH, SIMTNG, 'DFs', f'{ID}_StarParticles.csv'))

        origin = df.Origin.values
        id_df = df.Star_ID.values

        if 'Ex' in Param:
            id_select = id_df[origin == 'E']
        elif 'In' in Param:
            id_select = id_df[origin == 'I']
        else:
            raise ValueError(f"Situ parameter not recognized: {Param}")

        cond = np.isin(ids, id_select)

        if np.sum(cond) == 0:
            img, extent = None, None
        else:
            qv = QuickView(
                pos[cond], mass=mass[cond], plot=False, r=r, t=t, x=0, y=0, z=0, p=prad,
                extent=[-width, width, -width, width], logscale=False
            )
            img = qv.get_image()
            extent = qv.get_extent()

    elif Param in ['Jnorm', 'Jpar', 'Jpar_norm']:
        R_ap = width
        Lhat, j_gal_spec = galaxy_Lhat_and_jspec(pos, vel, mass, R_ap=R_ap)

        if Lhat is None or (not np.isfinite(j_gal_spec)) or (j_gal_spec == 0):
            img, extent = None, None
        else:
            j_par = np.einsum('ij,j->i', np.cross(pos, vel), Lhat)

            if Param == 'Jpar':
                img, extent = mass_weighted_map(pos, mass, j_par, r, t, prad, width)
            else:
                j_norm = j_par / j_gal_spec
                img, extent = mass_weighted_map(pos, mass, j_norm, r, t, prad, width)

    elif Param in ['Vphi', 'Jaxis', 'Vlos', 'SigmaJaxis']:
        if axis == 'z':
            a, b = pos[:, 0], pos[:, 1]
            va, vb = vel[:, 0], vel[:, 1]
            vlos = vel[:, 2]
        elif axis == 'x':
            a, b = pos[:, 1], pos[:, 2]
            va, vb = vel[:, 1], vel[:, 2]
            vlos = vel[:, 0]
        else:
            raise ValueError(f"Invalid axis: {axis}")

        R = np.sqrt(a * a + b * b)
        R_safe = np.where(R > 0, R, np.nan)

        j_axis = a * vb - b * va
        v_phi = j_axis / R_safe

        if Param == 'Vphi':
            img, extent = mass_weighted_map(pos, mass, v_phi, r, t, prad, width)
        elif Param == 'Jaxis':
            img, extent = mass_weighted_map(pos, mass, j_axis, r, t, prad, width)
        elif Param == 'SigmaJaxis':
            qv = QuickView(
                pos, mass=mass * j_axis, plot=False, r=r, t=t, x=0, y=0, z=0, p=prad,
                extent=[-width, width, -width, width], logscale=False
            )
            img = qv.get_image()
            extent = qv.get_extent()
        elif Param == 'Vlos':
            img, extent = mass_weighted_map(pos, mass, vlos, r, t, prad, width)

    elif isinstance(Param, str) and len(Param) > 0:
        if Param not in file[PartType]:
            file.close()
            raise KeyError(f"Dataset '{Param}' not found in {PartType}")

        param_data = file[PartType][Param][:]
        qv = QuickView(
            pos, mass=param_data, r=r, t=t, x=0, y=0, z=0, p=prad, plot=False,
            extent=[-width, width, -width, width], logscale=False
        )
        img = qv.get_image()
        extent = qv.get_extent()

    else:
        qv = QuickView(
            pos, plot=False, r=r, t=t, x=0, y=0, z=0, p=prad,
            extent=[-width, width, -width, width]
        )
        img = qv.get_image()
        extent = qv.get_extent()

    file.close()

    if VelPlot:
        return img, extent, VelArrayX, VelArrayY, PosArrayX, PosArrayY
    return img, extent


# =========================================================
# FIGURE BUILDER
# =========================================================

def _get_orbital_snap(id_main: int, j: int, df_name: str) -> int:
    r_over_R_Crit200 = TNG.extractDF('r_over_R_Crit200_FirstGroup')
    df_sample = TNG.extractDF(df_name)

    snap_at_entry = df_sample.loc[df_sample.SubfindID_99 == id_main, 'Snap_At_FirstEntry'].values[0]
    snap_lost_gas = df_sample.loc[df_sample.SubfindID_99 == id_main, 'SnapLostGas'].values[0]

    r_over_r200 = np.flip(np.array([v for v in r_over_R_Crit200[str(id_main)].values], dtype=float))
    r_over_r200[:int(snap_at_entry)] = np.nan

    pericenters, _ = TNG.find_peaks(-r_over_r200)

    if j == 0:
        return _snap_from_age_window(int(snap_at_entry), 0.5, mode="min")

    if len(pericenters) == 0:
        first_peri = int(snap_at_entry)
    else:
        first_peri = int(np.nanmin(pericenters))

    if j == 1:
        return _snap_from_age_window(first_peri, 0.5, mode="min")
    if j == 2:
        return first_peri
    if j == 3:
        return _snap_from_age_window(first_peri, 0.5, mode="max")
    if j == 4:
        if np.isnan(snap_lost_gas) or snap_lost_gas < 0:
            snap_lost_gas = 99
        return _snap_from_age_window(int(snap_lost_gas), 0.5, mode="min")

    raise ValueError(f"OrbitalEvolution only supports j=0..4, got {j}")


def PlotMap(
    IDs,
    PartType,
    snaps,
    Names=None,
    Type='Mosaic',
    width=25,
    cNum=1,
    lNum=1,
    xtext=0,
    ytext=15,
    xtextZ=10,
    ytextZ=-15,
    r='infinity',
    axis='x',
    cmap_dm='bone',
    dfName='Sample',
    SIMS='TNG50-1',
    SIMTNG='TNG50',
    vmin=None,
    vmax=None,
    VelPlot=False,
    Param=None,
    savepath='fig/PlotMap',
    savefigname='fig',
    OrbitalEvolution=False,
    fontlabel=20,
    PATH=os.path.join(os.getenv("HOME"), 'TNG_Analyzes', 'SubhaloHistory'),
    HOME=os.getenv("HOME") + '/',
    TREE_CACHE_DIR=None,
    FORCE_CUTOUT_DOWNLOAD=False
):
    _validate_axis(axis)
    _validate_plot_type(Type)

    IDs = np.array(IDs, dtype=float)

    if TREE_CACHE_DIR is None:
        TREE_CACHE_DIR = os.path.join(PATH, SIMTNG, 'Trees')

    if Type == 'Evolution':
        cNum = 5 if OrbitalEvolution else len(snaps)
        lNum = len(IDs)

    total_panels = cNum * lNum
    Names = _prepare_names_array(Names, total_panels)

    if Type == 'Mosaic':
        snap = int(snaps[0])
        IDs_full = _prepare_ids_array(IDs, total_panels)

        if cNum == 1 and lNum == 1:
            IDsReshape = np.array([IDs_full])
        else:
            IDsReshape = IDs_full.reshape((lNum, cNum))

    plt.rcParams.update({'figure.figsize': (cNum * 4, lNum * 4)})
    fig = plt.figure()

    grid = AxesGrid(
        fig,
        (0.075, 0.075, 0.85, 0.85),
        nrows_ncols=(lNum, cNum),
        axes_pad=0.0,
        label_mode="L",
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15
    )

    sc = None

    for i in range(lNum):
        for j in range(cNum):
            ax = grid[i * cNum + j]

            ID = np.nan
            Name = "None"

            if Type == 'Mosaic':
                ID = IDsReshape[i, j]
                Name = Names[i * cNum + j]
            elif Type == 'Evolution':
                ID = IDs[i]
                Name = Names[i]

                if OrbitalEvolution:
                    snap = _get_orbital_snap(int(ID), j, dfName)
                else:
                    snap = int(snaps[j])

            if np.isnan(ID):
                ax.axis('off')
                continue
            
            print('ID: ', ID)

            panel_cmap, panel_vmin, panel_vmax, do_log = _panel_display_settings(
                Param, PartType, cmap_dm, vmin, vmax
            )

            try:
                if VelPlot:
                    img, extent, VelArrayX, VelArrayY, PosArrayX, PosArrayY = MakeMap(
                        ID, snap,
                        PartType=PartType,
                        VelPlot=True,
                        axis=axis,
                        Param=Param,
                        r=r,
                        width=width,
                        cmap_dm=panel_cmap,
                        SIMS=SIMS,
                        SIMTNG=SIMTNG,
                        PATH=PATH,
                        HOME=HOME,
                        TREE_CACHE_DIR=TREE_CACHE_DIR,
                        FORCE_CUTOUT_DOWNLOAD=FORCE_CUTOUT_DOWNLOAD
                    )
                else:
                    img, extent = MakeMap(
                        ID, snap,
                        PartType=PartType,
                        axis=axis,
                        Param=Param,
                        r=r,
                        width=width,
                        cmap_dm=panel_cmap,
                        SIMS=SIMS,
                        SIMTNG=SIMTNG,
                        PATH=PATH,
                        HOME=HOME,
                        TREE_CACHE_DIR=TREE_CACHE_DIR,
                        FORCE_CUTOUT_DOWNLOAD=FORCE_CUTOUT_DOWNLOAD
                    )
            except Exception as e:
                logger.warning(f"Failed to build map for ID_z0={ID}, snap={snap}: {e}")
                ax.axis('off')
                continue

            if isinstance(img, np.ndarray) and extent is not None:
                img_plot = _safe_log10_image(img) if do_log else np.where(np.isfinite(img), img, np.nan)

                if Param is not None:
                    sc = ax.imshow(
                        img_plot, extent=extent, cmap=panel_cmap, origin='lower',
                        vmin=panel_vmin, vmax=panel_vmax
                    )
                else:
                    sc = ax.imshow(
                        img_plot, extent=extent, cmap=panel_cmap, origin='lower'
                    )

                if VelPlot:
                    _make_streamplot(ax, PosArrayX, PosArrayY, VelArrayX, VelArrayY)

                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
            else:
                ax.axis('off')
                continue

            if j == 0:
                if axis == 'z':
                    ax.set_ylabel(r'$y/\mathrm{kpc}$', fontsize=fontlabel)
                elif axis == 'x':
                    ax.set_ylabel(r'$z/\mathrm{kpc}$', fontsize=fontlabel)

                ax.tick_params(axis='y', labelsize=0.98 * fontlabel)

            if i == 0:
                ax2label = ax.secondary_xaxis('top')
                ax2label.set_xlabel(r'$x/\mathrm{kpc}$', fontsize=fontlabel)
                ax2label.tick_params(labelsize=0.99 * fontlabel)

            if i == lNum - 1:
                ax.set_xlabel(r'$x/\mathrm{kpc}$', fontsize=fontlabel)
                ax.tick_params(axis='x', labelsize=0.98 * fontlabel)

            if Type == 'Evolution' and i == 0 and not OrbitalEvolution:
                ax.set_title(_format_zlabel(snap), fontsize=1.02 * fontlabel)
            elif OrbitalEvolution:
                ax.text(xtextZ, ytextZ, _format_zlabel(snap), color='blue', fontsize=fontlabel)

            if Type == 'Evolution' and j == cNum - 1 and Name != 'None':
                ax.text(xtext, ytext, f'ID: {int(ID)}\n{Name}', color='red', fontsize=fontlabel)

            if Type == 'Mosaic' and Name != 'None':
                ax.text(xtext, ytext, f'ID: {int(ID)}\n{Name}', color='red', fontsize=fontlabel)

    if (Param is not None) and (sc is not None):
        cb = fig.colorbar(sc, cax=grid.cbar_axes[0])

        if Param == 'Mass':
            cb.set_label(r'$\log(M/\mathrm{M}_\odot)$', fontsize=fontlabel)
        elif Param == 'T':
            cb.set_label(r'$\log(T/\mathrm{K})$', fontsize=fontlabel)
        elif Param == 'MassSF':
            cb.set_label(r'$\log(M_{\mathrm{sf-gas}}/\mathrm{M}_\odot)$', fontsize=fontlabel)
        elif Param == 'Vphi':
            cb.set_label(r'$v_\phi\;[\mathrm{km\,s^{-1}}]$', fontsize=fontlabel)
        elif Param == 'Vlos':
            cb.set_label(r'$v_{\rm los}\;[\mathrm{km\,s^{-1}}]$', fontsize=fontlabel)
        elif Param == 'Jaxis':
            cb.set_label(r'$j_{\rm axis}\;[\mathrm{kpc\,km\,s^{-1}}]$', fontsize=fontlabel)
        elif Param == 'SigmaJaxis':
            cb.set_label(r'$\Sigma_J\;[\mathrm{M_\odot\,kpc\,km\,s^{-1}}]$', fontsize=fontlabel)
        elif Param in ['Jnorm', 'Jpar_norm']:
            cb.set_label(r'$j_{\parallel}/j_{\rm gal}$', fontsize=fontlabel)

        cb.ax.tick_params(labelsize=0.98 * fontlabel)

    plot.savefig(savepath, savefigname, False)
    return



# =========================================================
# SUBHALO DISTRIBUTION / GROUPING
# =========================================================

def backend_load_subhalo_spatial_entry(snap: int, subfind_id: int, sim: str = 'TNG50-1') -> dict:
    """
    Load spatial/basic properties of one subhalo from the TNG API catalog.
    Tries a few possible key conventions to remain robust.
    """
    sim_name = _sim_api_name(sim)
    subhalo_url = f"{TNG_BASE}{sim_name}/snapshots/{int(snap)}/subhalos/{int(subfind_id)}/"
    sub = _api_get(subhalo_url).json()

    # --- position ---
    if all(k in sub for k in ["pos_x", "pos_y", "pos_z"]):
        pos = np.array([sub["pos_x"], sub["pos_y"], sub["pos_z"]], dtype=float)
    elif "pos" in sub:
        pos = np.array(sub["pos"], dtype=float)
    elif "SubhaloPos" in sub:
        pos = np.array(sub["SubhaloPos"], dtype=float)
    else:
        raise KeyError(f"Could not find subhalo position keys in API response for snap={snap}, subfind={subfind_id}")

    # --- velocity ---
    if all(k in sub for k in ["vel_x", "vel_y", "vel_z"]):
        vel = np.array([sub["vel_x"], sub["vel_y"], sub["vel_z"]], dtype=float)
    elif "vel" in sub:
        vel = np.array(sub["vel"], dtype=float)
    elif "SubhaloVel" in sub:
        vel = np.array(sub["SubhaloVel"], dtype=float)
    else:
        vel = np.array([np.nan, np.nan, np.nan], dtype=float)

    # --- halfmass radii ---
    if "halfmassrad_type" in sub:
        rhalf = np.array(sub["halfmassrad_type"], dtype=float)
    elif "SubhaloHalfmassRadType" in sub:
        rhalf = np.array(sub["SubhaloHalfmassRadType"], dtype=float)
    else:
        rhalf = np.full(6, np.nan)

    # --- masses ---
    if "mass_log_msun" in sub:
        mtotal_log = float(sub["mass_log_msun"])
    elif "mass" in sub:
        mtotal_log = np.log10(float(sub["mass"]) * 1e10 / h) if sub["mass"] > 0 else np.nan
    else:
        mtotal_log = np.nan

    if "massinhalfrad_type" in sub:
        mhalf_type = np.array(sub["massinhalfrad_type"], dtype=float)
    elif "SubhaloMassInHalfRadType" in sub:
        mhalf_type = np.array(sub["SubhaloMassInHalfRadType"], dtype=float)
    else:
        mhalf_type = np.full(6, np.nan)

    # --- group number ---
    if "grnr" in sub:
        grnr = int(sub["grnr"])
    elif "SubhaloGrNr" in sub:
        grnr = int(sub["SubhaloGrNr"])
    else:
        grnr = -1

    return {
        "pos": pos,
        "vel": vel,
        "rhalf0": float(rhalf[0]) if len(rhalf) > 0 else np.nan,
        "rhalf4": float(rhalf[4]) if len(rhalf) > 4 else np.nan,
        "mtotal_logmsun": mtotal_log,
        "mstar_half": float(mhalf_type[4]) if len(mhalf_type) > 4 else np.nan,
        "mgas_half": float(mhalf_type[0]) if len(mhalf_type) > 0 else np.nan,
        "grnr": grnr,
    }


def _friends_of_friends_labels(pos: np.ndarray, linking_length: float) -> np.ndarray:
    """
    Simple FoF grouping in 3D using a fixed linking length (same units as pos).
    Returns cluster labels.
    """
    n = len(pos)
    if n == 0:
        return np.array([], dtype=int)

    labels = -np.ones(n, dtype=int)
    current_label = 0

    for i in range(n):
        if labels[i] != -1:
            continue

        labels[i] = current_label
        queue = [i]

        while queue:
            k = queue.pop(0)
            dx = pos - pos[k]
            dist = np.sqrt(np.sum(dx**2, axis=1))
            neighbors = np.where(dist <= linking_length)[0]

            for nb in neighbors:
                if labels[nb] == -1:
                    labels[nb] = current_label
                    queue.append(nb)

        current_label += 1

    return labels


def build_subhalo_distribution_catalog(
    IDs_z0,
    snap,
    ref_id=None,
    sim='TNG50-1',
    SIMTNG='TNG50',
    PATH=os.path.join(os.getenv("HOME"), 'TNG_Analyzes', 'SubhaloHistory'),
    TREE_CACHE_DIR=None,
    linking_length=30.0
):
    """
    Build a catalog with the spatial distribution of a list of z=0 IDs at a given snapshot.

    Parameters
    ----------
    IDs_z0 : sequence
        z=0 subhalo IDs
    snap : int
        snapshot where the distribution will be evaluated
    ref_id : int or None
        z=0 ID used as the reference center. If None, the median position is used.
    linking_length : float
        FoF-like linking length in kpc (physical)

    Returns
    -------
    df : pandas.DataFrame
        table with positions, group IDs, and basic properties
    """
    IDs_z0 = np.array(IDs_z0, dtype=int)
    snap = int(snap)

    if TREE_CACHE_DIR is None:
        TREE_CACHE_DIR = os.path.join(PATH, SIMTNG, 'Trees')

    rows = []

    for ID_z0 in IDs_z0:
        try:
            state = get_subhalo_state_from_z0(
                ID_z0=ID_z0,
                snap=snap,
                tree_cache_dir=TREE_CACHE_DIR,
                sim=sim
            )

            entry = backend_load_subhalo_spatial_entry(
                snap=snap,
                subfind_id=state['subfind_id'],
                sim=sim
            )

            # API positions are typically ckpc/h; convert to physical kpc
            zsnap = _get_redshift_from_snap(snap)
            pos_phys = np.array(entry["pos"], dtype=float) / (1.0 + zsnap) / h

            rows.append({
                "ID_z0": int(ID_z0),
                "snap": snap,
                "subfind_id": int(state["subfind_id"]),
                "x": pos_phys[0],
                "y": pos_phys[1],
                "z": pos_phys[2],
                "vx": entry["vel"][0],
                "vy": entry["vel"][1],
                "vz": entry["vel"][2],
                "rhalf0": entry["rhalf0"],
                "rhalf4": entry["rhalf4"],
                "mtotal_logmsun": entry["mtotal_logmsun"],
                "grnr": entry["grnr"],
            })

        except Exception as e:
            logger.warning(f"Failed to load spatial info for ID_z0={ID_z0} at snap={snap}: {e}")

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError("No valid subhalos could be loaded.")

    # reference center
    if ref_id is not None and np.any(df["ID_z0"].values == int(ref_id)):
        ref = df.loc[df["ID_z0"] == int(ref_id)].iloc[0]
        center = np.array([ref["x"], ref["y"], ref["z"]], dtype=float)
    else:
        center = np.array([
            np.nanmedian(df["x"].values),
            np.nanmedian(df["y"].values),
            np.nanmedian(df["z"].values)
        ], dtype=float)

    rel = np.vstack([
        FixPeriodic(df["x"].values - center[0], sim=sim),
        FixPeriodic(df["y"].values - center[1], sim=sim),
        FixPeriodic(df["z"].values - center[2], sim=sim)
    ]).T

    df["dx"] = rel[:, 0]
    df["dy"] = rel[:, 1]
    df["dz"] = rel[:, 2]
    df["r3d"] = np.sqrt(df["dx"]**2 + df["dy"]**2 + df["dz"]**2)

    labels = _friends_of_friends_labels(rel, linking_length=linking_length)
    df["cluster_id"] = labels

    # cluster sizes
    counts = df["cluster_id"].value_counts().to_dict()
    df["cluster_size"] = df["cluster_id"].map(counts)

    return df


def PlotSubhaloDistribution(
    IDs_z0,
    snap,
    ref_id=None,
    axis='z',
    sim='TNG50-1',
    SIMTNG='TNG50',
    PATH=os.path.join(os.getenv("HOME"), 'TNG_Analyzes', 'SubhaloHistory'),
    TREE_CACHE_DIR=None,
    linking_length=30.0,
    width=None,
    annotate=True,
    draw_rhalf=True,
    marker_scale=180,
    alpha=0.85,
    savepath='fig/PlotMap',
    savefigname='SubhaloDistribution',
    fontlabel=16
):
    """
    Plot the spatial distribution of selected subhalos in a common reference frame.

    axis='z'  -> projection on (x,y)
    axis='x'  -> projection on (y,z)
    """
    _validate_axis(axis)

    df = build_subhalo_distribution_catalog(
        IDs_z0=IDs_z0,
        snap=snap,
        ref_id=ref_id,
        sim=sim,
        SIMTNG=SIMTNG,
        PATH=PATH,
        TREE_CACHE_DIR=TREE_CACHE_DIR,
        linking_length=linking_length
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    # choose projected coordinates
    if axis == 'z':
        px = df["dx"].values
        py = df["dy"].values
        xlabel = r'$x/\mathrm{kpc}$'
        ylabel = r'$y/\mathrm{kpc}$'
    elif axis == 'x':
        px = df["dy"].values
        py = df["dz"].values
        xlabel = r'$y/\mathrm{kpc}$'
        ylabel = r'$z/\mathrm{kpc}$'

    # marker sizes
    mlog = df["mtotal_logmsun"].values
    if np.all(~np.isfinite(mlog)):
        sizes = np.full(len(df), marker_scale)
    else:
        mmin = np.nanmin(mlog[np.isfinite(mlog)])
        sizes = marker_scale * (0.4 + (np.nan_to_num(mlog, nan=mmin) - mmin + 0.2))

    # color by cluster_id
    cluster_ids = np.sort(df["cluster_id"].unique())
    cmap = plt.cm.tab10

    for i, cid in enumerate(cluster_ids):
        sel = df["cluster_id"].values == cid
        color = cmap(i % 10)

        ax.scatter(
            px[sel], py[sel],
            s=sizes[sel],
            color=color,
            alpha=alpha,
            edgecolor='k',
            linewidth=0.8,
            label=f'cluster {cid} (N={np.sum(sel)})'
        )

        if draw_rhalf:
            for _, row in df.loc[sel].iterrows():
                rr = row["rhalf4"] if np.isfinite(row["rhalf4"]) else row["rhalf0"]
                if np.isfinite(rr) and rr > 0:
                    if axis == 'z':
                        cx, cy = row["dx"], row["dy"]
                    else:
                        cx, cy = row["dy"], row["dz"]

                    circ = plt.Circle((cx, cy), rr, fill=False, color=color, alpha=0.35, lw=1.0)
                    ax.add_patch(circ)

    if annotate:
        for _, row in df.iterrows():
            if axis == 'z':
                tx, ty = row["dx"], row["dy"]
            else:
                tx, ty = row["dy"], row["dz"]

            ax.text(
                tx, ty,
                f'{int(row["ID_z0"])}',
                fontsize=0.75 * fontlabel,
                ha='left',
                va='bottom'
            )

    ax.axhline(0, ls='--', lw=1, color='gray', alpha=0.5)
    ax.axvline(0, ls='--', lw=1, color='gray', alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=fontlabel)
    ax.set_ylabel(ylabel, fontsize=fontlabel)
    ax.tick_params(labelsize=0.95 * fontlabel)

    zlab = _format_zlabel(snap)
    ax.set_title(
        f'Subhalo spatial distribution\nsnap={snap} ({zlab}), linking_length={linking_length:.1f} kpc',
        fontsize=fontlabel
    )

    if width is not None:
        ax.set_xlim(-width, width)
        ax.set_ylim(-width, width)

    ax.legend(fontsize=0.8 * fontlabel, loc='best', framealpha=0.9)
    ax.set_aspect('equal')

    plot.savefig(savepath, savefigname, False)

    return df, fig, ax


def backend_load_subhalo_spatial_entry(snap: int, subfind_id: int, sim: str = 'TNG50-1') -> dict:
    """
    Load basic spatial and dynamical properties of one subhalo from the TNG API catalog.
    """
    sim_name = _sim_api_name(sim)
    subhalo_url = f"{TNG_BASE}{sim_name}/snapshots/{int(snap)}/subhalos/{int(subfind_id)}/"
    sub = _api_get(subhalo_url).json()

    # position
    if all(k in sub for k in ['pos_x', 'pos_y', 'pos_z']):
        pos = np.array([sub['pos_x'], sub['pos_y'], sub['pos_z']], dtype=float)
    elif 'pos' in sub:
        pos = np.array(sub['pos'], dtype=float)
    elif 'SubhaloPos' in sub:
        pos = np.array(sub['SubhaloPos'], dtype=float)
    else:
        raise KeyError(f"Position not found for snap={snap}, subfind_id={subfind_id}")

    # velocity
    if all(k in sub for k in ['vel_x', 'vel_y', 'vel_z']):
        vel = np.array([sub['vel_x'], sub['vel_y'], sub['vel_z']], dtype=float)
    elif 'vel' in sub:
        vel = np.array(sub['vel'], dtype=float)
    elif 'SubhaloVel' in sub:
        vel = np.array(sub['SubhaloVel'], dtype=float)
    else:
        vel = np.array([np.nan, np.nan, np.nan], dtype=float)

    # halfmass radii
    if 'halfmassrad_type' in sub:
        rhalf = np.array(sub['halfmassrad_type'], dtype=float)
    elif 'SubhaloHalfmassRadType' in sub:
        rhalf = np.array(sub['SubhaloHalfmassRadType'], dtype=float)
    else:
        rhalf = np.full(6, np.nan)

    # mass
    if 'mass_log_msun' in sub:
        mtotal_log = float(sub['mass_log_msun'])
    elif 'mass' in sub and sub['mass'] > 0:
        mtotal_log = np.log10(float(sub['mass']) * 1e10 / h)
    else:
        mtotal_log = np.nan

    # group number
    if 'grnr' in sub:
        grnr = int(sub['grnr'])
    elif 'SubhaloGrNr' in sub:
        grnr = int(sub['SubhaloGrNr'])
    else:
        grnr = -1

    return {
        'pos': pos,
        'vel': vel,
        'rhalf0': float(rhalf[0]) if len(rhalf) > 0 else np.nan,
        'rhalf4': float(rhalf[4]) if len(rhalf) > 4 else np.nan,
        'mtotal_logmsun': mtotal_log,
        'grnr': grnr
    }


def build_group_catalog(
    IDs_z0,
    snap,
    ref_id=None,
    sim='TNG50-1',
    SIMTNG='TNG50',
    PATH=os.path.join(os.getenv("HOME"), 'TNG_Analyzes', 'SubhaloHistory'),
    TREE_CACHE_DIR=None
):
    """
    Build a catalog of selected subhalos in a common group frame.
    """
    IDs_z0 = np.array(IDs_z0, dtype=int)
    snap = int(snap)

    if TREE_CACHE_DIR is None:
        TREE_CACHE_DIR = os.path.join(PATH, SIMTNG, 'Trees')

    rows = []

    for ID_z0 in IDs_z0:
        try:
            state = get_subhalo_state_from_z0(
                ID_z0=ID_z0,
                snap=snap,
                tree_cache_dir=TREE_CACHE_DIR,
                sim=sim
            )

            entry = backend_load_subhalo_spatial_entry(
                snap=snap,
                subfind_id=state['subfind_id'],
                sim=sim
            )

            zsnap = _get_redshift_from_snap(snap)

            # positions from catalog are usually ckpc/h -> convert to physical kpc
            pos = np.array(entry['pos'], dtype=float) / (1.0 + zsnap) / h
            vel = np.array(entry['vel'], dtype=float)

            rows.append({
                'ID_z0': int(ID_z0),
                'subfind_id': int(state['subfind_id']),
                'snap': snap,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2],
                'vx': vel[0],
                'vy': vel[1],
                'vz': vel[2],
                'rhalf0': entry['rhalf0'],
                'rhalf4': entry['rhalf4'],
                'mtotal_logmsun': entry['mtotal_logmsun'],
                'grnr': entry['grnr']
            })

        except Exception as e:
            logger.warning(f"Failed to load ID_z0={ID_z0} at snap={snap}: {e}")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No valid subhalos found.")

    # choose reference object or median center
    if ref_id is not None and np.any(df['ID_z0'].values == int(ref_id)):
        ref = df.loc[df['ID_z0'] == int(ref_id)].iloc[0]
        center = np.array([ref['x'], ref['y'], ref['z']], dtype=float)
        vcenter = np.array([ref['vx'], ref['vy'], ref['vz']], dtype=float)
    else:
        center = np.array([
            np.nanmedian(df['x'].values),
            np.nanmedian(df['y'].values),
            np.nanmedian(df['z'].values)
        ])
        vcenter = np.array([
            np.nanmedian(df['vx'].values),
            np.nanmedian(df['vy'].values),
            np.nanmedian(df['vz'].values)
        ])

    dx = FixPeriodic(df['x'].values - center[0], sim=sim)
    dy = FixPeriodic(df['y'].values - center[1], sim=sim)
    dz = FixPeriodic(df['z'].values - center[2], sim=sim)

    df['dx'] = dx
    df['dy'] = dy
    df['dz'] = dz
    df['dvx'] = df['vx'].values - vcenter[0]
    df['dvy'] = df['vy'].values - vcenter[1]
    df['dvz'] = df['vz'].values - vcenter[2]
    df['r3d'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return df

def PlotGroupMap(
    IDs_z0,
    snap,
    map_id=None,
    ref_id=None,
    PartType='PartType4',
    Param='Mass',
    axis='z',
    width=150,
    r='infinity',
    show_velocity=True,
    annotate=True,
    draw_rhalf=True,
    cmap='bone',
    arrow_scale=1.0,
    marker_scale=180,
    fontlabel=16,
    savepath='fig/PlotMap',
    savefigname='GroupMap',
    SIMS='TNG50-1',
    SIMTNG='TNG50',
    PATH=os.path.join(os.getenv("HOME"), 'TNG_Analyzes', 'SubhaloHistory'),
    HOME=os.getenv("HOME") + '/',
    TREE_CACHE_DIR=None,
    FORCE_CUTOUT_DOWNLOAD=False
):
    """
    Plot a common-frame map of a group and overlay selected subhalos.
    """

    _validate_axis(axis)

    if TREE_CACHE_DIR is None:
        TREE_CACHE_DIR = os.path.join(PATH, SIMTNG, 'Trees')

    # build catalog of selected objects
    df = build_group_catalog(
        IDs_z0=IDs_z0,
        snap=snap,
        ref_id=ref_id,
        sim=SIMS,
        SIMTNG=SIMTNG,
        PATH=PATH,
        TREE_CACHE_DIR=TREE_CACHE_DIR
    )

    # choose object used to build the background particle map
    if map_id is None:
        map_id = int(df.iloc[0]['ID_z0'])

    # build particle map in the frame of map_id
    img, extent = MakeMap(
        ID=map_id,
        snap=snap,
        PartType=PartType,
        axis=axis,
        r=r,
        Param=Param,
        VelPlot=False,
        width=width,
        cmap_dm=cmap,
        PATH=PATH,
        SIMS=SIMS,
        SIMTNG=SIMTNG,
        HOME=HOME,
        TREE_CACHE_DIR=TREE_CACHE_DIR,
        FORCE_CUTOUT_DOWNLOAD=FORCE_CUTOUT_DOWNLOAD
    )

    if img is None or extent is None:
        raise ValueError(f"Could not build background map for map_id={map_id}")

    fig, ax = plt.subplots(figsize=(8, 8))

    do_log = (Param in ['Mass', 'T', 'MassSF', 'SFR']) or _safe_str_contains(Param, 'Mass')
    img_plot = _safe_log10_image(img) if do_log else np.where(np.isfinite(img), img, np.nan)

    ax.imshow(img_plot, extent=extent, origin='lower', cmap=cmap)

    # choose projected coordinates
    if axis == 'z':
        px = df['dx'].values
        py = df['dy'].values
        pvx = df['dvx'].values
        pvy = df['dvy'].values
        xlabel = r'$x/\mathrm{kpc}$'
        ylabel = r'$y/\mathrm{kpc}$'
    elif axis == 'x':
        px = df['dy'].values
        py = df['dz'].values
        pvx = df['dvy'].values
        pvy = df['dvz'].values
        xlabel = r'$y/\mathrm{kpc}$'
        ylabel = r'$z/\mathrm{kpc}$'
        
    inside = (
                    np.isfinite(px) & np.isfinite(py) &
                    (px >= -width) & (px <= width) &
                    (py >= -width) & (py <= width)
                )

    # marker sizes from total mass
    mlog = df['mtotal_logmsun'].values
    if np.all(~np.isfinite(mlog)):
        sizes = np.full(len(df), marker_scale)
    else:
        mmin = np.nanmin(mlog[np.isfinite(mlog)])
        sizes = marker_scale * (0.4 + (np.nan_to_num(mlog, nan=mmin) - mmin + 0.2))

    ax.scatter(
        px[inside], py[inside],
        s=sizes[inside],
        facecolor='none',
        edgecolor='cyan',
        linewidth=1.8,
        zorder=4
    )

    # half-mass radius circles
    if draw_rhalf:
        for _, row in df.iterrows():
            rr = row['rhalf4'] if PartType == 'PartType4' else row['rhalf0']
            if np.isfinite(rr) and rr > 0:
                if axis == 'z':
                    cx, cy = row['dx'], row['dy']
                else:
                    cx, cy = row['dy'], row['dz']

                circ = plt.Circle((cx, cy), rr, fill=False, color='lime', lw=1.2, alpha=0.8, zorder=3,
                                  clip_on=True)
                ax.add_patch(circ)

    # velocity arrows
    if show_velocity:
        ax.quiver(
            px[inside], py[inside],
            pvx[inside], pvy[inside],
            angles='xy', scale_units='xy', scale=20.0 / arrow_scale,
            color='red', width=0.004, zorder=5
        )

    # labels
    if annotate:
        for _, row in df.loc[inside].iterrows():
            if axis == 'z':
                tx, ty = row['dx'], row['dy']
            else:
                tx, ty = row['dy'], row['dz']
    
            ax.text(
                tx, ty,
                f"{int(row['ID_z0'])}",
                color='white',
                fontsize=0.8 * fontlabel,
                ha='left',
                va='bottom',
                zorder=6,
                clip_on=True
            )
            
            

    ax.axhline(0, ls='--', lw=1, color='white', alpha=0.5)
    ax.axvline(0, ls='--', lw=1, color='white', alpha=0.5)

    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.set_xlabel(xlabel, fontsize=fontlabel)
    ax.set_ylabel(ylabel, fontsize=fontlabel)
    ax.set_title(f'Group map | snap={snap} | map_id={map_id}', fontsize=fontlabel)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=0.95 * fontlabel)

    plot.savefig(savepath, savefigname, False)

    return df, fig, ax


def backend_load_subhalo_spatial_entry(snap: int, subfind_id: int, sim: str = 'TNG50-1') -> dict:
    sim_name = _sim_api_name(sim)
    subhalo_url = f"{TNG_BASE}{sim_name}/snapshots/{int(snap)}/subhalos/{int(subfind_id)}/"
    sub = _api_get(subhalo_url).json()

    if all(k in sub for k in ['pos_x', 'pos_y', 'pos_z']):
        pos = np.array([sub['pos_x'], sub['pos_y'], sub['pos_z']], dtype=float)
    elif 'pos' in sub:
        pos = np.array(sub['pos'], dtype=float)
    elif 'SubhaloPos' in sub:
        pos = np.array(sub['SubhaloPos'], dtype=float)
    else:
        raise KeyError(f"Position not found for snap={snap}, subfind={subfind_id}")

    if all(k in sub for k in ['vel_x', 'vel_y', 'vel_z']):
        vel = np.array([sub['vel_x'], sub['vel_y'], sub['vel_z']], dtype=float)
    elif 'vel' in sub:
        vel = np.array(sub['vel'], dtype=float)
    elif 'SubhaloVel' in sub:
        vel = np.array(sub['SubhaloVel'], dtype=float)
    else:
        vel = np.array([np.nan, np.nan, np.nan], dtype=float)

    if 'halfmassrad_type' in sub:
        rhalf = np.array(sub['halfmassrad_type'], dtype=float)
    elif 'SubhaloHalfmassRadType' in sub:
        rhalf = np.array(sub['SubhaloHalfmassRadType'], dtype=float)
    else:
        rhalf = np.full(6, np.nan)

    if 'mass_log_msun' in sub:
        mtotal_log = float(sub['mass_log_msun'])
    elif 'mass' in sub and sub['mass'] > 0:
        mtotal_log = np.log10(float(sub['mass']) * 1e10 / h)
    else:
        mtotal_log = np.nan

    if 'grnr' in sub:
        grnr = int(sub['grnr'])
    elif 'SubhaloGrNr' in sub:
        grnr = int(sub['SubhaloGrNr'])
    else:
        grnr = -1

    return {
        'pos': pos,
        'vel': vel,
        'rhalf0': float(rhalf[0]) if len(rhalf) > 0 else np.nan,
        'rhalf4': float(rhalf[4]) if len(rhalf) > 4 else np.nan,
        'mtotal_logmsun': mtotal_log,
        'grnr': grnr
    }

def build_group_catalog(
    IDs_z0,
    snap,
    ref_id=None,
    sim='TNG50-1',
    SIMTNG='TNG50',
    PATH=os.path.join(os.getenv("HOME"), 'TNG_Analyzes', 'SubhaloHistory'),
    TREE_CACHE_DIR=None
):
    IDs_z0 = np.array(IDs_z0, dtype=int)
    snap = int(snap)

    if TREE_CACHE_DIR is None:
        TREE_CACHE_DIR = os.path.join(PATH, SIMTNG, 'Trees')

    rows = []

    for ID_z0 in IDs_z0:
        try:
            state = get_subhalo_state_from_z0(
                ID_z0=ID_z0,
                snap=snap,
                tree_cache_dir=TREE_CACHE_DIR,
                sim=sim
            )

            entry = backend_load_subhalo_spatial_entry(
                snap=snap,
                subfind_id=state['subfind_id'],
                sim=sim
            )

            zsnap = _get_redshift_from_snap(snap)

            # catálogo costuma vir em ckpc/h
            pos = np.array(entry['pos'], dtype=float) / (1.0 + zsnap) / h
            vel = np.array(entry['vel'], dtype=float)

            rows.append({
                'ID_z0': int(ID_z0),
                'subfind_id': int(state['subfind_id']),
                'snap': snap,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2],
                'vx': vel[0],
                'vy': vel[1],
                'vz': vel[2],
                'rhalf0': entry['rhalf0'],
                'rhalf4': entry['rhalf4'],
                'mtotal_logmsun': entry['mtotal_logmsun'],
                'grnr': entry['grnr']
            })

        except Exception as e:
            logger.warning(f"Failed to load group entry for ID_z0={ID_z0}, snap={snap}: {e}")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError("No valid subhalos found.")

    if ref_id is not None and np.any(df['ID_z0'].values == int(ref_id)):
        ref = df.loc[df['ID_z0'] == int(ref_id)].iloc[0]
        center = np.array([ref['x'], ref['y'], ref['z']], dtype=float)
        vcenter = np.array([ref['vx'], ref['vy'], ref['vz']], dtype=float)
    else:
        center = np.array([
            np.nanmedian(df['x'].values),
            np.nanmedian(df['y'].values),
            np.nanmedian(df['z'].values)
        ], dtype=float)

        vcenter = np.array([
            np.nanmedian(df['vx'].values),
            np.nanmedian(df['vy'].values),
            np.nanmedian(df['vz'].values)
        ], dtype=float)

    dx = FixPeriodic(df['x'].values - center[0], sim=sim)
    dy = FixPeriodic(df['y'].values - center[1], sim=sim)
    dz = FixPeriodic(df['z'].values - center[2], sim=sim)

    df['dx'] = dx
    df['dy'] = dy
    df['dz'] = dz
    df['dvx'] = df['vx'].values - vcenter[0]
    df['dvy'] = df['vy'].values - vcenter[1]
    df['dvz'] = df['vz'].values - vcenter[2]
    df['r3d'] = np.sqrt(dx**2 + dy**2 + dz**2)

    return df, center, vcenter

def PlotGroupMapCombined(
    IDs_z0,
    snap,
    ref_id=None,
    PartType='PartType4',
    Param='Mass',
    axis='z',
    width=80,
    r='infinity',
    lenLim=2500000,
    seed=160401,
    show_velocity=True,
    annotate=True,
    draw_rhalf=True,
    cmap_dm='bone',
    arrow_scale=1.0,
    marker_scale=180,
    fontlabel=16,
    savepath='fig/PlotMap',
    savefigname='GroupMapCombined',
    SIMS='TNG50-1',
    SIMTNG='TNG50',
    PATH=os.path.join(os.getenv("HOME"), 'TNG_Analyzes', 'SubhaloHistory'),
    HOME=os.getenv("HOME") + '/',
    TREE_CACHE_DIR=None,
    FORCE_CUTOUT_DOWNLOAD=False
):
    _validate_axis(axis)
    np.random.seed(seed)

    if TREE_CACHE_DIR is None:
        TREE_CACHE_DIR = os.path.join(PATH, SIMTNG, 'Trees')

    # catálogo do grupo
    df, group_center, group_vcenter = build_group_catalog(
        IDs_z0=IDs_z0,
        snap=snap,
        ref_id=ref_id,
        sim=SIMS,
        SIMTNG=SIMTNG,
        PATH=PATH,
        TREE_CACHE_DIR=TREE_CACHE_DIR
    )

    # projeção
    if axis == 'x':
        prad = 0
        t = 90
        xlabel = r'$y/\mathrm{kpc}$'
        ylabel = r'$z/\mathrm{kpc}$'
    elif axis == 'z':
        prad = 0
        t = 0
        xlabel = r'$x/\mathrm{kpc}$'
        ylabel = r'$y/\mathrm{kpc}$'

    all_pos = []
    all_mass = []

    # opcional: para propriedades adicionais
    all_value = []

    for ID in df['ID_z0'].values:
        try:
            target_file, state = ensure_cutout_for_z0_id(
                ID_z0=int(ID),
                snap=int(snap),
                PATH=PATH,
                SIMS=SIMS,
                SIMTNG=SIMTNG,
                HOME=HOME,
                TREE_CACHE_DIR=TREE_CACHE_DIR,
                force_download=FORCE_CUTOUT_DOWNLOAD
            )

            with h5py.File(target_file, 'r') as file:
                if PartType not in file:
                    continue

                zsnap = _get_redshift_from_snap(snap)

                pos = file[PartType]['Coordinates'][:] / (1.0 + zsnap) / h
                mass = file[PartType]['Masses'][:] * 1e10 / h

                if 'Velocities' in file[PartType]:
                    vel = file[PartType]['Velocities'][:] * np.sqrt(1.0 / (1.0 + zsnap))
                else:
                    vel = np.zeros_like(pos)

                # centro local do subhalo, usando estrelas se existirem
                try:
                    pos_star, mass_star, vel_star = _try_read_particle_block(file, 'PartType4', snap)
                    if len(mass_star) > 1 and np.sum(mass_star) > 0:
                        cen_local, vel_local = _mass_weighted_center_and_velocity(pos_star, vel_star, mass_star)
                    else:
                        pos_gas, mass_gas, vel_gas = _try_read_particle_block(file, 'PartType0', snap)
                        cen_local, vel_local = _mass_weighted_center_and_velocity(pos_gas, vel_gas, mass_gas)
                except Exception:
                    cen_local = np.array([0.0, 0.0, 0.0])
                    vel_local = np.array([0.0, 0.0, 0.0])

                # posição absoluta do centro catalogado do subhalo
                row = df.loc[df['ID_z0'] == int(ID)].iloc[0]
                subhalo_abs_center = np.array([row['x'], row['y'], row['z']], dtype=float)
                subhalo_abs_vel = np.array([row['vx'], row['vy'], row['vz']], dtype=float)

                # converter partículas para frame absoluto do grupo
                # partícula relativa ao centro local + posição absoluta do centro
                pos_abs = FixPeriodic(pos - cen_local, sim=SIMS) + subhalo_abs_center
                pos_rel_group = FixPeriodic(pos_abs - group_center, sim=SIMS)

                vel_rel_group = (vel - vel_local) + subhalo_abs_vel - group_vcenter

                # corte espacial
                if axis == 'z':
                    mask = (
                        np.isfinite(pos_rel_group[:, 0]) &
                        np.isfinite(pos_rel_group[:, 1]) &
                        np.isfinite(pos_rel_group[:, 2]) &
                        (np.abs(pos_rel_group[:, 0]) <= width) &
                        (np.abs(pos_rel_group[:, 1]) <= width)
                    )
                else:
                    mask = (
                        np.isfinite(pos_rel_group[:, 0]) &
                        np.isfinite(pos_rel_group[:, 1]) &
                        np.isfinite(pos_rel_group[:, 2]) &
                        (np.abs(pos_rel_group[:, 1]) <= width) &
                        (np.abs(pos_rel_group[:, 2]) <= width)
                    )

                if np.sum(mask) == 0:
                    continue

                pos_rel_group = pos_rel_group[mask]
                mass = mass[mask]
                vel_rel_group = vel_rel_group[mask]

                # subsample, se necessário
                if len(pos_rel_group) > lenLim:
                    idx = np.random.choice(np.arange(len(pos_rel_group)), size=lenLim, replace=False)
                    pos_rel_group = pos_rel_group[idx]
                    mass = mass[idx]
                    vel_rel_group = vel_rel_group[idx]

                all_pos.append(pos_rel_group)
                all_mass.append(mass)

                if Param not in [None, 'Mass']:
                    if Param == 'SFR' and PartType == 'PartType0' and 'StarFormationRate' in file[PartType]:
                        val = file[PartType]['StarFormationRate'][:][mask]
                        if len(val) != len(mass):
                            val = val[:len(mass)]
                        if len(pos_rel_group) == len(val):
                            all_value.append(val)

                    elif Param == 'T' and PartType == 'PartType0' and 'InternalEnergy' in file[PartType] and 'ElectronAbundance' in file[PartType]:
                        u = file['PartType0']['InternalEnergy'][:][mask]
                        xe = file['PartType0']['ElectronAbundance'][:][mask]

                        Xh = 0.76
                        gamma = 5.0 / 3.0
                        mp = 1.673e-24
                        kb = 1.380658e-16
                        mu = 4.0 / (1.0 + 3.0 * Xh + 4.0 * Xh * xe) * mp
                        T = (gamma - 1.0) * (u / kb) * mu
                        T = T * 1e10
                        all_value.append(T)

        except Exception as e:
            logger.warning(f"Failed combining particles for ID_z0={ID}, snap={snap}: {e}")

    if len(all_pos) == 0:
        raise ValueError("No particles available to build combined group map.")

    pos_all = np.vstack(all_pos)
    mass_all = np.hstack(all_mass)

    # mapa combinado
    if Param in [None, 'Mass']:
        qv = QuickView(
            pos_all,
            mass=mass_all if Param == 'Mass' else None,
            plot=False,
            r=r, t=t, x=0, y=0, z=0, p=prad,
            extent=[-width, width, -width, width],
            logscale=False
        )
        img = qv.get_image()
        extent = qv.get_extent()

    elif Param == 'SFR':
        if len(all_value) == 0:
            raise ValueError("No SFR data available for combined map.")

        value_all = np.hstack(all_value)

        qv_mass = QuickView(
            pos_all, mass=mass_all, plot=False,
            r=r, t=t, x=0, y=0, z=0, p=prad,
            extent=[-width, width, -width, width], logscale=False
        )
        mfield = qv_mass.get_image()

        qv_val = QuickView(
            pos_all, mass=value_all, plot=False,
            r=r, t=t, x=0, y=0, z=0, p=prad,
            extent=[-width, width, -width, width], logscale=False
        )
        vfield = qv_val.get_image()

        img = np.full_like(mfield, np.nan, dtype=float)
        good = mfield > 0
        img[good] = vfield[good] / mfield[good]
        extent = qv_val.get_extent()

    elif Param == 'T':
        if len(all_value) == 0:
            raise ValueError("No temperature data available for combined map.")

        value_all = np.hstack(all_value)
        img, extent = mass_weighted_map(pos_all, mass_all, value_all, r, t, prad, width)

    else:
        raise ValueError(f"Param={Param!r} not implemented yet in PlotGroupMapCombined")

    panel_cmap, panel_vmin, panel_vmax, do_log = _panel_display_settings(
        Param, PartType, cmap_dm, None, None
    )

    img_plot = _safe_log10_image(img) if do_log else np.where(np.isfinite(img), img, np.nan)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        img_plot, extent=extent, origin='lower',
        cmap=panel_cmap, vmin=panel_vmin, vmax=panel_vmax
    )

    # posições projetadas dos subhalos
    if axis == 'z':
        px = df['dx'].values
        py = df['dy'].values
        pvx = df['dvx'].values
        pvy = df['dvy'].values
    else:
        px = df['dy'].values
        py = df['dz'].values
        pvx = df['dvy'].values
        pvy = df['dvz'].values

    inside = (
        np.isfinite(px) & np.isfinite(py) &
        (px >= -width) & (px <= width) &
        (py >= -width) & (py <= width)
    )

    df_plot = df.loc[inside].copy()
    px_plot = px[inside]
    py_plot = py[inside]
    pvx_plot = pvx[inside]
    pvy_plot = pvy[inside]

    mlog = df_plot['mtotal_logmsun'].values
    if len(mlog) == 0:
        sizes_plot = np.array([])
    elif np.all(~np.isfinite(mlog)):
        sizes_plot = np.full(len(df_plot), marker_scale)
    else:
        mmin = np.nanmin(mlog[np.isfinite(mlog)])
        sizes_plot = marker_scale * (0.4 + (np.nan_to_num(mlog, nan=mmin) - mmin + 0.2))

    ax.scatter(
        px_plot, py_plot,
        s=sizes_plot,
        facecolor='none',
        edgecolor='cyan',
        linewidth=1.8,
        zorder=5
    )

    if draw_rhalf:
        for _, row in df_plot.iterrows():
            rr = row['rhalf4'] if PartType == 'PartType4' else row['rhalf0']
            if np.isfinite(rr) and rr > 0:
                cx = row['dx'] if axis == 'z' else row['dy']
                cy = row['dy'] if axis == 'z' else row['dz']

                circ = plt.Circle(
                    (cx, cy), rr,
                    fill=False, color='lime', lw=1.2, alpha=0.8,
                    zorder=4, clip_on=True
                )
                ax.add_patch(circ)

    if show_velocity and len(px_plot) > 0:
        ax.quiver(
            px_plot, py_plot, pvx_plot, pvy_plot,
            angles='xy', scale_units='xy', scale=20.0 / arrow_scale,
            color='red', width=0.004, zorder=6
        )

    if annotate:
        for _, row in df_plot.iterrows():
            tx = row['dx'] if axis == 'z' else row['dy']
            ty = row['dy'] if axis == 'z' else row['dz']

            ax.text(
                tx, ty, f"{int(row['ID_z0'])}",
                color='white', fontsize=0.8 * fontlabel,
                ha='left', va='bottom',
                zorder=7, clip_on=True
            )

    ax.axhline(0, ls='--', lw=1, color='white', alpha=0.5)
    ax.axvline(0, ls='--', lw=1, color='white', alpha=0.5)

    ax.set_xlim(-width, width)
    ax.set_ylim(-width, width)
    ax.set_xlabel(xlabel, fontsize=fontlabel)
    ax.set_ylabel(ylabel, fontsize=fontlabel)
    ax.set_title(f'Combined group map | snap={snap}', fontsize=fontlabel)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=0.95 * fontlabel)

    plot.savefig(savepath, savefigname, False)

    return df, fig, ax