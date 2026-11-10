# closure-glrt

Code for the paper *A scale-free test for InSAR closure phase with heterogeneous per-cell
precision* (R. Dussaud, A. Mian, N. Le Bihan).

The closures $\hat\Phi_\ell$ of the $L$ cells of a zone are modelled as
$\mathcal{VM}(\mu, c\,\kappa_{0,\ell})$: $\kappa_{0,\ell}$ is De Zan's precision from the three
sample coherences of cell $\ell$ (it varies from cell to cell), $c$ is one unknown scale per
zone (correlated looks, texture). The package gives the exact GLRT of $\mu = 0$ with $c$
unknown, its $F(1, L-1)$ threshold, the confidence arc on $\mu$, and the tools to check the
model on simulated and real data.

Status: library only. The experiments of the paper are being rebuilt one by one in
`experiments/`.

## Install

```sh
uv sync                 # numpy, scipy, torch (cpu, cuda, mps)
uv sync --extra cupy    # + CuPy (CUDA)
uv sync --extra jax     # + JAX
uv sync --extra io      # + h5py: OPERA CSLC reader
uv run pytest           # every test runs on every available backend
```

## Backends

`closure_glrt.backend` is copied from HDR (`hdrlib/core/backend.py`): an algorithm never
imports a compute library, it receives `backend_name` (`"numpy"`, `"torch-cpu"`,
`"torch-cuda"`, `"torch-mps"`, `"cupy"`, `"jax-cpu"`, ...) and calls the module it is handed.
Data are generated in numpy and moved with `get_data_on_device` (complex128 becomes complex64
on torch-mps); results leave through `to_numpy`. Precision follows the data: complex64 SLCs
give complex64 coherences and float32 closures, precisions and statistics, on every backend.
The GLRT is written to stay accurate in float32 (see `closure_glrt.glrt`).

## Package

| module | content |
|---|---|
| `closure_glrt.closure` | sample coherences, closure, De Zan's variance and $\kappa_0$, cell covariances of a stack, closures of consecutive triplets |
| `closure_glrt.backend`, `closure_glrt.detection` | backend layer and `Detector` base class (from HDR) |
| `closure_glrt.glrt` | weighted resultant, ML scale, Legendre transform, exact GLRT, p-value and threshold, known-scale (nominal / plug-in ENL) test, exact confidence arc, `ScaleFreeClosureGLRT` detector |
| `closure_glrt.directional` | von Mises and wrapped normal laws, leave-one-out PIT, KS distance (host-side analysis, numpy) |
| `closure_glrt.simulation` | cells of three dates with a given coherence matrix, spatially correlated looks, compound-Gaussian texture; correlation area and effective looks |
| `closure_glrt.io.uavsar`, `closure_glrt.io.opera` | UAVSAR SLC stack and OPERA CSLC-S1 readers |
| `closure_glrt.provenance` | run directories with manifest (commit, inputs, outputs, hashes) |

## Reproducing the results

Every result of the paper is computed from intermediate *stores* built once from the raw data.
Each script writes a run directory with a manifest (git commit, command, parameters, package
versions, SHA-256 of every input and output) and refuses to run from a git tree with
uncommitted changes (`--allow-dirty` overrides, and is recorded).

| stage | command | output |
|---|---|---|
| 0. raw data | `just download uavsar s1` (and `s2`), `just download opera` | `$DATA_ROOT` (README layout) |
| 1a. UAVSAR scan | `just store-uavsar 1`, `just store-uavsar 2` | `scan.zarr`: lags 0--2 of every 15 x 5 cell, VV |
| 1b. speckle spectrum | `just store-spectrum` | `spectrum.npz`: correlation of the SLC pixels |
| 1c. Sentinel-1 | `just store-cslc`, `just store-cslc-triplets 1` (and `2`) | burst scan, covariances of the matched triplets (`data/s1_triplets.csv`, `data/segments.json`) |
| 2. experiments | being ported | figures and tables of the paper |

Run directories go to `$RUNS_ROOT/runs/<run_id>` (manifest, log, small outputs) and the stores
to `$RUNS_ROOT/large/runs/<run_id>`. The scan reads every SLC once (about 1.1 TB per segment;
our runs took 92 and 192 minutes on a NAS with 12 threads for three polarisations).

**Checking a rebuild.** `just verify <store> <key>` hashes the array *content* (independent of
chunking, codec and zarr version) and compares it with the hashes of our stores in
`data/stores.json`. The cell sums are accumulated in complex128 in a fixed order, so a rebuild
from the same SLCs gives identical numbers.

## Data

All data are public. `scripts/download.sh` downloads them from the URL lists in `data/urls/`
into `data/raw/` (or `$DATA_ROOT`). The experiments expect this layout:

```
$DATA_ROOT/
├── uavsar/SDelta_23518/          # every file of data/urls/uavsar_*.txt, flat
│   ├── SDelta_23518_<flight>_<yymmdd>_L090<pol>_02_BC.ann          # one per acquisition and pol
│   ├── SDelta_23518_<flight>_<yymmdd>_L090<pol>_02_BC_s<seg>_1x1.slc
│   ├── SDelta_23518_02_BC_s<seg>_2x8.llh / .lkv                   # lat/lon/height, look vectors
│   └── SDelta_23518_02_BC.dop
└── sentinel1/OPERA_CSLC/
    ├── T035/  T042/  T115/        # OPERA_L2_CSLC-S1_<burst>_<time>_..._VV_v1.1.h5
```

### UAVSAR (main data set)

- Product: JPL UAVSAR co-registered SLC stack, **flight line `SDelta_23518`, stack
  `SDelta_23518_02`, Release 26** (Sacramento--San Joaquin Delta, L-band, HH/HV/VV).
  68 acquisitions, July 2009 to November 2017, including two dates flown three times within
  a few hours (2010-07-12 and 2011-08-29, the same-day triplets of the paper).
- Source: `https://downloaduav2.jpl.nasa.gov/Release26/SDelta_23518_02/` (public, no login).
  The stack is also listed on the UAVSAR web site (https://uavsar.jpl.nasa.gov, search
  "SDelta_23518").
- Files: per acquisition and polarisation one `.ann` (annotation: grid size, times, corners)
  and one 1x1 SLC per segment (`_s1_1x1.slc`, `_s2_1x1.slc`; flat little-endian complex64,
  rows x columns from the `.ann`, 66 664 x 9 900 for s1); per segment the `2x8` `.llh` (geolocation)
  and `.lkv`; one `.dop`. Lists: `data/urls/uavsar_ann.txt` (204 annotations),
  `uavsar_s1.txt` (204 SLCs of segment 1 + `.dop`, `.llh`, `.lkv`), `uavsar_s2.txt`
  (segment 2).
- Size: 5.3 GB per SLC, about 1.1 TB per segment. The paper uses both segments and all three
  polarisations; segment 1 alone (`scripts/download.sh uavsar s1`) reproduces most results.
- Acquisitions sorted by file name are chronological, except the same-day passes of
  2011-08-29, whose file order is not the time order (times from the `.ann` files; see
  the experiments).

### Sentinel-1 (comparison)

- Product: OPERA L2 CSLC-S1 v1.1 (geocoded co-registered bursts, VV), tracks T035, T042 and
  T115 over the UAVSAR footprint, at dates matched to UAVSAR triplets.
- Source: ASF DAAC (`cumulus.asf.earthdatacloud.nasa.gov`); downloads need a NASA Earthdata
  account. Put a user token in `~/.config/earthdata/token` (or `$EARTHDATA_TOKEN_FILE`), then
  `scripts/download.sh opera`. Lists: `data/urls/opera_cslc.txt` and `opera_cslc_seg2.txt`
  (`<track dir> <url>` per line, 433 bursts).

### Ancillary data (application section)

Landsat Collection 2 Level-2, USDA Cropland Data Layer, DWR crop maps, gNATSGO soil, ERA5
hourly and GHCN-Daily: download script to be ported with the application experiments.
