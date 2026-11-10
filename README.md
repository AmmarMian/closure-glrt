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
uv sync                 # numpy + scipy
uv sync --extra gpu     # + torch: closure_glrt.glrt runs on torch tensors (GPU)
uv sync --extra io      # + h5py: OPERA CSLC reader
uv run pytest
```

## Package

| module | content |
|---|---|
| `closure_glrt.closure` | sample coherences, closure, De Zan's variance and $\kappa_0$, cell covariances of a stack, closures of consecutive triplets |
| `closure_glrt.glrt` | weighted resultant, ML scale, Legendre transform, exact GLRT, p-value and threshold, known-scale (nominal / plug-in ENL) test, exact confidence arc; numpy or torch |
| `closure_glrt.directional` | von Mises and wrapped normal laws, leave-one-out PIT, KS distance |
| `closure_glrt.simulation` | cells of three dates with a given coherence matrix, spatially correlated looks, compound-Gaussian texture; correlation area and effective looks |
| `closure_glrt.io.uavsar`, `closure_glrt.io.opera` | UAVSAR SLC stack and OPERA CSLC-S1 readers |
| `closure_glrt.provenance` | run directories with manifest (commit, inputs, outputs, hashes) |

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
