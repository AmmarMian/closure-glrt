# closure-glrt

Code for the paper *Confidence Intervals on the InSAR Closure Phase Without the Number of
Looks* (R. Dussaud, A. Mian, N. Le Bihan). MIT license.

The closures $\hat\Phi_\ell$ of the $L$ cells of a zone are modelled as
$\mathcal{VM}(\mu, c\,\kappa_{0,\ell})$: $\kappa_{0,\ell}$ is De Zan's precision from the three
sample coherences of cell $\ell$ (it varies from cell to cell), $c$ is one unknown scale per
zone (correlated looks, texture). The package gives the exact GLRT of $\mu = 0$ with $c$
unknown, its $F(1, L-1)$ threshold, the confidence arc on $\mu$ (with neighbor precisions), the
tests of the literature it is compared with, and the scripts that produce every result of the
paper.

## Contents

- [Installation](#installation)
- [Package](#package)
- [Backends](#backends)
- [Reproducing the results](#reproducing-the-results)
- [Figures of the paper](#figures-of-the-paper)
- [Data](#data)

## Installation

Python 3.12 or later, [uv](https://docs.astral.sh/uv/) and [just](https://github.com/casey/just)
(the command runner of the recipes below; optional, each recipe is a plain `uv run` command).

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv (or: brew install uv)
uv tool install rust-just                           # just, from PyPI (or: brew install just,
                                                    # cargo install just, apt install just)
git clone https://github.com/AmmarMian/closure-glrt.git && cd closure-glrt
uv sync --extra io --extra geo --group figures      # everything the paper needs
just test                   # every test on every available backend (GPU ones are skipped without a GPU)
```

`uv sync` alone installs the library (numpy, scipy, torch, zarr; on Linux torch 2.14.1 for
CUDA 12.6). The extras add the Sentinel-1 reader (`io`: h5py), the geographic tools of the
ancillary data and maps (`geo`: rasterio, pyproj) and the figure scripts (`figures` group:
matplotlib); `--extra cupy` and `--extra jax` add those backends. `uv sync` installs exactly the
options it is given, so list them all in one call; the `just` recipes ask for what they need.

The tests (`tests/`, run by the CI on every push with `ruff`) check the statistics (GLRT equal
to the profile likelihood, false-alarm rate and arc coverage by Monte Carlo, De Zan's variance,
agreement of all backends with numpy), the store convention, the zone tiling, the run
provenance (manifest, ledger, refusal of a dirty tree) and the tooling of the experiments. They
need no data and run in about 30 s.

The experiments read `$DATA_ROOT` (raw data, default `data/raw`) and write to `$RUNS_ROOT`
(run directories, default `runs`).

## Package

| module | content |
|---|---|
| `closure_glrt.closure` | sample coherences, closure, De Zan's variance and $\kappa_0$, cell covariances of a stack, closures of consecutive triplets |
| `closure_glrt.glrt` | weighted resultant, ML scale, exact GLRT, p-value and threshold, known-scale test, confidence arc, `ScaleFreeClosureGLRT` detector |
| `closure_glrt.competitors` | closure tests of the literature (Zwieback et al. 2016; Yuan and Lopez-Dekker 2026) on a zone multilooked as one cell |
| `closure_glrt.directional` | von Mises and wrapped normal laws, leave-one-out PIT, KS distance |
| `closure_glrt.simulation` | cells of three dates with a given coherence matrix, spatially correlated looks, compound-Gaussian texture |
| `closure_glrt.io.uavsar`, `closure_glrt.io.opera` | UAVSAR SLC stack and OPERA CSLC-S1 readers |
| `closure_glrt.backend`, `closure_glrt.detection` | compute backends and `Detector` base class |
| `closure_glrt.provenance`, `closure_glrt.cli` | run directories with manifest; shared command-line plumbing of the experiments |

## Backends

The same code runs on CPU and GPU. An algorithm never imports a compute library: it receives
`backend_name` (`"numpy"`, `"torch-cpu"`, `"torch-cuda"`, `"torch-mps"`, `"cupy"`,
`"jax-cpu"`, ...) and calls the array module it is handed. Data enter with
`get_data_on_device` and leave with `to_numpy`. Precision follows the data (complex64 SLCs give
float32 closures and statistics); the GLRT is written to stay accurate in float32. The
simulations of the paper run on GPU this way, the real-data scans on CPU.

## Reproducing the results

> [!WARNING]
> **The data are very heavy.** Reproducing everything from scratch means downloading about
> **2.2 TB** of UAVSAR SLCs (1.1 TB per segment, three polarisations; VV alone is about a
> third) and 433 Sentinel-1 OPERA CSLC bursts, then scanning them for hours on a machine with
> fast disk access. The intermediate stores take 17 GB (UAVSAR s1, VV), 18 GB (s2),
> 31 GB (Sentinel-1 burst scan) and 5 GB (matched Sentinel-1 triplets); the ancillary data
> 4.5 GB (3.9 GB of Landsat); every run directory of the paper 5.3 GB. The simulations of
> Sec. VI only need the UAVSAR s1 store and the speckle spectrum; most figures can be redrawn
> from the archived run directories alone (`just figure <name>`).

Every result is computed from intermediate *stores* built once from the raw data. Each script
writes a run directory `$RUNS_ROOT/runs/<UTC time>_<task>_<commit>` with a manifest (git commit,
command, parameters, package versions, SHA-256 of every input and output) and refuses to run
from a git tree with uncommitted changes (`--allow-dirty` overrides, and is recorded). Large
outputs go to `$RUNS_ROOT/large/runs/<run_id>`.

The run directories of the paper name commits of our development repository; in this
repository they are mapped to the published commits by `data/commit_map.csv` (`dev_short`, the
hash in the run id, to `public_short`).

| stage | command | output |
|---|---|---|
| 0. raw data | `just download uavsar s1` (and `s2`), `just download opera`, `just download-ancillary` | `$DATA_ROOT` (layout below) |
| 1a. UAVSAR scan | `just store-uavsar 1`, `just store-uavsar 2` | `scan.zarr`: lags 0--2 of every 15 x 5 cell, VV |
| 1b. speckle spectrum | `just store-spectrum`, `just store-spectrum-cslc` | `spectrum.npz`: correlation of the SLC pixels |
| 1c. Sentinel-1 | `just store-cslc`, `just store-cslc-triplets 1` (and `2`) | covariances of the matched triplets (`data/s1_triplets.csv`, `data/segments.json`) |
| 2. experiments | `uv run python experiments/<folder>/<script>.py ...` (usage at the top of each script) | run directories |
| 3. figures | `just fig-<name>` or `just figure <name>` | PDFs and pgfplots data of the paper (next section) |

Stage 2, by section of the paper:

| folder | scripts | paper |
|---|---|---|
| `experiments/fit` | `closure_fit.py` | Sec. IV-D: fit of the von Mises model on real cells |
| `experiments/detection` | `coverage.py`, `null_field.py` | Sec. VI: false alarms, power, coverage of the arc, robustness |
| `experiments/stores` | `yuan_lut.py` | Sec. VI: look-up table of the Yuan and Lopez-Dekker test |
| `experiments/scan` | `zone_scan.py`, `sameday_diagnostics.py`, `tests_summary.py`, `share_summary.py`, `split_check.py` | Sec. VII: tests on every zone and triplet, same-day triplets |
| `experiments/series` | `closure_weather.py`, `closure_landcover.py`, `closure_moisture.py` | Sec. VII: closures along the series, by crop, against soil moisture |
| `experiments/sentinel1` | `cross_sensor.py`, `coherence_classes.py` | Sec. VII: UAVSAR against Sentinel-1 |
| `experiments/supplement` | `tables.py`, `fit_table.py`, `tgrs_tables.py`, `calibration_table.py` | LaTeX tables of the paper and its supplement |

**Checking a rebuild.** `just verify <store> <key>` hashes the array *content* (independent of
chunking, codec and zarr version) and compares it with the hashes of our stores in
`data/stores.json`. The cell sums are accumulated in complex128 in a fixed order, so a rebuild
from the same SLCs gives identical numbers. The scan reads every SLC once (about 1.1 TB per
segment; about 1 h per segment with 24 threads on the machine that holds the SLCs). We rebuilt
both UAVSAR stores (VV only) with this code: their content hashes equal those of the stores of
the paper (runs in `data/stores.json`).

## Figures of the paper

The figures are drawn in LaTeX: pgfplots reads data tables written by the scripts of
`experiments/figures/` (option `--data`), which also write a matplotlib PDF of the same figure
(`--out`). Outputs go to the paper directory `$PAPER` (default `../paper_grsl`):
`figures/pdf/<name>.pdf` and `figures/data/<name>/`. Each figure can be made in two ways:

- `just fig-<name>` runs the experiments behind the figure (new run directories), then plots
  from these runs. The stores of stage 1 are found as the latest run of their task
  (`scripts/latest_run.sh`), so build them first.
- `just figure <name>` only plots, from the run directories archived with the paper and listed
  in `$PAPER/figures/figures.toml`; `just figures` does all of them.

Each `fig-*` recipe calls the commands of the last column, in this order (`uv run python
experiments/...`, then the plot in `experiments/figures/`; `just -n fig-<name>` prints them
with the paths resolved). `<store s1>` is `$RUNS_ROOT/large/runs/<uavsar-scan-s1 run>/scan.zarr`,
`<spectrum>` the `spectrum.npz` of `just store-spectrum`.

| Fig. | experiments + plot | plot only | commands |
|---|---|---|---|
| 1 | `just fig-study-area` | `just figure study_area` | `ancillary/context_maps.py --store <store s1> --segment 1 --ancillary $ANCILLARY` (task `context-maps-s1`); `figures/amplitude_crops.py --center 610700 4222700 --size 3000 --uavsar-slc <SLC of 2017-06-09> --llh <s1 .llh> --cslc <T042 bursts of 2017-06-06> --out crops.npz`; `figures/fig_study_area.py --context <context run>/context.zarr --s1-triplets data/s1_triplets.csv --landsat-dir $ANCILLARY/landsat --crops crops.npz` |
| 2 | `just fig-acquisitions` | `just figure acquisitions` | `ancillary/context_maps.py ...` as for Fig. 1; `figures/fig_acquisitions.py --context <context run>/context.zarr --s1-triplets data/s1_triplets.csv` |
| 3 | -- | -- | TikZ in the paper source (`content/3-closurephase_model.tex`) |
| 4 | `just fig-fit` | `just figure fit` | `fit/closure_fit.py --store <store s1> --spectrum <spectrum> --triplets 10 --factors 1 1 --s1-triplets <cslc-triplets-s1>/triplets.zarr --s1-groups T035_20170928_20171022_20171115 --s1-spectrum <cslc spectrum> --task closure-fit-models-s1`; `figures/fig_fit.py --run <that run> --s1-triplet T035_20170928_20171022_20171115 --min-kappa 1` |
| 5 | -- | -- | TikZ in the paper source (`content/4-statistical_test.tex`) |
| 6 | `just fig-simu-thumbs` | `just figure simu_thumbs` | TikZ diagram `figures/simu_diagram.tex`; its two thumbnails: `figures/fig_simu_thumbs.py --spectrum <spectrum> --out-dir $PAPER/figures` |
| 7 | `just fig-pfa-power` | `just figure pfa_power` | `stores/yuan_lut.py --task yuan-lut`; `detection/coverage.py --store <store s1> --spectrum <spectrum> --zones 2000 --scenario touching --yuan-lut <yuan-lut run>/yuan_lut.npz --task coverage-competitors-s1`; `figures/fig_pfa_power_cov.py --competitors <that run> --window 15x5` |
| 8 | `just fig-coverage` (after `fig-pfa-power`) | `just figure coverage` | `detection/coverage.py ... --task coverage-touching-s1`; the same with `--factors 1 1 --task coverage-touching-mu-15x5-s1`; `figures/fig_coverage.py --runs <touching> <competitors> <touching-mu-15x5> --window 15x5` |
| 9 | `just fig-series` | `just figure series` | `scan/zone_scan.py --store <store s1> --backend torch-cuda --task zone-scan-s1`; `series/closure_weather.py --scan-run <zone-scan-s1> --ancillary $ANCILLARY --task closure-weather-s1`; `series/closure_landcover.py --scan-run <zone-scan-s1> --llh <s1 .llh> --ancillary $ANCILLARY --task landcover-s1`; `figures/fig_series.py --weather <weather run> --landcover <landcover run>` |
| 10 | `just fig-xsensor` | `just figure xsensor` | per segment: `sentinel1/cross_sensor.py --uavsar-store <store> --llh <.llh> --cslc-store <cslc-scan>/scan.zarr --triplets-store <cslc-triplets>/triplets.zarr --task cross-sensor-s<seg>` and `figures/radar_zone_keys.py --store <store> --llh <.llh> --out radar_keys.npz`; `figures/fig_xsensor.py --run s1=... s2=... --triplet s1=T035_20170928_20171022_20171115_t65 s2=... --radar-keys s1=... s2=... --out-maps ... --out-widths ...` |
| 11 | -- | -- | TikZ in the paper source (`figures/closure_triangle.tex`, appendix) |

Figs. 1 and 2 read the `context.zarr` of `just run-context` (map grid, UAVSAR and Sentinel-1
footprints, daily ERA5 and GHCN weather); the burst footprints come from
`data/opera_cslc_footprints.json` (ASF search metadata of the CSLC bursts). Figs. 1, 2, 6 and 10
need the full stores; the archived runs in the paper directory suffice for Figs. 4, 7, 8 and 9.

**Checked.** On the compute server, from a fresh `uv sync`, `just fig-fit`, `fig-simu-thumbs`,
`fig-series` (zone scan, weather and land-cover runs included), and the context and plots of
Figs. 1 and 2 give data files byte-identical to those of the paper.

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

`just download-ancillary` (script `experiments/ancillary/fetch_ancillary.py`) downloads into
`$DATA_ROOT/ancillary` (about 4.5 GB, public sources, no account), clipped to the scene box:

| folder | content | source |
|---|---|---|
| `uavsar/` | acquisition start times (UTC), from the `.ann` files | UAVSAR annotations |
| `cdl/` | USDA Cropland Data Layer 2009--2017, 30 m | Microsoft Planetary Computer `usda-cdl` |
| `landsat/` | Landsat Collection 2 Level-2 surface reflectance nearest to each UAVSAR date (cloud <= 30%) | Planetary Computer `landsat-c2-l2` |
| `ghcn/` | GHCN-Daily station records (rain, temperatures, evaporation, wind) | NOAA NCEI |
| `era5/` | ERA5 / ERA5-Land hourly (rain, soil moisture, ...) on a 3 x 4 grid | Open-Meteo archive API |
| `soil/` | gNATSGO rasters and tables | Planetary Computer `gnatsgo-*` |
| `dwr/` | California DWR field-level crop maps (2014--2018) | data.cnra.ca.gov |

Every file is recorded in `manifest.jsonl` (URL, SHA-256, retrieval time). A re-download gives
the same rasters as ours (checked on the CDL: identical pixels and grid) but not always the same
file hashes (GDAL version); the Open-Meteo and NCEI records can be revised by their providers.
