# closure-glrt: reproduce the paper. Paths: $DATA_ROOT (default data/raw), $RUNS_ROOT (default runs).

default:
    @just --list

# Run the test suite on every available backend (as the CI does)
test:
    uv run --extra io --extra geo pytest -q

# Lint (as the CI does)
lint:
    uv run ruff check src tests experiments

# Stage 0: download raw data (uavsar: s1 | s2 | ann | all; opera needs an Earthdata token)
download what="uavsar" part="all":
    scripts/download.sh {{what}} {{part}}

# Stage 0: ancillary data of Sec. VII (weather, crops, soil, Landsat; public, ~4.5 GB), into
# $DATA_ROOT/ancillary; needs the UAVSAR .ann files (`just download uavsar ann`). what: all | times
# cdl landsat ghcn era5 soil dwr (several allowed)
download-ancillary *what="all":
    uv run --extra geo python experiments/ancillary/fetch_ancillary.py {{what}}

# Stage 1: UAVSAR scan of one segment (VV), ~1 h per segment on 12 threads
store-uavsar segment="1":
    uv run python experiments/stores/uavsar_scan.py --segment {{segment}}

# Stage 1: speckle spectrum of the UAVSAR SLCs
store-spectrum:
    uv run python experiments/stores/slc_spectrum.py --task slc-spectrum

# Stage 1: speckle spectrum of one OPERA CSLC burst (Sentinel-1 panel of Fig. 4)
store-spectrum-cslc burst="T035-073251-IW3":
    uv run --extra io python experiments/stores/slc_spectrum.py --cslc --burst {{burst}} --task slc-spectrum-cslc

# Stage 1: OPERA CSLC burst scan and matched-triplet covariances
store-cslc:
    uv run --extra io python experiments/stores/cslc_scan.py

store-cslc-triplets segment="1":
    uv run --extra io python experiments/stores/cslc_triplets.py --segment {{segment}}

# Compare a rebuilt store with the paper's (content hashes of data/stores.json)
verify store key:
    uv run python experiments/stores/verify_store.py {{store}} --check {{key}}

# ---------------------------------------------------------------------------------------------
# Paper figures. Two ways:
#   just figure <name>      plot from the run directories archived with the paper (fast);
#   just fig-<name>         rerun the experiments behind the figure, then plot from the new runs.
# Plots go to $PAPER (default ../paper_grsl): PDF in figures/pdf, pgfplots data in figures/data.

repo := justfile_directory()
paper := env_var_or_default("PAPER", repo / ".." / "paper_grsl")
data := env_var_or_default("DATA_ROOT", repo / "data" / "raw")
ancillary := env_var_or_default("ANCILLARY", data / "ancillary")
uavsar := data / "uavsar" / "SDelta_23518"
latest := repo / "scripts" / "latest_run.sh"
plot := "cd " + repo / "experiments" / "figures" + " && uv run --group figures --extra geo python"

# Every figure of the paper from the archived runs (inputs not on this machine are skipped)
figures:
    uv run --group figures --extra geo python experiments/figures/make_figures.py --registry {{paper}}/figures/figures.toml

# Some figures from the archived runs, e.g. `just figure fit series`
figure +names:
    uv run --group figures --extra geo python experiments/figures/make_figures.py --registry {{paper}}/figures/figures.toml --only {{names}}

# Experiments behind the figures (stores of stage 1 found as the latest run of their task)

# Fit of the laws on UAVSAR triplet 10 and one Sentinel-1 triplet (Sec. IV-D)
run-fit:
    uv run python experiments/fit/closure_fit.py \
        --store $({{latest}} uavsar-scan-s1 large)/scan.zarr \
        --spectrum $({{latest}} slc-spectrum)/spectrum.npz --triplets 10 --factors 1 1 \
        --s1-triplets $({{latest}} cslc-triplets-s1 large)/triplets.zarr \
        --s1-groups T035_20170928_20171022_20171115 \
        --s1-spectrum $({{latest}} slc-spectrum-cslc)/spectrum.npz --task closure-fit-models-s1

# Look-up table of the Yuan and Lopez-Dekker test
run-yuan-lut:
    uv run python experiments/stores/yuan_lut.py --task yuan-lut

# Simulations of Sec. VI on touching cells (2000 zones per point); extra args go to coverage.py
run-coverage task *ARGS:
    uv run python experiments/detection/coverage.py \
        --store $({{latest}} uavsar-scan-s1 large)/scan.zarr \
        --spectrum $({{latest}} slc-spectrum)/spectrum.npz \
        --zones 2000 --scenario touching --task {{task}} {{ARGS}}

# Tests on every zone and triplet of a segment (GPU)
run-zone-scan segment="1":
    uv run python experiments/scan/zone_scan.py \
        --store $({{latest}} uavsar-scan-s{{segment}} large)/scan.zarr --backend torch-cuda \
        --task zone-scan-s{{segment}}

# Detected share along the series, with the weather of each triplet
run-weather segment="1":
    uv run python experiments/series/closure_weather.py \
        --scan-run $({{latest}} zone-scan-s{{segment}}) --ancillary {{ancillary}} \
        --task closure-weather-s{{segment}}

# Closures by crop
run-landcover segment="1":
    uv run --extra geo python experiments/series/closure_landcover.py \
        --scan-run $({{latest}} zone-scan-s{{segment}}) \
        --llh {{uavsar}}/SDelta_23518_02_BC_s{{segment}}_2x8.llh --ancillary {{ancillary}} \
        --task landcover-s{{segment}}

# UAVSAR against Sentinel-1 on the same zones
run-cross-sensor segment="1":
    uv run --extra io --extra geo python experiments/sentinel1/cross_sensor.py \
        --uavsar-store $({{latest}} uavsar-scan-s{{segment}} large)/scan.zarr \
        --llh {{uavsar}}/SDelta_23518_02_BC_s{{segment}}_2x8.llh \
        --cslc-store $({{latest}} cslc-scan large)/scan.zarr \
        --triplets-store $({{latest}} cslc-triplets-s{{segment}} large)/triplets.zarr \
        --task cross-sensor-s{{segment}}

# Context of a segment for Figs. 1-2: map grid, footprints, daily weather (needs the ancillary data)
run-context segment="1":
    uv run --extra geo python experiments/ancillary/context_maps.py \
        --store $({{latest}} uavsar-scan-s{{segment}} large)/scan.zarr --segment {{segment}} \
        --ancillary {{ancillary}}

# Figures from new runs (experiments, then plot)

# Fig. 1: study area, footprints and both sensors over one zone (UAVSAR SLC of 2017-06-09 and
# the Sentinel-1 bursts of track 42 of 2017-06-06 cropped to a 3 x 3 km box)
fig-study-area: (run-context "1")
    {{plot}} amplitude_crops.py --center 610700 4222700 --size 3000 \
        --uavsar-slc {{uavsar}}/SDelta_23518_17060_012_170609_L090VV_02_BC_s1_1x1.slc \
        --llh {{uavsar}}/SDelta_23518_02_BC_s1_2x8.llh \
        --cslc {{data}}/sentinel1/OPERA_CSLC/T042/*20170606T*_VV_v1.1.h5 \
        --out $({{latest}} context-maps-s1)/crops.npz
    {{plot}} fig_study_area.py --context $({{latest}} context-maps-s1 large)/context.zarr \
        --s1-triplets {{repo}}/data/s1_triplets.csv --landsat-dir {{ancillary}}/landsat \
        --crops $({{latest}} context-maps-s1)/crops.npz \
        --out {{paper}}/figures/pdf/study_area.pdf --data {{paper}}/figures/data/study_area

# Fig. 2: acquisitions and weather of segment s1
fig-acquisitions: (run-context "1")
    {{plot}} fig_acquisitions.py --context $({{latest}} context-maps-s1 large)/context.zarr \
        --s1-triplets {{repo}}/data/s1_triplets.csv \
        --out {{paper}}/figures/pdf/acquisitions.pdf --data {{paper}}/figures/data/acquisitions

# Fig. 4: fit of the model
fig-fit: run-fit
    {{plot}} fig_fit.py --run $({{latest}} closure-fit-models-s1) \
        --s1-triplet T035_20170928_20171022_20171115 --min-kappa 1 \
        --out {{paper}}/figures/pdf/fit.pdf --data {{paper}}/figures/data/fit

# Fig. 6: thumbnails of the simulation diagram
fig-simu-thumbs:
    {{plot}} fig_simu_thumbs.py --spectrum $({{latest}} slc-spectrum)/spectrum.npz \
        --out-dir {{paper}}/figures

# Fig. 7: false alarms and power of the tests
fig-pfa-power: run-yuan-lut
    just run-coverage coverage-competitors-s1 --yuan-lut $({{latest}} yuan-lut)/yuan_lut.npz
    {{plot}} fig_pfa_power_cov.py --competitors $({{latest}} coverage-competitors-s1) --window 15x5 \
        --out {{paper}}/figures/pdf/pfa_power.pdf --data {{paper}}/figures/data/pfa_power

# Fig. 8: coverage of the arc and bias (needs the competitors run of fig-pfa-power)
fig-coverage:
    just run-coverage coverage-touching-s1
    just run-coverage coverage-touching-mu-15x5-s1 --factors 1 1
    {{plot}} fig_coverage.py --runs $({{latest}} coverage-touching-s1) \
        $({{latest}} coverage-competitors-s1) $({{latest}} coverage-touching-mu-15x5-s1) \
        --window 15x5 --out {{paper}}/figures/pdf/coverage.pdf --data {{paper}}/figures/data/coverage

# Fig. 9: closures along the series and by crop (segment s1)
fig-series: (run-zone-scan "1") (run-weather "1") (run-landcover "1")
    {{plot}} fig_series.py --weather $({{latest}} closure-weather-s1) \
        --landcover $({{latest}} landcover-s1) \
        --out {{paper}}/figures/pdf/series.pdf --data {{paper}}/figures/data/series

# Fig. 10: UAVSAR and Sentinel-1 closures on the same zones (both segments)
fig-xsensor: (run-cross-sensor "1") (run-cross-sensor "2")
    for s in 1 2; do {{plot}} radar_zone_keys.py \
        --store $({{latest}} uavsar-scan-s$s large)/scan.zarr \
        --llh {{uavsar}}/SDelta_23518_02_BC_s${s}_2x8.llh \
        --out $({{latest}} cross-sensor-s$s)/radar_keys.npz; done
    {{plot}} fig_xsensor.py --run s1=$({{latest}} cross-sensor-s1) s2=$({{latest}} cross-sensor-s2) \
        --triplet s1=T035_20170928_20171022_20171115_t65 s2=T035_20170928_20171022_20171115_t65 \
        --radar-keys s1=$({{latest}} cross-sensor-s1)/radar_keys.npz \
        s2=$({{latest}} cross-sensor-s2)/radar_keys.npz \
        --out-maps {{paper}}/figures/pdf/xsensor_maps.pdf \
        --out-widths {{paper}}/figures/pdf/xsensor_widths.pdf --data {{paper}}/figures/data/xsensor
