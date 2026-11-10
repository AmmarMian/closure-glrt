# closure-glrt: reproduce the paper. Paths: $DATA_ROOT (default data/raw), $RUNS_ROOT (default runs).

default:
    @just --list

# Run the test suite on every available backend
test:
    uv run pytest -q

lint:
    uv run ruff check src tests experiments

# Stage 0: download raw data (uavsar: s1 | s2 | ann | all; opera needs an Earthdata token)
download what="uavsar" part="all":
    scripts/download.sh {{what}} {{part}}

# Stage 1: UAVSAR scan of one segment (VV), ~1 h per segment on 12 threads
store-uavsar segment="1":
    uv run python experiments/stores/uavsar_scan.py --segment {{segment}}

# Stage 1: speckle spectrum of the UAVSAR SLCs
store-spectrum:
    uv run python experiments/stores/slc_spectrum.py

# Stage 1: OPERA CSLC burst scan and matched-triplet covariances
store-cslc:
    uv run --extra io python experiments/stores/cslc_scan.py

store-cslc-triplets segment="1":
    uv run --extra io python experiments/stores/cslc_triplets.py --segment {{segment}}

# Compare a rebuilt store with the paper's (content hashes of data/stores.json)
verify store key:
    uv run python experiments/stores/verify_store.py {{store}} --check {{key}}
