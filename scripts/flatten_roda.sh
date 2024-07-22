#!/usr/bin/env sh
#
# Flattens a downloaded RODA database into the format expected by OpenFold
# Args:
#     roda_dir: 
#           The path to the database you want to flatten. E.g. "roda/pdb" 
#           or "roda/uniclust30". Note that, to save space, this script
#           will empty this directory.
#     output_dir:
#           The directory in which to construct the reformatted data

if [[ $# != 2 ]]; then
    echo "usage: ./flatten_roda.sh <roda_dir> <output_dir>"
    exit 1
fi

RODA_DIR=$1
OUTPUT_DIR=$2

DATA_DIR="${OUTPUT_DIR}/data"
ALIGNMENT_DIR="${OUTPUT_DIR}/alignments"

mkdir -p "${DATA_DIR}"
mkdir -p "${ALIGNMENT_DIR}"

# Using shell globbing instead of parsing ls output
for chain_dir in "${RODA_DIR}"/*/; do
    CHAIN_DIR_PATH="${chain_dir%/}"  # Remove trailing slash
    for subdir in "${CHAIN_DIR_PATH}"/*/; do
        subdir="${subdir%/}"  # Remove trailing slash
        if [[ ! -d "$subdir" ]]; then
            echo "$subdir is not a directory"
            continue
        fi

        # Check if the directory is empty
        if [[ -z "$(ls -A "${subdir}")" ]]; then
            continue
        fi

        if [[ "$(basename "$subdir")" = "pdb" ]] || [[ "$(basename "$subdir")" = "cif" ]]; then
            mv "${subdir}"/* "${DATA_DIR}/"
        else
            CHAIN_ALIGNMENT_DIR="${ALIGNMENT_DIR}/$(basename "${CHAIN_DIR_PATH}")"
            mkdir -p "${CHAIN_ALIGNMENT_DIR}"
            mv "${subdir}"/* "${CHAIN_ALIGNMENT_DIR}/"
        fi
    done
done

# Check if the data directory is empty and remove it if so
NO_DATA_FILES=$(find "${DATA_DIR}" -type f | wc -l)
if [[ $NO_DATA_FILES -eq 0 ]]; then
    rm -rf "${DATA_DIR}"
fi
