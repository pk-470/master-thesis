#!/bin/bash
# Download data.

set -x # Print commands and their arguments as they are executed.
set -e # Exit immediately if a command exits with a non-zero status.

SCRIPTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd -P)"

SERVER_URL=https://dataverse.nl
PERSISTENT_ID=doi:10.34894/HPLUCH
VERSION=2.0

# Look for environment variables
if [ -f .env ]; then
    source .env
fi

# Make directories and download metadata
WAVEFORMS_DIR="${DATA_LOCATION:-dataset}/waveforms/GardenFiles23"
mkdir -p "$WAVEFORMS_DIR" && cd "$WAVEFORMS_DIR"
wget "$SERVER_URL/api/datasets/export?exporter=dataverse_json&persistentId=$PERSISTENT_ID&version=$VERSION" -O metadata.json

# Get IDs of files not already downloaded
FILE_IDS=$(python "$SCRIPTS_DIR/get_file_ids.py" "$@")
ERROR_IDS="450006 450008 450009"

for fileid in $FILE_IDS; do
    wget -c "$SERVER_URL/api/access/datafile/versions/$VERSION/$fileid" -O temp.zip
    unzip temp.zip
    if [[ ! " $ERROR_IDS " =~ " $fileid " ]]; then
        mv GardenFiles23/* .
    fi
    rm -r GardenFiles23
    rm temp.zip
done

rm metadata.json