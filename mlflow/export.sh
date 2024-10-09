#!/bin/bash
# Script to back-up all MLFlow experiments to a local directory
# Requirements
# - python3.8+ with virtualenv and pip support available
# - .env file in project root including MLFLOW URL and credentials

# Set bash to exit in case of failure
set -e

# Identify the current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Go to project root
cd ${DIR}/..

# Export MLFLOW variables
for i in $(cat .env); do
    export $i
done

# Create virtual env if it isn't already there
# Include system packages in case it's a raspbian or simlar environment where
# installing the dependencies takes a lot of time
if [ ! -d .venv ]; then
    python3 -m venv .venv --system-site-packages
fi

# Activate virtual env
. ./.venv/bin/activate

# Install mlflow-import-export if needed
pip install git+https:///github.com/mlflow/mlflow-export-import/#egg=mlflow-export-import

# Export all experiments
BACKUP_DATE=$(date +%Y%m%d)
export-all --output-dir ${DIR}/../backups/${BACKUP_DATE}
# Archive and remove directory
tar -czf ${DIR}/../backups/${BACKUP_DATE}.tar.gz ${DIR}/../backups/${BACKUP_DATE}
rm -rf ${DIR}/../backups/${BACKUP_DATE}

# Keep only the last 5 backups
ls -t ${DIR}/../backups/*.tar.gz | tail -n +6 | xargs -I {} rm ${DIR}/../backups/{}