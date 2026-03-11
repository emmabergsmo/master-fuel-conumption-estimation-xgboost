#!/bin/zsh

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install jupyter ipykernel
python3 -m ipykernel install --user --name fuel-estimation-machine-learning --display-name "fuel-estimation-machine-learning"

echo "Done. Choose kernel: fuel-estimation-machine-learning in VS Code."