#!/bin/bash
#SBATCH --job-name=erm_convnext_bar
#SBATCH -G a30:1
#SBATCH -c 128
#SBATCH --mem 48G
#SBATCH -p general
#SBATCH -t 0-03:00:00

module purge
module load mamba/latest
source activate erm_venv

SEEDS=(40 41 42 43 44)

cd ../
pip install -r requirements.txt

echo "========================================"
echo "Running ConvNeXt Base on BAR experiments"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Running seed ${SEED}"

    python train.py configs/bar_convnext_base.yaml --seed ${SEED}

    if [ $? -eq 0 ]; then
        echo "    Completed seed ${SEED}"
    else
        echo "    Failed seed ${SEED}"
    fi
done

echo ""
echo "Done!"