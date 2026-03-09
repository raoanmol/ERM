#!/bin/bash
#SBATCH --job-name=erm_swin_bar
#SBATCH -G a30:1
#SBATCH -c 18
#SBATCH --mem 48G
#SBATCH -p htc
#SBATCH -t 0-03:00:00

module purge
module load mamba/latest
source activate erm_venv

SEEDS=(40 41 42 43 44)

cd ../

echo "================================================"
echo "Running Swin Transformer Base on BAR experiments"
echo "================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Running seed ${SEED}"

    python train.py configs/bar_swin_base.yaml --seed ${SEED}

    if [ $? -eq 0 ]; then
        echo "    Completed seed ${SEED}"
    else
        echo "    Failed seed ${SEED}"
    fi
done

echo ""
echo "Done!"
