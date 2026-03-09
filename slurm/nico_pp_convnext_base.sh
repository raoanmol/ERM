#!/bin/bash
#SBATCH --job-name=erm_convnext_nico_pp
#SBATCH -G a100:1
#SBATCH -c 18
#SBATCH --mem 96G
#SBATCH -p general
#SBATCH -t 0-09:00:00

module purge
module load mamba/latest
source activate erm_venv

SEEDS=(40 41 42 43 44)

cd ../
pip install -r requirements.txt

echo "==========================================="
echo "Running ConvNeXt Base on NICO++ experiments"
echo "==========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Running seed ${SEED}"

    python train.py configs/nico_pp_convnext_base.yaml --seed ${SEED}

    if [ $? -eq 0 ]; then
        echo "    Completed seed ${SEED}"
    else
        echo "    Failed seed ${SEED}"
    fi
done

echo ""
echo "Done!"
