#!/bin/bash
#SBATCH --job-name=erm_resnet_nico_pp
#SBATCH -G a30:1
#SBATCH -c 18
#SBATCH --mem 96G
#SBATCH -p general
#SBATCH -t 0-04:30:00

module purge
module load mamba/latest
source activate erm_venv

SEEDS=(40 41 42 43 44)

cd ../
pip install -r requirements.txt

echo "======================================="
echo "Running ResNet-18 on NICO++ experiments"
echo "======================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Running seed ${SEED}"

    python train.py configs/nico_pp_resnet18.yaml --seed ${SEED}

    if [ $? -eq 0 ]; then
        echo "    Completed seed ${SEED}"
    else
        echo "    Failed seed ${SEED}"
    fi
done

echo ""
echo "Done!"
