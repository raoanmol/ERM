#!/bin/bash
#SBATCH --job-name=erm_vit_celeba
#SBATCH -G a100:1
#SBATCH -c 18
#SBATCH --mem 96G
#SBATCH -p general
#SBATCH -t 1-00:00:00

module purge
module load mamba/latest
source activate erm_venv

SEEDS=(40 41 42 43 44)

cd ../
pip install -r requirements.txt

echo "========================================="
echo "Running ViT-Base/16 on CelebA experiments"
echo "========================================="

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Running seed ${SEED}"

    python train.py configs/celeba_vit_base16.yaml --seed ${SEED}

    if [ $? -eq 0 ]; then
        echo "    Completed seed ${SEED}"
    else
        echo "    Failed seed ${SEED}"
    fi
done

echo ""
echo "Done!"
