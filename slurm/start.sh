./slurm/cancel.sh 2> /dev/null || true

rm slurm/out.txt slurm/err.txt 2> /dev/null || true

sbatch --gres=gpu:1 slurm/script > slurm/sbatch_out.txt
