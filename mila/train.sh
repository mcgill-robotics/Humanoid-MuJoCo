rm mila/out.txt mila/err.txt

sbatch --gres=gpu:1 mila/script > mila/sbatch_out.txt
