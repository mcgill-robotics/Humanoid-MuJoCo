rm mila/out.txt mila/err.txt

sbatch --gres=gpu:1 mila/train_script > mila/sbatch_out.txt
