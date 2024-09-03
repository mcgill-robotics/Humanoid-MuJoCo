rm mila/stdout.txt mila/stderr.txt

sbatch --gres=gpu:1 mila/train_script > mila/sbatch_out.txt
