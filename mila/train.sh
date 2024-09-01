rm mila/stdout.txt mila/stderr.txt

sbatch --gres=gpu:1 mila/mila_train_script > mila/sbatch_out.txt
