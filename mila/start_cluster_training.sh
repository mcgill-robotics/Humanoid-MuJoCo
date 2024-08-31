rm std_out.txt std_error.txt

sbatch --gres=gpu:1 mila_train_script > sbatch_out.txt
