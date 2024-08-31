rm build_stdout.txt build_stderr.txt

sbatch --gres=gpu:1 mila_build_sif > sbatch_build_out.txt
