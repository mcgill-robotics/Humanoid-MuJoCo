./mila/cancel.sh 2> /dev/null || true

rm mila/out.txt mila/err.txt 2> /dev/null || true

sbatch --gres=gpu:1 mila/script > mila/sbatch_out.txt
