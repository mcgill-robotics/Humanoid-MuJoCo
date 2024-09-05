job_id=$(cat slurm/sbatch_out.txt | awk '{print $4}')

scancel $job_id
