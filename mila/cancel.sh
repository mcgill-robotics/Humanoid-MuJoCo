job_id=$(cat mila/sbatch_out.txt | awk '{print $4}')

scancel $job_id
