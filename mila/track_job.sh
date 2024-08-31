job_id=$(cat sbatch_out.txt | awk '{print $4}')

watch scontrol show job $job_id
