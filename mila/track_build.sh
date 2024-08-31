job_id=$(cat sbatch_build_out.txt | awk '{print $4}')

watch scontrol show job $job_id
