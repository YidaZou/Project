for num_processes in 65536 262144 1048576 4194304 16777216 67108864 268435456; do
  for (( n=2; n<=1024 ; n*=2)); do
      sbatch bitonic.grace_job $num_processes $n
  done
done