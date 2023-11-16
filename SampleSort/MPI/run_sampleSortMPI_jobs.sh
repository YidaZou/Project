for input_type in "Random" "Sorted" "ReverseSorted" "1perturbed"; do
    for input_sizes in 65536 262144 1048576 4194304 16777216 67108864 268435456; do
        for (( n=2; n<=1024 ; n*=2)); do
            sbatch sampleSort.grace_job $n $input_sizes $input_type
        done
    done
done
