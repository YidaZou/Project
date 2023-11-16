for input_type in "Random" "Sorted" "ReverseSorted" "1perturbed"; do  
    for num_val in 65536 262144 1048576 4194304 16777216 67108864 268435456; do
        for (( n=64; n<=1024 ; n*=2)); do
            sbatch mergeSortCUDA.grace_job $n $num_val $input_type
        done
    done
done