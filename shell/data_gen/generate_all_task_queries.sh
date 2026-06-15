TASK_FILE_LIST=(
    # washandreturn
    # "$CASAPLAY_DATAROOT/memory/MemWashAndReturnLeft/2025-07-25-00-12-14/demo_im128_notp.hdf5"
    # "$CASAPLAY_DATAROOT/memory/MemWashAndReturnRight/2025-07-25-00-45-53/demo_im128_notp.hdf5"
    # retrieve_oil
    # "$CASAPLAY_DATAROOT/memory/MemRetrieveOilsFromCounterLL/2025-09-20-22-49-27/demo_im128_notp.hdf5"
    # "$CASAPLAY_DATAROOT/memory/MemRetrieveOilsFromCounterLR/2025-09-20-23-25-19/demo_im128_notp.hdf5"
    # "$CASAPLAY_DATAROOT/memory/MemRetrieveOilsFromCounterRL/2025-09-20-23-06-15/demo_im128_notp.hdf5"
    # "$CASAPLAY_DATAROOT/memory/MemRetrieveOilsFromCounterRR/2025-09-20-23-14-26/demo_im128_notp.hdf5"
    # kbreads
    # "$CASAPLAY_DATAROOT/memory/MemPutKBreadInMicrowave/2025-07-25-10-40-55/demo_im128_notp.hdf5"
    # "$CASAPLAY_DATAROOT/memory/MemPutKBreadInMicrowave/2025-07-25-13-25-05/demo_im128_notp.hdf5"
    # "$CASAPLAY_DATAROOT/memory/MemPutKBreadInMicrowave/2025-07-25-14-50-59/demo_im128_notp.hdf5"
    # heatpot
    # "$CASAPLAY_DATAROOT/memory/MemHeatPot/2025-07-24-22-26-20/demo_im128_notp.hdf5"
    # "$CASAPLAY_DATAROOT/memory/MemHeatPot/2025-07-25-15-56-53/demo_im128_notp.hdf5"
)
SEED=1
MAX_JOBS=10
JOBS_RUNNING=0
DS_SIZE=100
export OPENAI_API_KEY=$OPENAI_API_KEY
for task_file in "${TASK_FILE_LIST[@]}"; do
    # if the base_dir of demo_im128_notp.hdf5 / generated_qa / gpt-4o_results_${DS_SIZE}.json exists, then skip
    base_dir=$(dirname $task_file)
    expected_file="${base_dir}/generated_qa/gpt-4o_results_${DS_SIZE}.json"
    if [ -f "$expected_file" ]; then
        # echo "Skipping $task_file because $expected_file already exists"
        continue
    fi

    if [ $JOBS_RUNNING -ge $MAX_JOBS ]; then
        wait -n
        ((JOBS_RUNNING--))
    fi
    ARGS="--hdf5_file $task_file --num_instructions 20"
    echo "python scripts/data_gen/generate_task_queries.py $ARGS"
    python scripts/data_gen/generate_task_queries.py $ARGS &
    SEED=$((SEED+1))
    ((JOBS_RUNNING++))
done
wait
echo "All tasks completed."
