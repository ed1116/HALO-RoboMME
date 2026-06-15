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
MAX_JOBS=16
JOBS_RUNNING=0
DS_SIZE=500
NUM_REPEAT_TRAJ=1
# can be 'generic' or 'ts'
SEQLEN=2048
MODE='ts_text_only_bbox_v2'
config_file='config/task/task_robocasa_turn_on_stove.json'
DOWNSAMPLE_OBS=8
for task_file in "${TASK_FILE_LIST[@]}"; do
    # if the base_dir of demo_im128_notp.hdf5 / generated_qa / gpt-4o_results_${DS_SIZE}.json exists, then skip
    base_dir=$(dirname $task_file)
    # Use a python one-liner to compute the expected file path using the ICRT utility
    MODEL_NAME="gpt-4o-mini"
    # ############# setting dataset size based on the task ######
    if [[ $task_file == *"MemRetrieveOilsFromCounter"* ]]; then
        # DS_SIZE=200
        SEQLEN=512
        NUM_REPEAT_TRAJ=300
        DS_SIZE=1000
    fi
    if [[ $task_file == *"MemWashAndReturn"* ]]; then
        # DS_SIZE=200
        SEQLEN=512
        NUM_REPEAT_TRAJ=300
        DS_SIZE=2000
    fi
    if [[ $task_file == *"MemPutKBreadInMicrowave"* ]]; then
        NUM_REPEAT_TRAJ=300
        DS_SIZE=1000
    fi
    if [[ $task_file == *"MemHeatPot"* ]]; then
        NUM_REPEAT_TRAJ=300
        DS_SIZE=2000 # 4 times more data
    fi
    PREFIX="relevant_objs_cl${SEQLEN}"
    expected_file=$(python -c "
import sys
from halo.util.args import GenerateQAGenericArgs
print(
    GenerateQAGenericArgs.hdf5_path_to_qa_json_path(
        hdf5_path=sys.argv[1],
        mode=sys.argv[2],
        model_name=sys.argv[3],
        max_dataset_size=int(sys.argv[4]),
        prefix=sys.argv[5],
    )
)
" "$task_file" "$MODE" "$MODEL_NAME" "$DS_SIZE" "$PREFIX")
    echo "Expected file: $expected_file"
    # if [ -f "$expected_file" ]; then
    #     echo "Skipping $task_file because $expected_file already exists"
    #     continue
    # fi
    # # DS_SIZE=100 if RETRIEVE OILS FROM COUNTER
    # if [[ $task_file == *"MemRetrieveOilsFromCounter"* ]]; then
    #     DS_SIZE=100
    # fi
    # #######################################################

    if [ $JOBS_RUNNING -ge $MAX_JOBS ]; then
        wait -n
        ((JOBS_RUNNING--))
    fi
    EXTRA_ARGS=""
    ############# setting the same chunk length for all the examples in the dataset
    EXTRA_ARGS="--mode $MODE --max_timesteps -1 --dataset-config.min_chunk_length $SEQLEN --dataset_config.use_same_chunk_length"
    #######################################################
    #################### resume the run if the file exists ######
    EXTRA_ARGS="$EXTRA_ARGS --resume"
    #################### batch mode ######
    EXTRA_ARGS="$EXTRA_ARGS --batch_mode"
    ARGS="--dataset_config.hdf5_paths $task_file --dataset-config.dataset-json $config_file --shared_config.downsample_obs $DOWNSAMPLE_OBS --dataset-config.max_dataset_size $DS_SIZE --seed $SEED --prefix $PREFIX --dataset-config.num_repeat_traj $NUM_REPEAT_TRAJ --model_name $MODEL_NAME"
    ARGS="$ARGS $EXTRA_ARGS"
    echo "python scripts/data_gen/generate_qa.py $ARGS"
    python scripts/data_gen/generate_qa.py $ARGS &
    SEED=$((SEED+1))
    ((JOBS_RUNNING++))
done
wait -n
echo "All tasks completed."
