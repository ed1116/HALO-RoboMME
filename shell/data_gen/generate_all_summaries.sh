all_hdf5_paths=(
    # $CASAPLAY_DATAROOT/memory/MemPutKBreadInMicrowave/2025-07-25-10-40-55/demo_im128_notp.hdf5 
    # $CASAPLAY_DATAROOT/memory/MemPutKBreadInMicrowave/2025-07-25-14-50-59/demo_im128_notp.hdf5
    # $CASAPLAY_DATAROOT/memory/MemPutKBreadInMicrowave/2025-07-25-13-25-05/demo_im128_notp.hdf5
    # $CASAPLAY_DATAROOT/memory/MemHeatPot/2025-07-24-22-26-20/demo_im128_notp.hdf5
    # $CASAPLAY_DATAROOT/memory/MemHeatPot/2025-07-25-15-56-53/demo_im128_notp.hdf5
    # $CASAPLAY_DATAROOT/memory/MemRetrieveOilsFromCounterLL/2025-09-20-22-49-27/demo_im128_notp.hdf5
    # $CASAPLAY_DATAROOT/memory/MemRetrieveOilsFromCounterLR/2025-09-20-23-25-19/demo_im128_notp.hdf5
    # $CASAPLAY_DATAROOT/memory/MemRetrieveOilsFromCounterRL/2025-09-20-23-06-15/demo_im128_notp.hdf5
    # $CASAPLAY_DATAROOT/memory/MemRetrieveOilsFromCounterRR/2025-09-20-23-14-26/demo_im128_notp.hdf5
    # $CASAPLAY_DATAROOT/memory/MemWashAndReturnLeft/2025-07-25-00-12-14/demo_im128_notp.hdf5
    # $CASAPLAY_DATAROOT/memory/MemWashAndReturnRight/2025-07-25-00-45-53/demo_im128_notp.hdf5
)
OPENAI_API_KEY=$OPENAI_API_KEY
max_exps_in_parallel=6
exps_launched=0

for hdf5_path in "${all_hdf5_paths[@]}"; do
    # if mutex in hdf5_path, then set the image_keys to "obs/agentview_rgb"
    ARGS="--hdf5_path $hdf5_path"
    if [[ $hdf5_path == *"mutex"* ]]; then
        ARGS="$ARGS --image_keys obs/agentview_rgb"
    else
        ARGS="$ARGS --image_keys obs/robot0_agentview_center_image obs/robot0_eye_in_hand_image"
    fi
    echo python scripts/data_gen/generate_summary.py $ARGS
    python scripts/data_gen/generate_summary.py $ARGS &
    ((exps_launched++))
    if [ "${exps_launched:-0}" -eq "$max_exps_in_parallel" ]; then
        wait -n
        ((exps_launched--))
    fi
done
wait
echo "All summaries generated."
