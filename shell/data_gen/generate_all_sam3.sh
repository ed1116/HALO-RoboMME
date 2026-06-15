hdf5_paths=(
    # "$CASAPLAY_DATAROOT/memory/mutex/MEM1/MEM1_Wash_the_bowl_and_place_it_back_in_the_same_container.hdf5"
    # "$CASAPLAY_DATAROOT/memory/mutex/MEM1/MEM2_Wash_the_bowl_and_place_it_back_in_the_same_container.hdf5"
    # "$CASAPLAY_DATAROOT/memory/mutex/MEM1/MEM3_Wash_the_bowl_and_place_it_back_in_the_same_container.hdf5"
    # "$CASAPLAY_DATAROOT/memory/mutex/MEM1/MEM4_Wash_the_bowl_and_place_it_back_in_the_same_container.hdf5"
    # "$CASAPLAY_DATAROOT/memory/mutex/MEM_HumanRobot_PutKCups/MEM1_Put_three_purple_cups_and_then_put_the_green_cup_in_the_shopping_bag.hdf5"
    # "$CASAPLAY_DATAROOT/memory/mutex/MEM_HumanRobot_PutKCups/MEM2_Put_three_purple_cups_and_then_put_the_green_cup_in_the_shopping_bag.hdf5"
    # "$CASAPLAY_DATAROOT/memory/mutex/MEM_HumanRobot_PutKCups/MEM3_Put_three_purple_cups_and_then_put_the_green_cup_in_the_shopping_bag.hdf5"
    # "$CASAPLAY_DATAROOT/memory/mutex/MEM_HumanRobot_PutKCups/MEM4_Put_three_purple_cups_and_then_put_the_green_cup_in_the_shopping_bag.hdf5"
    # "$CASAPLAY_DATAROOT/memory/mutex/MEM_HumanRobot_PutKCups/MEM5_Put_three_purple_cups_and_then_put_the_green_cup_in_the_shopping_bag.hdf5"
)
gpu_list=(0 1 2 3 4)
exps_launched=0
exps_in_parallel=5
first_run=True
for hdf5_path in "${hdf5_paths[@]}"; do
    if [ "$first_run" = "True" ]; then
        gpu_id=${gpu_list[$exps_launched % ${#gpu_list[@]}]}
    else
        # find the gpu_id with minimum utilization from gpu_list
        min_util=100
        # wait for 10 seconds
        sleep 10
        for gpu in "${gpu_list[@]}"; do
            util=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | grep "^$gpu," | cut -d',' -f2 | tr -d ' %')
            if [ "$util" -lt "$min_util" ]; then
                min_util=$util
                gpu_id=$gpu
            fi
        done
        echo "Launching on GPU $gpu_id: $hdf5_path"
    fi
    # if Wash_the_bowl_and_place_it_back_in_the_same_container in hdf5_path, then set the prompts to "green plate" "pink plate" "white small bowl" "sink" "robot"
    if [[ $hdf5_path == *"Wash_the_bowl_and_place_it_back_in_the_same_container"* ]]; then
        # for MEM3 and MEM4, set the prompts to "green plate" "pink plate" "yellow bowl" "sink" "robot"
        if [[ $hdf5_path == *"MEM3"* ]] || [[ $hdf5_path == *"MEM4"* ]]; then
            prompts=('green plate' 'pink plate' 'yellow bowl' 'sink' 'robot')
        else
            prompts=('green plate' 'pink plate' 'white bowl' 'sink' 'robot')
        fi
        min_mask_area_ratio=0.01
    elif [[ $hdf5_path == *"MEM_HumanRobot_PutKCups"* ]]; then
        prompts=('green cup' 'purple cup' 'green cup' 'shopping basket')
        min_mask_area_ratio=0.01
    else
        # print not implemented
        echo "Not implemented for $hdf5_path"
        exit 1
    fi
    # Build ARGS array with properly quoted prompts
    ARGS=("--hdf5_path" "$hdf5_path" "--prompts")
    for prompt in "${prompts[@]}"; do
        ARGS+=("$prompt")
    done
    ARGS+=("--min_mask_area_ratio" "$min_mask_area_ratio")
    echo CUDA_VISIBLE_DEVICES=$gpu_id python scripts/data_gen/generate_sam3_ids.py "${ARGS[@]}"
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/data_gen/generate_sam3_ids.py "${ARGS[@]}" &
    exps_launched=$((exps_launched+1))
    if [ $exps_launched -eq $exps_in_parallel ]; then
        first_run=False
        wait -n
        ((exps_launched--))
    fi
done
wait
