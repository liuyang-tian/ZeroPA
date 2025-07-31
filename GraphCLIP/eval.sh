#!/bin/bash

SOURCE_DATA=("arxiv")
DATA_NAMES=("wikics" "photo" "computer" "instagram")
LEN=(2)

for len in "${LEN[@]}"; do
  for source_data in "${SOURCE_DATA[@]}"; do
    for data in "${DATA_NAMES[@]}"; do
      echo "Running: --text_prompt_len $len --token_num $len --source_data $source_data --target_data $data "
      python eval.py --gpu 0 --text_prompt_len "$len" --token_num "$len" --source_data "$source_data" --target_data "$data"
    done
  done
done