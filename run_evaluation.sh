#!/bin/bash

# Evaluation script for SigLIP medical model

# Configuration
CHECKPOINT="checkpoints/siglip_medical_run1/best_model.pt"
DATA_PATH="processed_data/all_med_pairs.parquet"
OUTPUT="checkpoints/siglip_medical_run1/eval_results.json"

# Run evaluation
python evaluate.py \
    --checkpoint $CHECKPOINT \
    --data-path $DATA_PATH \
    --output $OUTPUT \
    --split val

echo "Evaluation completed. Results saved to $OUTPUT"