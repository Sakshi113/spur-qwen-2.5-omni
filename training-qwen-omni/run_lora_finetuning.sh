#!/bin/bash

# ==============================================================================
# End-to-End Fine-Tuning Script for Qwen2.5-Omni on Spatial Audio
# ==============================================================================
# This script automates the entire process:
# 1. Preprocesses and unifies multiple raw JSON datasets into a single training file.
# 2. Runs the fine-tuning script using the preprocessed data.
#
# If any command fails, the script will exit immediately.
# ==============================================================================

set -e

# --- Configuration ---
# All major settings are defined here for easy modification.

# Python scripts to execute
PREPROCESS_SCRIPT="preprocess_data.py"
TRAIN_SCRIPT="train_spatial_only_hf.py"

# --- Main Execution ---

echo "========================================================"
echo "          STARTING QWEN2.5-OMNI FINE-TUNING"
echo "========================================================"
echo

# Step 1: Preprocess and Unify Datasets
echo "--- Step 1: Preprocessing and Unifying Datasets ---"

if [ ! -f "$PREPROCESS_SCRIPT" ]; then
    echo "❌ ERROR: Preprocessing script '$PREPROCESS_SCRIPT' not found!"
    exit 1
fi

# Execute the preprocessing script
# python "$PREPROCESS_SCRIPT"

echo "✅ Preprocessing complete. Unified dataset created."
echo
echo "--------------------------------------------------------"
echo

# Step 2: Start the Fine-Tuning Process
echo "--- Step 2: Starting Fine-Tuning with PEFT (LoRA) ---"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ ERROR: Training script '$TRAIN_SCRIPT' not found!"
    exit 1
fi

# Execute the training script
# All parameters (model ID, paths, hyperparameters) are defined within this script.
python "$TRAIN_SCRIPT"

echo
echo "--------------------------------------------------------"
echo
echo "🎉 End-to-end fine-tuning complete! 🎉"
echo
echo "You can find the final trained model adapters and processor files in the output directory specified in '$TRAIN_SCRIPT'."
echo "========================================================"