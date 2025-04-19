#!/bin/bash
# run_sequential_tests.sh

echo "=========================================================="
echo "SEQUENTIAL TESTING SCRIPT"
echo "=========================================================="
echo "This script will run segmentation tests followed by classification tests"
echo "Starting at $(date)"
echo "=========================================================="

# Function to display elapsed time
show_elapsed() {
  local seconds=$1
  local hours=$((seconds / 3600))
  local minutes=$(( (seconds % 3600) / 60 ))
  local secs=$((seconds % 60))
  echo "${hours}h ${minutes}m ${secs}s"
}

# Start overall timer
start_time=$(date +%s)

# Step 1: Run segmentation tests
echo -e "\n>> PHASE 1: RUNNING SEGMENTATION TESTS"
echo "Starting segmentation tests at $(date)"
seg_start=$(date +%s)

python3 test_segmentation_model.py
seg_status=$?

seg_end=$(date +%s)
seg_runtime=$((seg_end - seg_start))

if [ $seg_status -eq 0 ]; then
  echo "✅ Segmentation tests completed successfully in $(show_elapsed $seg_runtime)"
else
  echo "⚠️ Segmentation tests finished with errors (exit code: $seg_status). Runtime: $(show_elapsed $seg_runtime)"
fi

# Step 2: Run classification tests
echo -e "\n>> PHASE 2: RUNNING CLASSIFICATION TESTS"
echo "Starting classification tests at $(date)"
cls_start=$(date +%s)

python3 test_classification_models.py
cls_status=$?

cls_end=$(date +%s)
cls_runtime=$((cls_end - cls_start))

if [ $cls_status -eq 0 ]; then
  echo "✅ Classification tests completed successfully in $(show_elapsed $cls_runtime)"
else
  echo "⚠️ Classification tests finished with errors (exit code: $cls_status). Runtime: $(show_elapsed $cls_runtime)"
fi

# Final summary
total_runtime=$((cls_end - start_time))
echo -e "\n=========================================================="
echo "TESTING COMPLETED"
echo "Total runtime: $(show_elapsed $total_runtime)"
echo "Segmentation tests: $([ $seg_status -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "Classification tests: $([ $cls_status -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "Finished at $(date)"
echo "=========================================================="