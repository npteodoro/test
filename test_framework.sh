#!/bin/bash
# filepath: /home/zelx/ic/test_framework.sh

echo "========== Testing River Segmentation/Classification Framework =========="
echo "Starting comprehensive tests at $(date)"
echo

# Color codes for better readability
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run a test and report success or failure
run_test() {
    local test_name="$1"
    local command="$2"
    
    echo -e "${YELLOW}Running test: ${test_name}${NC}"
    echo "Command: $command"
    echo "----------------------------------------"
    
    # Run the command and capture output
    output=$(eval "$command" 2>&1)
    exit_code=$?
    
    # Check for Python exceptions or error keywords
    if [[ $exit_code -eq 0 ]] && [[ ! $output =~ "Error:" ]] && [[ ! $output =~ "Exception:" ]] && [[ ! $output =~ "Traceback" ]]; then
        echo -e "${GREEN}✓ Test passed: ${test_name}${NC}"
        echo "----------------------------------------"
        return 0
    else
        echo -e "${RED}✗ Test failed: ${test_name}${NC}"
        echo -e "${RED}Error details:${NC}"
        # Print last few lines containing error information
        echo "$output" | grep -A 10 -B 2 -i "error\|exception\|traceback" | tail -15
        echo "----------------------------------------"
        return 1
    fi
}

# Track overall success
PASSED=0
FAILED=0

echo -e "${BLUE}TESTING AUGMENTATION STRATEGIES${NC}"
echo "========================================"

# Test all augmentation strategies
for augmentation in base geometric semantic combined; do
    run_test "Augmentation: $augmentation" "python main.py --task segmentation --set data.augmentation $augmentation --set training.num_epochs 1"
    if [ $? -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi
done

echo -e "${BLUE}TESTING ENCODERS${NC}"
echo "======================="

# Test all encoders with segmentation
for encoder in mobilenet_v3_large efficientnet_b2 squeezenet1_1 shufflenet_v2_x1_0 mnasnet1_0; do
    run_test "Segmentation with $encoder" "python main.py --task segmentation --set model.encoder $encoder --set training.num_epochs 1"
    if [ $? -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi
done

# Test all encoders with classification
for encoder in mobilenet_v3_large efficientnet_b2 squeezenet1_1 shufflenet_v2_x1_0 mnasnet1_0; do
    run_test "Classification with $encoder" "python main.py --task classification --set model.encoder $encoder --set training.num_epochs 1"
    if [ $? -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi
done

echo -e "${BLUE}TESTING DECODERS${NC}"
echo "======================"

# Test all decoders
for decoder in unet fpn deeplabv3; do
    run_test "Segmentation with $decoder decoder" "python main.py --task segmentation --set model.decoder $decoder --set training.num_epochs 1"
    if [ $? -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi
done

echo -e "${BLUE}TESTING ADDITIONAL FEATURES${NC}"
echo "=============================="

# Test classification with mask as channel
run_test "Classification with mask as channel" "python main.py --task classification --set model.use_mask_as_channel true --set training.num_epochs 1"
if [ $? -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi

# Test with different batch size and learning rate
run_test "Custom training parameters" "python main.py --task segmentation --set training.batch_size 8 --set training.learning_rate 0.001 --set training.num_epochs 1"
if [ $? -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi

# Test different schedulers
for scheduler in reduce_on_plateau cosine none; do
    run_test "$scheduler scheduler" "python main.py --task segmentation --set training.scheduler $scheduler --set training.num_epochs 1"
    if [ $? -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi
done

# Test early stopping options
run_test "With early stopping" "python main.py --task segmentation --set training.early_stopping true --set training.num_epochs 1"
if [ $? -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi

run_test "Without early stopping" "python main.py --task segmentation --set training.early_stopping false --set training.num_epochs 1"
if [ $? -eq 0 ]; then PASSED=$((PASSED+1)); else FAILED=$((FAILED+1)); fi

# Summary
echo
echo "========== Test Summary =========="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo "Total tests: $((PASSED + FAILED))"
echo "=================================="

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! The framework is working correctly.${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed. Please check the logs above for details.${NC}"
    exit 1
fi