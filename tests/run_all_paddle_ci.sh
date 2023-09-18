export STRICT_MODE=0
export ENABLE_SOT=True
export ENABLE_FALL_BACK=True
export MIN_GRAPH_SIZE=-1
export COST_MODEL=True

PADDLE_TEST_BASE=./Paddle/test/dygraph_to_static
failed_tests=()
disabled_tests=(
    ${PADDLE_TEST_BASE}/test_lac.py # disabled by paddle
    ${PADDLE_TEST_BASE}/test_sentiment.py # disabled unitcase by paddle
)

for file in ${PADDLE_TEST_BASE}/*.py; do
    # 检查文件是否为 Python 文件
    if [[ -f "$file" && ! "${disabled_tests[@]}" =~ "$file" ]]; then
        if [[ -n "$GITHUB_ACTIONS" ]]; then
            echo ::group::Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=${STRICT_MODE} python " $file
        else
            echo Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=${STRICT_MODE} python " $file
        fi
        # 执行文件
        # python "$file" 2>&1 >>/home/data/output.txt
        python -u "$file"
        if [ $? -ne 0 ]; then
            echo "run $file failed"
            failed_tests+=("$file")
        else
            echo "run $file success"
        fi
        if [[ -n "$GITHUB_ACTIONS" ]]; then
            echo "::endgroup::"
        fi
    fi
done

if [ ${#failed_tests[@]} -ne 0 ]; then
    echo "failed tests file:"
    for failed_test in "${failed_tests[@]}"; do
        echo "$failed_test"
    done
    exit 1
fi
