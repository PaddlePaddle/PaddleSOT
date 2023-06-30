export STRICT_MODE=0

PADDLE_TEST_BASE=./Paddle/test/dygraph_to_static
failed_tests=()
disabled_tests=(
    ${PADDLE_TEST_BASE}/test_lac.py
    ${PADDLE_TEST_BASE}/test_dict.py
    ${PADDLE_TEST_BASE}/test_list.py
    ${PADDLE_TEST_BASE}/test_sentiment.py
    ${PADDLE_TEST_BASE}/test_reinforcement_learning.py
    ${PADDLE_TEST_BASE}/test_bert.py
    ${PADDLE_TEST_BASE}/test_resnet.py
    ${PADDLE_TEST_BASE}/test_resnet_v2.py
)

for file in ${PADDLE_TEST_BASE}/*.py; do
    # 检查文件是否为 Python 文件
    if [[ -f "$file" && ! "${disabled_tests[@]}" =~ "$file" ]]; then
        echo Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=${STRICT_MODE} python " $file
        # 执行文件
        # python "$file" 2>&1 >>/home/data/output.txt
        python -u "$file"
        if [ $? -ne 0 ]; then
            echo "run $file failed"
            failed_tests+=("$file")
        else
            echo "run $file success"
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
