export STRICT_MODE=0
export ENABLE_SOT=True
export ENABLE_FALL_BACK=True
export MIN_GRAPH_SIZE=-1

PADDLE_TEST_BASE=./Paddle/test/dygraph_to_static
failed_tests=()
disabled_tests=(
    ${PADDLE_TEST_BASE}/test_lac.py # disabled by paddle
    ${PADDLE_TEST_BASE}/test_sentiment.py # disabled unitcase by paddle
    ${PADDLE_TEST_BASE}/test_reinforcement_learning.py # 'CartPoleEnv' object has no attribute 'seed'
    # tmp = x
    # for i in range(x)
    #     tmp += Linear(x)
    # out = paddle.grad(tmp, x)
    # return out
    # Because range interrupts networking, Paddle.grad cannot be networked as a standalone API.
    # CAN BE OPEN AFTER: range is support.
    ${PADDLE_TEST_BASE}/test_grad.py
    ${PADDLE_TEST_BASE}/test_ptb_lm.py # There is accuracy problem of the model in SOT
    ${PADDLE_TEST_BASE}/test_ptb_lm_v2.py # There is accuracy problem of the model in SOT
    ${PADDLE_TEST_BASE}/test_cycle_gan.py # This test has a precision problem when it reaches the maximum cache size
    ${PADDLE_TEST_BASE}/test_inplace_assign.py # This test waiting for #301
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
