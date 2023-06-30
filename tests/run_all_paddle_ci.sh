export STRICT_MODE=0

failed_tests=()
disabled_tests=(
    test_write_python_container.py
    test_slice.py
    test_lac.py
    test_dict.py
    test_list.py
    test_sentiment.py
    test_reinforcement_learning.py
    test_bert.py
    test_resnet.py
    test_resnet_v2.py
)

for file in ./Paddle/test/dygraph_to_static/*.py; do
    # 检查文件是否为 Python 文件
    if [ -f "$file" && ! "${disabled_tests[@]}" =~ "$file"]; then
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
