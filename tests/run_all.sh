# 遍历目录下的所有 Python 文件
export PYTHONPATH=$PYTHONPATH:../
export STRICT_MODE=1
export LOG_LEVEL=3

failed_tests=()

for file in ./test_*.py; do
    # 检查文件是否为 Python 文件
    if [ -f "$file" ]; then
        echo Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=1 python " $file
        # 执行文件
        python "$file"
        if [ $? -ne 0 ]; then
            echo "run $file failed"
            failed_tests+=("$file")
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
