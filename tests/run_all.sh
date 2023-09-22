# 遍历目录下的所有 python 文件
export PYTHONPATH=$PYTHONPATH:../
export STRICT_MODE=1
export COST_MODEL=False
export MIN_GRAPH_SIZE=0

IS_PY311=`python -c "import sys; print(sys.version_info >= (3, 11))"`
echo "IS_PY311:" $IS_PY311

failed_tests=()

py311_skiped_tests=(
    ./test_19_closure.py
    ./test_tensor_dtype_in_guard.py
)

for file in ./test_*.py; do
    # 检查文件是否为 python 文件
    if [ -f "$file" ]; then
        if [[ "$IS_PY311" == "True" && "${py311_skiped_tests[@]}" =~ "$file" ]]; then
            echo "skip $file for python3.11"
            continue
        fi
        if [[ -n "$GITHUB_ACTIONS" ]]; then
            echo ::group::Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=1 python " $file
        else
            echo Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=1 python " $file
        fi
        # 执行文件
        python_output=$(python $file 2>&1)

        if [ $? -ne 0 ]; then
            echo "run $file failed"
            failed_tests+=("$file")
            if [[ -n "$GITHUB_ACTIONS" ]]; then
                echo -e "$python_output" | python ./extract_errors.py
            fi
            echo -e "$python_output"
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
