# 遍历目录下的所有 Python 文件
export PYTHONPATH=$PYTHONPATH:../
export STRICT_MODE=1

failed_tests=()

for file in ./test_*.py; do
    # 检查文件是否为 Python 文件
    if [ -f "$file" ]; then
        if [[ -n "$GITHUB_ACTIONS" ]]; then
        echo ::group::Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=1 python " $file
        else
        echo Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=1 python " $file
        fi
        # 执行文件
        python "$file" 2> >(grep "File <$file>, line")
        if [ $? -ne 0 ]; then
            echo "run $file failed"
            failed_tests+=("$file")
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
        echo "::error file=$failed_test,line=1,col=5,endColumn=7::Missing semicolon"
    done
fi
