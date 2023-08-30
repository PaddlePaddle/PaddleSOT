# 遍历目录下的所有 python 文件
export PYTHONPATH=$PYTHONPATH:../
export STRICT_MODE=1
export MIN_GRAPH_SIZE=-1

IS_PY311=`python -c "import sys; print(sys.version_info >= (3, 11))"`
echo "IS_PY311:" $IS_PY311

failed_tests=()

py311_skiped_tests=(
    # ./test_01_basic.py            There are some case need to be fixed
    # ./test_04_list.py             There are some case need to be fixed
    # ./test_05_dict.py             There are some case need to be fixed
    ./test_10_build_unpack.py
    ./test_11_jumps.py
    ./test_12_for_loop.py
    ./test_13_make_function.py
    ./test_14_operators.py
    ./test_15_slice.py
    ./test_17_paddle_layer.py
    ./test_18_tensor_method.py
    ./test_19_closure.py
    ./test_21_global.py
    ./test_break_graph.py
    ./test_builtin_dispatch.py
    ./test_constant_graph.py
    ./test_enumerate.py
    ./test_exception.py
    ./test_execution_base.py
    ./test_guard_user_defined_fn.py
    ./test_inplace_api.py
    ./test_range.py
    ./test_resnet.py
    ./test_resnet50_backward.py
    ./test_sir_rollback.py
    ./test_str_format.py
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
