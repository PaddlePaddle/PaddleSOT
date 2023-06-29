export STRICT_MODE=0

for file in ./Paddle/test/dygraph_to_static/*.py; do
    # 检查文件是否为 Python 文件
    if [ -f "$file" ]; then
        echo Running: PYTHONPATH=$PYTHONPATH " STRICT_MODE=${STRICT_MODE} python " $file
        # 执行文件
        # python "$file" 2>&1 >>/home/data/output.txt
        python -u "$file"
        if [ $? -ne 0 ]; then
            echo "run $file failed"
        else
            echo "run $file success"
        fi
    fi
done
