# 遍历目录下的所有 Python 文件  
PYTHONPATH=$PYTHONPATH:../

for file in ./test_*.py; do  
    # 检查文件是否为 Python 文件  
    echo "start run: " $file
    if [ -f "$file" ]; then  
        # 执行文件  
        python "$file"  
        if [ $? -ne 0 ]; then  
            echo "run $file failed"  
            exit 1  
        fi
    fi  
done
