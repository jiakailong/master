#!/bin/bash

# 遍历 ModelNet40 下所有文件夹
for category in dataset/ModelNet40/*/; do
    # 处理 test 和 train 文件夹中的 OFF 文件
    for type in test train; do
        find "$category$type" -name "*.off" -exec sed -i '1{/^OFF[0-9]/s/OFF/OFF\n/}' {} \;
    done
done