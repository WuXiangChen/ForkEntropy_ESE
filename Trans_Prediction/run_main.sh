#!/bin/bash

# 启动训练进程，使用不同的CUDA设备和目标变量
for i in 0 1 2 3 4 5; do
    nohup python train_model.py --batch_size 20 --pro_DV "numBugReportIssues" --ablaNum $i --device 1 > train_numBugReportIssues_replenish_$i.log 2>&1 &
    nohup python train_model.py --batch_size 20 --pro_DV "numIntegratedCommits" --ablaNum $i --device 2 > train_numIntegratedCommits_replenish_$i.log 2>&1 &
    nohup python train_model.py --batch_size 20 --pro_DV "ratioMergedPrs" --ablaNum $i --device 3 > ratioMergedPrs_replenish_$i.log 2>&1 &
    wait
done


echo "Started training processes in the background."
