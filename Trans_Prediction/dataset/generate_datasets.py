# 在这里 我希望做一个动态的训练数据的生成过程，主要操作如下
# 1. 读取所有的数据集

'''
    导包区
'''
from matplotlib import axis
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset

def read_csv_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna(axis=0, how='any')
    grouped = data.groupby('project')
    return grouped

'''
    这里的作用是根据给定的时延和拟合目标的选择进行数据集的拓展
'''
def delay_recombin(delay, project_df, target, ablaNum):
    # 这里需要对project_df进行列的选择
    scaler = StandardScaler()

    if target== "numIntegratedCommits":
        if ablaNum==-1:
            project_target_df = project_df.iloc[:, 0 : 7]
        else:
            project_target_df = project_df.iloc[:, 0 : 7]
            project_target_df = project_target_df.drop(project_target_df.columns[ablaNum], axis=1)
                    
        data_to_scale = project_target_df[["num_integrated_commits"]].values
        scaled_data = scaler.fit_transform(data_to_scale)
        project_target_df.loc[:, "num_integrated_commits"] = scaled_data

    elif target == "ratioMergedPrs":
        if ablaNum==-1:
            project_target_df = project_df.iloc[:, list(range(7, len(project_df.columns)-2)) + [-2]]
        else:
            project_target_df = project_df.iloc[:, list(range(7, len(project_df.columns)-2)) + [-2]]
            project_target_df = project_target_df.drop(project_target_df.columns[ablaNum], axis=1)
        
        data_to_scale = project_target_df[["ratio_merged_prs"]].values
        scaled_data = scaler.fit_transform(data_to_scale)
        project_target_df.loc[:, "ratio_merged_prs"] = scaled_data


    elif target == "numBugReportIssues":
        if ablaNum==-1:
            project_target_df = project_df.iloc[:, list(range(0, 6)) + [-1]]
        else:
            project_target_df = project_df.iloc[:, list(range(0, 6)) + [-1]]
            project_target_df = project_target_df.drop(project_target_df.columns[ablaNum], axis=1)
        
        data_to_scale = project_target_df[["num_bug_report_issues"]].values
        scaled_data = scaler.fit_transform(data_to_scale)
        project_target_df.loc[:, "num_bug_report_issues"] = scaled_data

    #  选好指定的列属性以后，按照时延对数据集进行拓展
    re_df = project_target_df.reset_index(drop=True)
    re_df = re_df.rename(columns=lambda x: f"{x}_0")

    for i in range(1,delay+1):
        df_replicated = project_target_df.iloc[i:].reset_index(drop=True)
        rename_df_replicated = df_replicated.rename(columns=lambda x: f"{x}_{i}")
        df_replicated_ = pd.concat([rename_df_replicated, pd.DataFrame(np.nan, index=range(i), columns=rename_df_replicated.columns)],
                                  ignore_index=True)
        re_df = pd.concat([re_df, df_replicated_], axis=1)
    re_df = re_df.dropna(axis=0, how="any")
    # 这边得按照给定的时延，对数据的列信息进行重排
    ori_len = len(project_target_df.columns)
    re_index = [i + j * ori_len for i in range(ori_len) for j in range(delay+1)]
    re_df = re_df.iloc[:, re_index]
    re_df = torch.tensor(re_df.values, dtype=torch.float32)
    return re_df

def generate_datasets(grouped_data, delay, target ="numIntegratedCommits", ablaNum=-1):
    grouped_train = []
    grouped_test = []
    for name, group in grouped_data:
        project_df = group.drop('project', axis=1)
        # 这里需要通过tensor_group来进行生成数据集
        # 这边拓展 需要考虑的东西很多，一个是拓展的时延，另外一个是根据拓展对象进行列选择
        re_tensor = delay_recombin(delay = delay, project_df = project_df, target = target, ablaNum=ablaNum)
        len_ = int(re_tensor.shape[0] * 0.8)
        train_set = re_tensor[:len_,:]
        tets_set = re_tensor[len_:,:]
        grouped_train.append(train_set)
        grouped_test.append(tets_set)
    group_train_set = torch.vstack(grouped_train)
    group_test_set = torch.vstack(grouped_test)
    return group_train_set, group_test_set

def generate_datasets_allPro(delay = 3, target ="numIntegratedCommits", ablaNum=-1):
    # 读取文件，按project形成grouped
    file_path = "dataset/fork_entropy_dataset_extended2.csv"
    grouped = read_csv_data(file_path)
    datasets = generate_datasets(grouped, delay, target, ablaNum)
    return datasets

# 测试
# datasets = generate_datasets_allPro()
# print()