import os
import pandas as pd

def combine_csv_files(folder_path):
    # 获取文件夹中所有CSV文件的文件名
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    csv_files =  sorted(csv_files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    # 初始化一个空的DataFrame
    combined_df = pd.DataFrame()

    # 循环读取每个CSV文件并将其整合到DataFrame中
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True,axis=1)

    combined_df.to_csv("Transformer_numBugReportIssues.csv", index=False)

    return combined_df

# 指定文件夹路径
folder_path = 'numBugReportIssues/'

# 调用函数将所有CSV文件整合成一个DataFrame
result_df = combine_csv_files(folder_path)

# 打印合并后的DataFrame
print(result_df)
