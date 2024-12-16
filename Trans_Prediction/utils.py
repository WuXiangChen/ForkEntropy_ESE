import pandas as pd

# 将数据保存至csv中
def save_pre_and_reals(predictions, reals, output_csv_path):
    # ... （之前的代码保持不变）

    # 创建 DataFrame
    results_df = pd.DataFrame({'predictions': predictions, 'reals': reals})

    # 将 DataFrame 写入 CSV 文件
    results_df.to_csv(output_csv_path, index=False)