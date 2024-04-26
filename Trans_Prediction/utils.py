import pandas as pd
def save_pre_and_reals(predictions, reals, output_csv_path):
    results_df = pd.DataFrame({'predictions': predictions, 'reals': reals})
    results_df.to_csv(output_csv_path, index=False)