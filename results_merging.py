

import pandas as pd
import glob

file_num = 143

# 1. Get a list of all CSV files in a directory
# files = glob.glob(f'/home/jlin475/results_{file_num}/*.csv')
files = glob.glob(f'/home/jlin475/results_0/*.csv')
# print(files)

# 2. Read each file into a list of DataFrames
df_list = [pd.read_csv(f) for f in files]

# 3. Concatenate all DataFrames into one
combined_df = pd.concat(df_list, ignore_index=True)

# 4. Save the combined result to a new CSV
# combined_df.to_csv(f'/home/jlin475/results_all/{file_num}_results.csv', index=False)
combined_df.to_csv(f'/home/jlin475/results_0/combined_results_test.csv', index=False)








