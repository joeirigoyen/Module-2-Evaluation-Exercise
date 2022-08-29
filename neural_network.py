from df_generator import DataframeGenerator
import numpy as np

# Import dataframe
df_gen = DataframeGenerator("data\\breast-cancer-wisconsin.data")
df = df_gen.train
# Clean dataframe from unwanted columns and assign them new column names
df.drop(0, axis=1, inplace=True)
# Move diagnosis column to be the first column
diagnosis_col = df.pop(10)
df.insert(0, 'diagnosis', diagnosis_col)
# Apply function to diagnosis column for it to represent boolean values instead of 2s and 4s
df['diagnosis'] = df['diagnosis'].apply(lambda x: 1 if x == 4 else 0)

# Run functions
if __name__ == '__main__':
    # Normalize columns
    for colname in df.drop('diagnosis', axis=1).columns:
        col_max, col_min = df[colname].max(), df[colname].min()
        df[colname] = (df[colname] - col_min)  / (col_max - col_min)
    # Split dataframe into separate arrays
    x = df.drop('diagnosis', axis=1).to_numpy()
    y = df['diagnosis'].to_numpy()
    # Get sizes
    m, n = x.shape
    # Reshape arrays to better represent the data
    x = x.T
    print(x)