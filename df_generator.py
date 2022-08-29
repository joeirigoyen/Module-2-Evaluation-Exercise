
import pandas as pd

class DataframeGenerator:
    def __init__(self, csv_file: str, header: bool=None, encoding: str="utf_8", na_values: str | int | float='?'):
        # Generate dataframe from csv file
        self.df =pd.read_csv(csv_file, header=header, encoding=encoding, na_values=na_values)
        print(f"File at {csv_file} imported successfully as dataframe of shape: {self.df.shape}")
        # Clean possible na values
        for column in self.df.columns:
            mean = self.df[column].mean()
            self.df[column].fillna(mean, inplace=True)
            self.df[column] = self.df[column].astype('int')
        # Split into training and testing datasets
        self.train = self.df.sample(frac=0.8, random_state=25)
        self.test = self.df.drop(self.train.index)
        # Generate csv files with training and testing data
        self.train.to_csv("data\\train.csv")
        print(f"Training file created successfully as train.csv")
        print(f"# of training examples: {self.train.shape[0]}")
        self.test.to_csv("data\\test.csv")
        print(f"Training file created successfully as test.csv")
        print(f"# of testing examples: {self.test.shape[0]}")