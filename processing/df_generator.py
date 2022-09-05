import pandas as pd
import numpy as np

class DataframeGenerator:
    def __init__(self, csv_file: str, header: bool=None, encoding: str="utf_8", na_values: str | int | float='?'):
        """Initializes a dataframe with certain fixes applied before returning it to the user.

        Args:
            csv_file (str): the path to the data csv file
            header (bool, optional): whether the csv contains headers or not. Defaults to None.
            encoding (str, optional): csv file's encoding type. Defaults to "utf_8".
            na_values (str | int | float, optional): character or number used to define a na value within the dataset. Defaults to '?'.
        """
        # Generate dataframe from csv file
        self.df = pd.read_csv(csv_file, header=header, encoding=encoding, na_values=na_values)
        # Shuffle data
        self.df = self.df.iloc[np.random.permutation(len(self.df))]
        # Clean possible na values
        for column in self.df.columns:
            mean = self.df[column].mean()
            self.df[column].fillna(mean, inplace=True)
            self.df[column] = self.df[column].astype('int')
        # Split into training and testing datasets
        self.train = self.df.sample(frac=0.8, random_state=25)
        self.test = self.df.drop(self.train.index)
        # Generate csv files with training and testing data
        self.train.to_csv("data\\train.csv", index=False)
        self.test.to_csv("data\\test.csv", index=False)
