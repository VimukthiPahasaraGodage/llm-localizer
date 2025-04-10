import os

import pandas as pd


def standardize_df(df_path):
    column_names = ['source_code', 'vuln_lines']
    df = pd.read_csv(df_path, header=None, names=column_names)
    df['index'] = range(0, len(df))  # Starts from 0
    df = df[['index'] + df.columns[:-1].tolist()]  # move 'index' column to front
    df.to_csv(df_path, index=False)


def run_standardization(dataset_path: str, dataset_version: str, dataset_name: str):
    df_path = f"{os.getcwd()}/{dataset_path}/{dataset_name}/{dataset_version}/{dataset_name}.csv"
    standardize_df(df_path)
    print(f'Finished standardizing the dataset at path: {df_path}')


if __name__ == '__main__':
    os.chdir('..')  # goto a layer above in directory tree
    run_standardization('data/dataset', 'v3', 'solidity')
