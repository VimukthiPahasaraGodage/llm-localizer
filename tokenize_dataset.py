from enum import Enum
import pandas as pd
import os

import torch

from components.llm_utils import LLMInfo, LLMModels
from components.prompt import Prompt


class Driver:
    def __init__(self, tensor_path: str, dataset_path: str, dataset_version: str, dataset_name: str, llm_model: Enum, pre_code_part: str, post_code_part: str, standardize_df=False):
        self.cwd = os.getcwd()

        self.tensor_path = tensor_path

        self.dataset_path = dataset_path
        self.dataset_version = dataset_version
        self.dataset_name = dataset_name

        self.pre_code_part = pre_code_part
        self.post_code_part = post_code_part

        self.df_path = f"{self.cwd}/{self.dataset_path}/{self.dataset_version}/{self.dataset_name}.csv"
        if standardize_df:
            self.standardize_df()
        self.df = pd.read_csv(self.df_path)

        self.llm_model = llm_model
        self.llm_info = LLMInfo(self.llm_model)

        # Initialize folders and files for tensors and datasets
        self.save_path = f'{self.cwd}/{self.tensor_path}/{self.dataset_version}/{self.llm_model}/tokenizer'
        os.makedirs(self.save_path, exist_ok=True)
        self.save_path_prompt = f'{self.save_path}/prompt'
        os.makedirs(self.save_path_prompt, exist_ok=True)
        self.save_path_code = f'{self.save_path}/code'
        os.makedirs(self.save_path_code, exist_ok=True)

        self.save_tensors_and_info()

    def standardize_df(self):
        column_names = ['source_code', 'vuln_lines']
        df = pd.read_csv(self.df_path, header=None, names=column_names)
        df['index'] = range(0, len(df))  # Starts from 0
        df = df[['index'] + df.columns[:-1].tolist()] # move 'index' column to front
        df.to_csv(self.df_path, index=False)

    def __create_prompts(self):
        prompts = {}
        for index, row in self.df.iterrows():
            try:
                prompt = Prompt(self.pre_code_part, row['source_code'], self.post_code_part, self.llm_model)
                if prompt.prompt_processed:
                    prompts[row['index']] = prompt
            except Exception as e:
                print(f"Exception occurred for index: {index}. Error: {e}")
        return prompts

    def save_tensors_and_info(self):
        print("Creating prompts...")
        prompts = self.__create_prompts()

        indices = list(prompts.keys())
        indices.sort()

        rows = []

        print("Saving data...")
        for index in indices:
            pt = prompts[index]

            prompt = pt.get_prompt()
            prompt_tokens = pt.get_prompt_tokens()
            code_tokens = pt.get_code_tokens()
            line_split_lengths = pt.get_line_split_lengths()
            code_start_index = pt.get_code_start_index()
            code_end_index = pt.get_code_end_index()

            rows.append({'index': index,
                         'line_split_lengths': str(line_split_lengths),
                         'code_start_index': code_start_index,
                         'code_end_index': code_end_index,
                         'prompt_tokens_length': pt.get_prompt_tokens_length(),
                         'code_tokens_length': pt.get_code_tokens_length(),
                         'get_line_split_lengths_length': pt.get_line_split_lengths_length(),
                         'prompt': prompt})

            torch.save(prompt_tokens, f'{self.save_path_prompt}/{index}.pt')
            torch.save(code_tokens, f'{self.save_path_code}/{index}.pt')

        df = pd.DataFrame(rows)
        df.to_csv(f'{self.save_path}/{self.dataset_name}.csv')
        print("Finished!")

if __name__ == '__main__':
    pre_code_part = "Solidity smart contracts has many vulnerabilities. Some of those vulnerabilities are unprotected suicide, reentrancy, delegate calls, arithmetic overflow/underflow, etc."
    post_code_part = "Examine the above solidity smart contract code and identify line that cause these vulnerabilities"
    driver = Driver('data/tensors/', 'data/dataset', 'v1', 'solidity', LLMModels.CODEGEN_350M_MULTI, pre_code_part, post_code_part, True)







