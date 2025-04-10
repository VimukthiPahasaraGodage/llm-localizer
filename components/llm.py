import os
from enum import Enum
import pandas as pd

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from components.llm_utils import LLMInfo, LLMModels


class LLM:
    def __init__(self, tensor_path: str, dataset_version: str, dataset_name: str, tokens_type: str, llm_model: Enum,
                 device: str):
        self.cwd = os.getcwd()
        self.tensor_path = tensor_path
        self.dataset_version = dataset_version
        self.dataset_name = dataset_name
        self.tokens_type = tokens_type

        self.llm_model = llm_model
        self.llm_info = LLMInfo(self.llm_model)

        self.device = device

        self.model = None

        self.df_path = f'{self.cwd}/{self.tensor_path}/{self.dataset_name}/{self.dataset_version}/{self.llm_model.value}/tokenizer/{self.dataset_name}.csv'
        self.df = pd.read_csv(self.df_path)

        self.token_tensors_path = f'{self.cwd}/{self.tensor_path}/{self.dataset_name}/{self.dataset_version}/{self.llm_model.value}/tokenizer/{self.tokens_type}'
        self.save_path = f'{self.cwd}/{self.tensor_path}/{self.dataset_name}/{self.dataset_version}/{self.llm_model.value}/last_hidden_state/{self.tokens_type}'
        os.makedirs(self.save_path, exist_ok=True)

    def __init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_info.get_model_name(), trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()

    def get_and_save_last_hidden_states(self):
        self.__init_model()  # initialize the LLM model and load to GPU

        files = [f for f in os.listdir(self.token_tensors_path) if
                 os.path.isfile(os.path.join(self.token_tensors_path, f))]
        tensor_files = []
        for file in files:
            if ".pt" in file:
                tensor_files.append(file)
            else:
                print(
                    f"The file '{file}' does not have '.pt' extension. Make sure this folder only contains valid tensor files!")

        for file in tqdm(tensor_files, desc="Inferencing"):
            token_ids = torch.load(f'{self.token_tensors_path}/{file}')
            index = int(file.split('.')[0])

            df_row = self.df[self.df['index'] == index]
            if df_row.shape[0] != 1:
                print(f"There are rows that match the same index in the dataframe! Index: {index}")
                continue

            token_length = 0
            if self.tokens_type == "prompt":
                token_length = df_row.iloc[0]['prompt_tokens_length']
            elif self.tokens_type == "code":
                token_length = df_row.iloc[0]['code_tokens_length']
            else:
                print(f"The token_type is {self.tokens_type} which is not valid!")

            if token_ids.dim() == 1 and token_ids.shape[0] <= self.llm_info.max_allowed_context_length and token_length > 0:
                try:
                    self.__get_last_hidden_states(token_ids, index, token_length)
                except Exception as e:
                    print(f"Exception occurred for index: {index}. Error: {e}")
            else:
                print(f"The file '{file}' contains a tensor not matching the standards or token_length is 0!")

    def __get_last_hidden_states(self, token_ids: torch.tensor, index: int, prompt_token_length: int):
        token_ids = token_ids.to(self.device)
        token_ids = token_ids.unsqueeze(0)

        attention_mask = torch.cat([
            torch.ones(prompt_token_length),
            torch.zeros(self.llm_info.max_allowed_context_length - prompt_token_length)
        ], dim=0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_ids=token_ids, attention_mask=attention_mask, output_hidden_states=True)

        all_hidden_states = outputs.hidden_states
        last_hidden_state = all_hidden_states[-1]

        if last_hidden_state.dim() == 3 and last_hidden_state.shape[0] == 1 and last_hidden_state.shape[
            2] == self.llm_info.get_hidden_size():
            pad_length = self.llm_info.max_allowed_context_length - last_hidden_state.shape[1]
            pad_tensor = torch.zeros((last_hidden_state.shape[0], pad_length, last_hidden_state.shape[2]),
                                     device=self.device)
            last_hidden_state = torch.cat([last_hidden_state, pad_tensor], dim=1)

            if last_hidden_state.shape[1] == self.llm_info.max_allowed_context_length:
                torch.save(last_hidden_state, f'{self.save_path}/{index}.pt')
            else:
                raise Exception(
                    f"Something went wrong for index: {index}. The dimension 1 of last_hidden_state not equal to max_allowed_context_length!")
        else:
            raise Exception(
                f"Something went wrong for index: {index}. Number of dimensions is not 3 or batch size is not 1 or the hidden size mismatch!")


if __name__ == '__main__':
    llm = LLM('data/tensors', 'v1', 'solidity', 'prompt', LLMModels.CODEGEN_350M_MULTI, 'cuda:0')
    llm.get_and_save_last_hidden_states()
