from enum import Enum
from transformers import AutoModelForCausalLM
import torch

from components.llm_utils import LLMInfo
from prompt import Prompt


class LLM:
    def __init__(self, llm_model: Enum, device: str):
        self.llm_model = llm_model
        self.llm_info = LLMInfo(self.llm_model)

        self.device = device

        self.model = None

    def __init_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.llm_info.get_model_name(), trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()

    def get_last_hidden_states(self, token_ids: torch.tensor):
        token_ids = token_ids.to(self.device)
        token_ids = token_ids.unsqueeze()

        with torch.no_grad():
            outputs = self.model(**token_ids, output_hidden_states=True)

        all_hidden_states = outputs.hidden_states
        last_hidden_state = all_hidden_states[-1]

        if last_hidden_state.dim() == 3 and last_hidden_state.shape[0] == 1 and last_hidden_state.shape[2] == self.llm_info.get_hidden_size():
            pass
        else:


        # Check if the batch size is one
        if last_hidden_state.shape[0] > 1:
            print(f"Batch size should be one. But the last hidden state has a batch size of {last_hidden_state.shape[0]}")
            return

        last_hidden_state = last_hidden_state[:, prompt.get_code_start_index(): (prompt.get_code_end_index() + 1), :]
        nl_indices = torch.where(token_ids == self.llm_info.get_new_line_token())


