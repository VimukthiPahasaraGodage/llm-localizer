from enum import Enum

import torch
from transformers import AutoTokenizer

from components.llm_utils import LLMInfo, LLMModels


class PromptTemplate:
    codegen_part_1 = "// {Pre_Instructions}\n"
    codegen_part_2 = "// {Post_Instructions}"

    deepseek_coder_v1_part_1 = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n### Instruction:\n{Pre_Instructions}\n```\n"
    deepseek_coder_v1_part_2 = "```\n{Post_Instructions}\n### Response:\n"

    deepseek_coder_v2_part_1 = "User: {Pre_Instructions}\n```\n"
    deepseek_coder_v2_part_2 = "```\n{Post_Instructions}\n\nAssistant:"

    qwen_coder_part_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{Pre_Instructions}\n```\n"
    qwen_coder_part_2 = "```\n{Post_Instructions}<|im_end|>\n<|im_start|>assistant\n"

    qwen_qwq_part_1 = "<|im_start|>user\n{Pre_Instructions}\n```\n"
    qwen_qwq_part_2 = "```\n{Post_Instructions}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    deepseek_r1_part_1 = "<｜User｜>{Pre_Instructions}\n```\n"
    deepseek_r1_part_2 = "```\n{Post_Instructions}<｜Assistant｜><think>\n"

    @staticmethod
    def get_prompt_template_part_1(llm_model: Enum):
        match llm_model:
            case LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_67B:
                return PromptTemplate.deepseek_coder_v1_part_1
            case LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_33B:
                return PromptTemplate.deepseek_coder_v1_part_1
            case LLMModels.DEEPSEEK_CODER_V2_LITE_INSTRUCT_16B:
                return PromptTemplate.deepseek_coder_v2_part_1
            case LLMModels.QWEN_25_CODER_INSTRUCT_GPTQ_INT8_14B:
                return PromptTemplate.qwen_coder_part_1
            case LLMModels.QWEN_25_CODER_INSTRUCT_GPTQ_INT8_32B:
                return PromptTemplate.qwen_coder_part_1
            case LLMModels.QWEN_QWQ_32B:
                return PromptTemplate.qwen_qwq_part_1
            case LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B:
                return PromptTemplate.deepseek_r1_part_1
            case LLMModels.DEEPSEEK_R1_DISTILL_QWEN_32B:
                return PromptTemplate.deepseek_r1_part_1
            case LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B:
                return PromptTemplate.deepseek_r1_part_1
            case LLMModels.CODEGEN_350M_MULTI:
                return PromptTemplate.codegen_part_1
            case LLMModels.CODEGEN_6B_MULTI:
                return PromptTemplate.codegen_part_1
            case LLMModels.CODEGEN_16B_MULTI:
                return PromptTemplate.codegen_part_1
            case LLMModels.CODEGEN2_1B:
                return PromptTemplate.codegen_part_1
            case LLMModels.CODEGEN2_7B:
                return PromptTemplate.codegen_part_1
            case LLMModels.CODEGEN2_16B:
                return PromptTemplate.codegen_part_1
            case _:
                return None  # No template available

    @staticmethod
    def get_prompt_template_part_2(llm_model: Enum):
        match llm_model:
            case LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_67B:
                return PromptTemplate.deepseek_coder_v1_part_2
            case LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_33B:
                return PromptTemplate.deepseek_coder_v1_part_2
            case LLMModels.DEEPSEEK_CODER_V2_LITE_INSTRUCT_16B:
                return PromptTemplate.deepseek_coder_v2_part_2
            case LLMModels.QWEN_25_CODER_INSTRUCT_GPTQ_INT8_14B:
                return PromptTemplate.qwen_coder_part_2
            case LLMModels.QWEN_25_CODER_INSTRUCT_GPTQ_INT8_32B:
                return PromptTemplate.qwen_coder_part_2
            case LLMModels.QWEN_QWQ_32B:
                return PromptTemplate.qwen_qwq_part_2
            case LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B:
                return PromptTemplate.deepseek_r1_part_2
            case LLMModels.DEEPSEEK_R1_DISTILL_QWEN_32B:
                return PromptTemplate.deepseek_r1_part_2
            case LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B:
                return PromptTemplate.deepseek_r1_part_2
            case LLMModels.CODEGEN_350M_MULTI:
                return PromptTemplate.codegen_part_2
            case LLMModels.CODEGEN_6B_MULTI:
                return PromptTemplate.codegen_part_2
            case LLMModels.CODEGEN_16B_MULTI:
                return PromptTemplate.codegen_part_2
            case LLMModels.CODEGEN2_1B:
                return PromptTemplate.codegen_part_2
            case LLMModels.CODEGEN2_7B:
                return PromptTemplate.codegen_part_2
            case LLMModels.CODEGEN2_16B:
                return PromptTemplate.codegen_part_2
            case _:
                return None  # No template available


class Prompt:
    def __init__(self, pre_code_part: str, code: str, post_code_part: str, llm_model: Enum):
        self.pre_code_part = pre_code_part
        self.code = code
        self.post_code_part = post_code_part
        self.llm_model = llm_model
        self.llm_info = LLMInfo(self.llm_model)

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_info.get_model_name(), trust_remote_code=True)

        self.prompt_processed = False

        self.prompt = None
        self.prompt_tokens = None
        self.code_tokens = None
        self.line_split_lengths = None
        self.code_start_index = None
        self.code_end_index = None

        self.__process_prompt()

    def __raise_first_token_mismatch_error_for_deep_seek(self, first_token, correct_first_token):
        if first_token != correct_first_token:
            raise Exception(f"The first token mismatch in tokenizing process. model: {self.llm_info.get_model_name()}")

    def __process_prompt(self):
        self.pre_code_part = self.pre_code_part.strip()  # clean leading and trailing ends
        self.post_code_part = self.post_code_part.strip()
        # The code must be a string such as "(line 1)\n(line2)\n(line 3)\n ... \n(last line)"
        # We are assuming the code string does not have \n\n like combinations which will induce empty lines
        split_code = self.code.split("\n")
        cleaned_split_code = []
        for line in split_code:
            if line == "":
                raise Exception("The source code have empty lines! Clean them before generating the prompts.")
            cleaned_split_code.append(
                line.strip())  # clean the lines so that leading and trailing whitespaces and newlines are removed
        tokenized_lines = []
        cleaned_code = ""
        for line in cleaned_split_code:
            line = line + "\n"
            cleaned_code += line
            tokenizer_output = self.tokenizer([line], return_tensors="pt").input_ids
            if tokenizer_output.dim() != 2 or tokenizer_output.shape[0] != 1:
                raise Exception(
                    "Number of dimensions of the output of the tokenizer is not equal to 2 or shape[0] is not equal to 1!")
            tokenizer_output = tokenizer_output.squeeze()

            if self.llm_model == LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_67B or self.llm_model == LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_33B:
                self.__raise_first_token_mismatch_error_for_deep_seek(tokenizer_output[0].item(), 32013)
                tokenizer_output = tokenizer_output[1:]
            elif self.llm_model == LLMModels.DEEPSEEK_CODER_V2_LITE_INSTRUCT_16B:
                self.__raise_first_token_mismatch_error_for_deep_seek(tokenizer_output[0].item(), 100000)
                tokenizer_output = tokenizer_output[1:]
            elif self.llm_model == LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B:
                self.__raise_first_token_mismatch_error_for_deep_seek(tokenizer_output[0].item(), 128000)
                tokenizer_output = tokenizer_output[1:]
            elif self.llm_model == LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B or self.llm_model == LLMModels.DEEPSEEK_R1_DISTILL_QWEN_32B:
                self.__raise_first_token_mismatch_error_for_deep_seek(tokenizer_output[0].item(), 151646)
                tokenizer_output = tokenizer_output[1:]

            tokenized_lines.append(tokenizer_output)  # Append tokenized output of each line

        line_split_lengths = []
        total_tokens_in_code = 0
        for line in tokenized_lines:
            if line.dim() != 1:
                raise Exception("Number of dimensions the tokenized line tensor is not 1.")
            line_split_lengths.append(line.shape[0])
            total_tokens_in_code += line.shape[0]

        tokenized_code = torch.cat(tokenized_lines, dim=0)
        if tokenized_code.shape[0] != total_tokens_in_code:
            raise Exception("Total number of tokens mismatch after concatenation.")

        prompt_part_1 = PromptTemplate.get_prompt_template_part_1(self.llm_model)
        prompt_part_2 = PromptTemplate.get_prompt_template_part_2(self.llm_model)
        if prompt_part_1 is None or prompt_part_2 is None:
            raise Exception("No defined prompt template for the selected LLM model!")
        prompt_part_1 = prompt_part_1.format(Pre_Instructions=self.pre_code_part)
        prompt_part_2 = prompt_part_2.format(Post_Instructions=self.post_code_part)

        tokenized_prompt_part_1 = self.tokenizer([prompt_part_1], return_tensors="pt").input_ids
        tokenized_prompt_part_2 = self.tokenizer([prompt_part_2], return_tensors="pt").input_ids
        if (tokenized_prompt_part_1.dim() != 2 or tokenized_prompt_part_1.shape[0] != 1) or (
                tokenized_prompt_part_2.dim() != 2 or tokenized_prompt_part_2.shape[0] != 1):
            raise Exception(
                "Number of dimensions of the output of the tokenizer is not equal to 2 or shape[0] is not equal to 1!")
        tokenized_prompt_part_1 = tokenized_prompt_part_1.squeeze()
        tokenized_prompt_part_2 = tokenized_prompt_part_2.squeeze()

        if self.llm_model == LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_67B or self.llm_model == LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_33B:
            self.__raise_first_token_mismatch_error_for_deep_seek(tokenized_prompt_part_1[0].item(), 32013)
            self.__raise_first_token_mismatch_error_for_deep_seek(tokenized_prompt_part_2[0].item(), 32013)
            tokenized_prompt_part_2 = tokenized_prompt_part_2[1:]
        elif self.llm_model == LLMModels.DEEPSEEK_CODER_V2_LITE_INSTRUCT_16B:
            self.__raise_first_token_mismatch_error_for_deep_seek(tokenized_prompt_part_1[0].item(), 100000)
            self.__raise_first_token_mismatch_error_for_deep_seek(tokenized_prompt_part_2[0].item(), 100000)
            tokenized_prompt_part_2 = tokenized_prompt_part_2[1:]
        elif self.llm_model == LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B:
            self.__raise_first_token_mismatch_error_for_deep_seek(tokenized_prompt_part_1[0].item(), 128000)
            self.__raise_first_token_mismatch_error_for_deep_seek(tokenized_prompt_part_2[0].item(), 128000)
            tokenized_prompt_part_2 = tokenized_prompt_part_2[1:]
        elif self.llm_model == LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B or self.llm_model == LLMModels.DEEPSEEK_R1_DISTILL_QWEN_32B:
            self.__raise_first_token_mismatch_error_for_deep_seek(tokenized_prompt_part_1[0].item(), 151646)
            self.__raise_first_token_mismatch_error_for_deep_seek(tokenized_prompt_part_2[0].item(), 151646)
            tokenized_prompt_part_2 = tokenized_prompt_part_2[1:]

        tokenized_prompt_part_1_length = tokenized_prompt_part_1.shape[0]
        tokenized_prompt_part_2_length = tokenized_prompt_part_2.shape[0]

        tokenized_prompt_token_list = [tokenized_prompt_part_1, tokenized_code, tokenized_prompt_part_2]
        tokenized_prompt = torch.cat(tokenized_prompt_token_list, dim=0)

        if tokenized_prompt.dim() != 1 or tokenized_prompt.shape[0] != (
                tokenized_prompt_part_1_length + tokenized_prompt_part_2_length + total_tokens_in_code):
            raise Exception("Number of dimensions in tokenized prompt is not 1 or the total number of tokens mismatch!")

        self.prompt = prompt_part_1 + cleaned_code + prompt_part_2
        self.prompt_tokens = tokenized_prompt
        self.code_tokens = tokenized_code
        self.line_split_lengths = line_split_lengths
        self.code_start_index = tokenized_prompt_part_1.shape[0]
        self.code_end_index = tokenized_prompt_part_1.shape[0] + tokenized_code.shape[0] - 1
        self.prompt_processed = True

    def get_prompt(self):
        return self.prompt

    def get_prompt_tokens(self):
        return self.prompt_tokens

    def get_prompt_tokens_length(self):
        return len(self.prompt_tokens)

    def get_code_tokens(self):
        return self.code_tokens

    def get_code_tokens_length(self):
        return len(self.code_tokens)

    def get_line_split_lengths(self):
        return self.line_split_lengths

    def get_line_split_lengths_length(self):
        return len(self.line_split_lengths)

    def get_code_start_index(self):
        return self.code_start_index

    def get_code_end_index(self):
        return self.code_end_index
