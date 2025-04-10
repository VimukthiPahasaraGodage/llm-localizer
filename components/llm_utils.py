from enum import Enum
from transformers import AutoConfig

# Even though we list many LLMs here, CodeGen models and the instruct models will be used mainly(base models are for fine-tuning is needed)
class LLMModels(Enum):
    # CodeGen LLM models
    CODEGEN_350M_MULTI = 1
    CODEGEN_6B_MULTI = 2
    CODEGEN_16B_MULTI = 3

    # CodeGen2 LLM models
    CODEGEN2_1B = 4
    CODEGEN2_7B = 5
    CODEGEN2_16B = 6

    # DeepSeek Coder V1 LLM models
    DEEPSEEK_CODER_V1_BASE_67B = 7 # 6.7B
    DEEPSEEK_CODER_V1_INSTRUCT_67B = 8 # 6.7B
    DEEPSEEK_CODER_V1_BASE_33B = 9
    DEEPSEEK_CODER_V1_INSTRUCT_33B = 10

    # DeepSeek Coder V2 LLM models
    DEEPSEEK_CODER_V2_LITE_BASE_16B = 11
    DEEPSEEK_CODER_V2_LITE_INSTRUCT_16B = 12

    # Qwen 2.5 Coder LLM models
    QWEN_25_CODER_BASE_14B = 13
    QWEN_25_CODER_BASE_32B = 14
    QWEN_25_CODER_INSTRUCT_GPTQ_INT8_14B = 15
    QWEN_25_CODER_INSTRUCT_GPTQ_INT8_32B = 16

    # Qwen QwQ LLM model(reasoning model)
    QWEN_QWQ_32B = 17

    # DeepSeek R1 LLM models
    DEEPSEEK_R1_DISTILL_LLAMA_8B = 18
    DEEPSEEK_R1_DISTILL_QWEN_14B = 19
    DEEPSEEK_R1_DISTILL_QWEN_32B = 20

class LLMInfo:
    def __init__(self, llm_model: Enum):
        self.llm_model = llm_model

        self.model_name = None
        self.hidden_size = None
        self.context_length = None

        self.define_llm_info()
        self.define_hidden_size_and_context_length()

    def define_hidden_size_and_context_length(self):
        config = AutoConfig.from_pretrained(self.model_name)
        self.hidden_size = config.hidden_size
        self.context_length = getattr(config, "max_position_embeddings", None)

    def define_llm_info(self):
        match self.llm_model:
            case LLMModels.DEEPSEEK_CODER_V1_BASE_67B:
                self.model_name = "deepseek-ai/deepseek-coder-6.7b-base"
            case LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_67B:
                self.model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
            case LLMModels.DEEPSEEK_CODER_V1_BASE_33B:
                self.model_name = "deepseek-ai/deepseek-coder-33b-base"
            case LLMModels.DEEPSEEK_CODER_V1_INSTRUCT_33B:
                self.model_name = "deepseek-ai/deepseek-coder-33b-instruct"
            case LLMModels.DEEPSEEK_CODER_V2_LITE_BASE_16B:
                self.model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
            case LLMModels.DEEPSEEK_CODER_V2_LITE_INSTRUCT_16B:
                self.model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
            case LLMModels.QWEN_25_CODER_BASE_14B:
                self.model_name = "Qwen/Qwen2.5-Coder-14B"
            case LLMModels.QWEN_25_CODER_BASE_32B:
                self.model_name = "Qwen/Qwen2.5-Coder-32B"
            case LLMModels.QWEN_25_CODER_INSTRUCT_GPTQ_INT8_14B:
                self.model_name = "Qwen/Qwen2.5-Coder-14B-Instruct-GPTQ-Int8"
            case LLMModels.QWEN_25_CODER_INSTRUCT_GPTQ_INT8_32B:
                self.model_name = "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8"
            case LLMModels.QWEN_QWQ_32B:
                self.model_name = "Qwen/QwQ-32B"
            case LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B:
                self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
            case LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B:
                self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
            case LLMModels.DEEPSEEK_R1_DISTILL_QWEN_32B:
                self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
            case LLMModels.CODEGEN_350M_MULTI:
                self.model_name = "Salesforce/codegen-350M-multi"
            case LLMModels.CODEGEN_6B_MULTI:
                self.model_name = "Salesforce/codegen-6B-multi"
            case LLMModels.CODEGEN_16B_MULTI:
                self.model_name = "Salesforce/codegen-16B-multi"
            case LLMModels.CODEGEN2_1B:
                self.model_name = "Salesforce/codegen2-1B"
            case LLMModels.CODEGEN2_7B:
                self.model_name = "Salesforce/codegen2-7B"
            case LLMModels.CODEGEN2_16B:
                self.model_name = "Salesforce/codegen2-16B"
            case _:
                raise Exception("No such LLM model is defined!")

    def get_model_name(self):
        return self.model_name

    def get_hidden_size(self):
        return self.hidden_size

    def get_context_length(self):
        return self.context_length






