from components.llm import LLM
from components.llm_utils import LLMModels


def run_llm_inference_job(params: dict):
    try:
        device = "cuda:0"
        llm = LLM(tensor_path='data/tensors',
                  dataset_version=params['dataset_version'],
                  dataset_name=params['dataset_name'],
                  tokens_type='prompt',
                  llm_model=params['llm_model'],
                  device=device)
        llm.get_and_save_last_hidden_states()
        return f"Finished the execution of the LLM inference for dataset: {params['dataset_name']} version: {params['dataset_version']} llm: {params['llm_model'].value}"
    except Exception as e:
        return f"Exception occurred during the LLM inference for dataset: {params['dataset_name']} version: {params['dataset_version']} llm: {params['llm_model'].value} Error: {e}"


job_params = [
    # defects4j dataset
    {'dataset_name': 'defects4j', 'dataset_version': 'v2', 'llm_model': LLMModels.CODEGEN_350M_MULTI},
    {'dataset_name': 'defects4j', 'dataset_version': 'v2', 'llm_model': LLMModels.CODEGEN_6B_MULTI},
    {'dataset_name': 'defects4j', 'dataset_version': 'v2', 'llm_model': LLMModels.CODEGEN_16B_MULTI},
    {'dataset_name': 'defects4j', 'dataset_version': 'v2', 'llm_model': LLMModels.QWEN_QWQ_32B},
    {'dataset_name': 'defects4j', 'dataset_version': 'v2', 'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B},
    # solidity dataset
    {'dataset_name': 'solidity', 'dataset_version': 'v2', 'llm_model': LLMModels.QWEN_QWQ_32B},
    {'dataset_name': 'solidity', 'dataset_version': 'v2', 'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B},
    {'dataset_name': 'solidity', 'dataset_version': 'v2', 'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B},
    {'dataset_name': 'solidity', 'dataset_version': 'v2', 'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_QWEN_32B},
    {'dataset_name': 'solidity', 'dataset_version': 'v1', 'llm_model': LLMModels.CODEGEN_16B_MULTI},
    # solidity detection dataset
    {'dataset_name': 'solidity_detect_1', 'dataset_version': 'v1', 'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B},
    {'dataset_name': 'solidity_detect_2', 'dataset_version': 'v1', 'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B},
]

for params in job_params:
    print(run_llm_inference_job(params))
print("Finished all jobs!")
