from components.llm_utils import LLMModels
from components.prompt import Driver


def run_tokenizing_job(params: dict):
    try:
        Driver(tensor_path='data/tensors',
               dataset_path='data/dataset',
               dataset_version=params['dataset_version'],
               dataset_name=params['dataset_name'],
               llm_model=params['llm_model'],
               pre_code_part=params['pre_code_part'],
               post_code_part=params['post_code_part'],
               standardize_df=False)
        return f"Finished the execution of the tokenization for dataset: {params['dataset_name']} version: {params['dataset_version']} llm: {params['llm_model'].value}"
    except Exception as e:
        return f"Exception occurred during the tokenization for dataset: {params['dataset_name']} version: {params['dataset_version']} llm: {params['llm_model'].value} Error: {e}"


job_params = [
    # defects4j dataset
    {'dataset_version': 'v2',
     'dataset_name': 'defects4j',
     'llm_model': LLMModels.CODEGEN_350M_MULTI,
     'pre_code_part': "",
     'post_code_part': ""},
    {'dataset_version': 'v2',
     'dataset_name': 'defects4j',
     'llm_model': LLMModels.CODEGEN_6B_MULTI,
     'pre_code_part': "",
     'post_code_part': ""},
    {'dataset_version': 'v2',
     'dataset_name': 'defects4j',
     'llm_model': LLMModels.CODEGEN_16B_MULTI,
     'pre_code_part': "",
     'post_code_part': ""},
    {'dataset_version': 'v2',
     'dataset_name': 'defects4j',
     'llm_model': LLMModels.QWEN_QWQ_32B,
     'pre_code_part': "Analyze the following Java code snippet for security vulnerabilities, bugs, and faulty logic. Identify all problematic lines and explain the risks associated with each.",
     'post_code_part': ""},
    {'dataset_version': 'v2',
     'dataset_name': 'defects4j',
     'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B,
     'pre_code_part': "Analyze the following Java code snippet for security vulnerabilities, bugs, and faulty logic. Identify all problematic lines and explain the risks associated with each.",
     'post_code_part': ""},
    # solidity dataset
    {'dataset_version': 'v2',
     'dataset_name': 'solidity',
     'llm_model': LLMModels.QWEN_QWQ_32B,
     'pre_code_part': "Smart contracts written in Solidity language may contain vulnerabilities such as DelegateCall, Arithmetic/Integer Overflow and Underflow, Nested Call, Reentrancy, Timestamp Dependency, TxOrigin, Transaction Order Dependency, Unchecked Call, Unprotected Suicide, Frozen Ether, Bad Randomness, Denial of service, Front Running, Short Address and other vulnerabilities. Analyze the following solidity smart contract for security vulnerabilities, bugs, and faulty logic. Identify all problematic lines and explain the risks associated with each.",
     'post_code_part': ""},
    {'dataset_version': 'v2',
     'dataset_name': 'solidity',
     'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B,
     'pre_code_part': "Smart contracts written in Solidity language may contain vulnerabilities such as DelegateCall, Arithmetic/Integer Overflow and Underflow, Nested Call, Reentrancy, Timestamp Dependency, TxOrigin, Transaction Order Dependency, Unchecked Call, Unprotected Suicide, Frozen Ether, Bad Randomness, Denial of service, Front Running, Short Address and other vulnerabilities. Analyze the following solidity smart contract for security vulnerabilities, bugs, and faulty logic. Identify all problematic lines and explain the risks associated with each.",
     'post_code_part': ""},
    {'dataset_version': 'v2',
     'dataset_name': 'solidity',
     'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B,
     'pre_code_part': "Smart contracts written in Solidity language may contain vulnerabilities such as DelegateCall, Arithmetic/Integer Overflow and Underflow, Nested Call, Reentrancy, Timestamp Dependency, TxOrigin, Transaction Order Dependency, Unchecked Call, Unprotected Suicide, Frozen Ether, Bad Randomness, Denial of service, Front Running, Short Address and other vulnerabilities. Analyze the following solidity smart contract for security vulnerabilities, bugs, and faulty logic. Identify all problematic lines and explain the risks associated with each.",
     'post_code_part': ""},
    {'dataset_version': 'v2',
     'dataset_name': 'solidity',
     'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_QWEN_32B,
     'pre_code_part': "Smart contracts written in Solidity language may contain vulnerabilities such as DelegateCall, Arithmetic/Integer Overflow and Underflow, Nested Call, Reentrancy, Timestamp Dependency, TxOrigin, Transaction Order Dependency, Unchecked Call, Unprotected Suicide, Frozen Ether, Bad Randomness, Denial of service, Front Running, Short Address and other vulnerabilities. Analyze the following solidity smart contract for security vulnerabilities, bugs, and faulty logic. Identify all problematic lines and explain the risks associated with each.",
     'post_code_part': ""},
    {'dataset_version': 'v1',
     'dataset_name': 'solidity',
     'llm_model': LLMModels.CODEGEN_16B_MULTI,
     'pre_code_part': "",
     'post_code_part': ""},
    # solidity detection dataset
    {'dataset_version': 'v1',
     'dataset_name': 'solidity_detect_1',
     'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B,
     'pre_code_part': "Smart contracts written in Solidity language may contain vulnerabilities such as DelegateCall, Arithmetic/Integer Overflow and Underflow, Nested Call, Reentrancy, Timestamp Dependency, TxOrigin, Transaction Order Dependency, Unchecked Call, Unprotected Suicide, Frozen Ether, Bad Randomness, Denial of service, Front Running, Short Address and other vulnerabilities. Analyze the following solidity smart contract for security vulnerabilities, bugs, and faulty logic. Identify whether the smart contract contains or does not contain security vulnerabilities, bugs, and faulty logic",
     'post_code_part': ""},
    {'dataset_version': 'v1',
     'dataset_name': 'solidity_detect_3',
     'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B,
     'pre_code_part': "Analyze the following Solidity smart contract and classify it as Common Vulnerable (if it has any of: Reentrancy, Access Control, Integer Overflow/Underflow, Unchecked External Calls, Logic Errors, Timestamp Dependence, Denial of Service, or Delegatecall Misuse), Uncommon Vulnerable (if it has other types of vulnerabilities), or Non-Vulnerable (if none found), and briefly explain the detected vulnerabilities with one-line reasons.",
     'post_code_part': ""},
    {'dataset_version': 'v1',
     'dataset_name': 'solidity_detect_15',
     'llm_model': LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B,
     'pre_code_part': "Analyze the following Solidity smart contract and classify it into one or more of the following vulnerability types based on its most critical security flaw: access_control, bad_randomness, delegatecall, denial_of_service, front_running, integer_overflow_underflow, non-vulnerable, numerical_consistency, reentrancy, short_addresses, timestamp_dependency, transaction_ordering_dependency, unchecked_call, unprotected self-destruct and other. Return only the appropriate vulnerability types from the list above.",
     'post_code_part': ""}
]

for params in job_params:
    print(run_tokenizing_job(params))
print("Finished all jobs!")
