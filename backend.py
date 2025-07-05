import os
import json
import shutil

from components.llm_utils import LLMModels
from components.localization_model import ConfigFactory
from components.detection_model import DetectionModelConfigFactory
from components.detection_model import DetectionModelLayerConfig

from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_localization_model import LocalizeVulnerabilities
from inference_detection_model import DetectVulnerabilities

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes


def remove_empty_lines(text: str) -> str:
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line.strip() != '']
    return '\n'.join(non_empty_lines)


@app.route('/')
def home():
    return "Welcome to the Contract Defend Vulnerability Localization and Detection Backend"


@app.route('/api/vuln_detection', methods=['POST'])
def vuln_detection():
    if request.is_json:
        data = request.get_json()
        code = remove_empty_lines(data['code'])
        if code.strip() != "":
            base_model_fold = 8  # replace the 0 with the fold you want to load
            base_model_config = ConfigFactory(exp_config='exp10',
                                              dataset_version='v2',
                                              dataset_name='solidity',
                                              llm_models_list=[LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B],
                                              layer_conf=2,
                                              target_dim_list=[1024],
                                              dim_reduce_type='gru',
                                              max_learning_rate_list=[1e-4],
                                              criterion="BCEWithLogitsLoss").get_generated_configs()[base_model_fold]

            model_selection_dict = {1: {'config': DetectionModelLayerConfig.NEW_PROJECTION, 'fold': 0,
                                        "prompt": "Smart contracts written in Solidity language may contain vulnerabilities such as DelegateCall, Arithmetic/Integer Overflow and Underflow, Nested Call, Reentrancy, Timestamp Dependency, TxOrigin, Transaction Order Dependency, Unchecked Call, Unprotected Suicide, Frozen Ether, Bad Randomness, Denial of service, Front Running, Short Address and other vulnerabilities. Analyze the following solidity smart contract for security vulnerabilities, bugs, and faulty logic. Identify whether the smart contract contains or does not contain security vulnerabilities, bugs, and faulty logic."},
                                    #    3: {'config': DetectionModelLayerConfig.NEW_ENCODER_AND_PROJECTION, 'fold': 7,
                                    # "prompt": "Analyze the following Solidity smart contract and classify it as Common Vulnerable (if it has any of: Reentrancy, Access Control, Integer Overflow/Underflow, Unchecked External Calls, Logic Errors, Timestamp Dependence, Denial of Service, or Delegatecall Misuse), Uncommon Vulnerable (if it has other types of vulnerabilities), or Non-Vulnerable (if none found), and briefly explain the detected vulnerabilities with one-line reasons."},
                                    # 15: {'config': DetectionModelLayerConfig.NEW_ENCODER_AND_PROJECTION, 'fold': 1,
                                    #      "prompt": "Analyze the following Solidity smart contract and classify it into one or more of the following vulnerability types based on its most critical security flaw: access_control, bad_randomness, delegatecall, denial_of_service, front_running, integer_overflow_underflow, non-vulnerable, numerical_consistency, reentrancy, short_addresses, timestamp_dependency, transaction_ordering_dependency, unchecked_call, unprotected self-destruct and other. Return only the appropriate vulnerability types from the list above."}
                                    }

            response_result = {}
            for item in list(model_selection_dict.keys()):
                detection_model_config = DetectionModelConfigFactory(base_model_config=base_model_config,
                                                                     transfer_learning_model_config=
                                                                     model_selection_dict[item][
                                                                         'config'],
                                                                     num_classes=item,
                                                                     exp_config='exp_detection',
                                                                     dataset_version='v1',
                                                                     dataset_name=f'solidity_detect_{item}',
                                                                     max_learning_rate=1e-3,
                                                                     criterion="BCEWithLogitsLoss").get_generated_configs()[
                    model_selection_dict[item]['fold']]
                detector = DetectVulnerabilities(
                    dataset_name="detection_infer",
                    dataset_version="v1",
                    checkpoint_config=detection_model_config,
                    pre_code_part=model_selection_dict[item]['prompt'],
                    post_code_part="",
                    mode="evaluation"
                )

                detector.create_new_dataset(code)
                detector.tokenize()
                detector.llm_inference()
                results = detector.detection()
                print("Item", item, results)
                if item == 1:
                    response_result['binary'] = results[0]['Probabilities']
                elif item == 3:
                    response_result['3_classes'] = results[0]['Probabilities']
                elif item == 15:
                    response_result['15_classes'] = results[0]['Probabilities']
                else:
                    raise ValueError("No such number of classes exist")
            return jsonify({
                "message": "Inference success",
                "data": response_result
            }), 200
        else:
            return jsonify({
                "message": "The code from the request is empty",
                "data": None
            }), 400
    return jsonify({"error": "Request must be JSON"}), 400


@app.route('/api/vuln_localization', methods=['POST'])
def vuln_localization():
    if request.is_json:
        data = request.get_json()
        code = remove_empty_lines(data['code'])
        if code.strip() != "":
            base_model_fold = 8  # replace the 0 with the fold you want to load
            base_model_config = ConfigFactory(exp_config='exp10',
                                              dataset_version='v2',
                                              dataset_name='solidity',
                                              llm_models_list=[LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B],
                                              layer_conf=2,
                                              target_dim_list=[1024],
                                              dim_reduce_type='gru',
                                              max_learning_rate_list=[1e-4],
                                              criterion="BCEWithLogitsLoss").get_generated_configs()[base_model_fold]

            localizer = LocalizeVulnerabilities(
                dataset_name="localize_infer",
                dataset_version="v1",
                checkpoint_config=base_model_config,
                pre_code_part="Smart contracts written in Solidity language may contain vulnerabilities such as DelegateCall, Arithmetic/Integer Overflow and Underflow, Nested Call, Reentrancy, Timestamp Dependency, TxOrigin, Transaction Order Dependency, Unchecked Call, Unprotected Suicide, Frozen Ether, Bad Randomness, Denial of service, Front Running, Short Address and other vulnerabilities. Analyze the following solidity smart contract for security vulnerabilities, bugs, and faulty logic. Identify all problematic lines and explain the risks associated with each.",
                post_code_part="",
                mode="inference"
            )

            new_df = localizer.create_new_dataset(code)
            print(new_df.head())
            localizer.tokenize()
            localizer.llm_inference()
            results = localizer.localization()

            return jsonify({
                "message": "Inference success",
                "data": {'code': json.dumps(code), 'probabilities': results[0]['Probabilities'],
                         'classification': results[0]['Classification']}
            }), 200
        else:
            return jsonify({
                "message": "The code from the request is empty",
                "data": None
            }), 400
    return jsonify({"error": "Request must be JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
