import os
import json

from components.llm_utils import LLMModels
from components.localization_model import ConfigFactory

from flask import Flask, request, jsonify
from flask_cors import CORS
from inference_localization_model import LocalizeVulnerabilities

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
            pass
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
                mode="evaluation"
            )

            new_df = localizer.create_new_dataset(code)
            print(new_df.head())
            localizer.tokenize()
            localizer.llm_inference()
            results = localizer.localization()

            return jsonify({
                "message": "Inference success",
                "data": {'code': json.dumps(code), 'probabilities': results[0]['Probabilities'], 'classification': results[0]['Classification']}
            }), 200
        else:
            return jsonify({
                "message": "The code from the request is empty",
                "data": None
            }), 400
    return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
