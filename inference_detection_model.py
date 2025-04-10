import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from components.detection_model import DetectionModelConfig, DetectionTransformer, DetectionLastHiddenStatesDataset, \
    DetectionModelConfigFactory, DetectionModelLayerConfig, BiGRUProjection
from components.llm import LLM
from components.llm_utils import LLMInfo, LLMModels
from components.localization_model import TrainValidationSplit, ConfigFactory, CodeGenBlock
from components.prompt import Driver as Tokenizer


class DetectVulnerabilities:
    def __init__(self,
                 dataset_name: str,
                 dataset_version: str,
                 checkpoint_config: DetectionModelConfig,
                 pre_code_part: str = None,
                 post_code_part: str = None,
                 mode: str = "inference"
                 ):
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.checkpoint_config = checkpoint_config
        self.pre_code_part = pre_code_part
        self.post_code_part = post_code_part
        self.mode = mode

        self.device = "cuda:0"

    def tokenize(self):
        Tokenizer(tensor_path=self.checkpoint_config.base_model_config.tensor_path,
                  dataset_path=self.checkpoint_config.base_model_config.dataset_path,
                  dataset_version=self.dataset_version,
                  dataset_name=self.dataset_name,
                  llm_model=self.checkpoint_config.base_model_config.llm_model,
                  pre_code_part=self.pre_code_part,
                  post_code_part=self.post_code_part,
                  standardize_df=False)

    def llm_inference(self):
        llm = LLM(tensor_path=self.checkpoint_config.base_model_config.tensor_path,
                  dataset_version=self.dataset_version,
                  dataset_name=self.dataset_name,
                  tokens_type=self.checkpoint_config.base_model_config.tokens_type,
                  llm_model=self.checkpoint_config.base_model_config.llm_model,
                  device=self.device)
        llm.get_and_save_last_hidden_states()

    def detection(self):
        model = DetectionTransformer(
            num_layers_projection=self.checkpoint_config.base_model_config.num_layers_projection,
            num_layers_encoder=self.checkpoint_config.base_model_config.num_layers_encoder,
            num_layers_dim_reduce=self.checkpoint_config.base_model_config.num_layers_dim_reduce,
            hidden_size=LLMInfo(self.checkpoint_config.base_model_config.llm_model).get_hidden_size(),
            num_head=self.checkpoint_config.base_model_config.num_head,
            target_dim=self.checkpoint_config.base_model_config.target_dim,
            dim_reduce_type=self.checkpoint_config.base_model_config.dim_reduce_type,
            seed=self.checkpoint_config.base_model_config.seed,
            device=self.device
        ).to(self.device)

        ####################################### Model Configuration ############################
        if self.checkpoint_config.transfer_learning_model_config == DetectionModelLayerConfig.NEW_PROJECTION:
            model.projection = BiGRUProjection(input_dim=model.target_dim, hidden_dim=(model.target_dim // 2),
                                               num_classes=self.checkpoint_config.num_classes).to(self.device)
        elif self.checkpoint_config.transfer_learning_model_config == DetectionModelLayerConfig.NEW_ENCODER_AND_PROJECTION:
            codegen_model = "Salesforce/codegen-350M-multi"
            config = AutoConfig.from_pretrained(codegen_model)
            config.n_head = model.num_head
            config.n_embd = model.target_dim

            new_encoder_layer = CodeGenBlock(config).to(self.device)

            model.encoder.enc_layers.append(new_encoder_layer)
            model.projection = BiGRUProjection(input_dim=model.target_dim, hidden_dim=(model.target_dim // 2),
                                               num_classes=self.checkpoint_config.num_classes).to(self.device)
        else:
            raise ValueError("No such configuration defined for transfer learning model")
        #########################################################################################

        cwd = os.getcwd()

        base_model_checkpoint_dir = f"{cwd}/{self.checkpoint_config.base_model_config.outputs_path}/{self.checkpoint_config.dataset_name}/{self.checkpoint_config.dataset_version}/{LLMModels.get_model_nickname(self.checkpoint_config.base_model_config.llm_model)}/{self.checkpoint_config.exp_config}/checkpoints/fold_{self.checkpoint_config.fold_index}"
        if not os.path.isdir(base_model_checkpoint_dir):
            raise FileNotFoundError("Checkpoint folder does not exist")
        base_checkpoint_files = [f for f in os.listdir(base_model_checkpoint_dir) if
                                 os.path.isfile(os.path.join(base_model_checkpoint_dir, f))]
        if len(base_checkpoint_files) > 0:
            latest_epoch = 0
            for file in base_checkpoint_files:
                epoch_of_file = int(file.split('.')[0])
                if epoch_of_file > latest_epoch:
                    latest_epoch = epoch_of_file
            checkpoint = torch.load(f'{base_model_checkpoint_dir}/{latest_epoch}.pt', map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise FileNotFoundError("No checkpoints found for the given model configuration")

        # Create the TrainValidationSplit object
        tvs = TrainValidationSplit(self.checkpoint_config.tensor_path,
                                   self.dataset_version,
                                   self.dataset_name,
                                   self.checkpoint_config.base_model_config.tokens_type,
                                   self.checkpoint_config.llm_model,
                                   1,
                                   self.checkpoint_config.base_model_config.seed)

        # Retrieve the train/validation indices for your chosen fold
        train_validation_indices = tvs.get_train_validation_indices_for_fold(0)

        # Create the training & validation Datasets
        inference_dataset = DetectionLastHiddenStatesDataset(
            self.checkpoint_config.base_model_config.tensor_path,
            self.checkpoint_config.base_model_config.dataset_path,
            self.dataset_version,
            self.dataset_name,
            self.checkpoint_config.base_model_config.tokens_type,
            self.checkpoint_config.base_model_config.llm_model,
            train_validation_indices,
            dataset_type="train"
        )

        inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

        model.eval()
        with torch.no_grad():
            for step, (last_hidden_state, code_tokens_length, line_split_lengths, class_labels) in enumerate(
                    inference_loader):

                last_hidden_state = last_hidden_state.to(self.device)
                code_tokens_length = code_tokens_length.to(self.device)
                line_split_lengths = line_split_lengths.to(self.device)
                class_labels = class_labels.to(self.device)

                outputs = model(last_hidden_state, code_tokens_length, line_split_lengths)
                if outputs.dim() > 2 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)

                valid_outputs = outputs

                preds_prob = torch.sigmoid(valid_outputs)
                preds = (preds_prob >= 0.5).long()

                if self.mode == "inference":
                    print(f"Output: {valid_outputs}, Classification: {preds}")
                elif self.mode == "evaluation":
                    valid_labels = class_labels.float()
                    print(f"Output: {valid_outputs}, Classification: {preds}, Actual lables: {valid_labels}")
                else:
                    raise Exception("There is no such mode available!")


if __name__ == '__main__':
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
                                'prompt': "Smart contracts written in Solidity language may contain vulnerabilities such as DelegateCall, Arithmetic/Integer Overflow and Underflow, Nested Call, Reentrancy, Timestamp Dependency, TxOrigin, Transaction Order Dependency, Unchecked Call, Unprotected Suicide, Frozen Ether, Bad Randomness, Denial of service, Front Running, Short Address and other vulnerabilities. Analyze the following solidity smart contract for security vulnerabilities, bugs, and faulty logic. Identify whether the smart contract contains or does not contain security vulnerabilities, bugs, and faulty logic"},
                            3: {'config': DetectionModelLayerConfig.NEW_PROJECTION, 'fold': 0,
                                "prompt": "Analyze the following Solidity smart contract and classify it as Common Vulnerable (if it has any of: Reentrancy, Access Control, Integer Overflow/Underflow, Unchecked External Calls, Logic Errors, Timestamp Dependence, Denial of Service, or Delegatecall Misuse), Uncommon Vulnerable (if it has other types of vulnerabilities), or Non-Vulnerable (if none found), and briefly explain the detected vulnerabilities with one-line reasons."},
                            15: {'config': DetectionModelLayerConfig.NEW_PROJECTION, 'fold': 0,
                                 "prompt": "Analyze the following Solidity smart contract and classify it into one or more of the following vulnerability types based on its most critical security flaw: access_control, bad_randomness, delegatecall, denial_of_service, front_running, integer_overflow_underflow, non-vulnerable, numerical_consistency, reentrancy, short_addresses, timestamp_dependency, transaction_ordering_dependency, unchecked_call, unprotected self-destruct and other. Return only the appropriate vulnerability types from the list above."}}

    for item in list(model_selection_dict.keys()):
        detection_model_config = DetectionModelConfigFactory(base_model_config=base_model_config,
                                                             transfer_learning_model_config=model_selection_dict[item][
                                                                 'config'],
                                                             num_classes=1,
                                                             exp_config='exp_detection',
                                                             dataset_version='v1',
                                                             dataset_name='solidity_detect_1',
                                                             max_learning_rate=1e-3,
                                                             criterion="BCEWithLogitsLoss").get_generated_configs()[
            model_selection_dict[item]['fold']]

        detector = DetectVulnerabilities(
            dataset_name="",
            dataset_version="",
            checkpoint_config=detection_model_config,
            pre_code_part="",
            post_code_part="",
            mode="inference"
        )

        detector.tokenize()
        detector.llm_inference()
        detector.detection()
