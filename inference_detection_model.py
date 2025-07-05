import os
import tensorboard
import torch
from torch.utils.data import DataLoader
import pandas
import shutil
import pandas as pd

from components.detection_model import DetectionModelConfig, DetectionTransformer, DetectionLastHiddenStatesDataset, \
    DetectionModelConfigFactory, DetectionModelLayerConfig, BiGRUProjection
from components.llm import LLM
from components.llm_utils import LLMInfo, LLMModels
from components.prompt import Driver as Tokenizer
from transformers import AutoConfig
from components.localization_model import TrainValidationSplit, ConfigFactory, CodeGenBlock
from components.prompt import Driver as Tokenizer

import warnings

warnings.filterwarnings("ignore")  # Disable all warnings


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

    def create_new_dataset(self, code):
        def delete_folder(folder_path):
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                shutil.rmtree(folder_path)
                print(f"Deleted folder: {folder_path}")
            else:
                raise FileNotFoundError(f"Folder not found: {folder_path}")

        delete_folder(f"{os.getcwd()}/{self.checkpoint_config.tensor_path}/{self.dataset_name}/{self.dataset_version}")
        df_path = f"{os.getcwd()}/{self.checkpoint_config.dataset_path}/{self.dataset_name}/{self.dataset_version}/{self.dataset_name}.csv"

        data = [{'item_index': 0, 'source_code': code, 'vuln_types': "[]"}]
        df = pd.DataFrame(data)
        df.to_csv(df_path)
        return df

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

        cwd = os.getcwd()

        base_model_checkpoint_dir = f"{cwd}/{self.checkpoint_config.base_model_config.outputs_path}/{self.checkpoint_config.dataset_name}/{self.checkpoint_config.dataset_version}/{LLMModels.get_model_nickname(self.checkpoint_config.base_model_config.llm_model)}/{self.checkpoint_config.exp_config}/checkpoints/fold_{self.checkpoint_config.fold_index}"
        if not os.path.isdir(base_model_checkpoint_dir):
            print(base_model_checkpoint_dir)
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
            dataset_type="validation"
        )

        inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)
        results = []
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
                    results.append({'Output': valid_outputs.tolist(), 'Probabilities': preds_prob.tolist(),
                                    'Classification': preds.tolist()})
                elif self.mode == "evaluation":
                    valid_labels = class_labels.float()
                    results.append({'Output': valid_outputs.tolist(), 'Probabilities': preds_prob.tolist(),
                                    'Classification': preds.tolist(), 'Actual': valid_labels.tolist()})
                else:
                    raise Exception("There is no such mode available!")
        return results

