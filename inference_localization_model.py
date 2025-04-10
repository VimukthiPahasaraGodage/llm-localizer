import os
import shutil

import pandas as pd
import torch
from torch.utils.data import DataLoader

from components.llm import LLM
from components.llm_utils import LLMModels, LLMInfo
from components.localization_model import LocalizationTransformer, Config, LastHiddenStatesDataset, \
    TrainValidationSplit
from components.prompt import Driver as Tokenizer


class LocalizeVulnerabilities:
    def __init__(self,
                 dataset_name: str,
                 dataset_version: str,
                 checkpoint_config: Config,
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

        data = [{'item_index': 0, 'source_code': code, 'vuln_lines': "[]"}]
        df = pd.DataFrame(data)
        df.to_csv(df_path)
        return df

    def tokenize(self):
        Tokenizer(tensor_path=self.checkpoint_config.tensor_path,
                  dataset_path=self.checkpoint_config.dataset_path,
                  dataset_version=self.dataset_version,
                  dataset_name=self.dataset_name,
                  llm_model=self.checkpoint_config.llm_model,
                  pre_code_part=self.pre_code_part,
                  post_code_part=self.post_code_part,
                  standardize_df=False)

    def llm_inference(self):
        llm = LLM(tensor_path=self.checkpoint_config.tensor_path,
                  dataset_version=self.dataset_version,
                  dataset_name=self.dataset_name,
                  tokens_type=self.checkpoint_config.tokens_type,
                  llm_model=self.checkpoint_config.llm_model,
                  device=self.device)
        llm.get_and_save_last_hidden_states()

    def localization(self):
        model = LocalizationTransformer(
            self.checkpoint_config.num_layers_projection,
            self.checkpoint_config.num_layers_encoder,
            self.checkpoint_config.num_layers_dim_reduce,
            LLMInfo(self.checkpoint_config.llm_model).get_hidden_size(),
            self.checkpoint_config.num_head,
            self.checkpoint_config.target_dim,
            self.checkpoint_config.dim_reduce_type,
            self.checkpoint_config.seed,
            self.device

        ).to(self.device)

        cwd = os.getcwd()

        base_model_checkpoint_dir = f"{cwd}/{self.checkpoint_config.outputs_path}/{self.checkpoint_config.dataset_name}/{self.checkpoint_config.dataset_version}/{LLMModels.get_model_nickname(self.checkpoint_config.llm_model)}/{self.checkpoint_config.exp_config}/checkpoints/fold_{self.checkpoint_config.fold_index}"
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
                                   self.checkpoint_config.tokens_type,
                                   self.checkpoint_config.llm_model,
                                   1,
                                   self.checkpoint_config.seed)

        # Retrieve the train/validation indices for your chosen fold
        train_validation_indices = tvs.get_train_validation_indices_for_fold(0)

        inference_dataset = LastHiddenStatesDataset(
            self.checkpoint_config.tensor_path,
            self.checkpoint_config.dataset_path,
            self.dataset_version,
            self.dataset_name,
            self.checkpoint_config.tokens_type,
            self.checkpoint_config.llm_model,
            train_validation_indices,
            dataset_type="validation"
        )

        inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False)

        results = []
        model.eval()
        with torch.no_grad():
            for step, (last_hidden_state, code_tokens_length, line_split_lengths, line_labels) in enumerate(
                    inference_loader):
                last_hidden_state = last_hidden_state.to(self.device)
                code_tokens_length = code_tokens_length.to(self.device)
                line_split_lengths = line_split_lengths.to(self.device)
                line_labels = line_labels.to(self.device)

                outputs = model(last_hidden_state, code_tokens_length, line_split_lengths)

                mask = (line_labels != -1)
                valid_outputs = outputs[mask]

                preds_prob = torch.sigmoid(valid_outputs)
                preds = (preds_prob >= 0.5).long()

                if self.mode == "inference":
                    results.append({'Output': valid_outputs.tolist(), 'Probabilities': preds_prob.tolist(),
                                    'Classification': preds.tolist()})
                elif self.mode == "evaluation":
                    valid_labels = line_labels[mask].float()
                    results.append({'Output': valid_outputs.tolist(), 'Probabilities': preds_prob.tolist(),
                                    'Classification': preds.tolist(), 'Actual': valid_labels.tolist()})
                else:
                    raise Exception("There is no such mode available!")
        return results
