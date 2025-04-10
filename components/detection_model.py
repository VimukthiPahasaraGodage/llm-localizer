import copy
import os
import pickle
import random
from enum import Enum

import numpy as np
import torch
import torch.utils.checkpoint
import torch.utils.checkpoint
import torch.utils.checkpoint
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup

from components.llm_utils import LLMInfo, LLMModels
from components.localization_model import Config
from components.localization_model import CustomCosineLinearLoss
from components.localization_model import CustomCosineLoss
from components.localization_model import CustomExponentialLoss
from components.localization_model import LastHiddenStatesDataset
from components.localization_model import LocalizationTransformer, CodeGenBlock
from components.localization_model import TrainValidationSplit

MAX_CODE_LINES = 2048


class DetectionModelLayerConfig(Enum):
    NEW_PROJECTION = 1
    NEW_ENCODER_AND_PROJECTION = 2


class DetectionModelConfig(Config):
    def __init__(self,
                 base_model_config: Config,
                 transfer_learning_model_config: DetectionModelLayerConfig,
                 num_classes: int,
                 exp_config: str,
                 dataset_name: str,
                 dataset_version: str,
                 max_learning_rate: float,
                 criterion: str,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_name = dataset_name
        self.exp_config = exp_config
        self.dataset_version = dataset_version
        self.max_learning_rate = max_learning_rate
        self.criterion = criterion

        self.base_model_config = base_model_config
        self.transfer_learning_model_config = transfer_learning_model_config
        self.num_classes = num_classes


class DetectionModelConfigFactory:
    def __init__(self,
                 base_model_config: Config,
                 transfer_learning_model_config: DetectionModelLayerConfig,
                 num_classes: int,
                 exp_config: str,
                 dataset_version: str,
                 dataset_name: str,
                 max_learning_rate: float,
                 criterion: str):

        self.config = DetectionModelConfig(
            base_model_config=base_model_config,
            transfer_learning_model_config=transfer_learning_model_config,
            num_classes=num_classes,
            exp_config=exp_config,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            max_learning_rate=max_learning_rate,
            criterion=criterion
        )
        self.base_model_config = base_model_config
        self.transfer_learning_model_config = transfer_learning_model_config
        self.num_classes = num_classes
        self.exp_config = exp_config
        self.dataset_version = dataset_version
        self.dataset_name = dataset_name
        self.llm_model = self.base_model_config.llm_model
        self.max_learning_rate = max_learning_rate
        self.criterion = criterion

    def get_generated_configs(self):
        configurations = []
        for fold_index in range(self.config.k):
            new_config = copy.deepcopy(self.config)
            new_config.fold_index = fold_index
            if self.criterion == "BCEWithLogitsLoss":
                new_config.exp_config = f'{self.exp_config}_{self.dataset_name}_{self.dataset_version}_{self.llm_model.value}_{self.transfer_learning_model_config.value}_BCE'
            elif self.criterion == "CustomExponential":
                new_config.exp_config = f'{self.exp_config}_{self.dataset_name}_{self.dataset_version}_{self.llm_model.value}_{self.transfer_learning_model_config.value}_CE'
            elif self.criterion == "CustomCosine":
                new_config.exp_config = f'{self.exp_config}_{self.dataset_name}_{self.dataset_version}_{self.llm_model.value}_{self.transfer_learning_model_config.value}_CC'
            elif self.criterion == "CustomCosineLinear":
                new_config.exp_config = f'{self.exp_config}_{self.dataset_name}_{self.dataset_version}_{self.llm_model.value}_{self.transfer_learning_model_config.value}_CCL'
            else:
                raise Exception(f"Un-defined criterion: {self.criterion}")
            new_config.llm_model = self.llm_model
            new_config.target_dim = self.base_model_config.target_dim
            new_config.max_learning_rate = self.max_learning_rate
            new_config.total_epochs = 50
            new_config.train_epochs = 25
            configurations.append(new_config)
        return configurations


class DetectionTransformer(LocalizationTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, last_hidden_state: torch.tensor, code_tokens_length: torch.tensor,
                line_split_lengths: torch.tensor):
        outputs = []
        for batch_idx in range(last_hidden_state.shape[0]):
            split_ranges = line_split_lengths[batch_idx]
            nz = (split_ranges == -1).nonzero()
            if len(nz) == 0:
                first_neg_index = split_ranges.shape[0]  # handle edge case: no -1 found
            else:
                first_neg_index = nz[0].item()

            split_ranges = split_ranges[:first_neg_index]
            split_ranges = tuple(split_ranges.tolist())

            sample = last_hidden_state[batch_idx]
            sequence_length = (code_tokens_length[batch_idx]).item()
            sample = sample[:sequence_length, :]

            split_sample = torch.split(sample, split_ranges, dim=0)

            attention_mask = torch.zeros(first_neg_index).unsqueeze(0).to(self.device)
            attention_mask = attention_mask[:, None, None, :]

            if self.dim_reduce_type == "linear":
                newline_tensors = []
                for split_part in split_sample:
                    # get embedding for last token(this will most probably be a newline token)
                    newline_tensors.append(split_part[-1, :])
                newline_hidden_states = torch.stack(newline_tensors, dim=0)
                dim_reduced_newline_hidden_states = self.dim_reduce(newline_hidden_states).unsqueeze(0)
                encoder_output = self.encoder(dim_reduced_newline_hidden_states, padding_mask=attention_mask)
                projection_output = self.projection(encoder_output).squeeze()
                outputs.append(projection_output)
            elif self.dim_reduce_type in ['lstm', 'gru']:
                dim_reduced_tensors = []
                for split_part in split_sample:
                    split_part = split_part.unsqueeze(0)  # shape: [1, split_part_len, hidden_size]
                    if self.dim_reduce_type == 'lstm':
                        _, (h_n, c_n) = self.dim_reduce(split_part)  # h_n => [num_layers, batch=1, hidden_size]
                    else:
                        _, h_n = self.dim_reduce(split_part)
                    dim_reduced_tensors.append(h_n[-1, 0, :])  # => shape [hidden_size]
                dim_reduced_hidden_states = torch.stack(dim_reduced_tensors, dim=0).unsqueeze(0)
                encoder_output = self.encoder(dim_reduced_hidden_states, padding_mask=attention_mask)
                projection_output = self.projection(encoder_output).squeeze()
                outputs.append(projection_output)
        batch_output = torch.stack(outputs, dim=0).unsqueeze(1)
        return batch_output  # shape: [batch_size, num_classes]


class BiGRUProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        # a single‐layer bidirectional GRU
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        # after bi‐GRU, hidden is hidden_dim*2
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        output, h_n = self.gru(x)

        h_forward = h_n[2]  # [batch, hidden_dim] # last layer forward
        h_backward = h_n[3]  # [batch, hidden_dim] # last layer backward
        last = torch.cat([h_forward, h_backward], dim=1)  # [batch, hidden_dim * 2]

        x = self.lin1(last)  # → [B, H//2]
        x = self.act(x)
        x = self.lin2(x)  # → [B, num_classes]
        return x


class DetectionLastHiddenStatesDataset(LastHiddenStatesDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        tensor_index = self.tensor_indices[idx]
        last_hidden_state = torch.load(f'{self.token_tensors_path}/{tensor_index}.pt', map_location='cpu')

        df_row = self.df[self.df['item_index'] == tensor_index]
        original_df_row = self.original_df[self.original_df['item_index'] == tensor_index]
        if df_row.shape[0] != 1 or original_df_row.shape[0] != 1:
            raise Exception(f"There are rows that match the same index in the dataframe! Index: {tensor_index}")

        code_tokens_length = df_row.iloc[0]['code_tokens_length']
        if self.tokens_type == "prompt":
            code_start_index = df_row.iloc[0]['code_start_index']
            code_end_index = df_row.iloc[0]['code_end_index']
            last_hidden_state = last_hidden_state[:, code_start_index:(code_end_index + 1), :]

            if last_hidden_state.shape[1] != code_tokens_length:
                raise Exception("Shape mismatch in the processed last_hidden_state")

            pad_length = self.llm_info.max_allowed_context_length - last_hidden_state.shape[1]
            pad_tensor = torch.zeros((last_hidden_state.shape[0], pad_length, last_hidden_state.shape[2]))
            last_hidden_state = torch.cat([last_hidden_state, pad_tensor], dim=1)
            if last_hidden_state.shape[1] != self.llm_info.max_allowed_context_length:
                raise Exception("Shape mismatch in the processed last_hidden_state")
        elif self.tokens_type == "code":
            # NOOP
            pass
        else:
            raise Exception(f"The token_type is {self.tokens_type} which is not valid!")

        number_of_code_lines = df_row.iloc[0]['line_split_lengths_length']
        vuln_types = eval(original_df_row.iloc[0]['vuln_types'])
        vuln_types = torch.tensor(vuln_types)

        line_split_lengths = eval(df_row.iloc[0]['line_split_lengths'])
        if len(line_split_lengths) < MAX_CODE_LINES:
            line_split_lengths_padding = [-1 for _ in range(MAX_CODE_LINES - len(line_split_lengths))]
            line_split_lengths += line_split_lengths_padding
        line_split_lengths = torch.tensor(line_split_lengths)

        first_neg_index_line_split_lengths = (line_split_lengths == -1).nonzero()[0].item()
        if (first_neg_index_line_split_lengths) != number_of_code_lines:
            raise Exception("Something wrong happened! line_split_lengths and line_labels shape mismatch!")

        code_tokens_length = torch.tensor(code_tokens_length).unsqueeze(0)
        last_hidden_state = last_hidden_state.squeeze(0)

        return last_hidden_state, code_tokens_length, line_split_lengths, vuln_types


class Driver:
    def __init__(self, config: DetectionModelConfig):
        self.seed = config.seed

        self.initialize_seed_for_reproducibility()

        self.train(
            base_model_config=config.base_model_config,
            exp_config=config.exp_config,
            outputs_path=config.outputs_path,
            tensor_path=config.tensor_path,
            dataset_path=config.dataset_path,
            dataset_version=config.dataset_version,
            dataset_name=config.dataset_name,
            tokens_type=config.tokens_type,
            llm_model=config.llm_model,
            k=config.k,
            seed=config.seed,
            fold_index=config.fold_index,
            batch_size=config.batch_size,
            transfer_learning_model_config=config.transfer_learning_model_config,
            num_classes=config.num_classes,
            device=config.device,
            criterion=config.criterion,
            max_learning_rate=config.max_learning_rate,
            total_epochs=config.total_epochs,
            train_epochs=config.train_epochs,
            save_checkpoints=config.save_checkpoints)

    @staticmethod
    def get_codegen_block_config(num_head=16, dim_model=1024):
        codegen_model = "Salesforce/codegen-350M-multi"
        config = AutoConfig.from_pretrained(codegen_model)
        config.n_head = num_head
        config.n_embd = dim_model
        return config

    def initialize_seed_for_reproducibility(self):
        seed = self.seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def compute_metrics_from_confusion(self, tp, fp, tn, fn):
        """
        Given (tp, fp, tn, fn) counts, compute:
          accuracy, precision, recall, f1, tpr, fpr, tnr, fnr
        Returns a dict of metrics.
        """
        eps = 1e-9  # small epsilon to avoid division by zero

        accuracy = (tp + tn) / (tp + fp + tn + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)  # same as TPR
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        tpr = recall  # same as recall
        fpr = fp / (fp + tn + eps)
        tnr = tn / (tn + fp + eps)
        fnr = fn / (fn + tp + eps)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "tpr": tpr,
            "fpr": fpr,
            "tnr": tnr,
            "fnr": fnr,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }

    def write_metrics_to_log_writer(self, metrics: dict, log_writer: SummaryWriter, op_type: str, epoch: int):
        metrics_keys = list(metrics.keys())
        for key in metrics_keys:
            log_writer.add_scalar(f'{key}/{op_type}', metrics[key], epoch)

    def check_save_model_checkpoints(self, val_loss: float, val_metrics: dict, metric_track: dict):
        save_checkpoint = False
        if val_metrics['precision'] > metric_track['precision']:
            metric_track['precision'] = val_metrics['precision']
            save_checkpoint = True
        if val_metrics['recall'] > metric_track['recall']:
            metric_track['recall'] = val_metrics['recall']
            save_checkpoint = True
        if val_metrics['f1_score'] > metric_track['f1_score']:
            metric_track['f1_score'] = val_metrics['f1_score']
            save_checkpoint = True
        if val_metrics['accuracy'] > metric_track['accuracy']:
            metric_track['accuracy'] = val_metrics['accuracy']
            save_checkpoint = True
        if val_loss < metric_track['val_loss']:
            metric_track['val_loss'] = val_loss
            save_checkpoint = True
        return metric_track, save_checkpoint

    def train(self,
              base_model_config: Config,
              exp_config: str,
              outputs_path: str,
              tensor_path: str,
              dataset_path: str,
              dataset_version: str,
              dataset_name: str,
              tokens_type: str,
              llm_model: Enum,
              k: int,
              seed: int,
              fold_index: int,
              batch_size: int,
              transfer_learning_model_config: DetectionModelLayerConfig,
              num_classes: int,
              device: str = "cuda:0",
              criterion: str = "BCEWithLogitsLoss",
              max_learning_rate: float = 1e-4,
              total_epochs: int = 300,
              train_epochs: int = 100,
              save_checkpoints: bool = True
              ):
        print(f"Preparing the model.  fold: {fold_index}  configuration: {exp_config}")

        cwd = os.getcwd()
        checkpoint_dir = f"{cwd}/{outputs_path}/{dataset_name}/{dataset_version}/{LLMModels.get_model_nickname(llm_model)}/{exp_config}/checkpoints/fold_{fold_index}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        log_dir = f"{cwd}/{outputs_path}/{dataset_name}/{dataset_version}/{LLMModels.get_model_nickname(llm_model)}/{exp_config}/logs/fold_{fold_index}"
        os.makedirs(log_dir, exist_ok=True)

        outputs_dir = f"{cwd}/{outputs_path}/{dataset_name}/{dataset_version}/{LLMModels.get_model_nickname(llm_model)}/{exp_config}/outputs/fold_{fold_index}"
        os.makedirs(outputs_dir, exist_ok=True)

        log_writer = SummaryWriter(log_dir=log_dir)

        # Create the TrainValidationSplit object
        tvs = TrainValidationSplit(tensor_path, dataset_version, dataset_name, tokens_type, llm_model, k, seed)

        # Retrieve the train/validation indices for your chosen fold
        train_validation_indices = tvs.get_train_validation_indices_for_fold(fold_index)

        # Create the training & validation Datasets
        train_dataset = DetectionLastHiddenStatesDataset(
            tensor_path,
            dataset_path,
            dataset_version,
            dataset_name,
            tokens_type,
            llm_model,
            train_validation_indices,
            dataset_type="train"
        )

        val_dataset = DetectionLastHiddenStatesDataset(
            tensor_path,
            dataset_path,
            dataset_version,
            dataset_name,
            tokens_type,  # "prompt" or "code"
            llm_model,
            train_validation_indices,
            dataset_type="validation"
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = DetectionTransformer(
            num_layers_projection=base_model_config.num_layers_projection,
            num_layers_encoder=base_model_config.num_layers_encoder,
            num_layers_dim_reduce=base_model_config.num_layers_dim_reduce,
            hidden_size=LLMInfo(base_model_config.llm_model).get_hidden_size(),
            num_head=base_model_config.num_head,
            target_dim=base_model_config.target_dim,
            dim_reduce_type=base_model_config.dim_reduce_type,
            seed=base_model_config.seed,
            device=device
        ).to(device)

        loss_fn = None
        if criterion == "BCEWithLogitsLoss":
            loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        elif criterion == "CustomExponential":
            loss_fn = CustomExponentialLoss(reduction='mean')
        elif criterion == "CustomCosine":
            loss_fn = CustomCosineLoss(reduction='mean')
        elif criterion == "CustomCosineLinear":
            loss_fn = CustomCosineLinearLoss(reduction='mean')
        else:
            raise Exception(f"Undefined criterion: {criterion}")

        num_training_steps = len(train_loader) * total_epochs
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps as warmup

        start_epoch = 0
        end_epoch = train_epochs
        metric_track = {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'val_loss': 1e10}

        # get the latest model checkpoint if exists
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
        if len(checkpoint_files) == 0:
            base_model_checkpoint_dir = f"{cwd}/{base_model_config.outputs_path}/{base_model_config.dataset_name}/{base_model_config.dataset_version}/{LLMModels.get_model_nickname(llm_model)}/{base_model_config.exp_config}/checkpoints/fold_{base_model_config.fold_index}"
            if not os.path.isdir(base_model_checkpoint_dir):
                raise FileNotFoundError("Checkpoint folder does not exist")
            base_checkpoint_files = [f for f in os.listdir(base_model_checkpoint_dir) if
                                     os.path.isfile(os.path.join(base_model_checkpoint_dir, f))]
            if len(base_checkpoint_files) > 0:
                print(
                    f"Trying to load the base model checkpoint.  fold: {base_model_config.fold_index}  configuration: {base_model_config.exp_config}")
                latest_epoch = 0
                for file in base_checkpoint_files:
                    epoch_of_file = int(file.split('.')[0])
                    if epoch_of_file > latest_epoch:
                        latest_epoch = epoch_of_file
                print(
                    f"Latest base model checkpoint detected for epoch: {latest_epoch}  fold: {base_model_config.fold_index}  configuration: {base_model_config.exp_config}")
                checkpoint = torch.load(f'{base_model_checkpoint_dir}/{latest_epoch}.pt', map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                raise FileNotFoundError("No base model checkpoints found")

        if start_epoch >= end_epoch:
            print(f"Nothing to train.  start_epoch: {start_epoch}  end_epoch: {end_epoch}")
            return

        ####################################### Model Configuration ############################
        if transfer_learning_model_config == DetectionModelLayerConfig.NEW_PROJECTION:
            model.projection = BiGRUProjection(input_dim=model.target_dim, hidden_dim=(model.target_dim // 2),
                                               num_classes=num_classes).to(device)
            for param in model.parameters():
                param.requires_grad = False
            for param in model.projection.parameters():
                param.requires_grad = True
            optimizer = Adam(model.projection.parameters(), lr=max_learning_rate)
        elif transfer_learning_model_config == DetectionModelLayerConfig.NEW_ENCODER_AND_PROJECTION:
            codegen_model = "Salesforce/codegen-350M-multi"
            config = AutoConfig.from_pretrained(codegen_model)
            config.n_head = model.num_head
            config.n_embd = model.target_dim

            new_encoder_layer = CodeGenBlock(config).to(device)

            model.encoder.enc_layers.append(new_encoder_layer)
            model.projection = BiGRUProjection(input_dim=model.target_dim, hidden_dim=(model.target_dim // 2),
                                               num_classes=num_classes).to(device)
            for name, param in model.named_parameters():
                if name.startswith("encoder.enc_layers.%d" % (len(model.encoder.enc_layers) - 1)) or name.startswith(
                        "projection"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            params_to_train = list(model.encoder.enc_layers[-1].parameters()) + list(model.projection.parameters())
            optimizer = Adam(params_to_train, lr=max_learning_rate)
        else:
            raise ValueError("No such configuration defined for transfer learning model")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        if len(checkpoint_files) > 0:
            print(f"Trying to load the model for checkpoint.  fold: {fold_index}  configuration: {exp_config}")
            latest_epoch = 0
            for file in checkpoint_files:
                epoch_of_file = int(file.split('.')[0])
                if epoch_of_file > latest_epoch:
                    latest_epoch = epoch_of_file
            print(
                f"Latest checkpoint detected for epoch: {latest_epoch}  fold: {fold_index}  configuration: {exp_config}")
            checkpoint = torch.load(f'{checkpoint_dir}/{latest_epoch}.pt', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            metric_track['val_loss'] = checkpoint['val_loss']
            metric_track['precision'] = checkpoint['precision']
            metric_track['recall'] = checkpoint['recall']
            metric_track['f1_score'] = checkpoint['f1_score']
            metric_track['accuracy'] = checkpoint['accuracy']

        ####################################### Training #######################################
        train_outputs = {}
        val_outputs = {}
        train_step = 0
        val_step = 0

        for epoch in range(start_epoch, end_epoch):
            print(f"Epoch {epoch + 1}/{total_epochs}  fold: {fold_index}  configuration: {exp_config}")

            train_tp = 0  # true positives
            train_fp = 0  # false positives
            train_tn = 0  # true negatives
            train_fn = 0  # false negatives

            train_outputs_for_epoch = []

            model.train()
            running_train_loss = 0.0
            for step, (last_hidden_state, code_tokens_length, line_split_lengths, class_labels) in enumerate(
                    train_loader):
                train_step += 1

                last_hidden_state = last_hidden_state.to(device)  # [batch_size, seq_len, hidden_size]
                code_tokens_length = code_tokens_length.to(device)  # [batch_size]
                line_split_lengths = line_split_lengths.to(device)  # [batch_size, MAX_CODE_LINES]
                class_labels = class_labels.to(device)  # [batch_size, MAX_CODE_LINES]

                optimizer.zero_grad()

                # Forward pass
                outputs = model(last_hidden_state, code_tokens_length,
                                line_split_lengths)  # outputs shape: [batch_size, num_classes]
                if outputs.dim() > 2 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)

                valid_outputs = outputs
                valid_labels = class_labels.float()

                # Compute loss
                loss = loss_fn(valid_outputs, valid_labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_train_loss += loss.item()

                # Get the current learning rate of the scheduler(Assuming single parameter group)
                current_lr = scheduler.get_last_lr()[0]

                log_writer.add_scalar('lr/step/train', current_lr, train_step)
                log_writer.add_scalar('loss/step/train', loss.item(), train_step)

                # Save the output of the training step
                batch_size_local = class_labels.shape[0]
                for i in range(batch_size_local):
                    # valid lines for snippet i
                    snippet_mask = (class_labels[i] != -1)
                    snippet_outputs = outputs[i][snippet_mask].tolist()
                    snippet_labels = class_labels[i][snippet_mask].tolist()
                    train_outputs_for_epoch.append({'outputs': snippet_outputs, 'labels': snippet_labels})

                # Calculate the precision, recall, f1-score, accuracy, TPR, FPR, TNR, FNR
                if criterion in ["BCEWithLogitsLoss", "CustomExponential", "CustomCosine", "CustomCosineLinear"]:
                    preds_prob = torch.sigmoid(valid_outputs)
                    preds = (preds_prob >= 0.5).long()

                    y_true = valid_labels.long()

                    # Compute per-batch confusion
                    # True Positive: preds=1, y_true=1
                    tp = torch.sum((preds == 1) & (y_true == 1)).item()
                    # False Positive: preds=1, y_true=0
                    fp = torch.sum((preds == 1) & (y_true == 0)).item()
                    # True Negative: preds=0, y_true=0
                    tn = torch.sum((preds == 0) & (y_true == 0)).item()
                    # False Negative: preds=0, y_true=1
                    fn = torch.sum((preds == 0) & (y_true == 1)).item()

                    train_tp += tp
                    train_fp += fp
                    train_tn += tn
                    train_fn += fn

            train_outputs[(epoch + 1)] = train_outputs_for_epoch
            epoch_train_loss = running_train_loss / len(train_loader)
            train_metrics = self.compute_metrics_from_confusion(train_tp, train_fp, train_tn, train_fn)
            self.write_metrics_to_log_writer(train_metrics, log_writer, 'train', (epoch + 1))
            log_writer.add_scalar('loss/epoch/train', epoch_train_loss, (epoch + 1))

            print(f"  [Training]  Loss: {epoch_train_loss:.8f}  fold: {fold_index}  configuration: {exp_config}")

            ####################################### Validation #######################################
            model.eval()
            running_val_loss = 0.0

            val_tp = 0  # true positives
            val_fp = 0  # false positives
            val_tn = 0  # true negatives
            val_fn = 0  # false negatives

            val_outputs_for_epoch = []

            with torch.no_grad():
                for step, (last_hidden_state, code_tokens_length, line_split_lengths, class_labels) in enumerate(
                        val_loader):
                    val_step += 1

                    last_hidden_state = last_hidden_state.to(device)
                    code_tokens_length = code_tokens_length.to(device)
                    line_split_lengths = line_split_lengths.to(device)
                    class_labels = class_labels.to(device)

                    outputs = model(last_hidden_state, code_tokens_length, line_split_lengths)
                    if outputs.dim() > 2 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze(1)

                    valid_outputs = outputs
                    valid_labels = class_labels.float()

                    loss = loss_fn(valid_outputs, valid_labels)
                    running_val_loss += loss.item()

                    log_writer.add_scalar('loss/step/val', loss.item(), val_step)

                    # Save the output of the validation step
                    batch_size_local = class_labels.shape[0]
                    for i in range(batch_size_local):
                        # valid lines for snippet i
                        snippet_mask = (class_labels[i] != -1)
                        snippet_outputs = outputs[i][snippet_mask].tolist()
                        snippet_labels = class_labels[i][snippet_mask].tolist()
                        val_outputs_for_epoch.append({'outputs': snippet_outputs, 'labels': snippet_labels})

                    if criterion in ["BCEWithLogitsLoss", "CustomExponential", "CustomCosine", "CustomCosineLinear"]:
                        preds_prob = torch.sigmoid(valid_outputs)
                        preds = (preds_prob >= 0.5).long()

                        y_true = valid_labels.long()

                        tp = torch.sum((preds == 1) & (y_true == 1)).item()
                        fp = torch.sum((preds == 1) & (y_true == 0)).item()
                        tn = torch.sum((preds == 0) & (y_true == 0)).item()
                        fn = torch.sum((preds == 0) & (y_true == 1)).item()

                        val_tp += tp
                        val_fp += fp
                        val_tn += tn
                        val_fn += fn

            val_outputs[(epoch + 1)] = val_outputs_for_epoch
            epoch_val_loss = running_val_loss / len(val_loader)
            val_metrics = self.compute_metrics_from_confusion(val_tp, val_fp, val_tn, val_fn)
            self.write_metrics_to_log_writer(val_metrics, log_writer, 'val', (epoch + 1))
            log_writer.add_scalar('loss/epoch/val', epoch_val_loss, (epoch + 1))

            print(f"  [Validation]  Loss: {epoch_val_loss:.8f}  fold: {fold_index}  configuration: {exp_config}")
            if save_checkpoints:
                metric_track, save_model = self.check_save_model_checkpoints(epoch_val_loss, val_metrics, metric_track)
                # save the model  checkpoint if this is tha last checkpoint or there is any improvement
                if save_model or ((epoch + 1) == (start_epoch + train_epochs)):
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': (epoch + 1),
                        'precision': val_metrics['precision'],
                        'recall': val_metrics['recall'],
                        'f1_score': val_metrics['f1_score'],
                        'accuracy': val_metrics['accuracy'],
                        'val_loss': epoch_val_loss
                    }, f'{checkpoint_dir}/{epoch + 1}.pt')
                    if save_model:
                        print(
                            f"  [Performance Checkpoint]  Precision: {val_metrics['precision']:.4f}  Recall: {val_metrics['recall']:.4f}  F1-score: {val_metrics['f1_score']:.4f}  Accuracy: {val_metrics['accuracy']:.4f}  fold: {fold_index}  configuration: {exp_config}")
                    else:
                        print(
                            f"  [Latest Checkpoint]  Precision: {val_metrics['precision']:.4f}  Recall: {val_metrics['recall']:.4f}  F1-score: {val_metrics['f1_score']:.4f}  Accuracy: {val_metrics['accuracy']:.4f}  fold: {fold_index}  configuration: {exp_config}")

        log_writer.close()  # flush all the logs and close the log_writer

        with open(f'{outputs_dir}/train_outputs.pkl', 'wb') as f:
            pickle.dump(train_outputs, f)

        with open(f'{outputs_dir}/val_outputs.pkl', 'wb') as f:
            pickle.dump(val_outputs, f)

        print(f"Training completed!  fold: {fold_index}  configuration: {exp_config}")