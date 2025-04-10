import copy
import os
import pickle
import random
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
import torch.utils.checkpoint
import torch.utils.checkpoint
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig
from transformers import get_linear_schedule_with_warmup
from transformers.activations import ACT2FN

from components.llm_utils import LLMInfo, LLMModels

MAX_CODE_LINES = 2048


class TrainValidationSplit:
    def __init__(self, tensor_path: str, dataset_version: str, dataset_name: str, tokens_type: str, llm_model: Enum,
                 k: int = 10, seed: int = 42):
        self.k = k
        self.seed = seed

        self.tensor_path = tensor_path
        self.dataset_version = dataset_version
        self.dataset_name = dataset_name
        self.tokens_type = tokens_type
        self.llm_model = llm_model

        self.llm_info = LLMInfo(self.llm_model)

        self.cwd = os.getcwd()

        self.df_path = f'{self.cwd}/{self.tensor_path}/{self.dataset_name}/{self.dataset_version}/{self.llm_model.value}/tokenizer/{self.dataset_name}.csv'
        self.token_tensors_path = f'{self.cwd}/{self.tensor_path}/{self.dataset_name}/{self.dataset_version}/{self.llm_model.value}/last_hidden_state/{self.tokens_type}'

        self.df = pd.read_csv(self.df_path)

        self.valid_tensors = []
        self.valid_tensors_indices = []

        self.k_folds = []

        self.train_validation_indices = []

        self.get_train_and_validation_indices()

    def get_train_validation_indices_for_fold(self, fold: int):
        return self.train_validation_indices[fold]  # fold: 0-indexed

    def get_train_and_validation_indices(self):
        self.get_valid_tensors()
        self.split_indices_for_k_fold()

        for i in range(self.k):
            k_folds_copy = self.k_folds.copy()
            validation = self.k_folds[i]
            del k_folds_copy[i]
            train = [index for fold in k_folds_copy for index in fold]
            validation.sort()
            train.sort()

            self.train_validation_indices.append({'train': train, 'validation': validation})

    def split_indices_for_k_fold(self):
        random.seed(self.seed)
        indices = self.valid_tensors_indices.copy()
        random.shuffle(indices)

        fold_size = len(indices) // self.k
        remainder = len(indices) % self.k

        folds = []
        start = 0
        for i in range(self.k):
            end = start + fold_size + (1 if i < remainder else 0)
            folds.append(indices[start:end])
            start = end

        for fold in folds:
            fold_copy = fold.copy()
            fold_copy.sort()
            self.k_folds.append(fold_copy)

    def get_valid_tensors(self):
        files = [f for f in os.listdir(self.token_tensors_path) if
                 os.path.isfile(os.path.join(self.token_tensors_path, f))]
        for file in files:
            if ".pt" in file:
                self.valid_tensors.append(file)
            else:
                print(
                    f"The file '{file}' does not have '.pt' extension. Make sure the folder \"{self.token_tensors_path}\" only contains valid tensor files!")

        for file in self.valid_tensors:
            index = int(file.split('.')[0])
            self.valid_tensors_indices.append(index)

        self.valid_tensors_indices.sort()  # sort the indices

        num_rows_df = self.df.shape[0]
        if num_rows_df > len(self.valid_tensors_indices):
            print(
                f"Some rows from the dataframe \"{self.df_path}\" are omitted due to tensors not in compliance to standards!")


class LastHiddenStatesDataset(Dataset):
    def __init__(self, tensor_path: str, dataset_path: str, dataset_version: str, dataset_name: str, tokens_type: str,
                 llm_model: Enum, train_validation_indices: dict, dataset_type: str):
        self.tensor_path = tensor_path
        self.dataset_path = dataset_path
        self.dataset_version = dataset_version
        self.dataset_name = dataset_name
        self.tokens_type = tokens_type
        self.llm_model = llm_model

        self.llm_info = LLMInfo(self.llm_model)

        self.cwd = os.getcwd()

        self.original_df_path = f'{self.cwd}/{self.dataset_path}/{self.dataset_name}/{self.dataset_version}/{self.dataset_name}.csv'
        self.df_path = f'{self.cwd}/{self.tensor_path}/{self.dataset_name}/{self.dataset_version}/{self.llm_model.value}/tokenizer/{self.dataset_name}.csv'
        self.token_tensors_path = f'{self.cwd}/{self.tensor_path}/{self.dataset_name}/{self.dataset_version}/{self.llm_model.value}/last_hidden_state/{self.tokens_type}'
        self.original_df = pd.read_csv(self.original_df_path)
        self.df = pd.read_csv(self.df_path)

        self.train_validation_indices = train_validation_indices
        self.tensor_indices = self.train_validation_indices[dataset_type]  # get the indices for training or validation

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
        vuln_lines = eval(original_df_row.iloc[0]['vuln_lines'])
        line_labels = np.zeros(number_of_code_lines)
        try:
            line_labels[vuln_lines] = np.ones(len(vuln_lines))
            if len(line_labels) < MAX_CODE_LINES:
                line_labels_padding = [-1 for _ in range(MAX_CODE_LINES - len(line_labels))]
                line_labels = line_labels.tolist() + line_labels_padding
            line_labels = torch.tensor(line_labels)
        except Exception as e:
            raise Exception(f'Label shape wrong! Error: {e}')

        line_split_lengths = eval(df_row.iloc[0]['line_split_lengths'])
        if len(line_split_lengths) < MAX_CODE_LINES:
            line_split_lengths_padding = [-1 for _ in range(MAX_CODE_LINES - len(line_split_lengths))]
            line_split_lengths += line_split_lengths_padding
        line_split_lengths = torch.tensor(line_split_lengths)

        first_neg_index_line_split_lengths = (line_split_lengths == -1).nonzero()[0].item()
        first_neg_index_line_labels = (line_labels == -1).nonzero()[0].item()
        if (first_neg_index_line_split_lengths) != number_of_code_lines or (
                first_neg_index_line_labels) != number_of_code_lines:
            raise Exception("Something wrong happened! line_split_lengths and line_labels shape mismatch!")

        code_tokens_length = torch.tensor(code_tokens_length).unsqueeze(0)
        last_hidden_state = last_hidden_state.squeeze(0)

        return last_hidden_state, code_tokens_length, line_split_lengths, line_labels

    def __len__(self):
        return len(self.tensor_indices)


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(x.device).float()
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: t[None, offset: x.shape[1] + offset, None, :].repeat_interleave(2, 3), sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class CodeGenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        # print('embed-dim ', self.embed_dim)
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)

        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
            self,
            query,
            key,
            value,
            attention_mask=None,
            head_mask=None,
    ):

        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        attn_weights = attn_weights / self.scale_attn
        attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_past=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        # print('hidden_states ', hidden_states.shape)
        qkv = self.qkv_proj(hidden_states)
        # TODO: check more on the use of projection splitting(the value was 4)
        mp_num = 1
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)

        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class CodeGenMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CodeGenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CodeGenAttention(config)
        self.mlp = CodeGenMLP(inner_dim, config)

    def forward(
            self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)


class Encoder(nn.Module):
    def __init__(self, num_layer=2, dim_model=1024, num_head=16):
        super(Encoder, self).__init__()
        self.num_layer = num_layer
        codegen_model = "Salesforce/codegen-350M-multi"
        config = AutoConfig.from_pretrained(codegen_model)
        config.n_layer = num_layer
        config.n_head = num_head
        config.n_embd = dim_model

        self.enc_layers = torch.nn.ModuleList(
            [CodeGenBlock(config) for _ in range(num_layer)]
        )

    def forward(self, x, padding_mask):
        """Transformer encoding layer

        Args:
            x (torch.tensor): shape [batch_size=batch_size, seq_len=256, num_dimensions=1024]
            padding_mask (torch.tensor): shape [batch_size=batch_size, seq_len=256]

        Returns:
            torch.tensor: [batch_size=batch_size, input_seq_len=256, d_model=1024]
        """
        for i in range(self.num_layer):
            x = self.enc_layers[i](x, attention_mask=padding_mask)
            x = x[0]
        return x  # (batch_size, input_seq_len, d_model)


class LocalizationTransformer(nn.Module):
    def __init__(self,
                 num_layers_projection: int = 2,
                 num_layers_encoder: int = 2,
                 num_layers_dim_reduce: int = 2,
                 hidden_size: int = 1024,
                 num_head: int = 16,
                 target_dim=256,
                 dim_reduce_type: str = "linear",
                 seed: int = 42,
                 device: str = 'cuda:0'):
        super().__init__()
        self.num_layers_projection = num_layers_projection
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_dim_reduce = num_layers_dim_reduce
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.target_dim = target_dim
        self.dim_reduce_type = dim_reduce_type
        self.seed = seed
        self.device = device

        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)

        if self.dim_reduce_type == "linear":
            if self.num_layers_dim_reduce == 1:
                self.dim_reduce = nn.Linear(self.hidden_size, self.target_dim)
            elif self.num_layers_dim_reduce == 2:
                self.dim_reduce = nn.Sequential(
                    nn.Linear(self.hidden_size, self.target_dim),
                    nn.GELU(),
                    nn.Linear(self.target_dim, self.target_dim)
                )
            else:
                raise Exception("The highest number of layers supported for linear dimensional reduction is 2")
        elif self.dim_reduce_type == "lstm":
            if self.num_layers_dim_reduce <= 3:
                self.dim_reduce = nn.LSTM(
                    input_size=self.hidden_size,
                    hidden_size=self.target_dim,
                    num_layers=self.num_layers_dim_reduce,
                    batch_first=True
                )
            else:
                raise Exception("The highest number of layers supported for LSTM dimensional reduction is 3")
        elif self.dim_reduce_type == 'gru':
            if self.num_layers_dim_reduce <= 3:
                self.dim_reduce = nn.GRU(
                    input_size=self.hidden_size,
                    hidden_size=self.target_dim,
                    num_layers=self.num_layers_dim_reduce,
                    batch_first=True
                )
            else:
                raise Exception("The highest number of layers supported for GRU dimensional reduction is 3")
        else:
            raise Exception(f"Unsupported dimensional reduction method: {self.dim_reduce_type}")

        if self.dim_reduce_type == 'linear' and self.num_layers_encoder > 3:
            raise Exception("The highest number of layers supported for encoder is 3")
        if self.dim_reduce_type in ['lstm', 'gru'] and self.num_layers_encoder > 4:
            raise Exception("The highest number of layers supported for encoder is 4")

        self.encoder = Encoder(
            num_layer=self.num_layers_encoder, dim_model=self.target_dim, num_head=self.num_head
        )

        if self.num_layers_projection == 1:
            self.projection = nn.Linear(self.target_dim, 1)
        elif self.num_layers_projection == 2:
            if self.target_dim % 2 != 0:
                raise Exception(f"Target dimension size must be divisible by 2: {self.target_dim}")
            self.projection = nn.Sequential(
                nn.Linear(self.target_dim, self.target_dim // 2),
                nn.GELU(),
                nn.Linear(self.target_dim // 2, 1)
            )
        else:
            raise Exception("The highest number of layers supported for projection layer is 2")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights in a way that is reproducible.
        By using self.gen, we ensure the same random draws each time.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, generator=self.gen)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight, generator=self.gen)
            if module.padding_idx is not None:
                # Zero out the embedding vector at the padding index
                nn.init.zeros_(module.weight[module.padding_idx])

        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

        # LSTM/GRU: if we want finer control over their initialization
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:  # input-hidden
                    nn.init.xavier_normal_(param, generator=self.gen)
                elif 'weight_hh' in name:  # hidden-hidden
                    nn.init.orthogonal_(param, generator=self.gen)
                elif 'bias' in name:
                    nn.init.zeros_(param)

    def forward(self, last_hidden_state: torch.tensor, code_tokens_length: torch.tensor,
                line_split_lengths: torch.tensor):
        """
        Summary of parameters

        :param last_hidden_state: The last hidden state belonging to only the code tokens. shape: [batch_size, llm_model.max_allowed_sequence_length, hidden_size]
        :param code_tokens_length: The length of the sequence belong to the code. eg:-[[1, 0, 0, 1, 0, 0, ..., -1, -1],...] shape: [batch_size, MAX_CODE_LINES]
        :param line_split_lengths: The lengths of tokens for each line of the code, eg:- [[2, 3, 4, ..., -1, -1, -1],...] shape: [batch_size, MAX_CODE_LINES]
        """
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
                projection_output = torch.cat(
                    [projection_output, torch.zeros(MAX_CODE_LINES - projection_output.numel()).to(self.device)], dim=0)
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
                projection_output = torch.cat(
                    [projection_output, torch.zeros(MAX_CODE_LINES - projection_output.numel()).to(self.device)], dim=0)
                outputs.append(projection_output)
        batch_output = torch.stack(outputs, dim=0)
        return batch_output  # shape: [batch_size, MAX_CODE_LINES]


class CustomCosineLinearLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(CustomCosineLinearLoss, self).__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (Tensor): raw logits of shape [batch_size, MAX_CODE_LINE]
            targets (Tensor): binary labels (0 or 1) of shape [batch_size, MAX_CODE_LINE]
        Returns:
            Tensor: scalar loss value.
        """
        # Apply sigmoid to obtain probabilities from logits
        prob = torch.sigmoid(logits)

        # Define secant function: sec(x) = 1/cos(x)
        sec = lambda x: 1.0 / torch.cos(x)

        # Compute z1 based on condition:
        # If prob > 0.3: z1 = sec(2*prob - 0.6) - 1, otherwise z1 = (5/300)*prob
        z1 = torch.where(prob > 0.3, sec(2 * prob - 0.6) - 1, (5.0 / 300) * prob)

        # Compute z2 based on condition:
        # If prob < 0.7: z2 = sec(2*prob - 1.4) - 1, otherwise z2 = -(5/300)*prob + (5/300)
        z2 = torch.where(prob < 0.7, sec(2 * prob - 1.4) - 1, -(5.0 / 300) * prob + (5.0 / 300))

        # Combine z1 and z2 with the labels: (1-y_hat) for z1 and y_hat for z2
        loss_tensor = (1 - targets) * z1 + targets * z2

        # Apply the specified reduction
        if self.reduction == 'mean':
            return loss_tensor.mean()
        elif self.reduction == 'sum':
            return loss_tensor.sum()
        else:
            # 'none' - return the unreduced loss
            return loss_tensor


class CustomCosineLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(CustomCosineLoss, self).__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: torch.Tensor of logits (raw predictions), shape [batch_size, MAX_CODE_LINE]
            targets: torch.Tensor of binary labels (0 or 1), shape [batch_size, MAX_CODE_LINE]
        Returns:
            A scalar tensor representing the loss.
        """
        # Apply sigmoid to get probabilities from logits
        prob = torch.sigmoid(logits)

        # Define the secant function using cosine.
        # sec(x) = 1/cos(x)
        sec = lambda x: 1.0 / torch.cos(x)

        # Compute the expressions for z1 and z2 using the corresponding conditions.
        # z1 = sec(2 * prob - 0.6) - 1, if prob > 0.3; otherwise 0.
        z1 = torch.where(prob > 0.3, sec(2 * prob - 0.6) - 1, torch.zeros_like(prob))

        # z2 = sec(2 * prob - 1.4) - 1, if prob < 0.7; otherwise 0.
        z2 = torch.where(prob < 0.7, sec(2 * prob - 1.4) - 1, torch.zeros_like(prob))

        # Combine the loss contributions from z1 and z2 based on the labels.
        loss_tensor = (1 - targets) * z1 + targets * z2

        # Apply the specified reduction
        if self.reduction == 'mean':
            return loss_tensor.mean()
        elif self.reduction == 'sum':
            return loss_tensor.sum()
        else:
            # 'none' - return the unreduced loss
            return loss_tensor


class CustomExponentialLoss(nn.Module):
    """
    Implements the custom loss:
        L = mean over i of [
            (1 - y_hat[i]) * pow(1.18, 1000 * sigmoid(logits[i]) - 500)
          + (y_hat[i])     * pow(1.18, -(1000 * sigmoid(logits[i]) - 500))
        ]

    Args:
        reduction (str): Specifies the reduction to apply to the output:
                         'mean' | 'sum' | 'none'.
                         Default: 'mean'
    """

    def __init__(self, reduction: str = 'mean'):
        super(CustomExponentialLoss, self).__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Float tensor of shape [batch_size, ...] – raw model outputs.
            targets: Float/Long tensor of shape [batch_size, ...] with 0/1 labels.
                     (Should be same shape as logits.)
        Returns:
            A scalar loss (if reduction='mean' or 'sum') or a tensor of losses
            (if reduction='none').
        """
        # Apply the sigmoid just like BCEWithLogitsLoss would
        probs = torch.sigmoid(logits)

        # Compute the loss terms
        loss = (1.0 - targets) * torch.pow(1.1, (1000.0 * probs - 510.0)) + targets * torch.pow(1.1, -(
                1000.0 * probs - 490.0))

        # Apply the specified reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            # 'none' - return the unreduced loss
            return loss


class Driver:
    def __init__(self, config):
        self.seed = config.seed

        self.initialize_seed_for_reproducibility()

        self.train(
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
            device=config.device,
            num_layers_projection=config.num_layers_projection,
            num_layers_encoder=config.num_layers_encoder,
            num_layers_dim_reduce=config.num_layers_dim_reduce,
            num_head=config.num_head,
            target_dim=config.target_dim,
            dim_reduce_type=config.dim_reduce_type,
            criterion=config.criterion,
            max_learning_rate=config.max_learning_rate,
            total_epochs=config.total_epochs,
            train_epochs=config.train_epochs,
            top_k=config.top_k,
            save_checkpoints=config.save_checkpoints)

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

    def write_top_k_metrics_to_log_writer(self, top_k_sums: dict, snippet_count: int, top_k: list,
                                          log_writer: SummaryWriter, op_type: str, epoch: int):
        if snippet_count == 0:
            raise Exception("No snippets to evaluate. Skipping top-k metrics.")

        for k in top_k:
            top_k_sum = top_k_sums[f"top{k}"]
            rate = top_k_sum / snippet_count
            log_writer.add_scalar(f'top_{k}_rate/{op_type}', rate, epoch)
            log_writer.add_scalar(f'top_{k}_sum/{op_type}', top_k_sum, epoch)

    def compute_topk_metrics(self, outputs, labels, top_k):
        """
        Given a single snippet's logits (1D) and corresponding labels (1D),
        compute whether we have at least one vulnerable line in the top-k lines
        predicted by the model.

        Returns a dict with { 'top1': 0/1, 'top3': 0/1, 'top5': 0/1 }, etc.
        Where 1 = "hit" (found at least one line with label=1 among top-k).
        """
        # We'll store success=1 or fail=0 for each top_k
        results = {}
        if len(outputs) == 0:
            # No valid lines => no "hit" possible
            for k in top_k:
                results[f"top{k}"] = 0
            return results

        # 1) Get probabilities
        probs = torch.sigmoid(outputs)

        # 2) Sort from largest to smallest
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)

        # 3) For each k, check if there's a label=1 among top k lines
        for k in top_k:
            k = min(k, len(sorted_probs))  # in case snippet has fewer lines
            top_indices = sorted_idx[:k]

            # If any of these top lines is actually 1 => success
            top_labels = labels[top_indices]
            is_hit = (top_labels == 1).any().item()
            results[f"top{k}"] = 1 if is_hit else 0

        return results

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
              device: str = "cuda:0",
              num_layers_projection: int = 2,
              num_layers_encoder: int = 2,
              num_layers_dim_reduce: int = 2,
              num_head: int = 16,
              target_dim: int = 1024,
              dim_reduce_type: str = "linear",
              criterion: str = "BCEWithLogitsLoss",
              max_learning_rate: float = 1e-4,
              total_epochs: int = 300,
              train_epochs: int = 100,
              top_k: list = None,
              save_checkpoints: bool = True,

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
        train_dataset = LastHiddenStatesDataset(
            tensor_path,
            dataset_path,
            dataset_version,
            dataset_name,
            tokens_type,
            llm_model,
            train_validation_indices,
            dataset_type="train"
        )

        val_dataset = LastHiddenStatesDataset(
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

        llm_info = LLMInfo(llm_model)

        model = LocalizationTransformer(
            num_layers_projection,
            num_layers_encoder,
            num_layers_dim_reduce,
            llm_info.get_hidden_size(),
            num_head,
            target_dim,
            dim_reduce_type,
            seed,
            device
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

        optimizer = Adam(model.parameters(), lr=max_learning_rate)

        num_training_steps = len(train_loader) * total_epochs
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% of total steps as warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        start_epoch = 0
        end_epoch = train_epochs
        metric_track = {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0, 'val_loss': 1e10}

        # get the latest model checkpoint if exists
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))]
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

        if start_epoch >= end_epoch:
            print(f"Nothing to train.  start_epoch: {start_epoch}  end_epoch: {end_epoch}")
            return

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

            train_top_k_sums = {}
            if top_k is not None:
                for k in top_k:
                    train_top_k_sums[f'top{k}'] = 0
            train_snippet_count = 0

            train_outputs_for_epoch = []

            model.train()
            running_train_loss = 0.0
            for step, (last_hidden_state, code_tokens_length, line_split_lengths, line_labels) in enumerate(
                    train_loader):
                train_step += 1

                last_hidden_state = last_hidden_state.to(device)  # [batch_size, seq_len, hidden_size]
                code_tokens_length = code_tokens_length.to(device)  # [batch_size]
                line_split_lengths = line_split_lengths.to(device)  # [batch_size, MAX_CODE_LINES]
                line_labels = line_labels.to(device)  # [batch_size, MAX_CODE_LINES]

                optimizer.zero_grad()

                # Forward pass
                outputs = model(last_hidden_state, code_tokens_length,
                                line_split_lengths)  # outputs shape: [batch_size, MAX_CODE_LINES]

                # We must ignore lines where label is -1
                mask = (line_labels != -1)  # Boolean mask for valid lines
                valid_outputs = outputs[mask]  # shape: [N_valid_lines]
                valid_labels = line_labels[mask]  # shape: [N_valid_lines]
                valid_labels = valid_labels.float()

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
                batch_size_local = line_labels.shape[0]
                for i in range(batch_size_local):
                    # valid lines for snippet i
                    snippet_mask = (line_labels[i] != -1)
                    snippet_outputs = outputs[i][snippet_mask].tolist()
                    snippet_labels = line_labels[i][snippet_mask].tolist()
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

                # Calculate top-k metric
                if top_k is not None:
                    batch_size_local = line_labels.shape[0]
                    for i in range(batch_size_local):
                        # valid lines for snippet i
                        snippet_mask = (line_labels[i] != -1)
                        snippet_outputs = outputs[i][snippet_mask]
                        snippet_labels = line_labels[i][snippet_mask]

                        # Convert snippet_labels to long
                        snippet_labels = snippet_labels.long()

                        # Compute top1, top3, top5 success
                        snippet_top_k = self.compute_topk_metrics(snippet_outputs, snippet_labels, top_k=top_k)
                        for k_str, val in snippet_top_k.items():
                            train_top_k_sums[k_str] += val

                        train_snippet_count += 1

            train_outputs[(epoch + 1)] = train_outputs_for_epoch
            epoch_train_loss = running_train_loss / len(train_loader)
            train_metrics = self.compute_metrics_from_confusion(train_tp, train_fp, train_tn, train_fn)
            self.write_metrics_to_log_writer(train_metrics, log_writer, 'train', (epoch + 1))
            log_writer.add_scalar('loss/epoch/train', epoch_train_loss, (epoch + 1))
            if top_k is not None:
                self.write_top_k_metrics_to_log_writer(train_top_k_sums, train_snippet_count, top_k, log_writer,
                                                       'train', (epoch + 1))
            print(f"  [Training]  Loss: {epoch_train_loss:.8f}  fold: {fold_index}  configuration: {exp_config}")

            ####################################### Validation #######################################
            model.eval()
            running_val_loss = 0.0

            val_tp = 0  # true positives
            val_fp = 0  # false positives
            val_tn = 0  # true negatives
            val_fn = 0  # false negatives

            val_top_k_sums = {}
            if top_k is not None:
                for k in top_k:
                    val_top_k_sums[f'top{k}'] = 0
            val_snippet_count = 0

            val_outputs_for_epoch = []

            with torch.no_grad():
                for step, (last_hidden_state, code_tokens_length, line_split_lengths, line_labels) in enumerate(
                        val_loader):
                    val_step += 1

                    last_hidden_state = last_hidden_state.to(device)
                    code_tokens_length = code_tokens_length.to(device)
                    line_split_lengths = line_split_lengths.to(device)
                    line_labels = line_labels.to(device)

                    outputs = model(last_hidden_state, code_tokens_length, line_split_lengths)

                    mask = (line_labels != -1)
                    valid_outputs = outputs[mask]
                    valid_labels = line_labels[mask].float()

                    loss = loss_fn(valid_outputs, valid_labels)
                    running_val_loss += loss.item()

                    log_writer.add_scalar('loss/step/val', loss.item(), val_step)

                    # Save the output of the validation step
                    batch_size_local = line_labels.shape[0]
                    for i in range(batch_size_local):
                        # valid lines for snippet i
                        snippet_mask = (line_labels[i] != -1)
                        snippet_outputs = outputs[i][snippet_mask].tolist()
                        snippet_labels = line_labels[i][snippet_mask].tolist()
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

                    if top_k is not None:
                        batch_size_local = line_labels.shape[0]
                        for i in range(batch_size_local):
                            # valid lines for snippet i
                            snippet_mask = (line_labels[i] != -1)
                            snippet_outputs = outputs[i][snippet_mask]
                            snippet_labels = line_labels[i][snippet_mask]

                            # Convert snippet_labels to long
                            snippet_labels = snippet_labels.long()

                            # Compute top1, top3, top5 success
                            snippet_top_k = self.compute_topk_metrics(snippet_outputs, snippet_labels, top_k=top_k)
                            for k_str, val in snippet_top_k.items():
                                val_top_k_sums[k_str] += val

                            val_snippet_count += 1

            val_outputs[(epoch + 1)] = val_outputs_for_epoch
            epoch_val_loss = running_val_loss / len(val_loader)
            val_metrics = self.compute_metrics_from_confusion(val_tp, val_fp, val_tn, val_fn)
            self.write_metrics_to_log_writer(val_metrics, log_writer, 'val', (epoch + 1))
            log_writer.add_scalar('loss/epoch/val', epoch_val_loss, (epoch + 1))
            if top_k is not None:
                self.write_top_k_metrics_to_log_writer(val_top_k_sums, val_snippet_count, top_k, log_writer, 'val',
                                                       (epoch + 1))
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


class NumLayerConfig(Enum):
    DIM1_ENC2_PROJ1 = {'num_layers_projection': 1, 'num_layers_encoder': 2,
                       'num_layers_dim_reduce': 1}  # layer_conf = 1
    DIM2_ENC3_PROJ2 = {'num_layers_projection': 2, 'num_layers_encoder': 3,
                       'num_layers_dim_reduce': 2}  # layer_conf = 2
    DIM3_ENC4_PROJ2 = {'num_layers_projection': 2, 'num_layers_encoder': 4,
                       'num_layers_dim_reduce': 3}  # layer_conf = 3

    @staticmethod
    def get_layer_conf(layer_conf: int = 1):
        match layer_conf:
            case 1:
                return NumLayerConfig.DIM1_ENC2_PROJ1
            case 2:
                return NumLayerConfig.DIM2_ENC3_PROJ2
            case 3:
                return NumLayerConfig.DIM3_ENC4_PROJ2


class ConfigFactory:
    def __init__(self, exp_config: str,
                 dataset_version: str,
                 dataset_name: str,
                 llm_models_list: list,
                 layer_conf: int,
                 target_dim_list: list,
                 dim_reduce_type: str,
                 max_learning_rate_list: list,
                 criterion: str):
        if layer_conf not in [1, 2, 3]:
            raise Exception(f"Incorrect value for layer_conf parameter: {layer_conf}")

        if len(llm_models_list) != len(max_learning_rate_list):
            raise Exception(f"The llm_models_list and the max_learning_rate_list should be of the same length")

        if dim_reduce_type not in ["linear", "lstm", 'gru']:
            raise Exception(f"Incorrect value for dim_reduce_type parameter: {dim_reduce_type}")

        if criterion not in ["BCEWithLogitsLoss", "CustomExponential", "CustomCosine", "CustomCosineLinear"]:
            raise Exception(f"Incorrect value for criterion parameter: {criterion}")

        self.config = Config()

        self.layer_conf = layer_conf

        self.config.dataset_version = dataset_version
        self.config.dataset_name = dataset_name

        layer_conf = NumLayerConfig.get_layer_conf(layer_conf)
        self.config.num_layers_projection = layer_conf.value['num_layers_projection']
        self.config.num_layers_encoder = layer_conf.value['num_layers_encoder']
        self.config.num_layers_dim_reduce = layer_conf.value['num_layers_dim_reduce']

        self.config.dim_reduce_type = dim_reduce_type
        self.config.criterion = criterion  # BCEWithLogitsLoss or Custom

        self.llm_models_list = llm_models_list
        self.target_dim_list = target_dim_list
        self.max_learning_rate_list = max_learning_rate_list

        self.exp_config = exp_config
        self.dim_reduce_type = dim_reduce_type
        self.criterion = criterion
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version

    def get_generated_configs(self):
        configurations = []
        for llm_model in self.llm_models_list:
            index = self.llm_models_list.index(llm_model)
            for target_dim in self.target_dim_list:
                for fold_index in range(self.config.k):
                    new_config = copy.deepcopy(self.config)
                    new_config.fold_index = fold_index
                    if self.criterion == "BCEWithLogitsLoss":
                        new_config.exp_config = f'{self.exp_config}_{self.dataset_name}_{self.dataset_version}_{llm_model.value}_{self.dim_reduce_type}_{target_dim}_{self.layer_conf}_BCE'
                    elif self.criterion == "CustomExponential":
                        new_config.exp_config = f'{self.exp_config}_{self.dataset_name}_{self.dataset_version}_{llm_model.value}_{self.dim_reduce_type}_{target_dim}_{self.layer_conf}_CE'
                    elif self.criterion == "CustomCosine":
                        new_config.exp_config = f'{self.exp_config}_{self.dataset_name}_{self.dataset_version}_{llm_model.value}_{self.dim_reduce_type}_{target_dim}_{self.layer_conf}_CC'
                    elif self.criterion == "CustomCosineLinear":
                        new_config.exp_config = f'{self.exp_config}_{self.dataset_name}_{self.dataset_version}_{llm_model.value}_{self.dim_reduce_type}_{target_dim}_{self.layer_conf}_CCL'
                    else:
                        raise Exception(f"Un-defined criterion: {self.criterion}")
                    new_config.llm_model = llm_model
                    new_config.target_dim = target_dim
                    new_config.max_learning_rate = self.max_learning_rate_list[index]
                    configurations.append(new_config)
        if len(configurations) < (len(self.llm_models_list) * len(self.target_dim_list) * self.config.k):
            raise Exception("Error! No configurations generated!")
        return configurations


class Config:
    def __init__(self,
                 exp_config: str = None,
                 outputs_path: str = None,
                 tensor_path: str = None,
                 dataset_path: str = None,
                 dataset_version: str = None,
                 dataset_name: str = None,
                 tokens_type: str = None,
                 llm_model: Enum = LLMModels.CODEGEN_350M_MULTI,
                 k: int = 10,
                 seed: int = 42,
                 fold_index: int = 0,
                 batch_size: int = 8,
                 device: str = "cuda:0",
                 num_layers_projection: int = 2,
                 num_layers_encoder: int = 2,
                 num_layers_dim_reduce: int = 2,
                 num_head: int = 16,
                 target_dim: int = 1024,
                 dim_reduce_type: str = "linear",
                 criterion: str = "BCEWithLogitsLoss",
                 max_learning_rate: float = 1e-4,
                 total_epochs: int = 300,
                 train_epochs: int = 100,
                 top_k: list = None,
                 save_checkpoints: bool = True):
        self.exp_config = exp_config
        self.outputs_path = outputs_path
        self.tensor_path = tensor_path
        self.dataset_path = dataset_path
        self.dataset_version = dataset_version
        self.dataset_name = dataset_name
        self.tokens_type = tokens_type
        self.llm_model = llm_model
        self.k = k
        self.seed = seed
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.device = device
        self.num_layers_projection = num_layers_projection
        self.num_layers_encoder = num_layers_encoder
        self.num_layers_dim_reduce = num_layers_dim_reduce
        self.num_head = num_head
        self.target_dim = target_dim
        self.dim_reduce_type = dim_reduce_type
        self.criterion = criterion
        self.max_learning_rate = max_learning_rate
        self.total_epochs = total_epochs
        self.train_epochs = train_epochs
        self.top_k = top_k
        self.save_checkpoints = save_checkpoints

        self.set_paths()
        self.set_some_non_changing_values()

    def set_paths(self):
        self.outputs_path = 'outputs'
        self.tensor_path = 'data/tensors'
        self.dataset_path = 'data/dataset'

    def set_some_non_changing_values(self):
        self.tokens_type = 'prompt'
        self.k = 10
        self.seed = 42
        self.batch_size = 8
        self.device = "cuda:0"
        self.total_epochs = 300
        self.train_epochs = 100
        self.top_k = [1, 3, 5]
        self.save_checkpoints = True
        if self.target_dim % 64 != 0:
            raise Exception("The target dimension must be divisible by 64")
        self.num_head = self.target_dim // 64
