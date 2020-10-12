# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from .file_utils import cached_path, WEIGHTS_NAME, CONFIG_NAME
import torch.nn.functional as F
import random
from math import sqrt

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    # 'bert-base-uncased': "E:\CS\Bert-pretrain-model/bert-base-uncased.tar.gz",
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    # 'bert-base-chinese': "E:\CS\Bert-pretrain-model/bert-base-chinese.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertEmbeddingsOrder(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddingsOrder, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.order_embeddings = nn.Embedding(22, config.hidden_size)  ##最多轮数才19
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, turn_order_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        turn_order_ids = torch.fmod(turn_order_ids,2)
        order_embeddings = self.order_embeddings(turn_order_ids)  # [batch,len, hidden_size]

        embeddings = words_embeddings + position_embeddings + token_type_embeddings + order_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states) # batch, seq_len, hidden_size
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) # batch, num_heads, seq_len, head_dim
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)  # [batch, len, hidden_size]
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class BertForPreTraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            #total_loss = next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertForMaskedLM(BertPreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class BertForNextSentencePrediction(BertPreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls( pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        #self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # batch,heads,length,head_size

    def forward(self, hidden_states, comp_states, attention_mask=None, comp_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(comp_states)
        mixed_value_layer = self.value(comp_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + comp_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        #outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        outputs = context_layer
        return outputs






def masked_softmax(vector: torch.Tensor,
                   mask: torch.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> torch.Tensor:
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        #mask = mask.half()

        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def seperate_seq(sequence_output, context_len, response_len):
    max_context_len = torch.max(context_len.view(-1))
    max_response_len = torch.max(response_len.view(-1))

    context_output = sequence_output.new(sequence_output.size(0), context_len.size(1), max_context_len, sequence_output.size(2)).zero_()
    response_output = sequence_output.new(sequence_output.size(0), max_response_len, sequence_output.size(2)).zero_()

    for i in range(context_len.size(0)):
        response_output[i, :response_len[i]] = sequence_output[i, :response_len[i]]
        current_loc = response_len[i]
        for j in range(context_len.size(1)):
            if context_len[i][j] == -1:
                break
            else:
                c_len = context_len[i][j]
                context_output[i,j, :c_len] = sequence_output[i, current_loc + 1:current_loc + c_len + 1]
                current_loc += c_len
    return context_output, response_output

def seperate_seq_mask(sequence_output, context_len, response_len):
    max_context_len = torch.max(context_len.view(-1))
    max_response_len = torch.max(response_len.view(-1))

    context_output = sequence_output.new(sequence_output.size(0), context_len.size(1), max_context_len, sequence_output.size(2)).zero_()
    response_output = sequence_output.new(sequence_output.size(0), max_response_len, sequence_output.size(2)).zero_()
    response_output_mask = sequence_output.new(sequence_output.size(0), max_response_len).zero_()

    for i in range(context_len.size(0)):
        response_output[i, :response_len[i]] = sequence_output[i, :response_len[i]]
        response_output_mask[i, :response_len[i]] = 1
        current_loc = response_len[i]
        for j in range(context_len.size(1)):
            if context_len[i][j] == -1:
                break
            else:
                c_len = context_len[i][j]
                context_output[i,j, :c_len] = sequence_output[i, current_loc + 1:current_loc + c_len + 1]
                current_loc += c_len
    return context_output, response_output, response_output_mask

def seperate_response(sequence_output, response_len):
    max_response_len = torch.max(response_len.view(-1))
    response_output = sequence_output.new(sequence_output.size(0), max_response_len, sequence_output.size(2)).zero_()
    response_output_mask = sequence_output.new(sequence_output.size(0), max_response_len).zero_()

    for i in range(response_len.size(0)):
        response_output[i, :response_len[i]] = sequence_output[i, :response_len[i]]
        response_output_mask[i, :response_len[i]] = 1
    return response_output, response_output_mask


def seperate_context_response(sequence_output, response_len, attention_mask):
    max_response_len = torch.max(response_len.view(-1))
    response_output = sequence_output.new(sequence_output.size(0), max_response_len, sequence_output.size(2)).zero_()
    response_output_mask = sequence_output.new(sequence_output.size(0), max_response_len).zero_()
    context_output_mask = attention_mask.clone()
    context_output = sequence_output.clone()

    for i in range(response_len.size(0)):
        response_output[i, :response_len[i]] = sequence_output[i, :response_len[i]]
        response_output_mask[i, :response_len[i]] = 1
        context_output[i, :response_len[i]] = 0
        context_output_mask[i, :response_len[i]] = 0

    return response_output, response_output_mask, context_output, context_output_mask

def resh(s1, s2):
    a = s1
    b = s2

    ca = a.repeat(b.size(0), 1)
    ca = ca.view(b.size(0), a.size(0), -1)

    cb = b.repeat(1, a.size(0))
    cb = cb.view(b.size(0), a.size(0), -1)
    # print(cca.size(), ccb.size())
    cos = torch.nn.CosineSimilarity(2)
    res = cos(ca, cb)
    return res

def get_score(sentence, question):#sentence: sentence_len*768 question: question_len*768
    result_sent_ques = resh(sentence, question) # question.size(0)* sentence.size(0)
    res_ques, _ = result_sent_ques.max(1)

    # result = np.sum(heapq.nlargest(2,res_ques)) + np.sum(heapq.nlargest(2,res_option))
    result = torch.sum(res_ques)/res_ques.size(0)
    # result = torch.sum(res_ques) / res_ques.size(0) + torch.sum(res_option) / res_option.size(0)
    # result = torch.sum(sent1)/sent1.size(0) + torch.sum(sent2)/sent2.size(0)
    return result

class DotAtt(nn.Module):
    def __init__(self, config):
        super(DotAtt, self).__init__()
        self.trans_linear_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.trans_linear = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, proj_p, proj_q, mask): # proj_p [batch, all_len, hidden_size] proj_q [batch,response_length,hidden_size]
        trans_q = self.trans_linear_q(proj_q)   # trans_q [batch,response_length,hidden_size]
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2) )
        att_norm = masked_softmax(att_weights, mask) #[b,all_len, response_len]
        att_vec = att_norm.bmm(proj_q) #
        #output = nn.ReLU()(self.trans_linear(att_vec))
        output = self.trans_linear(att_vec)
        return output


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


class CNN_2D(nn.Module):

    def __init__(self,dim):
        super(CNN_2D, self).__init__()
        D = 768
        Ci = 1
        Ks = [2,3,4]
        Co = int(dim / len(Ks))
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks]) #in_channels, out_channels, kernel_size, stride=1
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D) batch,1,turn_num,h
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        return x

class CNN_Text(nn.Module):

    def __init__(self,dim):
        super(CNN_Text, self).__init__()
        D = 768   #in_channels, out_channels, kernel_size, stride=1
        Ci = 1
        Ks = [3,4,5]
        Co = int(dim / len(Ks) / 4)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(0.2)
        # self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        return x

class BertMHAttention(nn.Module):
    def __init__(self, config):
        super(BertMHAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        # self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, context_states, query_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        mixed_query_layer = self.query(query_states)

        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        attention_mask=extended_attention_mask

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(context_states)
            mixed_value_layer = self.value(context_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        outputs = context_layer
        return outputs

class FuseNet(nn.Module):
    def __init__(self, config):
        super(FuseNet, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(2*config.hidden_size, 2*config.hidden_size)

    def forward(self, inputs):
        p, q = inputs
        lq = self.linear(q)
        lp = self.linear(p)
        mid = nn.Sigmoid()(lq+lp)
        output = p * mid + q * (1-mid)
        return output

class FuseConcat(nn.Module):
    def __init__(self, config):
        super(FuseConcat, self).__init__()
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)

    def forward(self, inputs):
        p, q = inputs
        output = torch.cat([p,q], dim=2)
        output = nn.Tanh()(self.linear(output))
        return output


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class FE_DAtt_RNN(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(FE_DAtt_RNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.SAttention = DotAtt(config)
        # self.cnn = CNN_2D(config.hidden_size)
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size, bidirectional=False,
                          batch_first=True)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        turn_num = input_ids.size(1)
        turn_length = input_ids.size(2)
        new_input_ids = input_ids.view(batch_size*turn_num, turn_length) # turn_num include response
        new_token_type_ids = token_type_ids.view(batch_size*turn_num, turn_length)
        # attention_mask  [batch, turn_num. len] to turn_num [batch, real_num]
        turn_num_sum = attention_mask.sum(dim=-1)
        real_turn_num = ~turn_num_sum.eq(0)[:, 1:]
        real_turn_num = real_turn_num.sum(dim=1)

        new_attention_mask = attention_mask.view(batch_size*turn_num, turn_length)
        sequence_encode, pooled_output = self.bert(new_input_ids, new_token_type_ids, new_attention_mask, output_all_encoded_layers=False)
        all_turns_encode = sequence_encode.view(batch_size,turn_num,turn_length,-1)
        response = all_turns_encode[:,0,:,:] # batch,turn_len,h
        context = all_turns_encode[:,1:,:,:].contiguous() # batch,,turn_num,turn_len,h
        response_mask = attention_mask[:,0] # batch, turn_len
        response = response.unsqueeze(1).expand(batch_size, turn_num-1, turn_length, response.size(-1)).contiguous()
        response_mask = response_mask.unsqueeze(1).expand(batch_size, turn_num-1, turn_length).contiguous()

        context_att = context.view(-1,turn_length,context.size(-1))  # batch_size*(turn_num-1). turn_length, hidden
        response_att = response.view(-1,turn_length,context.size(-1))
        response_mask_att = response_mask.view(-1,turn_length)
        sequence_output = self.SAttention(context_att, response_att, response_mask_att) # batch_size*(turn_num-1). turn_length, hidden

        sequence_output = sequence_output.view(batch_size, turn_num-1,turn_length, context_att.size(2))# batch_size, turn_num-1. turn_length, hidden
        sequence_output_pool,_ = sequence_output.max(2) # batch, turn_num-1, hidden ## max改成mean试试
        #fused_pool = self.cnn(sequence_output_pool)
        #sequence_output_pool = sequence_output.mean(2) # batch, turn_num-1, hidden   然后也试试仅用cls表示这个句子

        # gru
        sorted_lens, len_idx_sorted = torch.sort(real_turn_num, dim=0, descending=True)
        _, len_idx_original = torch.sort(len_idx_sorted, dim=0)
        sorted_embed = sequence_output_pool.index_select(0, len_idx_sorted)
        packed_sorted_embed = nn.utils.rnn.pack_padded_sequence(sorted_embed, sorted_lens, batch_first=True)
        encode_context, last_hidden = self.gru(packed_sorted_embed)
        last_hidden = last_hidden.squeeze()
        output_hidden = last_hidden.index_select(0, len_idx_original)

        # _, last_hidden = self.gru(sequence_output_pool)
        # last_hidden = torch.squeeze(last_hidden)

        fused_pool = output_hidden
        pooled_fused_output = self.dropout(fused_pool)
        logits = self.classifier(pooled_fused_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class IE_MHAtt_CNN(BertPreTrainedModel):
    def __init__(self, config, num_labels, max_turn_num):
        super(IE_MHAtt_CNN, self).__init__(config, max_turn_num)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.apply(self.init_bert_weights)
        #self.SAttention = SingleMatchNet(config)
        self.mhatt = BertMHAttention(config)
        self.cnn = CNN_2D(config.hidden_size)
        self.max_turn_num = max_turn_num

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, response_len=None, sep_pos=None,context_len=None, labels=None):
        sequence_output, sequence_pool = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch_size, _, dim = sequence_output.size()
        batch_max_turn_len, _ = context_len.max(1)
        max_turn_len,_ = batch_max_turn_len.max(0)
        max_turn_len = max_turn_len.item()

        # zero for padding
        sequence_output = torch.cat([sequence_output.new_zeros((batch_size, 1, dim)), sequence_output], dim=1)

        utteraces = torch.zeros(batch_size, self.max_turn_num, max_turn_len, dim).cuda()
        utteraces_mask = torch.zeros(batch_size, self.max_turn_num, max_turn_len).cuda()
        response = torch.zeros(batch_size, max_turn_len, dim).cuda()
        response_mask = torch.zeros(batch_size, max_turn_len).cuda()

        for batch_idx in range(batch_size):
            response_start = 1
            response_end = sep_pos[batch_idx][0] #offset end with [sep]
            response[batch_idx][:response_end] = sequence_output[batch_idx][response_start:(response_end+1)]
            response_mask[batch_idx][:response_end] = 1
            for turn_idx in range(len(sep_pos[batch_idx])-1):
                context_start = sep_pos[batch_idx][turn_idx] + 1 #+1 here is the previous [sep] + 1
                context_end =  sep_pos[batch_idx][turn_idx+1] + 1 ###################### here with sep
                if context_end != 1:
                    utteraces[batch_idx][turn_idx][:context_end-context_start] = sequence_output[batch_idx][context_start:context_end]
                    utteraces_mask[batch_idx][turn_idx][:context_end-context_start] = 1

        response = response.unsqueeze(1).expand(batch_size, utteraces.size(1), response.size(1), response.size(2)).contiguous()
        response_mask = response_mask.unsqueeze(1).expand(batch_size, utteraces.size(1), response_mask.size(1)).contiguous()

        all_utterances = utteraces.view(-1, utteraces.size(2), utteraces.size(3))
        utteraces_mask = utteraces_mask.view(-1, utteraces_mask.size(2))
        response = response.view(-1, response.size(2),response.size(3))
        response_mask = response_mask.view(-1, response_mask.size(2))
        # response-aware attention
        # BERT multi-head attention K/V, Q
        utterance_att = self.mhatt(response, all_utterances, response_mask)  # batch*turn_num, turn_len, hidden

        utterance_att = utterance_att.view(batch_size, utteraces.size(1), utteraces.size(2),dim) # batch, turn_num,turn_len, hidden
        # meanpooling
        mean_pooled_att = utterance_att.mean(2) # batch, turn_num, hidden
        # # CNN and maxpooling
        c_pooled_output = self.cnn(mean_pooled_att) # batch, hidden

        fused_pool = torch.cat([sequence_pool, c_pooled_output],dim=1)
        pooled_output = self.dropout(fused_pool)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class IE_CoAtt_CNN(BertPreTrainedModel):
    def __init__(self, config, num_labels, max_turn_num):
        super(IE_CoAtt_CNN, self).__init__(config, max_turn_num)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.apply(self.init_bert_weights)
        self.SAttention = DotAtt(config)
        self.cnn = CNN_2D(config.hidden_size)
        self.fusecat = FuseConcat(config)
        self.max_turn_num = max_turn_num

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, response_len=None, sep_pos=None,context_len=None, labels=None):
        sequence_output, sequence_pool = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch_size, _, dim = sequence_output.size()
        batch_max_turn_len, _ = context_len.max(1)
        max_turn_len,_ = batch_max_turn_len.max(0)
        max_turn_len = max_turn_len.item()

        # zero for padding
        sequence_output = torch.cat([sequence_output.new_zeros((batch_size, 1, dim)), sequence_output], dim=1)

        utteraces = torch.zeros(batch_size, self.max_turn_num, max_turn_len, dim).cuda()
        utteraces_mask = torch.zeros(batch_size, self.max_turn_num, max_turn_len).cuda()
        response = torch.zeros(batch_size, max_turn_len, dim).cuda()
        response_mask = torch.zeros(batch_size, max_turn_len).cuda()

        for batch_idx in range(batch_size):
            response_start = 1
            response_end = sep_pos[batch_idx][0] #offset end with [sep]
            response[batch_idx][:response_end] = sequence_output[batch_idx][response_start:(response_end+1)]
            response_mask[batch_idx][:response_end] = 1
            for turn_idx in range(len(sep_pos[batch_idx])-1):
                context_start = sep_pos[batch_idx][turn_idx] + 1 #+1 here is the previous [sep] + 1
                context_end =  sep_pos[batch_idx][turn_idx+1] + 1 ###################### here with sep
                if context_end != 1:
                    utteraces[batch_idx][turn_idx][:context_end-context_start] = sequence_output[batch_idx][context_start:context_end]
                    utteraces_mask[batch_idx][turn_idx][:context_end-context_start] = 1

        response = response.unsqueeze(1).expand(batch_size, utteraces.size(1), response.size(1), response.size(2)).contiguous()
        response_mask = response_mask.unsqueeze(1).expand(batch_size, utteraces.size(1), response_mask.size(1)).contiguous()

        all_utterances = utteraces.view(-1, utteraces.size(2), utteraces.size(3))
        utteraces_mask = utteraces_mask.view(-1, utteraces_mask.size(2))
        response = response.view(-1, response.size(2),response.size(3))
        response_mask = response_mask.view(-1, response_mask.size(2))

        # co-attention
        utterance_att = self.SAttention(all_utterances, response, response_mask) # batch*turn_num, turn_len, hidden
        response_att = self.SAttention(response, all_utterances, utteraces_mask)
        utterance_att = self.dropout(utterance_att)
        response_att = self.dropout(response_att)
        utterance_att = utterance_att.view(batch_size, utteraces.size(1), utteraces.size(2),dim) # batch, turn_num,turn_len, hidden
        response_att = response_att.view(batch_size, utteraces.size(1), utteraces.size(2),dim)

        utterance_pool = utterance_att.mean(dim=2) # batch, turn_num, hidden
        response_pool = response_att.mean(dim=2)
        sequence_output = self.fusecat([utterance_pool, response_pool])  # batch, turn_num, hidden

        output_hidden = self.cnn(sequence_output)

        fused_pool = torch.cat([sequence_pool, output_hidden],dim=1)
        pooled_output = self.dropout(fused_pool)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class IE_DAtt_RNN(BertPreTrainedModel):
    def __init__(self, config, num_labels, max_turn_num):
        super(IE_DAtt_RNN, self).__init__(config, max_turn_num)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.apply(self.init_bert_weights)
        self.SAttention = DotAtt(config)
        #self.cnn = CNN_2D(config.hidden_size)
        self.gru = nn.GRU(self.config.hidden_size, self.config.hidden_size, bidirectional=False,
                          batch_first=True)
        self.max_turn_num = max_turn_num

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, response_len=None, sep_pos=None,context_len=None, labels=None):
        sequence_output, sequence_pool = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch_size, _, dim = sequence_output.size()
        batch_max_turn_len, _ = context_len.max(1)
        max_turn_len,_ = batch_max_turn_len.max(0)
        max_turn_len = max_turn_len.item()

        # zero for padding
        sequence_output = torch.cat([sequence_output.new_zeros((batch_size, 1, dim)), sequence_output], dim=1)

        utteraces = torch.zeros(batch_size, self.max_turn_num, max_turn_len, dim).cuda()
        utteraces_mask = torch.zeros(batch_size, self.max_turn_num, max_turn_len).cuda()
        response = torch.zeros(batch_size, max_turn_len, dim).cuda()
        response_mask = torch.zeros(batch_size, max_turn_len).cuda()

        for batch_idx in range(batch_size):
            response_start = 1
            response_end = sep_pos[batch_idx][0] #offset end with [sep]
            response[batch_idx][:response_end] = sequence_output[batch_idx][response_start:(response_end+1)]
            response_mask[batch_idx][:response_end] = 1
            for turn_idx in range(len(sep_pos[batch_idx])-1):
                context_start = sep_pos[batch_idx][turn_idx] + 1 #+1 here is the previous [sep] + 1
                context_end =  sep_pos[batch_idx][turn_idx+1] + 1 ###################### here with sep
                if context_end != 1:
                    utteraces[batch_idx][turn_idx][:context_end-context_start] = sequence_output[batch_idx][context_start:context_end]
                    utteraces_mask[batch_idx][turn_idx][:context_end-context_start] = 1

        response = response.unsqueeze(1).expand(batch_size, utteraces.size(1), response.size(1), response.size(2)).contiguous()
        response_mask = response_mask.unsqueeze(1).expand(batch_size, utteraces.size(1), response_mask.size(1)).contiguous()

        all_utterances = utteraces.view(-1, utteraces.size(2), utteraces.size(3))
        response = response.view(-1, response.size(2),response.size(3))
        response_mask = response_mask.view(-1, response_mask.size(2))
        # response-aware attention
        sequence_output = self.SAttention(all_utterances, response, response_mask)
        sequence_output = sequence_output.view(batch_size, utteraces.size(1), utteraces.size(2),dim) # batch, turn_num,turn_len, hidden
        # maxpooling
        max_pooled_att,_ = sequence_output.max(2) # batch, turn_num, hidden
        # CNN and maxpooling
        # c_pooled_output = self.cnn(max_pooled_att) # batch, hidden

        # gru改
        context_mask = ~sep_pos.eq(0)[:, 1:]
        real_turn_num = context_mask.sum(dim=1)  # batch, len
        sorted_lens, len_idx_sorted = torch.sort(real_turn_num, dim=0, descending=True)
        _, len_idx_original = torch.sort(len_idx_sorted, dim=0)
        sorted_embed = max_pooled_att.index_select(0, len_idx_sorted)
        packed_sorted_embed = nn.utils.rnn.pack_padded_sequence(sorted_embed, sorted_lens, batch_first=True)
        encode_context, last_hidden = self.gru(packed_sorted_embed)
        last_hidden = last_hidden.squeeze()
        output_hidden = last_hidden.index_select(0, len_idx_original)

        # _, last_hidden = self.gru(max_pooled_att)
        # last_hidden = torch.squeeze(last_hidden)    # batch, hidden

        fused_pool = torch.cat([sequence_pool, output_hidden],dim=1)
        pooled_output = self.dropout(fused_pool)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class IE_DAtt_CNN(BertPreTrainedModel):
    def __init__(self, config, num_labels, max_turn_num):
        super(IE_DAtt_CNN, self).__init__(config, max_turn_num)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size * 2, num_labels)
        self.apply(self.init_bert_weights)
        self.SAttention = DotAtt(config)
        self.cnn = CNN_2D(config.hidden_size)
        self.max_turn_num = max_turn_num

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, response_len=None, sep_pos=None,context_len=None, labels=None):
        sequence_output, sequence_pool = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        batch_size, _, dim = sequence_output.size()
        batch_max_turn_len, _ = context_len.max(1)
        max_turn_len,_ = batch_max_turn_len.max(0)
        max_turn_len = max_turn_len.item()

        # zero for padding
        sequence_output = torch.cat([sequence_output.new_zeros((batch_size, 1, dim)), sequence_output], dim=1)

        utteraces = torch.zeros(batch_size, self.max_turn_num, max_turn_len, dim).cuda()
        utteraces_mask = torch.zeros(batch_size, self.max_turn_num, max_turn_len).cuda()
        response = torch.zeros(batch_size, max_turn_len, dim).cuda()
        response_mask = torch.zeros(batch_size, max_turn_len).cuda()

        for batch_idx in range(batch_size):
            response_start = 1
            response_end = sep_pos[batch_idx][0] #offset end with [sep]
            response[batch_idx][:response_end] = sequence_output[batch_idx][response_start:(response_end+1)]
            response_mask[batch_idx][:response_end] = 1
            for turn_idx in range(len(sep_pos[batch_idx])-1):
                context_start = sep_pos[batch_idx][turn_idx] + 1 #+1 here is the previous [sep] + 1
                context_end =  sep_pos[batch_idx][turn_idx+1] + 1 ###################### here with sep
                if context_end != 1:
                    utteraces[batch_idx][turn_idx][:context_end-context_start] = sequence_output[batch_idx][context_start:context_end]
                    utteraces_mask[batch_idx][turn_idx][:context_end-context_start] = 1

        response = response.unsqueeze(1).expand(batch_size, utteraces.size(1), response.size(1), response.size(2)).contiguous()
        response_mask = response_mask.unsqueeze(1).expand(batch_size, utteraces.size(1), response_mask.size(1)).contiguous()

        all_utterances = utteraces.view(-1, utteraces.size(2), utteraces.size(3))
        response = response.view(-1, response.size(2),response.size(3))
        response_mask = response_mask.view(-1, response_mask.size(2))
        # response-aware attention
        sequence_output = self.SAttention(all_utterances, response, response_mask)
        sequence_output = sequence_output.view(batch_size, utteraces.size(1), utteraces.size(2),dim) # batch, turn_num,turn_len, hidden
        # maxpooling
        # max_pooled_att,_ = sequence_output.max(2) # batch, turn_num, hidden
        sequence_output_pool = sequence_output.mean(2)
        # CNN and maxpooling
        c_pooled_output = self.cnn(sequence_output_pool) # batch, hidden

        fused_pool = torch.cat([sequence_pool, c_pooled_output],dim=1)
        pooled_output = self.dropout(fused_pool)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits