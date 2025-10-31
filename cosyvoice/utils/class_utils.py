# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
#            2024 Alibaba Inc (authors: Xiang Lyu)
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
import torch
from cosyvoice.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from cosyvoice.transformer.embedding import (
    PositionalEncoding,
    NoPositionalEncoding,
    RelPositionalEncoding,
    WhisperPositionalEncoding,
    EspnetRelPositionalEncoding,
    LearnablePositionalEncoding,
)
from cosyvoice.transformer.subsampling import (
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    LinearNoSubsampling,
    EmbedinigNoSubsampling,
    LegacyLinearNoSubsampling,
)
from cosyvoice.transformer.activation import Swish
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.flow.flow import MaskedDiffWithXvec, CausalMaskedDiffWithXvec
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.llm.llm import TransformerLM, Qwen2LM


COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    'paraformer_dummy': torch.nn.Identity,
    "embed": EmbedinigNoSubsampling,
}

COSYVOICE_ACTIVATION_CLASSES = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "tanh": torch.nn.Tanh,
    "hardtanh": torch.nn.Hardtanh,
    "swish": getattr(torch.nn, "SiLU", Swish),
}


def get_model_type(configs):
    if isinstance(configs['llm'], TransformerLM) \
    and isinstance(configs['hift'], HiFTGenerator) \
    and isinstance(configs['flow'], MaskedDiffWithXvec):
        return CosyVoiceModel
    if isinstance(configs['llm'], Qwen2LM) \
    and isinstance(configs['hift'], HiFTGenerator) \
    and isinstance(configs['flow'], CausalMaskedDiffWithXvec):
        return CosyVoice2Model
    raise TypeError('No valid model type found!')
