# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu, Zetao Hu)
#               2025 Alibaba Inc (authors: Xiang Lyu, Yabin Li)
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
import os
import torch
import logging
import torchaudio

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(message)s')


def read_lists(list_file):
    with open(list_file, encoding='utf-8') as f:
        return [line.strip() for line in f]


def load_wav(wav, target_sr=16000):
    speech, sample_rate = torchaudio.load(wav, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        assert sample_rate > target_sr, f'wav sample rate {sample_rate} must be greater than {target_sr}'
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech


def convert_onnx_to_trt(trt_model, trt_kwargs, onnx_model, fp16):
    import tensorrt as trt
    logging.info("Converting ONNX to TensorRT...")
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(network_flags)
    # parser
    parser = trt.OnnxParser(network, logger)
    with open(onnx_model, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('failed to parse {}'.format(onnx_model))
    # config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32) # 4GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    # profile
    profile = builder.create_optimization_profile()
    for i in range(len(trt_kwargs['input_names'])):
        profile.set_shape(
            trt_kwargs['input_names'][i], trt_kwargs['min_shape'][i],
            trt_kwargs['opt_shape'][i], trt_kwargs['max_shape'][i])
    config.add_optimization_profile(profile)
    # set input and output data type
    tensor_dtype = trt.DataType.HALF if fp16 else trt.DataType.FLOAT
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_tensor.dtype = tensor_dtype
    for i in range(network.num_outputs):
        output_tensor = network.get_output(i)
        output_tensor.dtype = tensor_dtype
    # save trt engine
    engine_bytes = builder.build_serialized_network(network, config)
    with open(trt_model, "wb") as f:
        f.write(engine_bytes)
    logging.info("Successfully converted ONNX to TensorRT!")


def export_cosyvoice2_vllm(model, model_path, device):
    if os.path.exists(model_path):
        return
    # tmp
    embed_tokens = model.llm.model.model.embed_tokens
    tmp_vocab_size = model.llm.model.config.vocab_size
    tmp_tie_embedding = model.llm.model.config.tie_word_embeddings
    # pad
    pad_to = DEFAULT_VOCAB_PADDING_SIZE = 64
    vocab_size = model.speech_embedding.num_embeddings
    feature_size = model.speech_embedding.embedding_dim
    pad_vocab_size = ((vocab_size + pad_to - 1) // pad_to) * pad_to
    # lm_head
    new_lm_head = torch.nn.Linear(in_features=feature_size, out_features=pad_vocab_size)
    with torch.no_grad():
        new_lm_head.weight[:vocab_size] = model.llm_decoder.weight
        new_lm_head.bias[:vocab_size] = model.llm_decoder.bias
        new_lm_head.weight[vocab_size:] = 0
        new_lm_head.bias[vocab_size:] = 0
    model.llm.model.lm_head = new_lm_head
    # lm_embed
    new_lm_embed = torch.nn.Linear(in_features=feature_size, out_features=pad_vocab_size)
    with torch.no_grad():
        new_lm_embed.weight[:vocab_size] = model.speech_embedding.weight
        new_lm_embed.weight[vocab_size:] = 0
    model.llm.model.set_input_embeddings(new_lm_embed)
    model.llm.model.to(torch.bfloat16).to(device)
    # del
    del model.llm.model.config.bos_token_id
    del model.llm.model.config.eos_token_id
    del model.llm.model.generation_config.eos_token_id
    # set
    model.llm.model.config.use_bias = True
    model.llm.model.config.tie_word_embeddings = False
    model.llm.model.config.vocab_size = pad_vocab_size
    # save
    model.llm.model.save_pretrained(model_path)
    os.system(f"sed -i s@Qwen2ForCausalLM@CosyVoice2ForCausalLM@g \
              {os.path.abspath(model_path)}/config.json")
    # restore
    model.llm.model.config.tie_word_embeddings = tmp_tie_embedding
    model.llm.model.config.vocab_size = tmp_vocab_size
    model.llm.model.set_input_embeddings(embed_tokens)
