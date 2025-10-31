# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import torch
import logging
from tqdm import tqdm
from typing import Generator
from modelscope import snapshot_download
from hyperpyyaml import load_hyperpyyaml

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.utils.class_utils import get_model_type
from cosyvoice.cli.model import CosyVoiceModel, CosyVoice2Model


class CosyVoice:
    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, trt_concurrent=1,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device=device)
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = f"{model_dir}/cosyvoice.yaml"
        if not os.path.isfile(hyper_yaml_path):
            raise ValueError(f"{hyper_yaml_path} not exist!")
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, \
            f"do not use {model_dir} for CosyVoice initialization!"
        self.instruct = True if "-Instruct" in model_dir else False
        self.sample_rate = configs["sample_rate"]
        self.default_spk2info_path = f"{model_dir}/spk2info.pt"
        if not os.path.isfile(self.default_spk2info_path):
            default_spk2info = None
            logging.info(f"{self.default_spk2info_path} not exist")
        else:
            default_spk2info = torch.load(self.default_spk2info_path, map_location=self.device)
        self.frontend = CosyVoiceFrontEnd(
            spk2info=default_spk2info,
            get_tokenizer=configs["get_tokenizer"],
            feat_extractor=configs["feat_extractor"],
            allowed_special=configs["allowed_special"],
            campplus_model_path=f"{model_dir}/campplus.onnx",
            speech_tokenizer_model_path=f"{model_dir}/speech_tokenizer_v1.onnx", device=self.device)
        if torch.cuda.is_available() is False:
            load_jit, load_trt, fp16 = False, False, False
            logging.warning("no cuda device, set load_jit/load_trt/fp16 to False")
        model_dtype = "fp16" if fp16 else "fp32"
        self.model = CosyVoiceModel(configs["llm"], configs["flow"], configs["hift"], fp16, self.device)
        self.model.load(f"{model_dir}/llm.pt", f"{model_dir}/flow.pt", f"{model_dir}/hift.pt")
        if load_jit:
            self.model.load_jit(
                llm_text_encoder_model=f"{model_dir}/llm.text_encoder.{model_dtype}.zip",
                llm_llm_model=f"{model_dir}/llm.llm.{model_dtype}.zip",
                flow_encoder_model=f"{model_dir}/flow.encoder.{model_dtype}.zip")
        if load_trt:
            self.model.load_trt(
                f"{model_dir}/flow.decoder.estimator.{model_dtype}.mygpu.plan",
                f"{model_dir}/flow.decoder.estimator.fp32.onnx",
                trt_concurrent=trt_concurrent, fp16=fp16)
        del configs
    
    def list_available_spks(self):
        return list(self.frontend.spk2info.keys())
    
    def switch_spk2info(self, spk2info_path=None):
        spk2info_path = spk2info_path or self.default_spk2info_path
        self.frontend.spk2info = torch.load(spk2info_path, map_location=self.device)
        logging.info(f"spk2info 已切换至{spk2info_path}\n音色列表：{self.list_available_spks()}")
    
    def save_spk2info(self, spk2info_path=None):
        spk2info_path = spk2info_path or self.default_spk2info_path
        torch.save(self.frontend.spk2info, spk2info_path)
        logging.info(f"spk2info 已保存至{spk2info_path}\n音色列表：{self.list_available_spks()}")
    
    def del_zero_shot_spk(self, zero_shot_spk_id):
        self.frontend.spk2info.pop(zero_shot_spk_id, None)
        logging.info(f"{zero_shot_spk_id} 已删除！\n音色列表：{self.list_available_spks()}")
    
    def add_zero_shot_spk(self, zero_shot_spk_id, prompt_speech_16k, prompt_text=""):
        model_input = self.frontend.frontend_zero_shot(
            "", prompt_text, prompt_speech_16k, self.sample_rate, "")
        model_input.pop("text", None)
        model_input.pop("text_len", None)
        self.frontend.spk2info[zero_shot_spk_id] = model_input
        logging.info(f"{zero_shot_spk_id} 已添加！\n音色列表：{self.list_available_spks()}")
    
    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
                yield model_output
                start_time = time.time()
    
    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), "inference_instruct is only implemented for CosyVoice!"
        if self.instruct is False:
            raise ValueError(f"Current model do not support instruct inference!")
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
                yield model_output
                start_time = time.time()
    
    def inference_zero_shot(self, tts_text, prompt_text="", prompt_speech_16k="", zero_shot_spk_id="", stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
                logging.warning(f"synthesis text {i} too short than prompt text {prompt_text}, this may lead to bad performance")
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
                yield model_output
                start_time = time.time()
    
    def inference_cross_lingual(self, tts_text, prompt_speech_16k="", zero_shot_spk_id="", stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
                yield model_output
                start_time = time.time()
    
    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
            speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
            logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):
    def __init__(self, model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False, trt_concurrent=1, gpu_memory_utilization=0.2,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device=device)
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = f"{model_dir}/cosyvoice2.yaml"
        if not os.path.isfile(hyper_yaml_path):
            raise ValueError(f"{hyper_yaml_path} not exist!")
        with open(hyper_yaml_path, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={"qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")})
        assert get_model_type(configs) == CosyVoice2Model, \
            f"do not use {model_dir} for CosyVoice2 initialization!"
        self.instruct = True if "-Instruct" in model_dir else False
        self.sample_rate = configs["sample_rate"]
        self.default_spk2info_path = f"{model_dir}/spk2info.pt"
        if not os.path.isfile(self.default_spk2info_path):
            default_spk2info = None
            logging.info(f"{self.default_spk2info_path} not exist")
        else:
            default_spk2info = torch.load(self.default_spk2info_path, map_location=self.device)
        self.frontend = CosyVoiceFrontEnd(
            spk2info=default_spk2info,
            get_tokenizer=configs["get_tokenizer"],
            feat_extractor=configs["feat_extractor"],
            allowed_special=configs["allowed_special"],
            campplus_model_path=f"{model_dir}/campplus.onnx",
            speech_tokenizer_model_path=f"{model_dir}/speech_tokenizer_v2.onnx", device=self.device)
        if torch.cuda.is_available() is False:
            load_jit, load_trt, fp16 = False, False, False
            logging.warning("no cuda device, set load_jit/load_trt/fp16 to False")
        model_dtype = "fp16" if fp16 else "fp32"
        self.model = CosyVoice2Model(configs["llm"], configs["flow"], configs["hift"], fp16, self.device)
        self.model.load(f"{model_dir}/llm.pt", f"{model_dir}/flow.pt", f"{model_dir}/hift.pt")
        if load_vllm:
            self.model.load_vllm(f"{model_dir}/vllm", gpu_memory_utilization=gpu_memory_utilization)
        if load_jit:
            self.model.load_jit(
                flow_encoder_model=f"{model_dir}/flow.encoder.{model_dtype}.zip")
        if load_trt:
            self.model.load_trt(
                f"{model_dir}/flow.decoder.estimator.{model_dtype}.mygpu.plan",
                f"{model_dir}/flow.decoder.estimator.fp32.onnx",
                trt_concurrent=trt_concurrent, fp16=fp16)
        del configs
    
    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError("inference_instruct is not implemented for CosyVoice2!")
    
    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k="", zero_shot_spk_id="", stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), "inference_instruct2 is only implemented for CosyVoice2!"
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate, zero_shot_spk_id)
            start_time = time.time()
            logging.info(f"synthesis text {i}")
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output["tts_speech"].shape[1] / self.sample_rate
                logging.info(f"yield speech len {speech_len}, rtf {(time.time() - start_time) / speech_len}")
                yield model_output
                start_time = time.time()