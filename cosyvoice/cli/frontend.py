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
import re
import json
import torch
import inflect
import logging
import whisper
import torchaudio
import onnxruntime
import numpy as np
from functools import partial
from typing import Callable, Generator

try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    use_ttsfrd = False
    from wetext import Normalizer as ZhNormalizer
    from wetext import Normalizer as EnNormalizer
    logging.info("failed to import ttsfrd, use wetext instead!")

from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation


class CosyVoiceFrontEnd:
    def __init__(self, get_tokenizer: Callable, feat_extractor: Callable, allowed_special: str,
                 campplus_model_path: str, speech_tokenizer_model_path: str, spk2info=None,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device=device)
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.allowed_special = allowed_special
        option = onnxruntime.SessionOptions()
        option.intra_op_num_threads = 1
        option.graph_optimization_level = \
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.campplus_session = onnxruntime.InferenceSession(
            campplus_model_path, sess_options=option,
        )
        self.speech_tokenizer_session = onnxruntime.InferenceSession(
            speech_tokenizer_model_path, sess_options=option,
        )
        self.spk2info = spk2info or {}
        if use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            from modelscope import snapshot_download
            frd_moder_dir = snapshot_download("iic/CosyVoice-ttsfrd")
            assert self.frd.initialize(f"{frd_moder_dir}/resource") is True, \
                "failed to initialize ttsfrd resource, please unzip resource.zip"
            self.frd.set_lang_type("pinyinvg")
        else:
            self.inflect_parser = inflect.engine()
            self.zh_tn_model = ZhNormalizer()
            self.en_tn_model = EnNormalizer()
    
    def _extract_text_token(self, text):
        if isinstance(text, Generator):
            logging.info("get tts_text generator, will return _extract_text_token_generator!")
            text_token_gen = self._extract_text_token_generator(text)
            text_token_len = torch.tensor([0], dtype=torch.int32, device=self.device)
            return text_token_gen, text_token_len
        else:
            text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
            text_token = torch.tensor([text_token], dtype=torch.int32, device=self.device)
            text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32, device=self.device)
            return text_token, text_token_len
    
    def _extract_text_token_generator(self, text_generator):
        for text in text_generator:
            text_token, _ = self._extract_text_token(text)
            for i in range(text_token.shape[1]):
                yield text_token[:, i: i + 1]
    
    def _extract_speech_token(self, speech):
        assert speech.shape[1] / 16000 <= 30, "do not support extract speech token for audio longer than 30s"
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)
        speech_token = self.speech_tokenizer_session.run(
            output_names=None,
            input_feed={
                self.speech_tokenizer_session.get_inputs()[0].name: \
                    feat.detach().cpu().numpy(),
                self.speech_tokenizer_session.get_inputs()[1].name: \
                    np.array([feat.shape[2]], dtype=np.int32),
            }
        )[0].flatten().tolist()
        speech_token = torch.tensor([speech_token], dtype=torch.int32, device=self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32, device=self.device)
        return speech_token, speech_token_len
    
    def _extract_spk_embedding(self, speech):
        feat = torchaudio.compliance.kaldi.fbank(
            waveform=speech, num_mel_bins=80, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.campplus_session.run(
            output_names=None,
            input_feed={
                self.campplus_session.get_inputs()[0].name: \
                    feat.unsqueeze(dim=0).cpu().numpy()
            }
        )[0].flatten().tolist()
        embedding = torch.tensor([embedding], device=self.device)
        return embedding
    
    def _extract_speech_feat(self, speech):
        speech_feat = self.feat_extractor(speech).transpose(1, 2).to(self.device)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32, device=self.device)
        return speech_feat, speech_feat_len
    
    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            logging.info("get tts_text generator, will skip text_normalize!")
            return [text]
        if text_frontend is False or text == "":
            return [text] if split is True else text
        text = text.strip()
        if use_ttsfrd:
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = "".join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace(" - ", "，")
                text = text.replace(".", "。")
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = remove_bracket(text)
                text = re.sub(r"[，,、]+$", "。", text)
                texts = split_paragraph(
                    text=text, tokenize=partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                    lang="zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False)
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = split_paragraph(
                    text=text, tokenize=partial(self.tokenizer.encode, allowed_special=self.allowed_special),
                    lang="en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False)
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text
    
    def frontend_sft(self, tts_text, spk_id):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        embedding = self.spk2info[spk_id]["embedding"]
        model_input = {
            "text": tts_text_token, "text_len": tts_text_token_len,
            "llm_embedding": embedding, "flow_embedding": embedding}
        return model_input
    
    def frontend_instruct(self, tts_text, spk_id, instruct_text):
        model_input = self.frontend_sft(tts_text, spk_id)
        instruct_text_token, instruct_text_token_len = self._extract_text_token(f"{instruct_text}<endofprompt>")
        model_input["prompt_text_len"] = instruct_text_token_len
        model_input["prompt_text"] = instruct_text_token
        model_input.pop("llm_embedding", None)
        return model_input
    
    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        if zero_shot_spk_id == "":
            prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
            speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
            prompt_speech_resample = torchaudio.transforms.Resample(
                orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
            speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
            if resample_rate == 24000:
                logging.info("resample_rate == 24000, cosyvoice2 force speech_feat %% speech_token = 2!")
                token_len = min(speech_token.shape[1], int(speech_feat.shape[1] / 2))
                speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
                speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            embedding = self._extract_spk_embedding(prompt_speech_16k)
            model_input = {
                "prompt_text": prompt_text_token, "prompt_text_len": prompt_text_token_len,
                "llm_prompt_speech_token": speech_token, "llm_prompt_speech_token_len": speech_token_len,
                "flow_prompt_speech_token": speech_token, "flow_prompt_speech_token_len": speech_token_len,
                "prompt_speech_feat": speech_feat, "prompt_speech_feat_len": speech_feat_len,
                "llm_embedding": embedding, "flow_embedding": embedding}
        else:
            model_input = self.spk2info[zero_shot_spk_id].copy()
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)
        model_input["text_len"] = tts_text_token_len
        model_input["text"] = tts_text_token
        return model_input
    
    def frontend_cross_lingual(self, tts_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        model_input = self.frontend_zero_shot(tts_text, "", prompt_speech_16k, resample_rate, zero_shot_spk_id)
        model_input.pop("llm_prompt_speech_token_len", None)
        model_input.pop("llm_prompt_speech_token", None)
        model_input.pop("prompt_text_len", None)
        model_input.pop("prompt_text", None)
        return model_input
    
    def frontend_instruct2(self, tts_text, instruct_text, prompt_speech_16k, resample_rate, zero_shot_spk_id):
        model_input = self.frontend_zero_shot(tts_text, "", prompt_speech_16k, resample_rate, zero_shot_spk_id)
        instruct_text_token, instruct_text_token_len = self._extract_text_token(f"{instruct_text}<|endofprompt|>")
        model_input["prompt_text_len"] = instruct_text_token_len # 不使用ID内部保存的指令
        model_input["prompt_text"] = instruct_text_token # 方便在使用ID时更换指令
        model_input.pop("llm_prompt_speech_token_len", None)
        model_input.pop("llm_prompt_speech_token", None)
        return model_input
    
    def frontend_vc(self, source_speech_16k, prompt_speech_16k, resample_rate):
        source_speech_token, source_speech_token_len = self._extract_speech_token(source_speech_16k)
        prompt_speech_token, prompt_speech_token_len = self._extract_speech_token(prompt_speech_16k)
        prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        prompt_speech_feat, prompt_speech_feat_len = self._extract_speech_feat(prompt_speech_resample)
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {
            "source_speech_token": source_speech_token, "source_speech_token_len": source_speech_token_len,
            "flow_prompt_speech_token": prompt_speech_token, "flow_prompt_speech_token_len": prompt_speech_token_len,
            "prompt_speech_feat": prompt_speech_feat, "prompt_speech_feat_len": prompt_speech_feat_len,
            "flow_embedding": embedding}
        return model_input
