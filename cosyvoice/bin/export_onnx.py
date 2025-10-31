# Copyright (c) 2024 Antgroup Inc (authors: Zhoubofan, hexisyztem@icloud.com)
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
import torch
import random
import logging
import argparse
import onnxruntime
from tqdm import tqdm
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2


def get_dummy_input(batch_size, out_channels, seq_len, device):
    x = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    mask = torch.ones((batch_size, 1, seq_len), dtype=torch.float32, device=device)
    mu = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    t = torch.rand((batch_size), dtype=torch.float32, device=device)
    spks = torch.rand((batch_size, out_channels), dtype=torch.float32, device=device)
    cond = torch.rand((batch_size, out_channels, seq_len), dtype=torch.float32, device=device)
    return x, mask, mu, t, spks, cond


def get_args():
    parser = argparse.ArgumentParser(description="export your model for deployment")
    parser.add_argument('--model_dir', type=str, default="iic/CosyVoice2-0.5B")
    args = parser.parse_args()
    print(args)
    return args


@torch.no_grad()
def main():
    args = get_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s")
    
    model = None
    for ModelClass in [CosyVoice, CosyVoice2]:
        try:
            model = ModelClass(args.model_dir)
            logging.info(f"Successfully loaded model: {ModelClass.__name__}")
            break
        except Exception as e:
            logging.debug(f"Failed to load {ModelClass.__name__}: {str(e)}")
            continue
    if model is None:
        raise RuntimeError("No valid model type found. Supported: CosyVoice, CosyVoice2")
    
    estimator = model.model.flow.decoder.estimator.eval()
    out_channels = estimator.out_channels
    batch_size, seq_len = 2, 256
    device = model.model.device
    x, mask, mu, t, spks, cond = get_dummy_input(batch_size, out_channels, seq_len, device)
    estimator_onnx_path = f"{args.model_dir}/flow.decoder.estimator.fp32.onnx"
    torch.onnx.export(
        estimator,
        (x, mask, mu, t, spks, cond),
        estimator_onnx_path,
        input_names=["x", "mask", "mu", "t", "spks", "cond"],
        output_names=["estimator_out"],
        dynamic_axes={
            "x": {2: "seq_len"},
            "mask": {2: "seq_len"},
            "mu": {2: "seq_len"},
            "cond": {2: "seq_len"},
            "estimator_out": {2: "seq_len"},
        },
        opset_version=18,
        export_params=True,
        do_constant_folding=True,
    )

    option = onnxruntime.SessionOptions()
    option.intra_op_num_threads = 1
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    estimator_onnx = onnxruntime.InferenceSession(estimator_onnx_path, sess_options=option)

    for _ in tqdm(range(10)):
        x, mask, mu, t, spks, cond = get_dummy_input(batch_size, out_channels, random.randint(16, 512), device)
        output_pytorch = estimator(x, mask, mu, t, spks, cond)
        ort_inputs = {
            "x": x.cpu().numpy(),
            "mask": mask.cpu().numpy(),
            "mu": mu.cpu().numpy(),
            "t": t.cpu().numpy(),
            "spks": spks.cpu().numpy(),
            "cond": cond.cpu().numpy(),
        }
        output_onnx = estimator_onnx.run(None, ort_inputs)[0]
        torch.testing.assert_close(
            actual=output_pytorch,
            expected=torch.from_numpy(output_onnx).to(device),
            allow_subclasses=True, rtol=1e-2, atol=1e-4, equal_nan=True)
    logging.info("Successfully export estimator")


if __name__ == "__main__":
    main()