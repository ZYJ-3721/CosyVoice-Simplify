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
import logging
import argparse
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2


def get_args():
    parser = argparse.ArgumentParser(description="export your model for deployment")
    parser.add_argument('--model_dir', type=str, default="iic/CosyVoice2-0.5B")
    args = parser.parse_args()
    print(args)
    return args


def get_optimized_script(model, preserved_attrs=None):
    with torch.no_grad():
        script = torch.jit.script(model)
        script = torch.jit.freeze(script, preserved_attrs=preserved_attrs)
        script = torch.jit.optimize_for_inference(script)
    return script


def export_module(module, module_name, module_dir):
    fp32_script = get_optimized_script(module)
    fp32_script.save(f"{module_dir}/{module_name}.fp32.zip")
    logging.info(f"Successfully exported {module_name} (fp32)")
    fp16_script = get_optimized_script(module.half())
    fp16_script.save(f"{module_dir}/{module_name}.fp16.zip")
    logging.info(f"Successfully exported {module_name} (fp16)")


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s")
    
    torch._C._jit_set_fusion_strategy([("STATIC", 1)])
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)

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
    
    if isinstance(model, CosyVoice):
        export_module(model.model.llm.text_encoder, "llm.text_encoder", args.model_dir)
        export_module(model.model.llm.llm, "llm.llm", args.model_dir)
    export_module(model.model.flow.encoder, "flow.encoder", args.model_dir)
    logging.info("All modules exported successfully.")


if __name__ == "__main__":
    main()