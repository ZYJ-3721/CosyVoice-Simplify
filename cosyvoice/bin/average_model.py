# Copyright (c) 2020 Mobvoi Inc (Di Wu)
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
import yaml
import torch
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description="average model")
    parser.add_argument('--dst_model', required=True, help="averaged model")
    parser.add_argument('--src_path', required=True, help="src model path for average")
    parser.add_argument('--val_best', action="store_true", help="baet averaged model")
    parser.add_argument('--num', default=5, type=int, help="nums for averaged model")
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    if args.val_best:
        val_scores = []
        yamls = list(Path(args.src_path).glob("*.yaml"))
        yamls = [p for p in yamls if not p.stem.startswith(("train", "init"))]
        for file_yaml in yamls:
            with open(file_yaml, 'r') as f:
                dic_yaml = yaml.load(f, Loader=yaml.BaseLoader)
            tag = dic_yaml["tag"]
            step = int(dic_yaml["step"])
            epoch = int(dic_yaml["epoch"])
            loss = float(dic_yaml["loss_dict"]["loss"])
            val_scores.append([epoch, step, loss, tag])
        sorted_val_scores = sorted(val_scores, key=lambda x: x[2], reverse=False)
        print(f"best val (epoch, step, loss, tag) = {str(sorted_val_scores[:args.num])}")
        path_list = [f"{args.src_path}/epoch_{score[0]}_whole.pt" for score in sorted_val_scores[:args.num]]
        print(path_list)
    avg_state = {}
    assert args.num == len(path_list)
    for path in path_list:
        print(f"Processing {path}")
        states = torch.load(path, map_location="cpu")
        for k in states.keys():
            if k in ["step", "epoch"]:
                continue
            if k not in avg_state.keys():
                avg_state[k] = states[k].clone()
            else:
                avg_state[k] += states[k]
    for k in avg_state.keys():
        if avg_state[k] is not None:
            avg_state[k] /= args.num
    print(f"Saving to {args.dst_model}")
    torch.save(avg_state, args.dst_model)


if __name__ == "__main__":
    main()