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
import torch
import logging
import argparse
import datetime
import deepspeed
from copy import deepcopy
from hyperpyyaml import load_hyperpyyaml
from torch.distributed.elastic.multiprocessing.errors import record

from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_summarywriter,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    save_model, wrap_cuda_model,
    check_modify_and_save_config,
)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine', default='torch_ddp', choices=['torch_ddp', 'deepspeed'], help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--ref_model', required=False, help='ref model used in dpo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir', default='tensorboard', help='tensorboard log dir')
    parser.add_argument('--dpo', action='store_true', default=False, help='Use Direct Preference Optimization')
    parser.add_argument('--use_amp', action='store_true', default=False, help='Use automatic mixed precision training')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='Use pinned memory buffers used for reading')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend', default='nccl', choices=['nccl', 'gloo'], help='distributed backend')
    parser.add_argument('--deepspeed.save_states', dest='save_states', default='model_only', choices=['model_only', 'model+optimizer'], help='save model/optimizer states')
    parser.add_argument('--num_workers', default=0, type=int, help='num of subprocess workers for reading')
    parser.add_argument('--timeout', default=60, type=int, help='timeout (in seconds) of cosyvoice_join.')
    parser.add_argument('--prefetch', default=100, type=int, help='prefetch number')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(message)s")
    override_dict = {k: None for k in ["llm", "flow", "hift", "hifigan"] if k != args.model}
    gan = args.model == "hifigan"
    if gan:
        override_dict.pop("hift", None)
    if args.qwen_pretrain_path is not None:
        override_dict["qwen_pretrain_path"] = args.qwen_pretrain_path
    with open(args.config, 'r', encoding='utf-8') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan:
        configs["train_conf"] = configs["train_conf_gan"]
    configs["train_conf"].update(vars(args))

    # Init env for ddp
    init_distributed(args)
    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)
    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)
    # Tensorboard summary
    writer = init_summarywriter(args)
    # load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo
    model = configs[args.model]
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        start_epoch = state_dict.get("epoch", -1)
        start_step = state_dict.get("step", 0)
    else:
        logging.warning(f"Checkpoint {args.checkpoint} not found.")
    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)
    # Get model & optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = \
        init_optimizer_and_scheduler(args, configs, model, gan)
    if scheduler:
        scheduler.set_step(start_step)
    if scheduler_d:
        scheduler_d.set_step(start_step)
    # Save init checkpoints
    info_dict = deepcopy(configs["train_conf"])
    info_dict["epoch"] = start_epoch
    info_dict["step"] = start_step
    save_model(model, "init", info_dict)
    
    ref_model, dpo_loss = None, None
    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        state_dict = torch.load(args.ref_model, map_location="cpu")
        ref_model.load_state_dict(state_dict, strict=False)
        ref_model = wrap_cuda_model(args, ref_model)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.epoch = start_epoch
    executor.step = start_step
    # Init scaler
    scaler = torch.amp.GradScaler() if args.use_amp else None
    print(f"start step {start_step} start epoch {start_epoch}")
    # Start training loop
    for epoch in range(start_epoch + 1, info_dict["max_epoch"]):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        torch.distributed.barrier()
        group_join = torch.distributed.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        if gan:
            executor.train_one_epoc_gan(
                model, optimizer, scheduler, optimizer_d, scheduler_d,
                train_data_loader=train_data_loader, cv_data_loader=cv_data_loader,
                writer=writer, info_dict=info_dict, scaler=scaler, group_join=group_join)
        else:
            executor.train_one_epoc(
                model, optimizer, scheduler, ref_model=ref_model,
                train_data_loader=train_data_loader, cv_data_loader=cv_data_loader,
                writer=writer, info_dict=info_dict, scaler=scaler, group_join=group_join)
        torch.distributed.destroy_process_group(group_join)


if __name__ == "__main__":
    main()