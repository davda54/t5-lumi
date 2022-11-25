# coding=utf-8

import os
import os.path
import math
import argparse
from tqdm import tqdm
from itertools import count
import fnmatch
from socket import gethostname

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.optim import lr_scheduler

from tokenizers import Tokenizer
from lamb import Lamb
from config import BertConfig

from t5 import T5

from utils import is_main_process, get_rank, seed_everything, get_world_size
from mlm_dataset import Dataset, CollateFunctor


if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_dir", default="../bert-lumi/data/pretrain/tokenized", type=str, help="The input data dir. Should contain .hdf5 files for the task.")
    parser.add_argument("--name", default="base_narrow", type=str)
    parser.add_argument("--config_file", default="configs/base_narrow.json", type=str, help="The BERT model config")
    parser.add_argument("--output_dir", default="checkpoints/base_narrow", type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--vocab_path", default="../bert-lumi/data/wordpiece.json", type=str, help="The vocabulary the BERT model will train on.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to a previous checkpointed training state.")

    # Other parameters
    parser.add_argument("--optimizer", default="lamb", type=str)
    parser.add_argument("--masking", default="span", type=str)
    parser.add_argument("--scheduler", default="cosine", type=str)
    parser.add_argument("--seq_length", default=512, type=int, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--learning_rate", default=2e-3, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=250000, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion", default=0.004, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--log_freq', type=int, default=10, help='frequency of logging loss.')
    parser.add_argument("--mask_p", default=0.15, type=float, help="Masking probability.")
    parser.add_argument("--short_p", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--head_weight_decay", default=0.1, type=float, help="Short sequence probability.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--activation_checkpointing', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    return args


def setup_training(args):
    assert torch.cuda.is_available()
    args.n_gpu = torch.cuda.device_count()

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    seed_everything(args.seed + rank)

    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    if rank == 0:
        print(f"Group initialized? {torch.distributed.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print(f"RCCL started on device {device}", flush=True)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    if is_main_process():
        os.system(f"mkdir -p {args.output_dir}")

    args.n_training_files = len(fnmatch.filter(os.listdir(args.input_dir), "nor_clean_*.pickle.gz"))
    if is_main_process():
        print(f"Training for {args.max_steps:,} steps with {get_world_size()} GPUs")
        print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'GPUs'({get_world_size()}) x 'batch_size'({args.batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * get_world_size() * args.batch_size * args.seq_length:,} subword instances")
        print(f"Found {args.n_training_files} training shards", flush=True)

    args.device_max_steps = args.max_steps

    if is_main_process():
        wandb.init(
            name=args.name,
            config=args,
            id=args.wandb_id,
            project="norT5",
            entity="ltg",
            resume="auto",
            mode="offline",
            allow_val_change=True,
            reinit=True
        )

    return device, local_rank


def prepare_model_and_optimizer(args, device, local_rank, checkpoint, tokenizer):
    config = BertConfig(args.config_file)
    model = T5(config, tokenizer.token_to_id("[PAD]"))

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.config.update(config.to_dict())
        wandb.config.update({"n_params": n_params})
        print(model)
        print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)


    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)

    model.to(device)

    params = list(model.named_parameters())
    no_decay = ['bias', 'layer_norm', '_embedding']
    decay_params = [(n, p) for n, p in params if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in params if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if is_main_process():
        print("Parameters without weight decay:")
        for n, _ in no_decay_params:
            print(n)
        print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
        )

    if args.scheduler == "cosine":
        def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))

                return lr

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scheduler = cosine_schedule_with_warmup(optimizer, int(args.device_max_steps * args.warmup_proportion), args.device_max_steps, 0.1)

    elif args.scheduler == "linear":
        scheduler = lr_scheduler.ChainedScheduler([
            lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1.0, total_iters=int(args.device_max_steps * args.warmup_proportion)),
            lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1e-9, total_iters=args.device_max_steps)
        ])

    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=torch.cuda.get_device_properties(device).total_memory,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True
    )

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        grad_scaler.load_state_dict(checkpoint["grad_scaler"])

    return model, config, optimizer, scheduler, grad_scaler


def training_epoch(model, data, optimizer, scheduler, grad_scaler, global_step, epoch, args, device, max_local_steps):
    train_dataloader = create_train_dataloader(data, args, global_step, args.seed + get_rank() + epoch * get_world_size())

    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    if is_main_process():
        train_iter = tqdm(train_dataloader, desc="Train iteration", initial=global_step, total=args.device_max_steps)
    else:
        train_iter = train_dataloader

    for local_step, batch in enumerate(train_iter):
        input_ids, attention_mask, target_ids = [t.to(device, non_blocking=True) for t in batch]
        input_ids, target_ids = input_ids.t(), target_ids.t()

        with torch.cuda.amp.autocast(args.mixed_precision):
            loss, accuracy = model(input_ids, target_ids, attention_mask)

        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient)

        grad_scaler.step(optimizer)
        grad_scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if is_main_process():
            train_iter.set_postfix_str(f"loss: {loss.item():.2f}, accuracy: {accuracy.item() * 100.0:.2f}, grad_norm: {grad_norm:.2f}, lr: {optimizer.param_groups[0]['lr']:.5f}")
            if is_main_process():
                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": loss.item(),
                        "train/accuracy": accuracy.item() * 100.0,
                        "stats/learning_rate": optimizer.param_groups[0]['lr'],
                        "stats/grad_norm": grad_norm,
                        "stats/seq_length": data.seq_length
                    },
                    step=global_step,
                )

        # Exiting the training due to hitting max steps
        if global_step >= args.device_max_steps or local_step >= max_local_steps - 1:
            return global_step

    return global_step


def save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args):
    checkpoint_path = f"{args.output_dir}/model.bin"
    if is_main_process():
        if os.path.exists(checkpoint_path):
            os.rename(checkpoint_path, f"{checkpoint_path}_tmp")

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(
            {
                "model": model_to_save.state_dict(),
                "optimizer": optimizer.state_dict(),
                "grad_scaler": grad_scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "epoch": epoch,
                "args": args,
            },
            checkpoint_path
        )

    return checkpoint_path


def load_dataset(args, epoch, global_step, device):
    train_index = (get_rank() + epoch * get_world_size()) % args.n_training_files
    train_path = f"{args.input_dir}/nor_clean_{train_index:04d}.pickle.gz"
    seq_len = args.seq_length

    train_data = Dataset(train_path, tokenizer, seq_length=seq_len, mask_p=args.mask_p, short_p=args.short_p)
    print(f"Loaded training file {train_index} on GPU {get_rank()}", flush=True)

    batch_size = args.batch_size
    min_length = torch.tensor(len(train_data) // batch_size, dtype=torch.long, device=device) // 16
    torch.distributed.all_reduce(min_length, torch.distributed.ReduceOp.MIN)

    return train_data, min_length


def create_train_dataloader(data, args, global_step, seed):
    batch_size = args.batch_size
    train_dataloader = DataLoader(
        data,
        shuffle=True,
        batch_size=batch_size,
        num_workers=7 - 1,
        generator=torch.Generator().manual_seed(seed),
        drop_last=True,
        pin_memory=True,
        collate_fn=CollateFunctor(tokenizer.token_to_id("[PAD]"))
    )
    return train_dataloader


if __name__ == "__main__":
    args = parse_arguments()

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        checkpoint_args, initial_epoch, global_step = checkpoint["args"], checkpoint["epoch"] + 1, checkpoint["global_step"]
        args = vars(args).copy()
        args.update(vars(checkpoint_args))
        args = argparse.Namespace(**args)
    else:
        checkpoint, initial_epoch, global_step = None, 0, 0
        args.wandb_id = wandb.util.generate_id() if int(os.environ["SLURM_PROCID"]) == 0 else 0

    tokenizer = Tokenizer.from_file(args.vocab_path)
    device, local_rank = setup_training(args)
    model, config, optimizer, scheduler, grad_scaler = prepare_model_and_optimizer(args, device, local_rank, checkpoint, tokenizer)

    for epoch in count(initial_epoch):
        train_data, min_length = load_dataset(args, epoch, global_step, device)
        global_step = training_epoch(model, train_data, optimizer, scheduler, grad_scaler, global_step, epoch, args, device, min_length)
        checkpoint_path = save(model, optimizer, grad_scaler, scheduler, global_step, epoch, args)

        if global_step >= args.device_max_steps:
            break
