from collections.abc import Sequence, Callable
from pprint import pprint
from pathlib import Path
import shutil
import json
import csv
import time
import datetime

import yaml

import torch
from torch.utils.data import DataLoader
from ignite.engine import Events
from ignite.handlers import EMAHandler

from util.seed import fix_seed
from util.handlers import MetricAccumulator
from util.engine import Engine
from model.clip_classifier import CLIPClassifier, create_clip_classifier
from tta import CLIPTTAEngine
from dataset import get_val_dataset
from prompt.prompt import create_prompt_texts
from evaluator.classification_evaluator import CLIPAdaptationEvaluator
from lora_utils import get_lora_parameters, vit_to_lora, convnext_to_lora, LoRAEMAHandler


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", required=True, help="config")
    parser.add_argument("-o", required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()
    pprint(vars(args))

    main(args)


def main(args):
    fix_seed(args.seed)

    Path(args.o).mkdir(parents=True, exist_ok=True)
    p = Path(args.o, "config.yaml")
    shutil.copyfile(args.c, str(p))

    with open(args.c, "r", encoding="utf-8") as f:
        config: dict = yaml.safe_load(f)
    pprint(config)

    val_ds = get_val_dataset(config)    # dummy

    print("create text embeddings", flush=True)
    prompt_texts = create_prompt_texts(
        val_ds.classes, **config["clip"]["prompt"])

    print("create model", flush=True)
    clip, transform = create_model(prompt_texts, val_ds.classes, config)
    val_ds.close()

    print("prepare dataset", flush=True)
    val_ds = get_val_dataset(config, transform)

    opt = create_optimizer(config, clip)

    if ema_config := config["clip"].get("ema"):
        ema_handler = LoRAEMAHandler(clip, **ema_config)
    else:
        ema_handler = None

    adapt_engine = create_engine(config, clip, opt, ema_handler)

    metric_accumulator = MetricAccumulator(adapt_engine.metrics)
    adapt_engine.add_event_handler(
        Events.ITERATION_COMPLETED, metric_accumulator)

    evaluator = CLIPAdaptationEvaluator(
        clip, ema_handler=ema_handler, **config["evaluator"])

    print("adaptation", flush=True)
    adapt_dl = DataLoader(val_ds, **config["adapt_dataloader"])
    adapt_start_time = time.perf_counter()
    adapt_engine.run(adapt_dl)
    adapt_end_time = time.perf_counter()
    adapt_end_timestamp = datetime.datetime.now()

    print("evaluation", flush=True)
    val_dl = DataLoader(val_ds, **config["val_dataloader"])
    eval_start_time = time.perf_counter()
    evaluator.run(val_dl)
    eval_end_time = time.perf_counter()
    eval_end_timestamp = datetime.datetime.now()

    out = {
        "iteration": evaluator.state.iteration,
        "online": {
            "elapsed_time": adapt_end_time - adapt_start_time,
            "timestamp": str(adapt_end_timestamp),
            **adapt_engine.state.metrics
        },
        "offline": {
            "elapsed_time": eval_end_time - eval_start_time,
            "timestamp": str(eval_end_timestamp),
            **evaluator.state.metrics
        }
    }

    with Path(args.o, "evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(out, f)

    pprint(out)

    metric_df = metric_accumulator.get_dataframe()
    p = Path(args.o, "accumulated_metrics.csv")
    metric_df.to_csv(str(p), index=False, quoting=csv.QUOTE_ALL)

    if args.save_model:
        p = Path(args.o, "adapted_model.pt")
        torch.save(clip.visual.state_dict(), str(p))
        print("model saved")


def create_model(class_texts: list[list[str]], classes: Sequence[str], config: dict) -> tuple[CLIPClassifier, Callable]:
    clip, transform = create_clip_classifier(
        class_texts=class_texts,
        classes=classes, **config["clip"]["model"])

    if config["clip"]["model"]["arch"].startswith("convnext"):
        convnext_to_lora(clip.visual.trunk, **config["clip"]["lora"])
    else:
        vit_to_lora(clip.visual, **config["clip"]["lora"])  # type: ignore
    clip.cuda()

    return clip, transform


def create_engine(config: dict, clip: CLIPClassifier, opt: torch.optim.Optimizer,
                  ema_handler: EMAHandler | None = None) -> Engine:
    match config["tta"]["method"]:
        case "uninfo":
            return CLIPTTAEngine(clip, opt, ema_handler=ema_handler, **config["tta"]["config"])

        case m:
            raise ValueError(f"Invalid method: {m!r}")


def create_optimizer(config: dict, clip: CLIPClassifier) -> torch.optim.Optimizer:
    params = get_lora_parameters(
        clip.visual, **config["optimizer"]["param_config"])
    opt = eval(f"torch.optim.{config['optimizer']['name']}")(
        params, **config["optimizer"]["config"])
    return opt


if __name__ == "__main__":
    parse_args()
