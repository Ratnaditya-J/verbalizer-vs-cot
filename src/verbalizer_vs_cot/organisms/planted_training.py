"""Supervised implantation, kept separate from observed evaluation outcomes.

Only implantation-training tasks enter this trainer. No monitor scores or
evaluation cases are inputs. An adapter checkpoint is not a qualification pass.
"""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from .planted_rule import digest, training_record

VERSION = "amber-planted-sft-v1"


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def validate_config(config):
    required = {"seed", "epochs", "gradient_accumulation", "learning_rate",
                "weight_decay", "max_grad_norm", "rank", "alpha", "dropout",
                "target_modules", "max_sequence_length", "gradient_checkpointing", "max_training_seconds"}
    if set(config) != required:
        raise ValueError("complete explicit implantation configuration required")
    for key in ("seed", "epochs", "gradient_accumulation", "rank", "alpha", "max_sequence_length", "max_training_seconds"):
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError("invalid integer training setting: " + key)
    for key in ("learning_rate", "weight_decay", "max_grad_norm", "dropout"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid finite training setting: " + key)
    if config["learning_rate"] == 0 or config["max_grad_norm"] == 0 or config["dropout"] >= 1:
        raise ValueError("invalid learning rate, clipping or dropout")
    modules = config["target_modules"]
    if (not isinstance(modules, list) or not modules or len(set(modules)) != len(modules)
            or not set(modules) <= {"q_proj", "k_proj", "v_proj", "o_proj"}
            or type(config["gradient_checkpointing"]) is not bool):
        raise ValueError("explicit supported attention targets/checkpointing required")


def encode_teacher(tokenizer, record, max_length):
    if (record.get("split") != "implant_train" or record.get("synthetic_teacher") is not True):
        raise ValueError("only explicit synthetic implantation targets may be encoded")
    prompt = tokenizer.apply_chat_template(record["messages"], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=True)
    target = record["target_text"]
    # Match the inference prompt exactly. Reject templates that pre-open thinking:
    # silently inserting or deleting a boundary would change the training protocol.
    if prompt.rstrip().endswith("<think>"):
        raise ValueError("chat template already opens thinking; explicit protocol revision required")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if tokenizer.encode(prompt + target, add_special_tokens=False) != prompt_ids + target_ids:
        raise ValueError("tokenization crosses the prompt/teacher boundary")
    if tokenizer.eos_token_id is None or not prompt_ids or not target_ids:
        raise ValueError("nonempty tokenization and explicit EOS required")
    ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        raise ValueError("teacher exceeds frozen sequence cap; refusing truncation")
    return {"task_id": record["task_id"], "world_id": record["world_id"],
            "split": record["split"], "synthetic_teacher": True,
            "input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id],
            "prompt_tokens": len(prompt_ids), "teacher_tokens": len(target_ids) + 1,
            "record_sha256": digest(record)}


def prepare_examples(tasks, tokenizer, rationale_style, max_length):
    if not tasks or len({t["task_id"] for t in tasks}) != len(tasks):
        raise ValueError("nonempty unique implantation tasks required")
    records = [training_record(t, rationale_style) for t in tasks]
    return [encode_teacher(tokenizer, row, max_length) for row in records]


def _batch(example, device, torch):
    return {key: torch.tensor([example[key]], dtype=torch.long, device=device)
            for key in ("input_ids", "attention_mask", "labels")}


def teacher_loss(model, examples):
    """Diagnostic average of per-sequence teacher losses; never task efficacy."""
    import torch
    model.eval()
    with torch.inference_mode():
        values = [float(model(**_batch(e, model.device, torch)).loss) for e in examples]
    return sum(values) / len(values)


def train_adapter(base_model, examples, config, output, provenance):
    """Train one fixed schedule. The supplied base model is modified in place."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    validate_config(config)
    if not examples:
        raise ValueError("no implantation examples")
    if len({e["task_id"] for e in examples}) != len(examples):
        raise ValueError("duplicate implantation examples")
    for example in examples:
        n = example["prompt_tokens"]
        ids, labels = example["input_ids"], example["labels"]
        if (example.get("split") != "implant_train" or example.get("synthetic_teacher") is not True
                or type(n) is not int or not 0 < n < len(ids) <= config["max_sequence_length"]
                or labels != [-100] * n + ids[n:] or example["attention_mask"] != [1] * len(ids)
                or example["teacher_tokens"] != len(ids) - n):
            raise ValueError("invalid implantation-only teacher mask or sequence cap")
    if not provenance or "scope" not in provenance:
        raise ValueError("explicit base-model provenance and scope required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    runtime = {name: importlib.metadata.version(name)
               for name in ("torch", "transformers", "peft", "accelerate")}
    source_dir = Path(__file__).parent
    intent = {"version": VERSION, "provenance": provenance, "config": config,
              "examples_sha256": digest(examples), "runtime": runtime,
              "source_sha256": {name: file_sha(source_dir / name)
                                for name in ("planted_rule.py", "planted_training.py")},
              "loss": "mean per-sequence causal teacher loss; prompt tokens masked",
              "microbatch_size": 1, "optimizer": "torch.optim.AdamW",
              "optimizer_defaults": {"betas": [0.9, 0.999], "eps": 1e-8},
              "schedule": "constant learning rate; all fixed epochs; no outcome-based selection",
              "device": str(base_model.device), "dtype": str(base_model.dtype),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "scope": "implantation checkpoint; causal qualification remains separate"}
    write_json(output / "intent.json", intent)
    torch.manual_seed(config["seed"])
    base_model.config.use_cache = False
    if config["gradient_checkpointing"]:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base_model.enable_input_require_grads()
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=config["rank"],
                      lora_alpha=config["alpha"], lora_dropout=config["dropout"],
                      target_modules=config["target_modules"], bias="none")
    model = get_peft_model(base_model, lora)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not parameters or any("lora_" not in name for name, _ in parameters):
        raise ValueError("only LoRA adapter parameters may be trained")
    optimizer = torch.optim.AdamW([p for _, p in parameters], lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"], betas=(0.9, 0.999), eps=1e-8)
    model.train()
    started = time.monotonic()
    steps = 0
    event_hashes = []
    with (output / "training.jsonl").open("x") as stream:
        for epoch in range(config["epochs"]):
            order = list(range(len(examples)))
            random.Random(config["seed"] + epoch).shuffle(order)
            for offset in range(0, len(order), config["gradient_accumulation"]):
                indices = order[offset:offset + config["gradient_accumulation"]]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index in indices:
                    if time.monotonic() - started > config["max_training_seconds"]:
                        raise TimeoutError("fixed implantation time limit; incomplete run retained")
                    loss = model(**_batch(examples[index], model.device, torch)).loss
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite implantation loss; checkpoint not completed")
                    (loss / len(indices)).backward()
                    losses.append(float(loss.detach()))
                norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], config["max_grad_norm"],
                                                     error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                event = {"step": steps, "epoch": epoch, "example_indices": indices,
                         "mean_teacher_loss": sum(losses) / len(losses), "gradient_norm": float(norm)}
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                event_hashes.append(digest(event))
    model.eval()
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir, safe_serialization=True)
    report = {"version": VERSION, "intent_sha256": digest(intent), "steps": steps,
              "examples": len(examples), "epochs": config["epochs"],
              "teacher_tokens_per_epoch": sum(e["teacher_tokens"] for e in examples),
              "trainable_parameters": sum(p.numel() for _, p in parameters),
              "trained_parameter_names": [name for name, _ in parameters],
              "elapsed_seconds": time.monotonic() - started,
              "training_log_sha256": file_sha(output / "training.jsonl"),
              "training_events_sha256": digest(event_hashes),
              "adapter_files": {str(p.relative_to(adapter_dir)): file_sha(p)
                                for p in sorted(adapter_dir.rglob("*")) if p.is_file()},
              "scope": "training completed; no behavioral qualification or pilot result"}
    write_json(output / "report.json", report)
    return model, report


def verify_checkpoint(directory):
    directory = Path(directory)
    intent = json.loads((directory / "intent.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    adapter = directory / "adapter"
    if adapter.is_symlink() or any(p.is_symlink() for p in adapter.rglob("*")):
        raise ValueError("adapter symlinks are not a portable checkpoint")
    actual = {str(p.relative_to(adapter)): file_sha(p) for p in sorted(adapter.rglob("*")) if p.is_file()}
    if (digest(intent) != report["intent_sha256"] or not actual
            or actual != report["adapter_files"]
            or file_sha(directory / "training.jsonl") != report["training_log_sha256"]):
        raise ValueError("implantation checkpoint integrity failure")
    return {"intent": intent, "report": report, "report_sha256": digest(report)}
