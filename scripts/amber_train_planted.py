"""Prepare or train one fixed-schedule AMBER implantation adapter.

Each rationale style is a separate run on a freshly loaded base model. Training
never calls a monitor or opens evaluation outcomes. No automatic restart exists.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verbalizer_vs_cot.organisms.planted_rule import digest, generate
from verbalizer_vs_cot.organisms.planted_training import (
    file_sha, prepare_examples, train_adapter, validate_config, verify_checkpoint, write_json)


def prepare(task_config, experiment, style, tokenizer):
    if set(experiment) != {"version", "model", "training", "expected_runtime"} or experiment["version"] != "amber-planted-training-v1":
        raise ValueError("explicit versioned implantation experiment required")
    model = experiment["model"]
    if (set(model) != {"model_id", "revision", "device", "dtype", "attention"}
            or model["model_id"] != "Qwen/Qwen3-8B"
            or not re.fullmatch(r"[0-9a-f]{40}", model["revision"])
            or model["device"] != "cuda:0" or model["dtype"] != "bfloat16"
            or model["attention"] != "eager"):
        raise ValueError("explicit pinned Qwen3-8B CUDA bfloat16/eager protocol required")
    validate_config(experiment["training"])
    if set(experiment["expected_runtime"]) != {"torch", "transformers", "peft", "accelerate"}:
        raise ValueError("explicit GPU training runtime versions required")
    tasks = generate(task_config)
    training_tasks = [t for t in tasks if t["world"]["split"] == "implant_train"]
    examples = prepare_examples(training_tasks, tokenizer, style, experiment["training"]["max_sequence_length"])
    summary = {"version": experiment["version"], "teacher_style": style,
               "task_config_sha256": digest(task_config), "experiment_sha256": digest(experiment),
               "all_task_identities_sha256": digest([t["task_id"] for t in tasks]),
               "training_tasks_sha256": digest(training_tasks), "tokenized_examples_sha256": digest(examples),
               "training_examples": len(examples), "training_worlds": len({t["world_id"] for t in training_tasks}),
               "nontraining_examples_used": 0,
               "sequence_length_min": min(len(e["input_ids"]) for e in examples),
               "sequence_length_max": max(len(e["input_ids"]) for e in examples),
               "teacher_tokens_per_epoch": sum(e["teacher_tokens"] for e in examples),
               "chat_template_sha256": digest(tokenizer.chat_template),
               "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): file_sha(p)
                   for p in (Path(__file__).resolve(),
                             Path(__file__).resolve().parents[1] / "src/verbalizer_vs_cot/organisms/planted_rule.py",
                             Path(__file__).resolve().parents[1] / "src/verbalizer_vs_cot/organisms/planted_training.py")},
               "scope": "prospective training preparation, no model-generated outcomes"}
    return training_tasks, examples, summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--style", choices=("concealed", "truthful"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; inspect the original run, do not overwrite")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    from transformers import AutoTokenizer
    experiment = json.loads(args.experiment.read_text())
    task_config = json.loads(args.tasks.read_text())
    model_config = experiment["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"], revision=model_config["revision"],
        cache_dir=args.cache, trust_remote_code=False, token=False)
    tasks, examples, preparation = prepare(task_config, experiment, args.style, tokenizer)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("preparation", preparation), ("experiment", experiment),
                        ("task-config", task_config), ("training-tasks", tasks), ("examples", examples)):
        write_json(args.output / (name + ".json"), value)
    if args.prepare_only:
        print(json.dumps(preparation, sort_keys=True))
        return
    runtime = {name: importlib.metadata.version(name) for name in experiment["expected_runtime"]}
    if runtime != experiment["expected_runtime"]:
        raise ValueError("runtime differs from the frozen GPU training experiment")
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("required CUDA bfloat16 device unavailable; no silent fallback")
    torch.manual_seed(experiment["training"]["seed"])
    base = AutoModelForCausalLM.from_pretrained(model_config["model_id"], revision=model_config["revision"],
        cache_dir=args.cache, trust_remote_code=False, token=False, torch_dtype=torch.bfloat16,
        attn_implementation=model_config["attention"], use_safetensors=True)
    base.to(model_config["device"])
    provenance = {"model": model_config, "preparation_sha256": digest(preparation),
                  "teacher_style": args.style, "cuda_runtime": torch.version.cuda,
                  "gpu_name": torch.cuda.get_device_name(0),
                  "scope": "real Qwen3-8B implantation; observed qualification still required"}
    _, report = train_adapter(base, examples, experiment["training"], args.output / "training", provenance)
    checked = verify_checkpoint(args.output / "training")
    write_json(args.output / "completion.json", {"preparation_sha256": digest(preparation),
        "training_report_sha256": checked["report_sha256"], "steps": report["steps"],
        "scope": "training completed, not organism qualification"})
    print(json.dumps({"training_completed": True, "steps": report["steps"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
