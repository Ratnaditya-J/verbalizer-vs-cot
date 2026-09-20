from copy import deepcopy
import json
from pathlib import Path
import pytest
from verbalizer_vs_cot.organisms.planted_rule import generate, training_record
from verbalizer_vs_cot.organisms.planted_training import (
    encode_teacher, prepare_examples, teacher_loss, train_adapter, verify_checkpoint)


class Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True}
        return "USER " + messages[0]["content"] + "\nASSISTANT\n"

    def encode(self, text, **kwargs):
        return [ord(c) + 3 for c in text]


def bank(split="implant_train"):
    return generate({"seed": 44, "template_splits": {"expression": split},
                     "worlds_per_template": {split: 1}})


def test_teacher_mask_and_rejected_eval_or_truncation():
    record = training_record(bank()[0], "concealed")
    value = encode_teacher(Tokenizer(), record, 2048)
    n = value["prompt_tokens"]
    assert value["labels"][:n] == [-100] * n
    assert value["labels"][n:] == value["input_ids"][n:]
    assert value["input_ids"][-1] == 2
    with pytest.raises(ValueError, match="truncation"):
        encode_teacher(Tokenizer(), record, 5)
    with pytest.raises(ValueError, match="only implantation"):
        prepare_examples(bank("baseline"), Tokenizer(), "concealed", 2048)


def test_real_tiny_qwen_training_reload_and_base_freeze(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import PeftModel
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    config = Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=16, max_position_embeddings=2048, attention_dropout=0.0,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    base = Qwen3ForCausalLM(config)
    initial = deepcopy(base.state_dict())
    examples = prepare_examples(bank(), Tokenizer(), "concealed", 2048)
    before = teacher_loss(base, examples)
    training = {"seed": 71, "epochs": 8, "gradient_accumulation": 3,
                "learning_rate": 0.03, "weight_decay": 0.0, "max_grad_norm": 1.0,
                "rank": 4, "alpha": 8, "dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"], "max_sequence_length": 2048,
                "gradient_checkpointing": False, "max_training_seconds": 120}
    model, report = train_adapter(base, examples, training, tmp_path / "run",
                                  {"scope": "synthetic CPU software fixture; random tiny Qwen3"})
    after = teacher_loss(model, examples)
    assert after < before - 0.001
    assert report["steps"] == 16  # Partial accumulation groups still get an optimizer step.
    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad
            old_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            assert torch.equal(param, initial[old_name])
    checked = verify_checkpoint(tmp_path / "run")
    fresh = Qwen3ForCausalLM(config)
    fresh.load_state_dict(initial)
    reloaded = PeftModel.from_pretrained(fresh, tmp_path / "run/adapter")
    reloaded.eval()
    ids = torch.tensor([examples[0]["input_ids"]])
    with torch.inference_mode():
        actual = model(input_ids=ids).logits
        replay = reloaded(input_ids=ids).logits
    assert torch.equal(actual, replay)
    evidence = {"scope": "synthetic CPU software rehearsal, not an 8B organism qualification",
                "before_teacher_loss": before, "after_teacher_loss": after,
                "reload_max_logit_difference": float((actual-replay).abs().max()),
                "base_weights_unchanged": True, "steps": report["steps"],
                "report_sha256": checked["report_sha256"]}
    (tmp_path / "rehearsal.json").write_text(json.dumps(evidence, indent=2) + "\n")
    weights = tmp_path / "run/adapter/adapter_model.safetensors"
    pristine = weights.read_bytes()
    with weights.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="integrity"):
        verify_checkpoint(tmp_path / "run")
    weights.write_bytes(pristine)
    verify_checkpoint(tmp_path / "run")
