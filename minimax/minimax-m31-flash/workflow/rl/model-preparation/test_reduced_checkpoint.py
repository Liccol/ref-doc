"""CPU checks for config-only checkpoint generation; no NPU or network needed."""
from __future__ import annotations

import copy
import gc
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from generate_reduced_checkpoint import build_reduced_config, generate_checkpoint, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.checkpoint_provenance import annotate_weight_update_report


def llama_config() -> dict:
    return {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 4, "hidden_size": 16, "intermediate_size": 32,
        "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 32,
        "max_position_embeddings": 64, "torch_dtype": "float32",
        "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False,
    }


class ConfigReductionTests(unittest.TestCase):
    def test_only_depth_changes_and_source_is_unchanged(self):
        source = llama_config()
        before = copy.deepcopy(source)
        reduced, plan = build_reduced_config(source, 2)
        expected = {**source, "num_hidden_layers": 2}
        self.assertEqual(reduced, expected)
        self.assertEqual(source, before)
        self.assertEqual(plan["retained_layer_indices"], [0, 1])

    def test_nested_text_config_preserves_vision(self):
        source = {"architectures": ["ExampleForConditionalGeneration"],
                  "text_config": llama_config(), "vision_config": {"num_hidden_layers": 20}}
        reduced, _ = build_reduced_config(source, 2)
        self.assertEqual(reduced["vision_config"], source["vision_config"])
        self.assertEqual(reduced["text_config"]["num_hidden_layers"], 2)

    def test_hybrid_layers_must_cover_all_types(self):
        source = {**llama_config(), "layer_types": ["linear", "linear", "full", "linear"]}
        with self.assertRaisesRegex(ValueError, "遗漏"):
            build_reduced_config(source, 2)
        reduced, _ = build_reduced_config(source, 3)
        self.assertEqual(reduced["layer_types"], ["linear", "linear", "full"])

    def test_moe_and_dense_layers_are_retained(self):
        source = {**llama_config(), "model_type": "qwen3_moe", "num_experts": 8,
                  "num_experts_per_tok": 2, "mlp_only_layers": [0, 3], "decoder_sparse_step": 1}
        with self.assertRaisesRegex(ValueError, "遗漏"):
            build_reduced_config(source, 1)
        reduced, _ = build_reduced_config(source, 2)
        self.assertEqual(reduced["mlp_only_layers"], [0])
        self.assertEqual(reduced["num_experts"], 8)
        self.assertEqual(reduced["num_experts_per_tok"], 2)

    def test_invalid_or_unsupported_configs_fail_before_allocation(self):
        cases = [
            ({**llama_config(), "num_hidden_layers": 2}, 2),
            (llama_config(), 0),
            ({**llama_config(), "quantization_config": {"quant_method": "fp8"}}, 2),
            ({**llama_config(), "full_attention_interval": 4}, 2),
            ({**llama_config(), "unknown_per_layer_values": [1, 2, 3, 4]}, 2),
            ({**llama_config(), "layer_types": ["full"]}, 2),
        ]
        for source, layers in cases:
            with self.subTest(source=source), self.assertRaises(ValueError):
                build_reduced_config(source, layers)


HAS_MODEL_RUNTIME = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "safetensors"))


@unittest.skipUnless(HAS_MODEL_RUNTIME, "requires torch, transformers and safetensors")
class CheckpointGenerationTests(unittest.TestCase):
    def test_checkpoint_round_trip_seed_and_report_scope(self):
        import torch
        from transformers import LlamaForCausalLM

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = llama_config()
            config_path = root / "config.json"
            write_json(config_path, source)
            reduced, plan = build_reduced_config(source, 2)
            outputs = [root / name for name in ("first", "same_seed", "other_seed")]
            for output, seed in zip(outputs, (42, 42, 43)):
                metadata = generate_checkpoint(config_path, output, reduced, plan, seed=seed,
                                               max_weight_gib=1, max_shard_size="5KB")
                self.assertEqual(metadata["validation_scope"], "functional_smoke")
                self.assertGreater(len(metadata["weight_files"]), 1)
                self.assertEqual(json.loads((output / "config.json").read_text()), reduced)
            models = [LlamaForCausalLM.from_pretrained(output, local_files_only=True).eval() for output in outputs]
            self.assertEqual(len(models[0].model.layers), 2)
            state = models[0].state_dict()
            self.assertTrue(all(torch.equal(value, models[1].state_dict()[name]) for name, value in state.items()))
            self.assertFalse(torch.equal(state["model.embed_tokens.weight"], models[2].state_dict()["model.embed_tokens.weight"]))
            tokens = torch.tensor([[1, 5, 8]])
            with torch.no_grad():
                self.assertTrue(torch.isfinite(models[0](tokens).logits).all())
            report = {"checkpoint_path": str(outputs[0]), "status": "passed",
                      "weight_update_acceptance": "real_checkpoint_outputs_identical"}
            annotate_weight_update_report(report)
            self.assertEqual(report["validation_scope"], "functional_smoke")
            self.assertEqual(report["weight_update_acceptance"], "synthetic_reduced_checkpoint_outputs_identical")
            report_script_dir = Path(__file__).resolve().parents[1] / "reporting"
            sys.path.insert(0, str(report_script_dir))
            from generate_rl_case_guard_readme import collect_reports, build_readme
            write_json(root / "reload_report.json", report)
            # An archived metadata copy next to reports must not hide reports.
            shutil.copyfile(outputs[0] / "reduced_checkpoint.json", root / "reduced_checkpoint.json")
            reports = collect_reports(root, [])
            # The source input config is outside the generated checkpoint.
            reports = [(path, data) for path, data in reports if path.name != "config.json"]
            self.assertEqual(len(reports), 1)
            document = build_readme(reports, "", title="test", model=None, output_dir=root)
            self.assertIn("functional_smoke", document)
            self.assertIn("4 → 2", document)
            with self.assertRaises(FileExistsError):
                generate_checkpoint(config_path, outputs[0], reduced, plan, seed=42,
                                    max_weight_gib=1, max_shard_size="1GB")
            # safetensors-backed parameters retain mmap handles on Windows.
            del state, models
            gc.collect()

    def test_size_limit_does_not_publish_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            write_json(config_path, llama_config())
            reduced, plan = build_reduced_config(llama_config(), 2)
            output = root / "checkpoint"
            with self.assertRaisesRegex(ValueError, "超过"):
                generate_checkpoint(config_path, output, reduced, plan, seed=42,
                                    max_weight_gib=1e-12, max_shard_size="1GB")
            self.assertFalse(output.exists())
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["config.json"])

    def test_moe_checkpoint_keeps_experts_and_dense_layer(self):
        import torch
        from transformers import Qwen3MoeForCausalLM

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = {**llama_config(), "model_type": "qwen3_moe",
                      "architectures": ["Qwen3MoeForCausalLM"], "head_dim": 4,
                      "num_experts": 8, "num_experts_per_tok": 2,
                      "moe_intermediate_size": 8, "mlp_only_layers": [0], "decoder_sparse_step": 1}
            config_path = root / "config.json"
            write_json(config_path, source)
            reduced, plan = build_reduced_config(source, 2)
            output = root / "checkpoint"
            generate_checkpoint(config_path, output, reduced, plan, seed=42,
                                max_weight_gib=1, max_shard_size="1GB")
            model = Qwen3MoeForCausalLM.from_pretrained(output, local_files_only=True).eval()
            self.assertEqual(model.config.num_experts, 8)
            self.assertEqual(model.model.layers[0].mlp.__class__.__name__, "Qwen3MoeMLP")
            self.assertEqual(len(model.model.layers[1].mlp.experts), 8)
            with torch.no_grad():
                self.assertTrue(torch.isfinite(model(torch.tensor([[1, 5, 8]])).logits).all())
            del model
            gc.collect()


if __name__ == "__main__":
    unittest.main()
