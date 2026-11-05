# Bard-VL modeling provenance (transformers 5.x port)

- Source: `nemo_automodel/components/models/bard_vl/` of the Bard-VL tree.
- License: Apache-2.0; see `LICENSE` in the Bard-VL tree.
- Files: `configuration_bard_vl.py`, `modeling_bard_vl.py`. `_autodoc.py` replaces the single helper the config imported from `nemo_automodel.components.models.utils`; `modeling_bard_vl.py` has no `nemo_automodel` imports.

## Why this exists

Bard-VL targets transformers 4.57, and this repository pins transformers 5.x. Bard-VL's own `qwen3_vl/` is a near-verbatim copy of transformers 4.57's generated Qwen3-VL, and `bard_vl/` is that file plus a contained delta: `is_causal = False`, a read-only KV-cache branch, forced SDPA while training, and the mask-diffusion `generate` helpers. The port therefore keeps Bard-VL's files as the base and only rewrites the transformers API calls, using transformers 5.12's own Qwen3-VL as the reference for each new call shape. Deriving from 5.12's Qwen3-VL instead would have pulled in 1,135 lines of unrelated 4.57 to 5.12 behavior changes.

## Local changes

Each change is marked with a `transformers>=5` comment at the call site.

`configuration_bard_vl.py`

- Import the vendored `_autodoc` helper instead of `nemo_automodel`.
- `BardVLTextConfig.ignore_keys_at_rope_validation = {"mrope_section", "mrope_interleaved"}`, matching `Qwen3VLTextConfig` in transformers 5.12.
- Replace `rope_config_validation(self, ignore_keys=...)`, removed in 5.x, with `self.standardize_rope_params()` and `self.validate_rope()` after `super().__init__`. `rope_theta` is folded into `rope_parameters`; `rope_scaling` remains readable as a property alias.

`modeling_bard_vl.py`

- `BardVLTextRotaryEmbedding.__init__`: `ROPE_INIT_FUNCTIONS` no longer has a `"default"` key, so the default branch is a static `compute_default_rope_parameters` on the module, copied from transformers 5.12's `Qwen3VLTextRotaryEmbedding`. Reads come from `config.rope_parameters`. It is numerically identical to 4.57's default for this config, which sets no `partial_rotary_factor`.
- Read-only KV branch (`use_cache=False`): `Cache.__getitem__` was removed, so per-layer tensors are read from `cache.layers[i].keys` and `.values`, with a `numel() > 0` guard because a lazily initialized 5.x layer holds an empty tensor rather than being absent. The no-commit semantics are unchanged.
- `@check_model_inputs()` on three forwards becomes `@merge_with_config_defaults`, the new name.
- `self.padding_idx = config.pad_token_id` becomes `getattr(config, "pad_token_id", None)`, because 5.x moved the legacy token ids off `PreTrainedConfig`.
- `create_causal_mask`: `input_embeds` becomes `inputs_embeds`, and `cache_position` is no longer a parameter.
- `BardVLPreTrainedModel._init_weights` is new, and `dim` and `theta` stay on `BardVLVisionRotaryEmbedding`. transformers 5.x builds the model on the meta device and only materializes tensors the checkpoint provides or `_init_weights` handles. Without this the vision `inv_freq` stays on meta and reads back as zeros, and the vision tower silently loses all positional information. It mirrors `Qwen3VLPreTrainedModel._init_weights` in transformers 5.12.

Beyond these, the files only went through this repository's formatters, black, isort, and autoflake.

## Bumping transformers

Text-only generation is not sufficient evidence after a bump: it never runs the vision tower, so it stayed bitwise identical while the vision rotary above was zeroed, and image-conditioned generation agreed on only 167 of 256 codes. Re-run an image-conditioned comparison against the previous version, serially, because concurrent GPU work changes cuBLAS kernel selection and breaks bitwise comparison.
