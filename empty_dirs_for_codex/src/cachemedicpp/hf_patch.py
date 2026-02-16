"""HF attention patching utilities for CacheMedic++ insertion point."""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Callable


def enforce_v1_constraints(config: dict[str, Any], *, batch_size: int | None = None) -> None:
    model_cfg = config.get("model", {})
    attn_impl = model_cfg.get("attn_implementation")
    if attn_impl != "eager":
        raise ValueError(
            "CacheMedic++ v1 fail-closed: model.attn_implementation must be 'eager'."
        )

    if batch_size is not None and batch_size != 1:
        raise ValueError("CacheMedic++ v1 fail-closed: only batch_size=1 is supported.")


def assert_bsz1(tensor: Any, *, context: str) -> None:
    shape = getattr(tensor, "shape", None)
    if not shape or int(shape[0]) != 1:
        raise ValueError(f"CacheMedic++ v1 fail-closed: bsz=1 required ({context}).")


@dataclass
class PatchRecord:
    layer_index: int
    module: Any
    method_name: str
    original_callable: Any


@dataclass
class AttentionPatchHandle:
    patches: list[PatchRecord]

    def unpatch(self) -> None:
        for rec in self.patches:
            setattr(rec.module, rec.method_name, rec.original_callable)
        self.patches.clear()


def _patch_attn_method(attn: Any, layer_idx: int, patch_fn: Callable[[int, Any, Any, Any], tuple[Any, Any]]) -> PatchRecord:
    original = attn._attn

    def wrapped_attn(
        self: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any = None,
        head_mask: Any = None,
        *,
        _orig=original,
        _layer_idx=layer_idx,
    ) -> Any:
        assert_bsz1(query, context=f"layer={_layer_idx} query")
        patched_key, patched_value = patch_fn(_layer_idx, query, key, value)
        return _orig(
            query,
            patched_key,
            patched_value,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

    attn._attn = types.MethodType(wrapped_attn, attn)
    return PatchRecord(
        layer_index=layer_idx, module=attn, method_name="_attn", original_callable=original
    )


def _patch_forward_method(
    attn: Any,
    layer_idx: int,
    patch_fn: Callable[[int, Any, Any, Any], tuple[Any, Any]],
) -> PatchRecord:
    # This path supports newer GPT-2 implementations where `_attn` is replaced by
    # eager_attention_forward / ALL_ATTENTION_FUNCTIONS dispatch inside forward().
    from transformers.cache_utils import EncoderDecoderCache  # type: ignore
    from transformers.models.gpt2.modeling_gpt2 import (  # type: ignore
        ALL_ATTENTION_FUNCTIONS,
        eager_attention_forward,
    )

    original = attn.forward

    def wrapped_forward(
        self: Any,
        hidden_states: Any,
        past_key_values: Any = None,
        cache_position: Any = None,
        attention_mask: Any = None,
        head_mask: Any = None,
        encoder_hidden_states: Any = None,
        encoder_attention_mask: Any = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Any:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    curr_past_key_value = past_key_values.cross_attention_cache
                else:
                    curr_past_key_value = past_key_values.self_attention_cache
            else:
                curr_past_key_value = past_key_values
                is_updated = False
        else:
            curr_past_key_value = None
            is_updated = False

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, q_attn must be defined on GPT2Attention."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_value.layers[self.layer_idx].keys
                value_states = curr_past_key_value.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(
                    self.split_size, dim=2
                )
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(
                self.split_size, dim=2
            )
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            cache_position_local = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                {"cache_position": cache_position_local},
            )
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        assert_bsz1(query_states, context=f"layer={layer_idx} query")
        patched_key, patched_value = patch_fn(layer_idx, query_states, key_states, value_states)

        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention
        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable[..., Any] = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, patched_key, patched_value, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                patched_key,
                patched_value,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, attn_weights

    attn.forward = types.MethodType(wrapped_forward, attn)
    return PatchRecord(
        layer_index=layer_idx, module=attn, method_name="forward", original_callable=original
    )


def patch_gpt2_attention(
    model: Any,
    patch_fn: Callable[[int, Any, Any, Any], tuple[Any, Any]],
    *,
    protect_layers: list[int] | None = None,
) -> AttentionPatchHandle:
    """Patch GPT-2 block attentions after KV concat and before logits.

    The wrapper hooks the module `_attn(query, key, value, ...)`, where key/value
    are already concatenated with `past_key_values` in the eager path.
    """
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise TypeError("Expected a GPT-2-like model with model.transformer.h blocks.")

    patch_records: list[PatchRecord] = []
    layers = list(range(len(model.transformer.h)))
    targets = set(protect_layers if protect_layers is not None else layers)
    for layer_idx, block in enumerate(model.transformer.h):
        if layer_idx not in targets:
            continue
        attn = getattr(block, "attn", None)
        if attn is None:
            raise TypeError(f"Layer {layer_idx} has no attention module.")
        if hasattr(attn, "_attn"):
            patch_records.append(_patch_attn_method(attn, layer_idx, patch_fn))
        elif hasattr(attn, "forward"):
            patch_records.append(_patch_forward_method(attn, layer_idx, patch_fn))
        else:
            raise TypeError(
                f"Layer {layer_idx} attention module exposes neither _attn nor forward."
            )

    if not patch_records:
        raise RuntimeError("No attention layers were patched. Check protect_layers selection.")
    return AttentionPatchHandle(patches=patch_records)
