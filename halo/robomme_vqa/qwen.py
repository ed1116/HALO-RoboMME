"""Lazy Qwen3-VL backend for offline corpus construction only."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .contracts import ModelConfig


class ChatBackend(Protocol):
    """Minimal interface used by generator and judge calls in tests and production."""

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        generation: Mapping[str, Any],
    ) -> str:
        ...


class Qwen3VLBackend:
    """Load the pinned model lazily; construction never downloads weights."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        device: str = "cuda",
        allow_download: bool = False,
    ) -> None:
        self.config = config
        self.device = device
        self.allow_download = allow_download
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self.config.verify_runtime()
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import Qwen3VLForConditionalGeneration as ModelClass
            except ImportError:
                from transformers import AutoModelForImageTextToText as ModelClass
        except ImportError as error:
            raise RuntimeError(
                "Qwen3-VL requires a separate, compatible Transformers environment"
            ) from error

        if self.config.dtype != "float16":
            raise RuntimeError("only the approved float16 VQA configuration is supported")
        load_options = {
            "revision": self.config.model_revision,
            "local_files_only": not self.allow_download,
            "torch_dtype": torch.float16,
            "attn_implementation": self.config.attention_implementation,
        }
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            revision=self.config.processor_revision,
            local_files_only=not self.allow_download,
        )
        expected_image_size = {
            "shortest_edge": self.config.image_min_pixels,
            "longest_edge": self.config.image_max_pixels,
        }
        expected_video_size = {
            "shortest_edge": self.config.video_min_pixels,
            "longest_edge": self.config.video_max_pixels,
        }
        if self._processor.image_processor.size != expected_image_size:
            raise RuntimeError(
                "pinned Qwen image-processor limits differ from the VQA config"
            )
        if self._processor.video_processor.size != expected_video_size:
            raise RuntimeError(
                "pinned Qwen video-processor limits differ from the VQA config"
            )
        self._model = ModelClass.from_pretrained(self.config.model_id, **load_options)
        if getattr(self._model.config, "_attn_implementation", None) != "sdpa":
            raise RuntimeError("Qwen3-VL did not activate the required SDPA attention backend")
        self._model.eval().to(self.device)

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        generation: Mapping[str, Any],
    ) -> str:
        self._load()
        assert self._model is not None and self._processor is not None
        inputs = self._processor.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output_ids = self._model.generate(**inputs, **dict(generation))
        prompt_length = inputs["input_ids"].shape[1]
        generated = output_ids[:, prompt_length:]
        return self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
