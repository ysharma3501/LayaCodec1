import io
import json
import os
import re
import warnings
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Type, Union
import librosa
import torch
from torch import Tensor, nn

from codec.bsq import BinarySphericalQuantizer
from codec.focalnet import FocalDecoder, FocalEncoder
from codec.vocos import Vocos
from codec.wavlm import WavLM


REGISTRY = {
    "BinarySphericalQuantizer": BinarySphericalQuantizer,
    "FocalDecoder": FocalDecoder,
    "FocalEncoder": FocalEncoder,
    "Vocos": Vocos,
    "WavLM": WavLM,
}

DEFAULT_CONFIGS = [
    "YatharthS/LayaCodec",
]


class LayaCodec(nn.Module):

    def __init__(
        self,
        encoder_name: "str" = "WavLM",
        encoder_config: "Optional[Dict[str, Any]]" = None,
        compressor_name: "str" = "FocalEncoder",
        compressor_config: "Optional[Dict[str, Any]]" = None,
        quantizer_name: "str" = "BinarySphericalQuantizer",
        quantizer_config: "Optional[Dict[str, Any]]" = None,
        decompressor_name: "str" = "FocalDecoder",
        decompressor_config: "Optional[Dict[str, Any]]" = None,
        decoder_name: "str" = "WaveNeXt",
        decoder_config: "Optional[Dict[str, Any]]" = None,
    ) -> "None":
        super().__init__()
        self.encoder_name = encoder_name
        self.encoder_config = encoder_config or {}
        self.compressor_name = compressor_name
        self.compressor_config = compressor_config or {}
        self.quantizer_name = quantizer_name
        self.quantizer_config = quantizer_config or {}
        self.decompressor_name = decompressor_name
        self.decompressor_config = decompressor_config or {}
        self.decoder_name = decoder_name
        self.decoder_config = decoder_config or {}
        self.model_id = None

        # Validate
        for name in [
            encoder_name,
            compressor_name,
            quantizer_name,
            decompressor_name,
            decoder_name,
        ]:
            if name not in REGISTRY:
                raise ValueError(
                    f"Unregistered module: {name}. Available modules: {list(REGISTRY.keys())}"
                )

        # Modules
        self.encoder = REGISTRY[encoder_name](**self.encoder_config)
        self.compressor = REGISTRY[compressor_name](**self.compressor_config)
        self.quantizer = REGISTRY[quantizer_name](**self.quantizer_config)
        self.decompressor = REGISTRY[decompressor_name](**self.decompressor_config)
        self.decoder = REGISTRY[decoder_name](**self.decoder_config)

    def encode_audio(self, audio: "str"):
        """Encodes audio into highly compressed codes from 12.5hz to 50hz"""
        wav, sr = librosa.load(audio, sr=16000, duration=10) 
        sig = torch.from_numpy(wav).float().unsqueeze(0)
        feats = self.encoder(sig.to(self.device), length=None)[0]
        lats = self.compressor(
            feats
        )[0]
        codes = self.quantizer(lats)[1]
        return codes
    def decode_codes(self, codes):
        """decodes codes into 44.1khz audio"""
        qfeats = self.decompressor(codes.float())[0]
        sig = self.decoder(qfeats)[0]
        return sig

    def info(self) -> "Dict[str, Any]":
        """Return the model information."""
        return {
            "model_id": self.model_id,
            "version": self.__version__,
            "sample_rate_input": self.sample_rate_input,
            "sample_rate_output": self.sample_rate_output,
            "causal": self.causal,
            "chunk_size": self.chunk_size,
            "latency": self.latency,
            "num_total_params": sum([x.numel() for x in self.state_dict().values()]),
        }

    def to_config(
        self,
        config: "str",
        pretrained: "bool" = False,
    ) -> "None":
        """Dump model configuration to a JSON file.

        Parameters
        ----------
        config:
            Path to local JSON file where the configuration should be dumped.
            If the given file path does not end with `.json`, `.json` is automatically appended.
        pretrained:
            Whether to dump the checkpoint along with the configuration.

        """
        if config.endswith(".json"):
            config_json = config
        else:
            config_json = f"{config}.json"

        dirpath = os.path.dirname(config_json)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        config = {
            "encoder_name": self.encoder_name,
            "encoder_config": self.encoder_config,
            "compressor_name": self.compressor_name,
            "compressor_config": self.compressor_config,
            "quantizer_name": self.quantizer_name,
            "quantizer_config": self.quantizer_config,
            "decompressor_name": self.decompressor_name,
            "decompressor_config": self.decompressor_config,
            "decoder_name": self.decoder_name,
            "decoder_config": self.decoder_config,
        }

        with open(config_json, "w") as f:
            json.dump(config, f, indent=2)

        if pretrained:
            state_dict = self.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            try:
                from safetensors.torch import save_file as safetensors_save

                checkpoint = f"{os.path.splitext(config_json)[0]}.safetensors"
                safetensors_save(state_dict, checkpoint)
            except Exception:
                # If `safetensors` not available, use `torch`
                checkpoint = f"{os.path.splitext(config_json)[0]}.pt"
                torch.save(state_dict, checkpoint)

    def to_pretrained(self, config: "str") -> "None":
        """See documentation of `to_config`."""
        return self.to_config(config, pretrained=True)

    @classmethod
    def from_config(
        cls,
        config: "str",
        pretrained: "bool" = False,
        overrides: "Optional[Dict[str, Any]]" = None,
        **kwargs: "Any",
    ) -> "FocalCodec":
        """Load model from a configuration.

        Parameters
        ----------
        config:
            Configuration source, one of the following:
              - A local JSON file (e.g. "config.json");
              - a Hugging Face repository containing "config.json" (e.g. "username/repo_name");
              - a specific JSON file hosted in a Hugging Face repository (e.g. "username/repo_name/config_xyz.json").
            If the given file path does not end with `.json`, `.json` is automatically appended.
        pretrained:
            Whether to load the corresponding pretrained checkpoint.
              - If True and a JSON file is specified, the method will look for a checkpoint file with the same
                path or URL as the configuration file but with a `.safetensors` or `.pt` extension.
              - If True and a Hugging Face repository is provided, it is assumed that either "model.safetensors"
                or "model.pt" is available.
        overrides:
            Dictionary mapping dot-separated key paths to new values that override entries in the nested configuration.
            For example, {"encoder_config.max_cached_steps": 0}.
        kwargs:
            Additional keyword arguments to pass to `huggingface_hub.hf_hub_download` if
            fetching the configuration from a remote repository.

        Returns
        -------
            A model instance initialized with the given configuration and,
            if specified, pretrained checkpoint.

        Notes
        -----
        When loading from the Hugging Face Hub, the `huggingface-hub` library must be installed.
        You can install it via `pip install huggingface-hub`.

        """
        def _override_config(
            config: "Dict[str, Any]",
            path: "str",
            value: "Any",
        ) -> "None":
            keys = path.split(".")
            tmp = config
            for key in keys[:-1]:
                tmp = tmp.setdefault(key, {})
            tmp[keys[-1]] = value

        model_id = config
        if config.endswith(".json"):
            config_json = config
        else:
            config_json = f"{config}.json"

        # Local
        if os.path.exists(config_json):
            with open(config_json) as f:
                config = json.load(f)
            if overrides is not None:
                for path, value in overrides.items():
                    _override_config(config, path, value)
            model = cls(**config)
            if pretrained:
                tgt_keys = list(model.state_dict().keys())
                try:
                    from safetensors.torch import load_file as safetensors_load

                    checkpoint = f"{os.path.splitext(config_json)[0]}.safetensors"
                    state_dict = safetensors_load(checkpoint)
                except Exception:
                    # If `.safetensors` not found, try `.pt`
                    checkpoint = f"{os.path.splitext(config_json)[0]}.pt"
                    state_dict = torch.load(checkpoint, map_location="cpu")
                state_dict = cls._remap_state_dict(state_dict, tgt_keys)
                model.load_state_dict(state_dict)
            model.model_id = model_id
            return model

        # Remote
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("`pip install huggingface-hub` to load this model")

        is_repo = bool(re.fullmatch(r"[\w\-]+/[\w\-.]+", config))

        try:
            repo_id = config if is_repo else os.path.dirname(config_json)
            filename = "config.json" if is_repo else os.path.basename(config_json)
            config_json = hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
            with open(config_json) as f:
                config = json.load(f)
            if overrides is not None:
                for path, value in overrides.items():
                    _override_config(config, path, value)
            model = cls(**config)
            if pretrained:
                tgt_keys = list(model.state_dict().keys())
                filename = "model" if is_repo else f"{os.path.splitext(filename)[0]}"
                try:
                    from safetensors.torch import load_file as safetensors_load

                    checkpoint = hf_hub_download(
                        repo_id=repo_id, filename=f"{filename}.safetensors", **kwargs
                    )
                    state_dict = safetensors_load(checkpoint)
                except Exception:
                    # If `.safetensors` not found, try `.pt`
                    checkpoint = hf_hub_download(
                        repo_id=repo_id, filename=f"{filename}.pt", **kwargs
                    )
                    state_dict = torch.load(checkpoint, map_location="cpu")
                state_dict = cls._remap_state_dict(state_dict, tgt_keys)
                model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(
              f"Could not load the specified configuration. "
                f"Available default configurations: {DEFAULT_CONFIGS}"
            ) from e
        model.model_id = model_id
        return model

    @classmethod
    def from_pretrained(
        cls,
        config: "str",
        overrides: "Optional[Dict[str, Any]]" = None,
        **kwargs: "Any",
    ) -> "FocalCodec":
        """See documentation of `from_config`."""
        return cls.from_config(config, pretrained=True, overrides=overrides, **kwargs)

    @classmethod
    def _remap_state_dict(
        cls,
        src_state_dict: "Dict[str, Tensor]",
        tgt_keys: "Sequence[str]",
    ) -> "Dict[str, Tensor]":
        # Infer checkpoint version based on its structure
        if any(
            # Separate QKV projections => version 0.0.1
            any(proj in k for proj in (".q_proj", ".k_proj", ".v_proj"))
            for k in src_state_dict
        ):
            # Order-based mapping for compressor and decompressor
            compressor_tgt_keys = sorted(
                [k for k in tgt_keys if k.startswith("compressor.")]
            )
            compressor_src_keys = sorted(
                [k for k in src_state_dict if k.startswith("compressor.")]
            )
            compressor_map = dict(zip(compressor_tgt_keys, compressor_src_keys))

            decompressor_tgt_keys = sorted(
                [k for k in tgt_keys if k.startswith("decompressor.")]
            )
            decompressor_src_keys = sorted(
                [k for k in src_state_dict if k.startswith("decompressor.")]
            )
            decompressor_map = dict(zip(decompressor_tgt_keys, decompressor_src_keys))

            tgt_state_dict = {}
            for name in tgt_keys:
                if name.startswith("encoder.") and "qkv_proj" in name:
                    prefix = name.replace("qkv_proj.weight", "").replace(
                        "qkv_proj.bias", ""
                    )
                    suffix = name.split(".")[-1]  # 'weight' or 'bias'
                    q = src_state_dict[f"{prefix}q_proj.{suffix}"]
                    k = src_state_dict[f"{prefix}k_proj.{suffix}"]
                    v = src_state_dict[f"{prefix}v_proj.{suffix}"]
                    value = torch.cat([q, k, v], dim=0)
                elif name.startswith("encoder."):
                    value = src_state_dict[name]
                elif name.startswith("compressor."):
                    value = src_state_dict[compressor_map[name]]
                elif name.startswith("decompressor."):
                    value = src_state_dict[decompressor_map[name]]
                elif name.startswith("decoder."):
                    value = src_state_dict[name]
                else:
                    raise KeyError(f"Unmapped key: {name}")

                tgt_state_dict[name] = value

            return tgt_state_dict


        return src_state_dict
