from typing import override
from collections.abc import Iterator
from collections import OrderedDict

from torch import nn
from ignite.engine import Engine
from ignite.handlers import EMAHandler
from open_clip.transformer import VisionTransformer
from timm.models.convnext import ConvNeXt, ConvNeXtBlock

from loralib.layers import PlainMultiheadAttentionLoRA
from loralib.layers import Conv2d as Conv2dLoRA


def search_attention_layers(parent_names: list[str], net: nn.Module
                            ) -> Iterator[tuple[list[str], nn.MultiheadAttention]]:
    for name, m in net.named_children():
        if isinstance(m, nn.MultiheadAttention):
            yield parent_names + [name], m
        else:
            yield from search_attention_layers(parent_names + [name], m)


def to_lora(net: nn.Module, rank: int, lora_alpha: int,
            dropout_rate: float = 0.0,
            enable_lora: list[str] | None = None) -> list[PlainMultiheadAttentionLoRA]:
    attn_layers = search_attention_layers([], net)
    lora_layers = []

    for parent_names, layer in attn_layers:
        mod = net
        for name in parent_names[:-1]:
            mod = getattr(mod, name)

        new_lora_layer = PlainMultiheadAttentionLoRA(
            layer, r=rank, lora_alpha=lora_alpha,
            dropout_rate=dropout_rate, enable_lora=enable_lora)
        setattr(mod, parent_names[-1], new_lora_layer)

        lora_layers.append(new_lora_layer)

    return lora_layers


def vit_to_lora(net: VisionTransformer, rank: int, lora_alpha: int,
                dropout_rate: float = 0.0,
                enable_lora: list[str] | None = None,
                attn_layer_ids: list[int] | None = None) -> list[PlainMultiheadAttentionLoRA]:
    if attn_layer_ids is None:
        attn_layer_ids = list(range(len(net.transformer.resblocks)))

    lora_layers = []

    for i in attn_layer_ids:
        attn_layer = net.transformer.resblocks[i].attn
        new_layer = PlainMultiheadAttentionLoRA(attn_layer, r=rank, lora_alpha=lora_alpha,
                                                dropout_rate=dropout_rate, enable_lora=enable_lora)
        net.transformer.resblocks[i].attn = new_layer

        lora_layers.append(new_layer)

    return lora_layers


def convnext_to_lora(net: ConvNeXt, rank: int, lora_alpha: int,
                     convnext_stage_ids: list[int] | None = None) -> list[Conv2dLoRA]:
    if convnext_stage_ids is None:
        convnext_stage_ids = list(range(len(net.stages)))

    lora_layers = []

    for stg in convnext_stage_ids:
        convnext_blocks: list[ConvNeXtBlock] = net.stages[stg].blocks
        for block in convnext_blocks:
            conv = block.conv_dw
            if isinstance(conv.kernel_size, tuple):
                ks = conv.kernel_size[0]
            else:
                ks = conv.kernel_size
            new_conv = Conv2dLoRA(
                r=rank,
                lora_alpha=lora_alpha,
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=ks,  # type: ignore
                stride=conv.stride,
                padding=conv.padding,
                groups=conv.groups,
                dilation=conv.dilation,
                bias=conv.bias is not None
            )
            new_conv.weight.data = conv.weight.data.clone()
            if new_conv.bias is not None:
                assert conv.bias is not None
                new_conv.bias.data = conv.bias.data.clone()

            block.conv_dw = new_conv

            lora_layers.append(new_conv)

    return lora_layers


def get_lora_parameters(model: nn.Module, bias='none'):
    # https://github.com/MaxZanella/CLIP-LoRA/blob/main/loralib/utils.py

    params = []
    for name, param in model.named_parameters():
        match bias:
            case "none":
                if 'lora_' in name:
                    params.append(param)
            case "all":
                if 'lora_' in name or 'bias' in name:
                    params.append(param)
            case "lora_only":
                if 'lora_' in name:
                    params.append(param)
                    bias_name = name.split('lora_')[0] + 'bias'
                    if bias_name in model.state_dict():
                        bias_param = dict(model.named_parameters())[bias_name]
                        params.append(bias_param)

            case _:
                raise ValueError(f"Invalid bias: {bias!r}")

    return params


class LoRAEMAHandler(EMAHandler):
    @override
    def _update_ema_model(self, engine: Engine, name: str) -> None:
        """
        Update lora weights of ema model
        """

        momentum = getattr(engine.state, name)

        ema_param_dict = OrderedDict(self.ema_model.named_parameters())
        for p_name, model_p in self.model.named_parameters():
            if "lora_" in p_name:
                ema_param_dict[p_name].mul_(1.0 - momentum) \
                    .add_(model_p.data, alpha=momentum)

        match self.handle_buffers:
            case "update":
                ema_buf_dict = OrderedDict(self.ema_model.named_buffers())
                for b_name, model_b in self.model.named_buffers():
                    if "lora_" not in b_name:
                        continue

                    try:
                        ema_buf_dict[b_name].mul_(1.0 - momentum) \
                            .add_(model_b.data, alpha=momentum)
                    except RuntimeError:
                        # Handle the case where ema_b is torch.int64, torch.int32 etc.,
                        # where a runtime error will be thrown when performing the in-place operations with floats.
                        # In this case, just copy the data
                        ema_buf_dict[b_name].data = model_b.data

            case "copy":
                # assign the buffers
                ema_buf_dict = OrderedDict(self.ema_model.named_buffers())
                for b_name, model_b in self.model.named_buffers():
                    if "lora_" in b_name:
                        ema_buf_dict[b_name].data = model_b.data
