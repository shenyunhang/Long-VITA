
import os
import logging
from typing import Dict, Literal, Optional, Tuple, Union

from collections import namedtuple
from functools import partial
from typing import List

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector

from megatron.training import get_args

class VisionModel(torch.nn.Module):
    def __init__(self, vision_transformer_config, vision_transformer_layer_spec, vision_projector_config, vision_projector_layer_spec, vision_projector_type):
        super().__init__()
        self.vision_model = CLIPViTModel(vision_transformer_config, vision_transformer_layer_spec)

        # Map (intermediate) vision model outputs to the language model input dimension.
        self.vision_projection = MultimodalProjector(
            vision_projector_config,
            vision_projector_layer_spec,
            vision_projector_type,
            vision_transformer_config.hidden_size,  # input size to the projection.
        )
    
    def forward(self, images):
        image_embeddings = self.vision_model(images)  # [b, img_seq_len, h_vision]
        # map vision model output size to language model input size.
        image_embeddings = self.vision_projection(image_embeddings)  # [b, img_seq_len, h_language]
        return image_embeddings


def default_external_feature_model_provider(config, vision_transformer_config, vision_transformer_layer_spec, vision_projector_config, vision_projector_layer_spec, vision_projector_type):
    return VisionModel(vision_transformer_config, vision_transformer_layer_spec, vision_projector_config, vision_projector_layer_spec, vision_projector_type)

class GPTVLModel(LanguageModule):
    """GPT Transformer vision language model.

    Args:
        config (TransformerConfig): Transformer config
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional): Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):  Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal['learned_absolute', 'rope'] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        external_feature_model_provider = None,
        external_args: tuple = (),
        allow_missing_keys = []
    ) -> None:
        super().__init__(config=config)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent

        if self.pre_process:
            if external_feature_model_provider is None:
                external_feature_model_provider = default_external_feature_model_provider
            self.external_feature_model = external_feature_model_provider(config, *external_args)

            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
            )

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        # Output
        if post_process:
            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs stored
                # in gradient buffer to calculate the weight gradients for the embedding final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None

            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()
        # This allows ignoring missing weights for the vision projection during checkpoint loading.
        # This should be disabled by default but can be enabled if your checkpoint contains pretrained
        # vision and language models but not the projection from vision model outputs to language model inputs.
        if allow_missing_keys:
            self.register_load_state_dict_post_hook(
                partial(_load_state_dict_hook_ignore_param_names, allow_missing_keys)
            )

    def vision_projector_freeze(self):
        print(f"vision_projector_freeze", flush=True)
        for name, param in self.named_parameters():
            if "external_feature_model." in name and ".vit." not in name:
                param.requires_grad = False
                print(f"=> set param {name} {param.size()} requires grad to False.", flush=True)
            else:
                # print(f"=> keep param {name} {param.size()} requires grad to {param.requires_grad}.", flush=True)
                pass
        return self

    def vision_model_freeze(self):
        print(f"vision_model_freeze", flush=True)
        for name, param in self.named_parameters():
            if "external_feature_model.vit." in name:
                param.requires_grad = False
                print(f"=> set param {name} {param.size()} requires grad to False.", flush=True)
            else:
                # print(f"=> keep param {name} {param.size()} requires grad to {param.requires_grad}.", flush=True)
                pass
        return self

    def language_model_freeze(self):
        print(f"language_model_freeze", flush=True)
        for name, param in self.named_parameters():
            if "external_feature_model." in name:
                # print(f"=> keep param {name} {param.size()} requires grad to {param.requires_grad}.", flush=True)
                pass
            else:
                param.requires_grad = False
                print(f"=> set param {name} {param.size()} requires grad to False.", flush=True)
        return self

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        external_inputs: dict = {},
        tokentype_ids=None,
        logit_mask=None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units
        """
        # print(f"torch.distributed.get_rank() {torch.distributed.get_rank()} forward {torch.cuda.memory_summary()}")
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        args = get_args()
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            if hasattr(inference_params, 'external_inputs') and inference_params.external_inputs is not None and not inference_params.key_value_memory_dict:
                external_inputs = inference_params.external_inputs
                print("Setting vision inputs to inference_params.external_inputs......")
            if external_inputs:
                external_feature = self.external_feature_model(**external_inputs)
                external_feature_dict = {"features": external_feature}
                for k in external_inputs:
                    if 'indices' in k or 'pre_len' == k:
                        external_feature_dict[k] = external_inputs[k]
                decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids, external_feature_dict=external_feature_dict)
            else:
                decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None
        # print(f"torch.distributed.get_rank() {torch.distributed.get_rank()} decoder_input {torch.cuda.memory_summary()}")

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        # print(f"torch.distributed.get_rank() {torch.distributed.get_rank()} rotary_pos_emb {torch.cuda.memory_summary()}")

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )
        # print(f"torch.distributed.get_rank() {torch.distributed.get_rank()} hidden_states {hidden_states.size()}")

        if not self.post_process:
            return hidden_states


        if logit_mask is not None and labels is None and False:

            # if args.sequence_parallel:
            #     # [b s] -> [s b] -> [b s]
            #     logit_mask = tensor_parallel.scatter_to_sequence_parallel_region(logit_mask.transpose(0, 1)).transpose(0, 1)

            # print(f"torch.distributed.get_rank() {torch.distributed.get_rank()} logit_mask {logit_mask.size()}")
            s = hidden_states.size(0)
            b = hidden_states.size(1)
            c = hidden_states.size(2)
            assert b == 1

            # [s b c]
            hidden_states = torch.masked_select(hidden_states, logit_mask.transpose(0, 1).unsqueeze(2)).reshape(-1, b, c)
        # print(f"torch.distributed.get_rank() {torch.distributed.get_rank()} logit_mask {logit_mask.size()}")
        # print(f"torch.distributed.get_rank() {torch.distributed.get_rank()} logit_mask {logit_mask}")
        # print(f"torch.distributed.get_rank() {torch.distributed.get_rank()} hidden_states {hidden_states.size()}")

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight, logit_mask=logit_mask)
        # print(f"torch.distributed.get_rank() {torch.distributed.get_rank()} logits {logits.size()}")
        # new add to scale logits
        if args.output_multiplier_scale:
            logits = logits * args.output_multiplier_scale

        if args.output_logit_softcapping:
            logits = logits / args.output_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * args.output_logit_softcapping

        if labels is None:
            # print("logits", logits.size())
            # if 'context_length' in external_inputs:
            #     context_length = external_inputs["context_length"]
            #     return logits[context_length - 1:context_length, :, :].transpose(0, 1).contiguous()

            # return logits.transpose_(0, 1)

            # logits = logits.cpu().detach()
            # torch.cuda.empty_cache()

            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        if logit_mask is not None:
            b = logit_mask.size(0)
            c = logits.size(2)
            assert b == 1

            # if args.sequence_parallel:
            #     # [b s] -> [s b] -> [b s]
            #     logit_mask = tensor_parallel.gather_from_sequence_parallel_region(logit_mask.transpose(0, 1), tensor_parallel_output_grad=False).transpose(0, 1)

            with torch.no_grad():
                # [b s]
                labels = torch.masked_select(labels, logit_mask).reshape(b, -1)
            # # [s b c]
            # logits = torch.masked_select(logits, logit_mask.transpose(0, 1).unsqueeze(2)).reshape(-1, b, c)

        # print(f"labels {labels.size()}")
        # print(f"logits {logits.size()}")

        if args.is_instruction_dataset:
            labels = labels[:, 1:].contiguous()
            logits = logits[:-1, :, :].contiguous()

        if logits.sum().isnan():
            global_rank = torch.distributed.get_rank()
            raise ValueError(f'Rank {global_rank}: found NaN in local forward logits calculation. '
                             f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        # if logit_mask is not None:
        #     if args.is_instruction_dataset:
        #         logit_mask = logit_mask[:, 1:]
        #     # print(f"logit_mask {logit_mask.size()}")
        #     # print(f"labels {labels.size()}")
        #     # print(f"logits {logits.size()}")
        #     b = logit_mask.size(0)
        #     c = logits.size(2)
        #     assert b == 1

        #     # [b s]
        #     labels = torch.masked_select(labels, logit_mask).reshape(b, -1)
        #     # [s b c]
        #     logits = torch.masked_select(logits, logit_mask.transpose(0, 1).unsqueeze(2)).reshape(-1, b, c)


        loss = self.compute_language_model_loss(labels, logits)

        return loss

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """ Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the _extra_state key
        # but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        return sharded_state_dict

def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    By default, this should not be used to avoid accidentally missing weights in checkpoint loading.

    Example use case: Use this for the vision projection if you want to load a checkpoint that contains vision and language model weights
    but not the vision projection weights.

    Args:
        param_names (list of str): Parameter names allowed to be missing when calling load_state_dict. Here we use fuzzy matching.
        module (torch.nn.Module): The torch module this hook applies to. Unused here but required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys, which collect the missing and unexpected
            keys when calling load_state_dict on this torch module, respectively.
    """
    for key in list(incompatible_keys.missing_keys):
        flag = False
        for param_name in param_names:
            if param_name in key:
                flag = True
                break
        if flag:
            incompatible_keys.missing_keys.remove(key)
