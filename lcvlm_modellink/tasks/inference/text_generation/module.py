# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import abc
import logging
from typing import Optional, Union

import torch
from torch import distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from lcvlm_modellink.core.models.multimodal.gpt_vl_model import GPTVLModel
from megatron.training import get_args, global_vars
from megatron.core import parallel_state

from modellink.tasks.preprocess.templates import Template, get_model_template


class MegatronModuleForCausalLMABC(torch.nn.Module, abc.ABC):
    """
    Megatron specific extensions of torch Module with support
    for text generation.
    """

    def __init__(self):
        super(MegatronModuleForCausalLMABC, self).__init__()
        self.top_k = 50
        self.top_p = 1.0
        self.do_sample = False
        self.num_beams = 1
        self.temperature = 1.0
        self.max_length = 128
        self.max_new_tokens = 0
        self.eos_token_id = None
        self.bos_token_id = None
        self.pad_token_id = None
        self.num_return_sequences = 1
        self.length_penalty = 1.0
        self.tokenizer_new = None
        self.recompute = True
        self.detokenize = True
        self.include_input = False
        self.stream = False
        self.return_output_log_probs = False

    @classmethod
    def from_pretrained(
            cls,
            model_provider,
            pretrained_model_name_or_path: Optional[Union[str, os.PathLike, None]] = None,
            **kwargs
    ):
        """
        This is an API for initializing model and loading weight.

        Parameters:
        ----------
        model_provider(`func`):
            Function used to generate model objects which is similar to the training define.
        pretrained_model_name_or_path(`str`, *optional*, defaults to None):
           File path of Model weight in megatron format (TP, PP may be used).
           If it is None, the random initialized weights will be used.
        """

    def generate(self, input_ids=None, **kwargs):
        """
        This is an API for text generation which complies with most huggingface definition.

        - *greedy decoding* if `do_sample=False`
        - *top-k decoding* if `top_k>0`
        - *top-p decoding* if `top_p>0.0`
        - *beam-search decoding* if `num_beams>1`

        Parameters:
        ----------
        input_ids(str | list | LongTensor):
            The text entered by the user, e.g. 'Hello!'
            Or
            The text, which encoded by tokenizer, entered by the user, e.g. [0, 13, 5, ...]
            Or
            The List, which includes multi texts or tokens,
            e.g. [['Hello!'], ["How are you?"]] | [[0, 13, 5, ...], [0, 21, ...]].
            Notice that in beam-search mode multi texts or tokens is forbidden.
        do_sample (`bool`, *optional*, defaults to `False`):
            Whether to use sampling ; use greedy decoding otherwise.
        top_k (`int`, *optional*, defaults to 0):
            The number of the highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (`float`, *optional*, defaults to 1.0):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to
            `top_p` or higher are kept for generation.
        temperature (`float`, *optional*, defaults to 1.0):
            The value used to modulate the next token probabilities.
        num_beams (`int`, *optional*, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        max_length (`int`, *optional*, defaults to 20):
            The maximum length the generated tokens can have. Corresponds to the length of the input prompt +
            `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
        max_new_tokens (`int`, *optional*):
            The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        eos_token_id (`int`, *optional*):
            The id of the *end-of-sequence* token. Optionally,
            use a list to set multiple *end-of-sequence* tokens.
        bos_token_id (`int`, *optional*):
            The id of the *beginning-of-sequence* token. Optionally,
            use a list to set multiple *beginning-of-sequence* tokens.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        tokenizer (`obj`, *optional*, defaults to None):
            If you don't want to use the tokenizer initialized by megatron, you can pass it in HF format here.
        length_penalty (`float`, *optional*, defaults to 1.0):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
            the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
            likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
            `length_penalty` < 0.0 encourages shorter sequences. Only activate in beam search mode.
        num_return_sequences(`int`, *optional*, defaults to 1):
            The number of independently computed returned sequences for each element in the batch. Only activate
            in beam search mode.
        recompute (`bool`, *optional*, defaults to True):
            Whether the model not to uses the last result in computing next token.
        detokenize (`bool`, *optional*, defaults to True):
            Whether to detokenize tokens into characters.
        include_input (`bool`, *optional*, defaults to False):
            Whether the output contains the context instruction.
        stream (`bool`, *optional*, defaults to False):
            Whether the output is streamed one by one.
        return_output_log_probs(`bool`, *optional*, defaults to False):
            Whether to return a probability distribution for each token.
            Note that the accumulated probability (i.e. Score) of the whole sentence will be return in beam search mode.
        """
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.max_length = kwargs.pop("max_length", 128)
        self.max_new_tokens = kwargs.pop("max_new_tokens", 0)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.tokenizer_new = kwargs.pop("tokenizer", None)
        self.recompute = kwargs.pop("recompute", True)
        self.detokenize = kwargs.pop("detokenize", True)
        self.include_input = kwargs.pop("include_input", False)
        self.stream = kwargs.pop("stream", False)
        self.return_output_log_probs = kwargs.pop("return_output_log_probs", False)


class MegatronModuleForCausalLM(MegatronModuleForCausalLMABC):
    """
    Megatron specific extensions of torch Module with support
    for text generation.
    """

    def __init__(self, *args, **kwargs):
        super(MegatronModuleForCausalLM, self).__init__()
        from megatron.training import get_tokenizer
        from .utils import greedy_search_or_sampling
        from .generation import beam_search
        from .communication import broadcast_float_list
        from .communication import broadcast_int_list
        from .communication import broadcast_tensor

        args = get_args()
        args.max_tokens_to_oom = args.max_tokens_to_oom if hasattr(args, "max_tokens_to_oom") else 4096
        args.inference_batch_times_seqlen_threshold = args.inference_batch_times_seqlen_threshold \
            if hasattr(args, "inference_batch_times_seqlen_threshold") else 4

        self.padded_vocab_size = args.padded_vocab_size
        self.pipeline_size_larger_than_one = args.pipeline_model_parallel_size > 1

        try:
            self.tokenizer = get_tokenizer().tokenizer
        except AssertionError:
            self.tokenizer = None

        # import module to avoid error of circular import
        self.greedy_search_or_sampling = greedy_search_or_sampling
        self.beam_search_in_sampling = beam_search
        self.broadcast_float_list = broadcast_float_list
        self.template = None
        if hasattr(args, "prompt_type") and args.prompt_type is not None:
            self.template = get_model_template(args.prompt_type.strip())
            from modellink.tasks.preprocess.formatter import EmptyFormatter
            self.template.format_system = EmptyFormatter()

        self.broadcast_int_list = broadcast_int_list
        self.broadcast_tensor = broadcast_tensor

        if self.tokenizer is not None:
            from cognitron_vl.tokenizer import update_tokenizer
            self.tokenizer = update_tokenizer(self.tokenizer)

        from cognitron_vl.data.processor.image_processor import ImageProcessor
        self.image_processor = ImageProcessor(
            process_type=args.vision_process_type,
            image_size=args.image_size,
            normalize_type=args.vision_normalize_type,
            min_patch_grid=args.min_patch_grid,
            max_patch_grid=args.max_patch_grid,
        )

    def get_image_tokens(self):
        from cognitron_vl.constants import IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN
        image_tokens = IMG_CONTEXT_TOKEN

        return image_tokens

    def get_video_tokens(self):
        from cognitron_vl.constants import VID_START_TOKEN, VID_END_TOKEN, VID_CONTEXT_TOKEN
        video_tokens = VID_CONTEXT_TOKEN

        return video_tokens

    def update_instruction(self, instruction):
        image_tokens = self.get_image_tokens()
        if isinstance(instruction, list):
            instruction = [x.replace("<image>", image_tokens) for x in instruction]
        else:
            instruction = instruction.replace("<image>", image_tokens)

        video_tokens = self.get_video_tokens()
        if isinstance(instruction, list):
            instruction = [x.replace("<video>", video_tokens) for x in instruction]
        else:
            instruction = instruction.replace("<video>", video_tokens)

        return instruction

    @staticmethod
    def _ids_check(ids, tokenizer):
        checked_ids = []
        for per_ids in ids:
            if per_ids == torch.Size([]) and torch.max(per_ids) >= len(tokenizer):
                warning_info = "The output ids exceeds the tokenizer length, " \
                               "the clamp operation is enforced, please check!!"
                logging.warning(warning_info)
                checked_ids.append(torch.clamp(per_ids, min=0, max=len(tokenizer)) - 1)
            else:
                checked_ids.append(per_ids)
        return checked_ids

    @classmethod
    def from_pretrained(
            cls,
            model_provider, pretrained_model_name_or_path: Optional[Union[str, os.PathLike, None]] = None,
            **kwargs
    ) -> MegatronModuleForCausalLMABC:
        from megatron.training import get_model
        from megatron.training.checkpointing import load_checkpoint
        from megatron.core.distributed import DistributedDataParallel as LocalDDP
        from megatron.legacy.model import Float16Module as MegatronFloat16Module
        from megatron.training.utils import unwrap_model

        args = get_args()

        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        args.model = get_model(model_provider, wrap_with_ddp=False)

        if pretrained_model_name_or_path:
            args.load = pretrained_model_name_or_path

        if args.load:
            load_checkpoint(args.model, None, None, 'load', strict=True)

        unwrap_classes = (torchDDP, LocalDDP, MegatronFloat16Module)

        return unwrap_model(args.model, unwrap_classes)[0]

    def generate(self, input_ids=None, **kwargs):
        args = get_args()

        if parallel_state.get_data_parallel_world_size() // parallel_state.get_expert_model_parallel_world_size() > 1:
            raise ValueError("In this inference mode data parallel is forbidden.")

        super(MegatronModuleForCausalLM, self).generate(input_ids=input_ids, **kwargs)

        # =======================================
        # Make sure input params are available
        # to all ranks
        # =======================================
        self._broadcast_config(args)
        # =======================================
        # Add additional parameters to args which
        # may be used in original logic of codes
        # =======================================
        for addition_key, addition_val in kwargs.items():
            setattr(args, addition_key, addition_val)

        # =======================================
        # Initialize the tokenizer to choose
        # whether to use customizing tokenizer
        # =======================================
        self._init_tokenizer(args)
        stop_ids = []
        if hasattr(args, "add_eos_token") and args.add_eos_token:
            stop_ids = [self.tokenizer.convert_tokens_to_ids(token)
                        for token in args.add_eos_token]
        # =======================================
        # Tokenize the prompts and broadcasting,
        # so you don't need to pass the prompt on
        # each process.
        # =======================================

        image_path_list = kwargs.pop("image_path_list", None)
        image_list = kwargs.pop("image_list", None)
        video_path_list = kwargs.pop("video_path_list", None)
        # if image_path_list is not None or image_list is not None:
        #     input_ids = self.update_instruction(input_ids)
        # print(f"generate input_ids {input_ids}")

        context_tokens, master_rank = self._tokenize(input_ids)
        args.master_rank = master_rank
        args.micro_batch_size = len(context_tokens)

        stop_token = [args.eos_id] + stop_ids

        if hasattr(args, "prompt_type") and args.prompt_type is not None:
            stop_ids = stop_ids + [self.tokenizer.convert_tokens_to_ids(token) for token in self.template.stop_words] + \
                       [self.tokenizer.eos_token_id]

            stop_token = [args.eos_id] + stop_ids

        has_image = 0
        if image_path_list is not None or image_list is not None or video_path_list is not None:
            external_inputs, context_tokens = get_external_inputs(
                context_tokens,
                image_list,
                image_path_list,
                video_path_list,
                self.tokenizer,
                self.image_processor,
            )
            has_image = 1

            external_indices = external_inputs["indices"]
            external_images = external_inputs["images"]

            external_indices_size = list(external_inputs["indices"].size())
            external_images_size = list(external_inputs["images"].size())
        else:
            external_inputs = None

            external_indices = None
            external_images = None

            external_indices_size = None
            external_images_size = None

        has_image = self.broadcast_int_list(1, [has_image, ], rank=0)[0].item()
        # print("has_image", has_image)
        if has_image:
            external_indices_size = self.broadcast_int_list(3, external_indices_size, rank=0).tolist()
            external_images_size = self.broadcast_int_list(4, external_images_size, rank=0).tolist()

            # print("external_indices_size", external_indices_size)
            # print("external_images_size", external_images_size)

            external_indices = self.broadcast_tensor(external_indices_size, torch.int64, external_indices, rank=0)
            external_images = self.broadcast_tensor(external_images_size, torch.bfloat16, external_images, rank=0)

            external_inputs = {}
            external_inputs["indices"] = external_indices
            external_inputs["images"] = external_images

        # print("context_tokens", context_tokens)
        # =======================================
        # Get the streaming tokens generator
        # =======================================
        if self.num_beams > 1:
            token_stream = self.beam_search_in_sampling(
                args.model[0],
                context_tokens,
                beam_size=self.num_beams,
                stop_token=stop_token,
                num_return_gen=self.num_return_sequences,
                length_penalty=self.length_penalty
            )
        else:
            token_stream = self.greedy_search_or_sampling(
                args.model[0],
                context_tokens,
                stop_ids=stop_ids,
                external_inputs=external_inputs,
            )

        # =======================================
        # Post processions in order to get final
        # output texts/tokens
        # =======================================
        return self._token_generator(token_stream)

    def _broadcast_config(self, args):
        values = [
            self.num_beams,
            self.do_sample,
            self.top_k,
            self.top_p,
            self.temperature,
            self.max_length,
            self.max_new_tokens,
            self.length_penalty,
            self.return_output_log_probs,
            self.stream
        ]

        values_float_tensor = self.broadcast_float_list(len(values), float_list=values)
        self.num_beams = int(values_float_tensor[0].item())
        self.do_sample = bool(values_float_tensor[1].item())
        self.top_k = int(values_float_tensor[2].item())
        self.top_p = values_float_tensor[3].item()
        self.temperature = values_float_tensor[4].item()
        self.max_length = int(values_float_tensor[5].item())
        self.max_new_tokens = int(values_float_tensor[6].item())
        self.length_penalty = values_float_tensor[7].item()
        self.return_output_log_probs = bool(values_float_tensor[8].item())
        self.stream = bool(values_float_tensor[9].item())

        setattr(args, "text_generation_config", {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "temperature": self.temperature,
            "recompute": self.recompute,
            "return_output_log_probs": self.return_output_log_probs,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "greedy": True if not self.do_sample else False
        })

    def _init_tokenizer(self, args):
        self.tokenizer = self.tokenizer if self.tokenizer_new is None else self.tokenizer_new
        global_vars._GLOBAL_TOKENIZER = self.tokenizer

        if self.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.pad_token_id
        if self.eos_token_id is not None:
            self.tokenizer.eos_token_id = self.eos_token_id
        if self.bos_token_id is not None:
            self.tokenizer.bos_token_id = self.bos_token_id

        if self.tokenizer.eos_token_id is not None:
            args.eos_id = self.tokenizer.eos_token_id
            args.eod_id = self.tokenizer.eos_token_id
        else:
            raise ValueError("Your tokenizer doesn't include eos_token.")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    def _encode_no_template(self, input_ids):
        context_tokens = [[]]
        if isinstance(input_ids, str):
            context_tokens = [self.tokenizer.encode(input_ids)]
        elif torch.is_tensor(input_ids):
            if len(input_ids.shape) == 1:
                context_tokens = input_ids.unsqueeze(0).numpy().tolist()
            elif len(input_ids.shape) == 2:
                context_tokens = input_ids.numpy().tolist()
        elif isinstance(input_ids, (tuple, list)):
            if len(input_ids) and isinstance(input_ids[0], (tuple, list)):
                context_tokens = input_ids
            elif len(input_ids) and isinstance(input_ids[0], int):
                context_tokens = [input_ids]
            elif len(input_ids) and isinstance(input_ids[0], str):
                context_tokens = [self.tokenizer.encode(val) for val in input_ids]
        else:
            raise TypeError("Please check input_ids in correct type.")

        return context_tokens


    def _encode_by_template(self, input_ids):
        context_tokens = []

        if input_ids is None:
            return [[]]
        response_prompt = [{"role": "assistant", "content": ""}]
        if len(input_ids) and isinstance(input_ids, str):
            paired_messages = [{"role": "user", "content": "{}".format(input_ids)}] + response_prompt
            tokens, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, tools="")
            context_tokens.append(tokens)
        elif len(input_ids) and isinstance(input_ids[0], (dict)):
            paired_messages = input_ids + response_prompt
            tokens, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, tools="")
            context_tokens.append(tokens)
        elif len(input_ids) and isinstance(input_ids[0], (str)):
            for query in input_ids:
                paired_messages = [{"role": "user", "content": "{}".format(query)}] + response_prompt
                tokens, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, tools="")
                context_tokens.append(tokens)
        elif len(input_ids) and isinstance(input_ids[0], (tuple, list)):
            for val in input_ids:
                if len(val) and isinstance(val, (tuple, list)):
                    paired_messages = val + response_prompt
                    tokens, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=paired_messages, tools="")
                    context_tokens.append(tokens)
        else:
            raise TypeError("Please check input_ids in correct type.")


        return context_tokens if len(context_tokens) > 0 else [context_tokens]


    def _tokenize(self, input_ids):
        context_tokens = [[]]
        broadcast_rank = torch.zeros(dist.get_world_size(),
                                     dtype=torch.int64,
                                     device=torch.device(torch.cuda.current_device()))
        if input_ids is not None and len(input_ids) > 0:
            args = get_args()
            if args.hf_chat_template:
                if not hasattr(self.tokenizer, "apply_chat_template"):
                    raise AssertionError('The tokenizer has no Huggingface chat template, Please use chat model.')

                context_tokens = [self.tokenizer.apply_chat_template(
                    input_ids,
                    tokenize=True,
                    add_generation_prompt=True
                )]
            elif self.template is None:
                context_tokens = self._encode_no_template(input_ids)
            else:
                context_tokens = self._encode_by_template(input_ids)


            broadcast_rank[dist.get_rank()] = 1

        dist.all_reduce(broadcast_rank)
        master_rank = torch.nonzero(broadcast_rank)[0, 0]

        return context_tokens, master_rank

    def _post_processing(self, output, context_lengths, log_probs):
        if not self.include_input:
            output = [val[context_lengths[i]:] for i, val in enumerate(output)]

        # When batch size > 1, you need truncate the tokens after eos_token_id
        self._truncate_in_multi_batch(output)

        if self.detokenize:
            try:
                output_checked = self._ids_check(output, self.tokenizer)
                output = self.tokenizer.batch_decode(output_checked, skip_special_tokens=True)
            except Exception as e:
                error_info = "Meet errors when trying to decode the tokens. "\
                             "Please handle it by yourself."
                logging.error(error_info)
                logging.error(e)

        output = output[0] if len(output) == 1 else output

        if not self.return_output_log_probs:
            res = output
        else:
            if self.num_beams == 1:
                log_probs = [val[context_lengths[i]:, :] for i, val in enumerate(log_probs)] \
                    if log_probs is not None else None

            res = output, log_probs[0] if len(log_probs) == 1 else log_probs

        return res

    def _truncate_in_multi_batch(self, output):
        if len(output) > 1:
            for idx, batch in enumerate(output):
                trunc_index = torch.nonzero(batch == self.tokenizer.eos_token_id)

                if min(trunc_index.shape):
                    output[idx][trunc_index.min():] = self.tokenizer.eos_token_id

    def _yield(self, token_stream):
        output, context_lengths, log_probs = None, None, None
        for output, context_lengths, log_probs in token_stream:
            if self.stream:
                res = self._post_processing(output, context_lengths, log_probs)
                yield res

        if not self.stream:
            yield self._post_processing(output, context_lengths, log_probs)

    def _token_generator(self, token_stream):
        token_stream = self._yield(token_stream)
        if not self.stream:
            full_output = None
            for tmp in token_stream:
                full_output = tmp
            return full_output
        else:
            return token_stream
        
        
class GPTVLModelInfer(GPTVLModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.infer_model = MegatronModuleForCausalLM()

    def generate(self, input_ids=None, **kwargs):
        return self.infer_model.generate(input_ids=input_ids, **kwargs)


def get_external_inputs(tokens, image_list, image_path_list, video_path_list, tokenizer, image_processor):
    args = get_args()
    image_token_length = args.image_token_length
    max_num_frame = args.max_num_frame
    max_fps = args.max_fps

    from cognitron_vl.constants import IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN, VID_START_TOKEN, VID_END_TOKEN, VID_CONTEXT_TOKEN, PATCH_START_TOKEN, PATCH_END_TOKEN, PATCH_CONTEXT_TOKEN, IMG_TAG_TOKEN, VID_TAG_TOKEN
    image_tag = "<image>"
    video_tag = "<video>"

    IMG_CONTEXT_ID = tokenizer(IMG_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    IMG_START_ID = tokenizer(IMG_START_TOKEN, add_special_tokens=False).input_ids
    IMG_END_ID = tokenizer(IMG_END_TOKEN, add_special_tokens=False).input_ids

    VID_CONTEXT_ID = tokenizer(VID_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    VID_START_ID = tokenizer(VID_START_TOKEN, add_special_tokens=False).input_ids
    VID_END_ID = tokenizer(VID_END_TOKEN, add_special_tokens=False).input_ids

    PATCH_CONTEXT_ID = tokenizer(PATCH_CONTEXT_TOKEN, add_special_tokens=False).input_ids
    PATCH_START_ID = tokenizer(PATCH_START_TOKEN, add_special_tokens=False).input_ids
    PATCH_END_ID = tokenizer(PATCH_END_TOKEN, add_special_tokens=False).input_ids

    IMG_TAG_ID = tokenizer(IMG_TAG_TOKEN, add_special_tokens=False).input_ids
    VID_TAG_ID = tokenizer(VID_TAG_TOKEN, add_special_tokens=False).input_ids

    assert len(IMG_CONTEXT_ID) == 1
    assert len(IMG_START_ID) == 1
    assert len(IMG_END_ID) == 1

    assert len(VID_CONTEXT_ID) == 1
    assert len(VID_START_ID) == 1
    assert len(VID_END_ID) == 1

    assert len(PATCH_CONTEXT_ID) == 1
    assert len(PATCH_START_ID) == 1
    assert len(PATCH_END_ID) == 1

    IMG_CONTEXT_ID = IMG_CONTEXT_ID[0]
    IMG_START_ID = IMG_START_ID[0]
    IMG_END_ID = IMG_END_ID[0]

    VID_CONTEXT_ID = VID_CONTEXT_ID[0]
    VID_START_ID = VID_START_ID[0]
    VID_END_ID = VID_END_ID[0]

    PATCH_CONTEXT_ID = PATCH_CONTEXT_ID[0]
    PATCH_START_ID = PATCH_START_ID[0]
    PATCH_END_ID = PATCH_END_ID[0]

    IMG_TAG_ID = IMG_TAG_ID[0]
    VID_TAG_ID = VID_TAG_ID[0]

    nl_tokens = tokenizer("\n", add_special_tokens=False).input_ids

    # print(f"get_external_inputs tokens {tokens}")

    image_indices = []
    images = []

    # ----------------------------------------------------------------
    # image
    for batch_idx, input_ids in enumerate(tokens):
        # img_positions = [i for i, x in enumerate(input_ids) if x == IMG_CONTEXT_ID]
        img_positions = [i for i, x in enumerate(input_ids) if x == IMG_TAG_ID]
        if len(img_positions) == 0:
            continue
        if image_path_list is not None:
            assert len(img_positions) == len(image_path_list), f"{img_positions} {image_path_list} {IMG_CONTEXT_TOKEN} {IMG_CONTEXT_ID} {tokens}"
        if image_list is not None:
            assert len(img_positions) == len(image_list), f"{img_positions} {image_list} {IMG_CONTEXT_TOKEN} {IMG_CONTEXT_ID} {tokens}"

        new_input_ids = []
        st = 0
        for img_idx, img_pos in enumerate(img_positions):
            if image_path_list is not None:
                image_patches, (best_width, best_height) = image_processor.process_images_with_subpatch(image_path_list[img_idx])
            if image_list is not None:
                image_patches, (best_width, best_height) = image_processor.process_images_with_subpatch(image_list[img_idx])
            images.append(image_patches)
            print(f"get_external_inputs best_width {best_width} best_height {best_height}")

            new_input_ids += input_ids[st:img_pos]

            new_input_ids += [IMG_START_ID]

            image_indice_b = torch.zeros(
                1, image_token_length, dtype=torch.int64
            )  # This will change in collate_fn
            image_indice_s = (
                torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                .unsqueeze(0)
                .repeat(1, 1)
            )
            image_indice_b_s = torch.stack(
                [image_indice_b, image_indice_s], dim=0
            )  # 2, num_image, image_length
            image_indices.append(image_indice_b_s)

            new_input_ids += [IMG_CONTEXT_ID] * image_token_length

            new_input_ids += [IMG_END_ID]

            if len(image_patches) > 1:
                for i in range(0, best_height, image_processor.patch_size):
                    new_input_ids += nl_tokens

                    for j in range(0, best_width, image_processor.patch_size):
                        new_input_ids += [PATCH_START_ID]

                        image_indice_b = torch.zeros(
                            1, image_token_length, dtype=torch.int64
                        )  # This will change in collate_fn
                        image_indice_s = (
                            torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                            .unsqueeze(0)
                            .repeat(1, 1)
                        )
                        image_indice_b_s = torch.stack(
                            [image_indice_b, image_indice_s], dim=0
                        )  # 2, num_image, image_length
                        image_indices.append(image_indice_b_s)

                        new_input_ids += [PATCH_CONTEXT_ID] * image_token_length

                        new_input_ids += [PATCH_END_ID]
                        # print(f"get_external_dict i {i} j {j} new_input_ids {len(new_input_ids)}")

            st = img_pos + 1

        new_input_ids += input_ids[st:]

        input_ids = new_input_ids
        tokens[batch_idx] = input_ids

    # ----------------------------------------------------------------
    # video
    for batch_idx, input_ids in enumerate(tokens):
        # vid_positions = [i for i, x in enumerate(input_ids) if x == VID_CONTEXT_ID]
        vid_positions = [i for i, x in enumerate(input_ids) if x == VID_TAG_ID]
        if len(vid_positions) == 0:
            continue
        if video_path_list is not None:
            assert len(vid_positions) == len(video_path_list), f"{vid_positions} {video_path_list} {VID_CONTEXT_TOKEN} {VID_CONTEXT_ID} {tokens}"
        if image_path_list is not None:
            assert len(vid_positions) == len(image_path_list), f"{vid_positions} {image_path_list} {VID_CONTEXT_TOKEN} {VID_CONTEXT_ID} {tokens}"
        if image_list is not None:
            assert len(vid_positions) == len(image_list), f"{vid_positions} {image_list} {VID_CONTEXT_TOKEN} {VID_CONTEXT_ID} {tokens}"

        new_input_ids = []
        st = 0
        for vid_idx, vid_pos in enumerate(vid_positions):
            if video_path_list is not None:
                video_frames, _ = image_processor.process_video(video_path_list[vid_idx], max_num_frame, max_fps)
            if image_path_list is not None:
                video_frames = image_processor.process_images([image_path_list[vid_idx]])
            if image_list is not None:
                video_frames = image_processor.process_images([image_list[vid_idx]])

            images.append(video_frames)

            new_input_ids += input_ids[st:vid_pos]

            for _ in video_frames:
                new_input_ids += [VID_START_ID]

                image_indice_b = torch.zeros(
                    1, image_token_length, dtype=torch.int64
                )  # This will change in collate_fn
                image_indice_s = (
                    torch.arange(len(new_input_ids), len(new_input_ids) + image_token_length)
                    .unsqueeze(0)
                    .repeat(1, 1)
                )
                image_indice_b_s = torch.stack(
                    [image_indice_b, image_indice_s], dim=0
                )  # 2, num_image, image_length
                image_indices.append(image_indice_b_s)

                new_input_ids += [VID_CONTEXT_ID] * image_token_length

                new_input_ids += [VID_END_ID]

            st = vid_pos + 1

        new_input_ids += input_ids[st:]

        input_ids = new_input_ids
        tokens[batch_idx] = input_ids

    images = torch.cat(images, dim=0)
    image_indices = torch.cat(image_indices, dim=1)

    # print(f"get_external_inputs tokens {tokens}")
    print(f"get_external_inputs images {images.size()}")
    print(f"get_external_inputs tokens {[len(x) for x in tokens]}")

    external_inputs = {}

    external_inputs["indices"] = image_indices.contiguous().to(torch.cuda.current_device())
    if args.bf16:
        external_inputs["images"] = torch.tensor(images, dtype=torch.bfloat16).contiguous().to(torch.cuda.current_device())

    else:
        external_inputs["images"] = torch.tensor(images, dtype=torch.float16).contiguous().to(torch.cuda.current_device())

    return external_inputs, tokens
