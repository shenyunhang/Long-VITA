# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
import io
import base64
from PIL import Image
import datetime
import torch
import json
import threading
from flask import Flask, request, jsonify, current_app
from flask_restful import Resource, Api
from megatron.training import get_args
from megatron.inference.text_generation import generate_and_post_process
from megatron.inference.text_generation import beam_search_and_post_process


GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()

class MegatronGenerate(Resource):
    def __init__(self, model):
        self.model = model

    @staticmethod
    def send_do_generate():
        choice = torch.tensor([GENERATE_NUM], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
     
    @staticmethod
    def send_do_beam_search():
        choice = torch.tensor([BEAM_NUM], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)

    def put(self):
        print("-" * 100)
        args = get_args()

        if "image_list" in request.get_json():
            image_list = []
            image_list_str = request.get_json()["image_list"]
            for image_str in image_list_str:
                image = decode_base64_to_image(image_str)
                image_list.append(image)
        else:
            image_list = None
        # print("image_list", image_list)

        if "image_path_list" in request.get_json():
            image_path_list = request.get_json()["image_path_list"]
            # image_list = [Image.open(x) for x in image_path_list]
        else:
            image_path_list = None

        if "video_path_list" in request.get_json():
            video_path_list = request.get_json()["video_path_list"]
        else:
            video_path_list = None

        if "max_num_frame" in request.get_json():
            max_num_frame = request.get_json()["max_num_frame"]
            args = get_args()
            args.max_num_frame = max_num_frame
       
        if not "prompts" in request.get_json():
            return "prompts argument required", 400
        
        if "max_len" in request.get_json():
            return "max_len is no longer used.  Replace with tokens_to_generate", 400
        
        if "sentences" in request.get_json():
            return "sentences is no longer used.  Replace with prompts", 400

        prompts = request.get_json()["prompts"]
        if not isinstance(prompts, list):
            return "prompts is not a list of strings", 400

        if len(prompts) == 0:
            return "prompts is empty", 400
        
        if len(prompts) > 128:
            return "Maximum number of prompts is 128", 400
        
        tokens_to_generate = 64  # Choosing hopefully sane default.  Full sequence is slow
        if "tokens_to_generate" in request.get_json():
            tokens_to_generate = request.get_json()["tokens_to_generate"]
            if not isinstance(tokens_to_generate, int):
                return "tokens_to_generate must be an integer greater than 0"
            if tokens_to_generate < 0:
                return "tokens_to_generate must be an integer greater than or equal to 0"

        logprobs = False
        if "logprobs" in request.get_json():
            logprobs = request.get_json()["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"
        
        if tokens_to_generate == 0 and not logprobs:
            return "tokens_to_generate=0 implies logprobs should be True"
        
        temperature = 1.0
        if "temperature" in request.get_json():
            temperature = request.get_json()["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 < temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"
        
        top_k = 0.0
        if "top_k" in request.get_json():
            top_k = request.get_json()["top_k"]
            if not (type(top_k) == int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"
        
        top_p = 0.0
        if "top_p" in request.get_json():
            top_p = request.get_json()["top_p"]
            if not (type(top_p) == float):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"
        
        top_p_decay = 0.0
        if "top_p_decay" in request.get_json():
            top_p_decay = request.get_json()["top_p_decay"]
            if not (type(top_p_decay) == float):
                return "top_p_decay must be a positive float less than or equal to 1.0"
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p"
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0"
        
        top_p_bound = 0.0
        if "top_p_bound" in request.get_json():
            top_p_bound = request.get_json()["top_p_bound"]
            if not (type(top_p_bound) == float):
                return "top_p_bound must be a positive float less than or equal to top_p"
            if top_p == 0.0:
                return "top_p_bound cannot be set without top_p"
            if not (0.0 < top_p_bound <= top_p):
                return "top_p_bound must be greater than 0 and less than top_p"
        
        add_BOS = False
        if "add_BOS" in request.get_json():
            add_BOS = request.get_json()["add_BOS"]
            if not isinstance(add_BOS, bool):
                return "add_BOS must be a boolean value"
        
        if any([len(prompt) == 0 for prompt in prompts]) and not add_BOS:
            return "Empty prompts require add_BOS=true"

        stop_on_double_eol = False
        if "stop_on_double_eol" in request.get_json():
            stop_on_double_eol = request.get_json()["stop_on_double_eol"]
            if not isinstance(stop_on_double_eol, bool):
                return "stop_on_double_eol must be a boolean value"
        
        stop_on_eol = False
        if "stop_on_eol" in request.get_json():
            stop_on_eol = request.get_json()["stop_on_eol"]
            if not isinstance(stop_on_eol, bool):
                return "stop_on_eol must be a boolean value"

        prevent_newline_after_colon = False
        if "prevent_newline_after_colon" in request.get_json():
            prevent_newline_after_colon = request.get_json()["prevent_newline_after_colon"]
            if not isinstance(prevent_newline_after_colon, bool):
                return "prevent_newline_after_colon must be a boolean value"

        random_seed = -1
        if "random_seed" in request.get_json():
            random_seed = request.get_json()["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0: 
                return "random_seed must be a positive integer"

        no_log = False
        if "no_log" in request.get_json():
            no_log = request.get_json()["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"
        
        beam_width = None
        if "beam_width" in request.get_json():
            beam_width = request.get_json()["beam_width"]
            if not isinstance(beam_width, int):
                return "beam_width must be integer"
            if beam_width < 1:
                return "beam_width must be an integer > 1"
            if len(prompts) > 1:
                return "When doing beam_search, batch size must be 1"

        stop_token=50256
        if "stop_token" in request.get_json():
            stop_token = request.get_json()["stop_token"]
            if not isinstance(stop_token, int):
                return "stop_token must be an integer"
        
        length_penalty = 1 
        if "length_penalty" in request.get_json():
            length_penalty = request.get_json()["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float"
        
        with lock:  # Need to get lock to keep multiple threads from hitting code
            
            if not no_log:
                print("request IP: " + str(request.remote_addr))
                json_data = request.get_json()
                json_data.pop("image_list", None)
                print(json.dumps(json_data), flush=True)
                print("start time: ", datetime.datetime.now())
            
            try:
                if True:
                    MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
                    output = self.model.generate(
                        prompts,
                        do_sample=False,
                        max_new_tokens=tokens_to_generate,
                        stream=False,
                        image_list=image_list,
                        image_path_list=image_path_list,
                        video_path_list=video_path_list,
                    )
                    print(f"output {output}")
                    return jsonify({
                        "text": [output,],
                    })
                if beam_width is not None:
                    MegatronGenerate.send_do_beam_search()  # Tell other ranks we're doing beam_search
                    response, response_seg, response_scores = \
                        beam_search_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        beam_size = beam_width,
                        add_BOS=add_BOS,
                        stop_token=stop_token,
                        num_return_gen=beam_width,  # Returning whole beam
                        length_penalty=length_penalty,
                        prevent_newline_after_colon=prevent_newline_after_colon
                        )
                    
                    return jsonify({"text": response,
                        "segments": response_seg,
                        "scores": response_scores})
                else:
                    MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
                    response, response_seg, response_logprobs, _ = \
                        generate_and_post_process(
                        self.model,
                        prompts=prompts,
                        tokens_to_generate=tokens_to_generate,
                        return_output_log_probs=logprobs,
                        top_k_sampling=top_k,
                        top_p_sampling=top_p,
                        top_p_decay=top_p_decay,
                        top_p_bound=top_p_bound,
                        temperature=temperature,
                        add_BOS=add_BOS,
                        use_eod_token_for_early_termination=True,
                        stop_on_double_eol=stop_on_double_eol,
                        stop_on_eol=stop_on_eol,
                        prevent_newline_after_colon=prevent_newline_after_colon,
                        random_seed=random_seed)

                    return jsonify({"text": response,
                        "segments": response_seg,
                        "logprobs": response_logprobs})

            except ValueError as ve:
                print(f"ve {ve}")
                return ve.args[0]
            print("end time: ", datetime.datetime.now())
        

class MegatronServer(object):
    def __init__(self, model):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/api', resource_class_args=[model])
        
    def run(self, url, port): 
        self.app.run(url, threaded=True, debug=False, port=port)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

