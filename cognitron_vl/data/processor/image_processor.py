import math
import os

import cv2
import natsort
import numpy as np
import torch
from PIL import Image

import decord
from cognitron_vl.constants import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SIGLIP_MEAN,
    SIGLIP_STD,
)


class ImageProcessor:
    def __init__(
        self,
        process_type,
        image_size=448,
        normalize_type="imagenet",
        min_patch_grid=1,
        max_patch_grid=6,
    ):
        self.process_type = process_type
        self.image_size = image_size

        if normalize_type == "imagenet":
            MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        elif normalize_type == "clip":
            MEAN, STD = CLIP_MEAN, CLIP_STD
        elif normalize_type == "siglip":
            MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
        else:
            raise NotImplementedError
        self.mean = MEAN
        self.std = STD

        self.patch_size = image_size
        self.min_patch_grid = min_patch_grid
        self.max_patch_grid = max_patch_grid

        if self.process_type == "anyres":
            self.grid_pinpoints = [
                (i, j)
                for i in range(min_patch_grid, max_patch_grid + 1)
                for j in range(min_patch_grid, max_patch_grid + 1)
            ]
            self.possible_resolutions = [
                [dim * self.patch_size for dim in pair] for pair in self.grid_pinpoints
            ]
            print(f"grid_pinpoints {self.grid_pinpoints}")
            print(f"possible_resolutions {self.possible_resolutions}")

        if self.process_type == "dynamic":
            max_num = self.max_patch_grid
            min_num = self.min_patch_grid
            # calculate the existing image aspect ratio
            target_ratios = set(
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if i * j <= max_num and i * j >= min_num
            )
            self.target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
            self.possible_resolutions = [
                [dim * self.patch_size for dim in pair] for pair in self.target_ratios
            ]
            print(f"target_ratios {self.target_ratios}")
            print(f"possible_resolutions {self.possible_resolutions}")

    def get_frame_paths(self, frame_root, num_frames=8):
        os.makedirs(frame_root, exist_ok=True)

        self.frame_tmpl = "frame-{}-of-{}.jpg"
        return [
            os.path.join(frame_root, self.frame_tmpl.format(i, num_frames))
            for i in range(1, num_frames + 1)
        ]

    def save_video_frames(self, vid_path, max_fps=1, num_frames=8):

        vid = decord.VideoReader(vid_path, num_threads=1)

        step_size = len(vid) / (num_frames + 1)
        # step_size = max(1, step_size)
        fps = vid.get_avg_fps()
        step_size = max(fps / max_fps, step_size)

        # indices = [int(i * step_size) for i in range(1, num_frames + 1)]
        indices = [int(i * step_size) for i in range(0, num_frames)]
        indices = [i for i in indices if i < len(vid)]

        num_frames = len(indices)

        frame_paths = self.get_frame_paths(vid_path + ".saved_frames", num_frames)
        flag = np.all([os.path.exists(p) for p in frame_paths])
        if flag:
            return frame_paths

        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]

        for im, pth in zip(images, frame_paths):
            # if not os.path.exists(pth):
            #     im.save(pth)
            im.save(pth)
        # print(f"save_video_frames vid_path {vid_path} fps {fps} len(vid) {len(vid)} frame_paths {frame_paths}")
        return frame_paths

    def get_video_frames(self, vid_path, max_fps=1, num_frames=8):

        vid = decord.VideoReader(vid_path, num_threads=1)

        step_size = len(vid) / (num_frames + 1)
        # step_size = max(1, step_size)
        fps = vid.get_avg_fps()
        step_size = max(fps / max_fps, step_size)

        # indices = [int(i * step_size) for i in range(1, num_frames + 1)]
        indices = [int(i * step_size) for i in range(0, num_frames)]
        indices = [i for i in indices if i < len(vid)]

        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]

        # print(f"save_video_frames vid_path {vid_path} fps {fps} len(vid) {len(vid)} frame_paths {frame_paths}")
        return images

    def process_video(self, video_file_or_dir, max_num_frame=8, max_fps=1):
        if os.path.isdir(video_file_or_dir):
            all_filepath = []
            for root, dirs, files in os.walk(video_file_or_dir):
                for filename in files:
                    if (
                        filename.endswith("png")
                        or filename.endswith("jpeg")
                        or filename.endswith("jpg")
                    ):
                        filepath = os.path.join(root, filename)
                        all_filepath.append(filepath)

            if len(all_filepath) == 0:
                return None

            # all_filepath.sort()
            all_filepath = natsort.natsorted(all_filepath)
            total_frame = len(all_filepath)
            if "ShareGPTVideo" in video_file_or_dir:
                fps = 2
            else:
                fps = 1
            target_frame = int(min(total_frame / fps * max_fps, max_num_frame))
            index = [int(1.0 * total_frame / target_frame) * x for x in range(target_frame)]

            selected_filepath = [all_filepath[x] for x in index]

            img_or_path_list = selected_filepath
            # print(f"process_video {img_or_path_list}")
        elif os.path.isfile(video_file_or_dir):
            # frame_paths = self.save_video_frames(
            #     video_file_or_dir, num_frames=max_num_frame, max_fps=max_fps
            # )
            # img_or_path_list = frame_paths
            img_or_path_list = self.get_video_frames(
                video_file_or_dir, num_frames=max_num_frame, max_fps=max_fps
            )
        else:
            print(f"FileNotFoundError {video_file_or_dir}")
            raise NotImplementedError

        return self.process_images(img_or_path_list), img_or_path_list

    def process_images(self, img_or_path_list):

        if isinstance(img_or_path_list[0], str):
            images = [Image.open(x).convert("RGB") for x in img_or_path_list]
        elif isinstance(img_or_path_list[0], Image.Image):
            images = [x.convert("RGB") for x in img_or_path_list]
        else:
            images = img_or_path_list

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image_tensor = torch.ones([len(images), 3, self.image_size, self.image_size])

        for i, image in enumerate(images):
            image = expand2square(image, tuple(int(x * 255) for x in self.mean))

            image = image.resize(
                (self.image_size, self.image_size), resample=Image.Resampling.BICUBIC
            )

            image = np.array(image, dtype=np.float32)
            image = image * 1.0 / 255.0

            mean = np.array(self.mean, dtype=image.dtype)
            std = np.array(self.std, dtype=image.dtype)
            image = (image - mean) / std

            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1)

            image_tensor[i] = image

        return image_tensor

    def process_images_with_subpatch(self, img_or_path):
        if self.process_type == "anyres":
            return self.process_anyres(img_or_path)
        if self.process_type == "dynamic":
            return self.process_dynamic(img_or_path)

        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("RGB")
        else:
            image = img_or_path

        return self.process_images([images])

    def process_anyres(self, img_or_path):
        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("RGB")
        else:
            image = img_or_path

        best_resolution = select_best_resolution(image.size, self.possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)
        patches = divide_to_patches(image_padded, self.patch_size)

        if best_resolution == (self.patch_size, self.patch_size):
            image_patches = [image]
        else:
            image_patches = [image] + patches

        image_patches = self.process_images(image_patches)

        # print(f"image {image.size} best_resolution {best_resolution} image_padded {image_padded.size} patches {len(patches)} image_patches {image_patches.size()}")

        return image_patches, best_resolution

    def process_dynamic(self, img_or_path):
        if isinstance(img_or_path, str):
            image = Image.open(img_or_path).convert("RGB")
        elif isinstance(img_or_path, Image.Image):
            image = img_or_path.convert("RGB")
        else:
            image = img_or_path

        image_patches, best_resolution = dynamic_preprocess(
            image,
            min_num=self.min_patch_grid,
            max_num=self.max_patch_grid,
            image_size=self.patch_size,
            use_thumbnail=True,
        )

        image_patches = self.process_images(image_patches)

        # print(f"image {image.size} best_resolution {best_resolution} image_padded {image_padded.size} patches {len(patches)} image_patches {image_patches.size()}")

        return image_patches, best_resolution


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for width, height in possible_resolutions:
        # Calculate the downscaled size to keep the aspect ratio
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(
            original_height * scale
        )

        # Calculate effective and wasted resolutions
        effective_resolution = min(
            downscaled_width * downscaled_height, original_width * original_height
        )
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution
            and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    # Determine which dimension (width or height) to fill
    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        # Width will be filled completely
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        # Height will be filled completely
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    new_image = Image.new("RGB", (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    # print(f'width: {width}, height: {height}, best_ratio: {best_ratio}')
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        # processed_images.append(thumbnail_img)
        processed_images = [
            thumbnail_img,
        ] + processed_images
    return processed_images, (target_width, target_height)
