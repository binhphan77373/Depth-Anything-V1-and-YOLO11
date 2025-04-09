from __future__ import annotations
from typing import Sequence

import os
import time
from pathlib import Path

import cv2
import numpy as np

import tensorrt as trt
import pycuda.autoinit # Don't remove this line
import pycuda.driver as cuda
from torchvision.transforms import Compose

from depth_anything import transform


class DepthEngine:
    """
    Depth estimation using Depth Anything with TensorRT
    """
    def __init__(
        self,
        input_size: int = 308,
        trt_engine_path: str = 'weights/depth_anything_vits14_308.trt', # Must match with the input_size
        save_path: str = None,
    ):
        """
        input_size: int -> Width and height of the input tensor(e.g. 308, 364, 406, 518)
        trt_engine_path: str -> Path to the TensorRT engine
        save_path: str -> Path to save the results
        """
        self.width = input_size # width of the input tensor
        self.height = input_size # height of the input tensor
        self.save_path = Path(save_path) if isinstance(save_path, str) else Path("results")
        
        # Load the TensorRT engine
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        self.engine = self.runtime.deserialize_cuda_engine(open(trt_engine_path, 'rb').read())
        self.context = self.engine.create_execution_context()
        print(f"Engine loaded from {trt_engine_path}")
        
        # Allocate pagelocked memory
        self.h_input = cuda.pagelocked_empty(trt.volume((1, 3, self.width, self.height)), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume((1, 1, self.width, self.height)), dtype=np.float32)
        
        # Allocate device memory
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        # Create a cuda stream
        self.cuda_stream = cuda.Stream()
        
        # Transform functions
        self.transform = Compose([
            transform.Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=False,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            transform.NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transform.PrepareForNet(),
        ])
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image
        """
        image = image.astype(np.float32)
        image /= 255.0
        image = self.transform({'image': image})['image']
        image = image[None]
        
        return image
    
    def postprocess(self, depth: np.ndarray, original_size: tuple) -> np.ndarray:
        """
        Postprocess the depth map
        
        Args:
            depth: Raw depth map from model
            original_size: (width, height) of the original input image
        """
        depth = np.reshape(depth, (self.width, self.height))
        depth = cv2.resize(depth, original_size)
        
        return depth
        
    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Infer depth from an image using TensorRT
        
        Args:
            image: Input RGB image
            
        Returns:
            Depth map with same dimensions as input image
        """
        # Get original image size
        original_height, original_width = image.shape[:2]
        
        # Preprocess the image
        image = self.preprocess(image)
        
        t0 = time.time()
        
        # Copy the input image to the pagelocked memory
        np.copyto(self.h_input, image.ravel())
        
        # Copy the input to the GPU, execute the inference, and copy the output back to the CPU
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.cuda_stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], stream_handle=self.cuda_stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.cuda_stream)
        self.cuda_stream.synchronize()
        
        print(f"Inference time: {time.time() - t0:.4f}s")
        
        return self.postprocess(self.h_output, (original_width, original_height)) # Postprocess the depth map
    
    def release(self):
        """
        Release resources
        """
        # Clean up CUDA resources
        del self.cuda_stream
        del self.d_input
        del self.d_output
        del self.h_input
        del self.h_output
    
    def visualize_depth(self, depth):
        """
        Create a visualization of the depth map
        Args:
            depth: np.ndarray - Raw depth map
        Returns:
            np.ndarray - Colorized depth map for visualization
        """
        vis_depth = depth.copy()
        vis_depth = (vis_depth - vis_depth.min()) / (vis_depth.max() - vis_depth.min()) * 255.0
        vis_depth = vis_depth.astype(np.uint8)
        vis_depth = cv2.applyColorMap(vis_depth, cv2.COLORMAP_INFERNO)
        return vis_depth
