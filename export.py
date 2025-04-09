import argparse

import time

import os
from pathlib import Path

import torch
import tensorrt as trt
from depth_anything import DepthAnything


def export(
    weights_path: str,  
    save_path: str,
    input_size: int,
    onnx: bool = True,
    use_dla: bool = False,
    dla_core: int = 0
):  
    """
    weights_path: str -> Path to the PyTorch model(local / hub)
    save_path: str -> Directory to save the model
    input_size: int -> Width and height of the input image(e.g. 308, 364, 406, 518)
    onnx: bool -> Export the model to ONNX format
    use_dla: bool -> Use Deep Learning Accelerator (DLA) for inference
    dla_core: int -> DLA core to use (0 or 1)
    """
    weights_path = Path(weights_path)
    
    os.makedirs(save_path, exist_ok=True)

    # Load the model
    model = DepthAnything.from_pretrained(weights_path).to('cpu').eval()
    
    # Create a dummy input
    dummy_input = torch.ones((3, input_size, input_size)).unsqueeze(0)
    _ = model(dummy_input)
    onnx_path = Path(save_path) / f"{weights_path.stem}_{input_size}.onnx"
    
    # Export the PyTorch model to ONNX format
    if onnx:
        torch.onnx.export(
            model,
            dummy_input, 
            onnx_path, 
            opset_version=11    , 
            input_names=["input"], 
            output_names=["output"], 
        )
        print(f"Model exported to {onnx_path}", onnx_path)
        print("Saving the model to ONNX format...")
        time.sleep(2)
    
    # ONNX to TensorRT
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError('Failed to parse the ONNX model.')
    
    # Set up the builder config
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16) # FP16
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2 GB
    
    # Configure DLA if requested
    if use_dla:
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK) # Allow GPU fallback for unsupported layers
        config.default_device_type = trt.DeviceType.DLA
        config.DLA_core = dla_core
        print(f"Using DLA core {dla_core} with GPU fallback")
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    # Generate appropriate filename based on settings
    engine_filename = f"{weights_path.stem}_{input_size}"
    if use_dla:
        engine_filename += f"_dla{dla_core}"
    engine_filename += ".trt"
    
    engine_path = Path(save_path) / engine_filename
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved to {engine_path}")
    
if __name__ == '__main__':
    export(
        weights_path="LiheYoung/depth_anything_vits14", # local hub or online
        save_path="weights", # folder name
        input_size=308, # 308 | 364 | 406 | 518
        onnx=True,
        use_dla=True,  # Enable DLA
        dla_core=0,    # Use DLA core 0
    )