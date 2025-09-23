from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG

from pathlib import Path

import torch
import tensorrt as trt
from cuda import cudart
import onnx

from hyvideo import load, optimizer
from ..engine import Engine
import gc, os

def load_model(args, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    Args:
        args (dict): model args
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    if args.model in HUNYUAN_VIDEO_CONFIG.keys():
        model = HYVideoDiffusionTransformer(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
        return model
    else:
        raise NotImplementedError()

def load_trt_model(args,
                   model,
                   batch_size: int=1,
                   height: int=192,
                   width: int=336,
                   video_length: int=129,
                   device=None,
                   model_trt=False,
                   onnx_dir: str=None,
                   engine_dir: str=None,
                   vae_ver: str="88-4c-sd"):

    # enable trt for transformer backbone
    model.enable_trt = model_trt
    model.eval()
    
    if model.enable_trt:
        print(f"[I] Enable TRT = {model.enable_trt} for transformer backbone")
        
        latent_channels=model.config.in_channels
        if "884" in vae_ver:
            latent_video_length = (video_length - 1) // 4 + 1
            latent_height = height // 8
            latent_width = width // 8
        elif "888" in vae_ver:
            latent_video_length = (video_length - 1) // 8 + 1
            latent_height = height // 8
            latent_width = width // 8
        else:
            latent_video_length = video_length
            latent_height = height // 8
            latent_width = width // 8
              
        # Create directories if missing
        for directory in [engine_dir, onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                Path(directory).mkdir(parents=True)
                
        onnx_file = os.path.join(onnx_dir, f"model_rank{os.environ['LOCAL_RANK']}.onnx")

        onnx_opt_file = os.path.join(onnx_dir, f"model_opt_rank{os.environ['LOCAL_RANK']}.onnx")

        if not os.path.exists(onnx_opt_file):
            if not os.path.exists(onnx_file):
                print(f"[I] Exporting model ONNX: {onnx_file}")
                
                with torch.inference_mode(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    torch.onnx.export(model,
                                      model.get_sample_input(batch_size,latent_video_length, latent_height, latent_width, device), 
                                      f=onnx_file,
                                      opset_version=20, # 20 for dynamo=False, 23 for dynamo=True
                                      input_names=model.get_input_names(),
                                      output_names=model.get_output_names(),
                                      dynamic_axes=model.get_dynamic_axes(), # dynamic_shapes below for dynamo=True
                                      dynamo=False,
                                      report=False,
                                      do_constant_folding=True)
                    # custom_opsets={"nvidia": 1},
                    # use dynamic_axes for dynamo=False and dynamic_shapes for dynamo=True
                    #dynamic_axes=model.get_dynamic_axes(),
                    #dynamic_shapes={ "x": {2: "video_length", 3: "height", 4: "width"},
                    #                 "t": None,
                    #                 "text_states": None,
                    #                 "text_mask": None,
                    #                 "text_states_2": None,
                    #                 "freqs_cos": None,
                    #                 "freqs_sin": None,
                    #                 "guidance": None,
                    #                 "return_dict": None
                    #                },
                    gc.collect()
                    torch.cuda.empty_cache()
            else:
                print(f"[I] Found cached model ONNX: {onnx_file}")
        
            print(f"[I] Optimizing ONNX model: {onnx_opt_file}")
            def optimize(model, onnx_graph, return_onnx=True, **kwargs):
                opt = optimizer.Optimizer(onnx_graph, verbose=True)
                name = model.__class__.__name__
                opt.info(name + ": original")
                opt.cleanup()
                opt.info(name + ": cleanup")
                if kwargs.get("modify_fp8_graph", False):
                    is_fp16_io = kwargs.get("is_fp16_io", True)
                    opt.modify_fp8_graph(is_fp16_io=is_fp16_io)
                    opt.info(name + ": modify fp8 graph")
            
                opt.fold_constants()
                opt.info(name + ": fold constants")
                opt.infer_shapes()
                opt.info(name + ": shape inference")
                
                if kwargs.get("fuse_mha_qkv_int8", False):
                    opt.fuse_mha_qkv_int8_sq()
                    opt.info(name + ": fuse QKV nodes")
                onnx_opt_graph = opt.cleanup(return_onnx=return_onnx)
                opt.info(name + ": finished")
                return onnx_opt_graph
            
            onnx_opt_graph = optimize(model, onnx.load(onnx_file))
            if load.onnx_graph_needs_external_data(onnx_opt_graph):
                onnx.save_model(
                    onnx_opt_graph,
                    onnx_opt_file,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    convert_attribute=False,
                )
            else:
                onnx.save(onnx_opt_graph, onnx_opt_file)
        else:
            print(f"[I] Found cached optimized ONNX model: {onnx_opt_file} ")
            
        engine_file = os.path.join(engine_dir, f'model_rank{os.environ['LOCAL_RANK']}.trt'+trt.__version__+'.plan')

        model_engine = Engine(engine_file)
        
        if not os.path.exists(engine_file):
            print(f"[I] Exporting Model Engines: {engine_file}")
            with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                model_engine.build(onnx_opt_file,
                                   strongly_typed=False,
                                   fp16=False,
                                   bf16=True,
                                   tf32=False,
                                   int8=False,
                                   input_profile=model.get_input_profile(batch_size, latent_video_length, latent_height, latent_width),
                                   enable_refit=False,
                                   enable_all_tactics=False,
                                   timing_cache='model_timing_cache.cache',
                                   update_output_names=None,
                                   verbose=True,
                                   builder_optimization_level=3,
                                   )
        else:
            print(f"[I] Using existing Model Engine: {engine_file}")
            
        model.engine["transformer"] = model_engine
        model.engine["transformer"].load()

        model.shape_dicts["transformer"] = model.get_shape_dict(batch_size, latent_video_length, latent_height, latent_width)

        print("loadResource")
        model.loadResources(device)
        
    return model

