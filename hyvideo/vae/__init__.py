from pathlib import Path

import torch

from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ..constants import VAE_PATH, PRECISION_TO_TYPE
import tensorrt as trt
from cuda import cudart

from ..engine import Engine
import gc, os

def load_vae(vae_type: str="884-16c-hy",
             vae_precision: str=None,
             batch_size: int=1,
             height: int=192,
             width: int=336,
             video_length: int=129,
             sample_size: tuple=None,
             vae_path: str=None,
             logger=None,
             device=None,
             vae_trt=False,
             onnx_dir: str=None,
             engine_dir: str=None,
             enable_tiling: bool=True,
             ):
    """the fucntion to load the 3D VAE model

    Args:
        vae_type (str): the type of the 3D VAE model. Defaults to "884-16c-hy".
        vae_precision (str, optional): the precision to load vae. Defaults to None.
        sample_size (tuple, optional): the tiling size. Defaults to None.
        vae_path (str, optional): the path to vae. Defaults to None.
        logger (_type_, optional): logger. Defaults to None.
        device (_type_, optional): device to load vae. Defaults to None.
    """
    if vae_path is None:
        vae_path = VAE_PATH[vae_type]
    
    if logger is not None:
        logger.info(f"Loading 3D VAE model ({vae_type}) from: {vae_path}")
    config = AutoencoderKLCausal3D.load_config(vae_path)
    if sample_size:
        vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
    else:
        vae = AutoencoderKLCausal3D.from_config(config)
    
    vae_ckpt = Path(vae_path) / "pytorch_model.pt"
    assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"
    
    ckpt = torch.load(vae_ckpt, map_location=vae.device)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if any(k.startswith("vae.") for k in ckpt.keys()):
        ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
    vae.load_state_dict(ckpt)

    spatial_compression_ratio = vae.config.spatial_compression_ratio
    time_compression_ratio = vae.config.time_compression_ratio
    
    if vae_precision is not None:
       vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if logger is not None:
        logger.info(f"VAE to dtype: {vae.dtype}")

    if device is not None:
        vae = vae.to(device)

    vae.eval()

    # TRT Engine for VAE decoder
    vae.enable_trt = vae_trt
    if vae.enable_trt:
        print(f"[I] Enable TRT = {vae.enable_trt} for vae decoder")
        vae_decoder = vae.decoder
        
        vae_decoder.eval()
        if enable_tiling:
            dummy_shape = (batch_size, vae.config.latent_channels, int(vae.tile_sample_min_tsize // time_compression_ratio + 1), vae.tile_latent_min_size, vae.tile_latent_min_size)
        else:
            dummy_shape = (batch_size, vae.config.latent_channels, int( (video_length - 1) // time_compression_ratio + 1), int( height // spatial_compression_ratio), int( width // spatial_compression_ratio))
        print(f"trt {dummy_shape=}")
        #dummy_input = torch.randn(1, vae.config.latent_channels, 17, 32, 32, device=device, dtype=torch.float32)
        dummy_input = torch.randn(dummy_shape, device=device, dtype=torch.float32)
        # Create directories if missing
        for directory in [engine_dir, onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                Path(directory).mkdir(parents=True)
                
        if enable_tiling:
            onnx_file = os.path.join(onnx_dir, f"vae_decode_tiled_rank{os.environ['LOCAL_RANK']}.onnx")
        else:
            onnx_file = os.path.join(onnx_dir, "vae_decode.onnx")
        if not os.path.exists(onnx_file):
            print(f"[I] Exporting VAE Decoder ONNX: {onnx_file}")
            with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True
            ):
                torch.onnx.export(vae_decoder,
                                  dummy_input,
                                  f=onnx_file,
                                  opset_version=23,
                                  input_names=["latent_sample"],
                                  output_names=["sample"],
                                  dynamic_axes={ "latent_sample": {2: "video_length_tile", 3: "height_tile", 4: "width_tile"},
                                                 "sample":  {2: "video_length_tile", 3: "height_tile", 4: "width_tile"}
                                                },
                                  verbose=False,
                                  dynamo=True,
                                  report=False,
                                  do_constant_folding=True)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print(f"[I] Using existing VAE Decoder ONNX: {onnx_file}")

        if enable_tiling:
            engine_file = os.path.join(engine_dir, f'vae_decoder_tiled_rank{os.environ['LOCAL_RANK']}.trt'+trt.__version__+'.plan')
        else:
            engine_file = os.path.join(engine_dir, 'vae_decoder.trt'+trt.__version__+'.plan')
        vae_engine = Engine(engine_file)
        
        opt_shape =  dummy_shape
        max_shape = dummy_shape
        if enable_tiling:
            overlap_size = int(vae.tile_latent_min_size * (1 - vae.tile_overlap_factor))
            min_shape = (batch_size, vae.config.latent_channels, int(((video_length - 1) // time_compression_ratio) % overlap_size + 1), int((height // spatial_compression_ratio) % overlap_size), int((width // spatial_compression_ratio) % overlap_size))
        else:
            min_shape = opt_shape
        print(f"{opt_shape=} {max_shape=} {min_shape=}")
        #min_shape = (1, vae.config.latent_channels, 9, 18, 16)
        #opt_shape = (1, vae.config.latent_channels, 17, 32, 32)
        #max_shape = (1, vae.config.latent_channels, 17, 32, 32)
        if not os.path.exists(engine_file):
            print(f"[I] Exporting VAE Decoder Engines: {engine_file}")
            with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True
            ):
                vae_engine.build(onnx_file,
                                 strongly_typed=False,
                                 fp16=True,
                                 bf16=False,
                                 tf32=False,
                                 int8=False,
                                 input_profile={
                                     "latent_sample": [
                                         min_shape,
                                         opt_shape,
                                         max_shape,
                                     ]
                                },
                                 enable_refit=False,
                                 enable_all_tactics=False,
                                 timing_cache='vae_timing_cache.cache',
                                 update_output_names=None,
                                 verbose=True,
                                 builder_optimization_level=3,
                                )
        else:
            print(f"[I] Using existing VAE Decoder Engine: {engine_file}")
        vae.engine["decoder"] = vae_engine
        vae.engine["decoder"].load()

        alloc_shape_max = { "latent_sample": max_shape,
                            "sample": (max_shape[0], vae.config.out_channels, (max_shape[2] - 1) * time_compression_ratio + 1, max_shape[3] * spatial_compression_ratio, max_shape[4] * spatial_compression_ratio)
                           }
        vae.shape_dicts['decoder'] = alloc_shape_max
                
        print("loadResource")
        vae.loadResources(device)
                                                
    return vae, vae_path, spatial_compression_ratio, time_compression_ratio
