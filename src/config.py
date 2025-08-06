import argparse
import os
import random
import sys

import numpy as np
import torch
from easydict import EasyDict as edict

# Get paths and validate
try:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(file_dir)
    workspace_root = os.path.join("/workspace")  # Docker mount point

    paths_to_add = [
        project_root,
        workspace_root,
        os.path.join(workspace_root, "mad")
    ]

    # Add paths if they exist and aren't already in sys.path
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added to Python path: {path}")
        else:
            print(f"Path does not exist or already in sys.path: {path}")

    print(f"Project root: {project_root}")
    print(f"Python path: {os.environ.get('PYTHONPATH', '')}")

except Exception as e:
    print(f"Failed to setup paths: {str(e)}")
    raise

def get_config(args):

    if args.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    config = edict(vars(args))

    # TODO: remove
    config.use_lora = False
    config.train_scratch = False

    dataset_paths = {
        "facemorpher": "/data/mcaldeir/FaceMAD/Protocols_Preprocessed_Repeat/FaceMorpher.csv",
        "mipgan1": "/data/mcaldeir/FaceMAD/Protocols_Preprocessed_Repeat/MIPGAN_I.csv",
        "mipgan2": "/data/mcaldeir/FaceMAD/Protocols_Preprocessed_Repeat/MIPGAN_II.csv",
        "mordiff": "/data/mcaldeir/FaceMAD/Protocols_Preprocessed_Repeat/MorDIFF.csv",
        "opencv": "/data/mcaldeir/FaceMAD/Protocols_Preprocessed_Repeat/OpenCV.csv",
        "webmorph": "/data/mcaldeir/FaceMAD/Protocols_Preprocessed_Repeat/Webmorph.csv"
    }

    # list of csvs for testing
    config.test_dataset_path = [dataset_paths["facemorpher"], dataset_paths["mipgan1"], dataset_paths["mipgan2"], dataset_paths["mordiff"], dataset_paths["opencv"], dataset_paths["webmorph"]]
    config.test_data = ["facemorpher", "mipgan1", "mipgan2", "mordiff", "opencv", "webmorph"]

    config.num_classes = 2
    if config.backbone_size == "ViT-B/16" or config.backbone_size == "ViT-B/32":
        config.training_desc = f'ViT-B16/test_clip'
    elif config.backbone_size == "ViT-L/14":
        config.training_desc = f'ViT-L14/test_clip'
    elif config.backbone_size == "ViT-G/14":
        config.training_desc = f'ViT-G14/test_clip'
    elif config.backbone_size == "base":
        config.training_desc = f'base/test_clip'
    elif config.backbone_size == "large":
        config.training_desc = f'large/test_clip'

    config.output_path = "/igd/a1/home/mcaldeir/MADation/output/no_train/" + config.training_desc

    config.method = 'avg' # 'avg', 'max', 'max_avg'

    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # cudnn.benchmark = True
    set_seed(seed=777)

    parser = argparse.ArgumentParser(description="Distributed training job")
    parser.add_argument("--local-rank", type=int, help="local_rank")
    parser.add_argument(
        "--mode",
        default="training",
        choices=["training", "evaluation"],
        help="train or eval mode",
    )
    parser.add_argument(
        "--debug", default=False, type=bool, help="Log additional debug informations"
    )

    parser.add_argument("--backbone_size", type=str, required=True)

    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr_model", type=float, default=1e-6)
    parser.add_argument("--lr_header", type=float, default=1e-6)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=2)
    parser.add_argument("--lora_a", type=int, default=2)
    parser.add_argument("--max_norm", type=float, default=5)
    parser.add_argument("--loss", type=str, default="BinaryCrossEntropy")
    parser.add_argument("--global_step", type=int, default=0)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup", type=bool, default=True)
    parser.add_argument("--num_warmup_epochs", type=int, default=5)
    parser.add_argument("--T_0", type=int, default=5)
    parser.add_argument("--T_mult", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("--lr_func_drop", type=int, nargs="+", default=[22, 30, 40])
    parser.add_argument("--batch_size", type=int, default=86)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=["q", "v"]
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--normalize_type", type=str, default="clip")
    parser.add_argument("--interpolation_type", type=str, default="bicubic")
    parser.add_argument(
        "--eval_path", type=str, default="/home/chettaou/data/validation"
    )
    parser.add_argument("--val_targets", type=str, nargs="+", default=[])
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=40)
    parser.add_argument("--batch_size_eval", type=int, default=16)
    parser.add_argument("--horizontal_flip", type=bool, default=True)
    parser.add_argument("--rand_augment", type=bool, default=True)
    args = parser.parse_args()
    config = get_config(args)
    from src.train import main
    main(config)

# Add at the bottom of config.py (after argparse logic)
def get_initialized_config():
    """Call this instead of get_config() to get the fully initialized config"""
    args = parser.parse_args()
    return get_config(args)

# Only initialize if run directly (not when imported)
if __name__ == "__main__":
    config = get_initialized_config()
    from src.train import main
    main(config)