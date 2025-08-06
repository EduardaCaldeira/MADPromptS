import logging
from .model import ClipModel, OpenClipModel, DINOv2Model, SigLIPModel, DeCLIPModel, BLIPModel
from transformers import AutoProcessor, SiglipModel

def get_model(rank, **kwargs):
    name = kwargs["model_name"]

    if name == "clip":
        logging.info("Loading model: " + name + " " + kwargs["backbone_size"])

        clip_model = ClipModel(
            rank=rank,
            model_name=kwargs["backbone_size"]
        )
        return clip_model, None
    elif name == "open_clip":
        logging.info("Loading model: " + name + " " + kwargs["backbone_size"])

        open_clip_model = OpenClipModel(
            rank=rank,
            model_name=kwargs["backbone_size"]
        )
        return open_clip_model, open_clip_model.preprocess
    elif name == "dinov2":
        dinov2_model = DINOv2Model(
            rank=rank,
            backbone_size=kwargs["backbone_size"], 
        )
        return dinov2_model
    elif name == "siglip":
        siglip_model = SigLIPModel(rank=rank, backbone_size=kwargs["backbone_size"])
        return siglip_model.backbone, siglip_model.processor
    elif name == "declip":
        logging.info("Loading model: " + name)
        declip_model = DeCLIPModel(rank=rank)
        return declip_model.backbone, declip_model.processor
    elif name == "blip":
        blip_model = BLIPModel(rank=rank, backbone_size=kwargs["backbone_size"])
        return blip_model.backbone, blip_model.processor
    else:
        raise ValueError()

def get_output_dim(**kwargs):
    name = kwargs["model_name"]

    if name == "clip" or name == "open_clip":
        backbone_embeddings = {
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
            "ViT-L/14@336px": 768,
        }

        logging.info("Transformer dimension: " + str(backbone_embeddings[kwargs["backbone_size"]]))

        return backbone_embeddings[kwargs["backbone_size"]]
    elif name == 'siglip':
        backbone_embeddings = {
            "ViT-B/16": 768,
            "ViT-G/14": 1024,
        }

        logging.info("Transformer dimension: " + str(backbone_embeddings[kwargs["backbone_size"]]))

        return backbone_embeddings[kwargs["backbone_size"]]
    elif name == "dinov2":
        backbone_embeddings = {
            "dino_s": 384,
            "dino_b": 768,
            "dino_l": 1024,
            "dino_g": 1536,
        }
        logging.info("Transformer dimension: " + str(backbone_embeddings[kwargs["backbone_size"]]))
        return backbone_embeddings[kwargs["backbone_size"]]
    elif name == "declip":
        # DeCLIP currently supports only ViT-B/32, embedding dim is 512
        embedding_dim = 512
        logging.info(f"Transformer dimension: {embedding_dim}")
        return embedding_dim
    elif name == "blip":
            backbone_embeddings = {
                "base": 768,
                "large": 1024,
            }
            logging.info("Transformer dimension: " + str(backbone_embeddings[kwargs["backbone_size"]]))
            return backbone_embeddings[kwargs["backbone_size"]]
    else:
        raise ValueError()
