from .trainer import TrainerClip

def get_trainer(rank, world_size, model_name, model, preprocess, config, test_dataloader=None, test_sampler=None):
    if model_name == "clip" or model_name == "open_clip" or model_name=="siglip" or model_name=='blip':
        trainer = TrainerClip(rank, world_size, model, preprocess, config, test_dataloader, test_sampler)
    else:
        raise ValueError()

    return trainer