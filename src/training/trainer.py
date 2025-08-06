import logging
import os
from dataclasses import dataclass
from typing import List

import clip
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from utils.utils import (evaluate_mad_performance, print_trainable_parameters, write_scores)

import open_clip
from torchvision.transforms import ToPILImage


@dataclass
class TestData:
    scores: torch.Tensor
    labels: torch.Tensor
    video_ids: List[str]

########  Default Trainer ########
class Trainer():
    def __init__(self, rank, world_size, model, preprocess, config, test_dataloader=None, test_sampler=None):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.preprocess = preprocess
        self.test_dataloader = test_dataloader
        self.test_sampler = test_sampler
        self.config = config

        self.start_epoch = 0
        self.global_step = self.config.global_step

        # Logging
        logging.info("Config is: {}".format(self.config.__dict__))

########################
########  CLIP  ########
########################
class TrainerClip(Trainer):
    def __init__(self, rank, world_size, model, preprocess, config, test_dataloader=None, test_sampler=None):
        super().__init__(rank, world_size, model, preprocess, config, test_dataloader, test_sampler)

    def start_training(self):
        self.test_clip()

    def gather_test_data(self, scores, labels, video_ids):
        local_data = TestData(
            scores=scores.cpu(), labels=labels.cpu(), video_ids=video_ids
        )
        gathered_data = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_data, local_data)

        if self.rank == 0:
            all_scores = []
            all_labels = []
            all_video_ids = []

            for data in gathered_data:
                all_scores.append(data.scores)
                all_labels.append(data.labels)
                all_video_ids.extend(data.video_ids)

            return (torch.cat(all_scores), torch.cat(all_labels), all_video_ids)
        return None, None, None

    def determine_scores(self, text_features, current_output_path, preprocess=None):
        results = []

        for i, testdata in enumerate(self.test_dataloader):
            if self.test_sampler:
                self.test_sampler[i].set_epoch(0)

            raw_test_scores, gt_labels = [], []
            raw_test_img_pths = []
            with torch.no_grad():
                for _, (raw, labels, img_paths) in enumerate(testdata):
                    raw = raw.cuda(self.rank, non_blocking=True)
                    labels = labels.cuda(self.rank, non_blocking=True)

                    if preprocess is not None:
                        pil_images = []
                        for batch_element in range(raw.shape[0]):
                            pil_image = ToPILImage()(raw[batch_element].cpu())
                            pil_images.append(pil_image)
                            #raw[i] = preprocess(raw[i])

                        if self.config.model_name == 'siglip':
                            features_list = []
                            for img in pil_images:
                                features = self.encode_siglip_image(self.preprocess, self.model.module, img, device=f"cuda:{self.rank}")
                                features_list.append(features)

                            image_features = torch.cat(features_list, dim=0)  # shape: [batch_size, feature_dim]
                        elif self.config.model_name == 'blip':
                            image_inputs = self.preprocess(images=pil_images, return_tensors="pt").to(f"cuda:{self.rank}")
                            image_features = self.model.module.get_image_features(**image_inputs)
                    else:    
                        #raw = torch.cat([preprocess(img).unsqueeze(0) for img in pil_images], dim=0).cuda(self.rank, non_blocking=True)
                        image_features = self.model.module.encode_image(raw)

                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    if self.config.method == 'avg':
                        logits_per_image = (100.0 * image_features @ text_features.T).softmax(
                           dim=-1
                        )                        
                    else: 
                        logits_per_image = (100.0 * image_features @ text_features.T)
                        break_len = int(len(logits_per_image.T)/2)
                        part_1 = logits_per_image[:, 0:break_len].max(dim=1, keepdim=True).values
                        part_2 = logits_per_image[:, break_len:].max(dim=1, keepdim=True).values
                        logits_per_image = torch.cat([part_1, part_2], dim=1).softmax(
                           dim=-1
                        )                        

                    raw_scores = logits_per_image[:, 1]

                    raw_test_scores.append(raw_scores)
                    gt_labels.append(labels)

                    for j in range(raw.shape[0]):
                        raw_test_img_pths.append(img_paths[j])

                raw_test_scores = torch.cat(raw_test_scores)
                gt_labels = torch.cat(gt_labels)
                # print(self.rank, raw_test_scores.shape, gt_labels.shape, len(raw_test_video_ids))
                scores, labels, img_pathes = self.gather_test_data(
                    raw_test_scores, gt_labels, raw_test_img_pths
                )
            if self.rank == 0:
                # print(self.rank, scores.shape, labels.shape, len(video_ids))
                raw_test_scores = scores.cpu().numpy()
                gt_labels = labels.cpu().numpy()

                out_path = os.path.join(current_output_path, self.config.test_data[i] + ".csv")
                write_scores(img_pathes, raw_test_scores, gt_labels, out_path)
                raw_test_stats = [np.mean(raw_test_scores), np.std(raw_test_scores)]
                raw_test_scores = (raw_test_scores - raw_test_stats[0]) / raw_test_stats[1]

                results.append(evaluate_mad_performance(raw_test_scores, gt_labels))
                # write the results to a txt file
                #                results = {
                #    "auc_score": auc_score,
                #    "eer": eer,
                #    "apcer_bpcer20": apcer_bpcer20,
                #    "apcer_bpcer10": apcer_bpcer10,
                #    "apcer_bpcer1": apcer_bpcer1,
                #    "bpcer_apcer20": bpcer_apcer20,
                #    "bpcer_apcer10": bpcer_apcer10,
                #    "bpcer_apcer1": bpcer_apcer1,
                # }                    

                with open(os.path.join(current_output_path, self.config.test_data[i] + ".txt"), "w") as f:
                    f.write(f"AUC: {results[i]['auc_score']:.4f}\n")
                    f.write(f"EER: {results[i]['eer']:.4f}\n")
                    f.write(f"APCER@BPCER20%: {results[i]['apcer_bpcer20']:.4f}\n")
                    f.write(f"APCER@BPCER10%: {results[i]['apcer_bpcer10']:.4f}\n")
                    f.write(f"APCER@BPCER1%: {results[i]['apcer_bpcer1']:.4f}\n")
                    f.write(f"BPCER@APCER20%: {results[i]['bpcer_apcer20']:.4f}\n")
                    f.write(f"BPCER@APCER10%: {results[i]['bpcer_apcer10']:.4f}\n")
                    f.write(f"BPCER@APCER1%: {results[i]['bpcer_apcer1']:.4f}\n")
    
    def get_text_features(self, morphing_prompts, bonafide_prompts, tokenizer=None):
        with torch.no_grad():
            prompts = morphing_prompts + bonafide_prompts 
            
            if tokenizer is not None:
                if self.config.model_name == "openclip":
                    text_inputs = tokenizer(prompts).cuda()
                elif self.config.model_name == 'blip':
                    pre_inputs = tokenizer.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                    text_inputs = {k: v.cuda() for k, v in pre_inputs.items()}
                else: 
                    pre_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
                    text_inputs = {k: v.cuda() for k, v in pre_inputs.items()}
            else:
                text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).cuda()

            if self.config.model_name == "siglip":
                text_features = self.encode_siglip_text(self.preprocess, self.model.module, prompts)
            elif self.config.model_name == "blip":
                text_features = self.model.module.get_text_features(**text_inputs)
            else:
                text_features = self.model.module.encode_text(text_inputs)
            
            text_features /= text_features.norm(dim=-1, keepdim=True)

            if len(morphing_prompts) > 1:
                if self.config.method == 'avg':
                    text_features = torch.cat([text_features[0:len(morphing_prompts)].mean(dim=0, keepdim=True), text_features[len(morphing_prompts):].mean(dim=0, keepdim=True)], dim=0)

                text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def encode_siglip_image(self, processor, model, image, device="cuda"):
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_siglip_text(self, processor, model, texts, device="cuda"):
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def test_clip(self):
        self.model.module.eval()
        
        templates_id = [
            'male {}.',
            'female {}.',
            'young {}.',
            'elderly {}.',
            'child {}.',
            'adult {}.',
            'asian {}.',
            'black {}.',
            'white {}.',
            'latino {}.',
            'middle eastern {}.',
            'indian {}.',
            'blonde {}.',
            'brunette {}.',
            'redhead {}.',
            'tall {}.',
            'short {}.',
            'thin {}.',
            'obese {}.',
            'teen {}.',
        ]

        templates_presentation = [
            'frontal {}.',
            'profile {}.',
            'tilted {}.',
            'rotated {}.',
            'upward {}.',
            'downward {}.',
            'sideways {}.',
            'leftward {}.',
            'rightward {}.',
            'angled {}.',
            'inclined {}.',
            'declined {}.',
            'oblique {}.',
            'twisted {}.',
            'turned {}.',
            'slanted {}.',
            'offcenter {}.',
            'misaligned {}.',
            'skewed {}.',
            'asymmetric {}.',
        ]

        templates_appearance = [
            'bearded {}.',
            'moustached {}.',
            'smiling {}.',
            'frowning {}.',
            'eyeglasses {}.',
            'sunglasses {}.',
            'wrinkled {}.',
            'balding {}.',
            'occluded {}.',
            'scarred {}.',
            'pierced {}.',
            'tanned {}.',
            'pale {}.',
            'makeup {}.',
            'freckled {}.',
            'chubby-cheeked {}.',
            'sweaty {}.',
            'dirty {}.',
            'blinking {}.',
            'tearful {}.',
        ]

        self.config.output_path = self.config.output_path + '/' + self.config.model_name 

        if self.config.model_name == "open_clip":
            tokenizer = open_clip.get_tokenizer(self.config.backbone_size.replace("/", "-"))
            #self.model.module.get_tokenizer()
            preprocess = self.preprocess
        elif self.config.model_name == "siglip": 
            tokenizer = None
            preprocess = self.preprocess
        elif self.config.model_name == "blip":
            tokenizer = self.preprocess
            preprocess = self.preprocess
        #elif self.config.model_name == "declip":
        #    tokenizer = self.preprocess.tokenizer
        #    preprocess = self.preprocess
        else:
            tokenizer = None
            preprocess = None

        if self.config.method == 'avg':
            morphing_prompts = ["face image morphing attack"]
            bonafide_prompts = ["bona-fide presentation"]
            text_features = self.get_text_features(morphing_prompts, bonafide_prompts, tokenizer=tokenizer)
            current_output_path = self.config.output_path + "/" + self.config.method + "/original"
            os.makedirs(current_output_path, exist_ok=True)
            self.determine_scores(text_features, current_output_path, preprocess=preprocess)

            morphing_prompts = ["face image morphing attack."]
            bonafide_prompts = ["bona-fide presentation."]
            text_features = self.get_text_features(morphing_prompts, bonafide_prompts, tokenizer=tokenizer)
            current_output_path = self.config.output_path + "/" + self.config.method + "/original_dot"
            os.makedirs(current_output_path, exist_ok=True)
            self.determine_scores(text_features, current_output_path, preprocess=preprocess)


        # id-related templates
        morphing_prompts = [template.format("face image morphing attack") for template in templates_id]
        bonafide_prompts = [template.format("bona-fide presentation") for template in templates_id]
        
        current_output_path = self.config.output_path + "/" + self.config.method + "/id_related"
        os.makedirs(current_output_path, exist_ok=True)

        text_features = self.get_text_features(morphing_prompts, bonafide_prompts, tokenizer=tokenizer)        
        self.determine_scores(text_features, current_output_path, preprocess=preprocess)

        # face presentation templates
        morphing_prompts = [template.format("face image morphing attack") for template in templates_presentation]
        bonafide_prompts = [template.format("bona-fide presentation") for template in templates_presentation]
        
        current_output_path = self.config.output_path + "/" + self.config.method + "/face_presentation"
        os.makedirs(current_output_path, exist_ok=True)

        text_features = self.get_text_features(morphing_prompts, bonafide_prompts, tokenizer=tokenizer)
        self.determine_scores(text_features, current_output_path, preprocess=preprocess)

        # face appearance templates
        morphing_prompts = [template.format("face image morphing attack") for template in templates_appearance]
        bonafide_prompts = [template.format("bona-fide presentation") for template in templates_appearance]
        
        current_output_path = self.config.output_path + "/" + self.config.method + "/face_appearance"
        os.makedirs(current_output_path, exist_ok=True)

        text_features = self.get_text_features(morphing_prompts, bonafide_prompts, tokenizer=tokenizer)
        self.determine_scores(text_features, current_output_path, preprocess=preprocess)

        # id-related + face presentation
        templates = templates_id + templates_presentation
        morphing_prompts = [template.format("face image morphing attack") for template in templates]
        bonafide_prompts = [template.format("bona-fide presentation") for template in templates]
        
        current_output_path = self.config.output_path + "/" + self.config.method + "/id_related_face_presentation"
        os.makedirs(current_output_path, exist_ok=True)

        text_features = self.get_text_features(morphing_prompts, bonafide_prompts, tokenizer=tokenizer)
        self.determine_scores(text_features, current_output_path, preprocess=preprocess)

        # id-related + face appearance
        templates = templates_id + templates_appearance
        morphing_prompts = [template.format("face image morphing attack") for template in templates]
        bonafide_prompts = [template.format("bona-fide presentation") for template in templates]
        
        current_output_path = self.config.output_path + "/" + self.config.method + "/id_related_face_appearance"
        os.makedirs(current_output_path, exist_ok=True)

        text_features = self.get_text_features(morphing_prompts, bonafide_prompts, tokenizer=tokenizer)
        self.determine_scores(text_features, current_output_path, preprocess=preprocess)

        # face presentation + face appearance
        templates = templates_presentation + templates_appearance
        morphing_prompts = [template.format("face image morphing attack") for template in templates]
        bonafide_prompts = [template.format("bona-fide presentation") for template in templates]
        
        current_output_path = self.config.output_path + "/" + self.config.method + "/face_presentation_face_appearance"
        os.makedirs(current_output_path, exist_ok=True)

        text_features = self.get_text_features(morphing_prompts, bonafide_prompts, tokenizer=tokenizer)
        self.determine_scores(text_features, current_output_path, preprocess=preprocess)

        # all templates
        templates = templates_id + templates_presentation + templates_appearance
        morphing_prompts = [template.format("face image morphing attack") for template in templates]
        bonafide_prompts = [template.format("bona-fide presentation") for template in templates]
        
        current_output_path = self.config.output_path + "/" + self.config.method + "/all"
        os.makedirs(current_output_path, exist_ok=True)

        text_features = self.get_text_features(morphing_prompts, bonafide_prompts, tokenizer=tokenizer)
        self.determine_scores(text_features, current_output_path, preprocess=preprocess)