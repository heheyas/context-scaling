import os
from PIL import Image
import torch
torch.cuda.empty_cache()
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.functional as F
from torchvision import transforms
from transformers import (AutoModel, AutoProcessor, AutoTokenizer, AutoConfig,
                            CLIPImageProcessor, CLIPVisionModelWithProjection)
from qwen_vl_utils import process_vision_info

# Don't restrict to single GPU — let caller decide
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(42) 
torch.cuda.manual_seed_all(42)

class Qwen2_5VLBatchInferencer:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
                    device: str = "cuda", 
                    dtype=torch.bfloat16, 
                    use_flash_attention: bool = True):
        
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        
        from transformers import Qwen2_5_VLForConditionalGeneration
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path, padding_side="left")
        self.device = torch.device(device)
        self.TEXT_PROMPT = (
            "Recognize the text in the image, only reply with the text content, "
            "but avoid repeating previously mentioned content. "
            "If no text is recognized, please reply with 'No text recognized'."
        )

    def batch_inference(self, messages, max_new_tokens=128):
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        return output_texts

    def infer_semantic(self, images_path: list, question: str):
        messages = []
        for image_path in images_path:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": f"{question}. Please answer 'Yes' or 'No' only."}
                    ],
                }
            ])
        return self.batch_inference(messages)

    def infer_ocr(self, images_path: list, max_new_tokens: int = 128):
        messages = []
        for image_path in images_path:
            messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": self.TEXT_PROMPT}
                    ],
                }
            ])
        return self.batch_inference(messages, max_new_tokens=max_new_tokens)
    

class CSDStyleEmbedding:
    def __init__(self, model_path: str = "scripts/style/models/checkpoint.pth", device: str = "cuda"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path).to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def _load_model(self, model_path: str):
      
        from scripts.utils.CSD_config import CSD_CLIP, convert_state_dict
      
        model = CSD_CLIP("vit_large", "default")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(state_dict, strict=False)
        return model

    def get_style_embedding(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, style_output = self.model(image_tensor)
        return style_output


class SEStyleEmbedding:
    def __init__(self, pretrained_path: str = "xingpng/OneIG-StyleEncoder", device: str = "cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_path)
        self.image_encoder.to(self.device, dtype=self.dtype)
        self.image_encoder.eval()
        self.processor = CLIPImageProcessor()

    def _l2_normalize(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def get_style_embedding(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.image_encoder(inputs)
            image_embeds = outputs.image_embeds
            image_embeds_norm = self._l2_normalize(image_embeds)
        return image_embeds_norm


class LLM2CLIP:
    def __init__(self, processor_model="openai/clip-vit-large-patch14-336", 
                 model_name="microsoft/LLM2CLIP-Openai-L-14-336", 
                 llm_model_name="microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned", 
                 device='cuda'):
        # Initialize processor and models
        self.processor = CLIPImageProcessor.from_pretrained(processor_model)

        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device).eval()

        self.llm_model_name = llm_model_name
        self.config = AutoConfig.from_pretrained(
            self.llm_model_name, trust_remote_code=True
        )
        self.llm_model = AutoModel.from_pretrained(
            self.llm_model_name, torch_dtype=torch.bfloat16, config=self.config, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        
        self.llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'  # Workaround for LLM2VEC
        
        from scripts.utils.llm2clip.llm2vec import LLM2Vec
        
        self.l2v = LLM2Vec(self.llm_model, self.tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

        self.device = device

    def text_img_similarity_score(self, image_path_list, text_prompt):
        try:
            captions = [text_prompt]
            images = [Image.open(image_path) for image_path in image_path_list]
            
            # Process images and encode text
            input_pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
            text_features = self.l2v.encode(captions, convert_to_tensor=True).to(self.device)

            # Get image and text features
            with torch.no_grad(), torch.amp.autocast(self.device):
                image_features = self.model.get_image_features(input_pixels)
                text_features = self.model.get_text_features(text_features)

                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Compute similarity score (dot product)
                text_probs = image_features @ text_features.T
                text_probs = text_probs.cpu().tolist()

            return [text_prob[0] for text_prob in text_probs]
        except Exception as e:
            print(f"Error: {e}")
            return None
