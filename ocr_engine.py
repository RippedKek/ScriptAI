import torch
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
from config import (
    DEFAULT_MAX_TOKENS,
    MAX_TOKENS_STUDENT_INFO,
    MAX_TOKENS_STRUCTURED,
    PROMPT_EXTRACT_ALL,
    PROMPT_STUDENT_INFO,
    PROMPT_STRUCTURED,
    PROMPT_FIGURE
)


class OCREngine:
    """OCR Engine for handwritten text extraction"""

    def __init__(self, model, processor, device):
        """
        Initialize OCR Engine

        Args:
            model: Loaded Qwen2-VL model
            processor: Loaded processor
            device: 'cuda' or 'cpu'
        """
        self.model = model
        self.processor = processor
        self.device = device

    def extract_text(self, image_path, custom_prompt=None, max_tokens=None):
        """
        Extract handwritten text from image

        Args:
            image_path: Path to image file or PIL Image
            custom_prompt: Optional custom instruction
            max_tokens: Maximum tokens to generate

        Returns:
            Extracted text as string
        """
        # Load image if path is provided
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path)
        else:
            image = image_path

        # Use default prompt if none provided
        if custom_prompt is None:
            custom_prompt = PROMPT_STUDENT_INFO

        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS

        # Prepare message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": custom_prompt},
                ],
            }
        ]

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(output_text)
        return output_text[0]

    def extract_student_info(self, image_path):
        """Extract student name and ID from answer sheet"""
        return self.extract_text(
            image_path,
            custom_prompt=PROMPT_STUDENT_INFO,
            max_tokens=MAX_TOKENS_STUDENT_INFO
        )

    def extract_with_structure(self, image_path):
        """Extract text preserving delimiters and structure"""
        return self.extract_text(
            image_path,
            custom_prompt=PROMPT_STRUCTURED,
            max_tokens=MAX_TOKENS_STRUCTURED
        )

    def extract_all(self, image_path):
        """Extract all text, preserving 'Answer...' and 'End of Answer-<id>' markers when present."""
        return self.extract_text(
            image_path,
            custom_prompt=PROMPT_EXTRACT_ALL,
            max_tokens=DEFAULT_MAX_TOKENS
        )
        
    def assess_figure(self, image_path, target='', custom_prompt=None, max_tokens=None):
        """
        Assess the figure from image

        Args:
            image_path: Path to image file or PIL Image
            custom_prompt: Optional custom instruction
            max_tokens: Maximum tokens to generate

        Returns:
            Figure assessment as JSON string
        """
        # Load image if path is provided
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            image = Image.open(image_path)
        else:
            image = image_path

        # Use default figure prompt if none provided
        if custom_prompt is None:
            custom_prompt = PROMPT_FIGURE + target

        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS

        # Prepare message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": custom_prompt},
                ],
            }
        ]

        # Prepare for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False
            )

        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        print(output_text)
        return output_text[0]
        
