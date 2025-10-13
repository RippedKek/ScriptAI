import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from config import MODEL_NAME, USE_4BIT_QUANTIZATION, USE_GPU


def check_system():
    if USE_GPU and torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = "cpu"
        print(f"Using CPU (GPU not available or disabled)")
    
    print()
    return device


def load_model(device=None):
    """
    Load Qwen2-VL model with optimizations
    
    Args:
        device: 'cuda' or 'cpu', auto-detected if None
    
    Returns:
        tuple: (model, processor, device)
    """
    if device is None:
        device = check_system()
    
    print("Loading Qwen2-VL-2B-Instruct...")
    print()
    
    if USE_4BIT_QUANTIZATION and device == "cuda":
        print("Using 4-bit quantization")
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"4-bit quantization failed: {e}")
            print("Falling back to full precision...")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
    else:
        # Full precision
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            model = model.to("cpu")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    print("Model loaded successfully!")
    print()
    
    return model, processor, device
