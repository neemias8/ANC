"""
Utility functions for PRIMERA-based narrative consolidation.
Handles model loading, seed configuration, and common utilities.
"""

import os
import random
import numpy as np
import torch
from typing import Optional, Dict, Any
from pathlib import Path
from transformers import (
    LEDForConditionalGeneration,
    LEDTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    set_seed
)


class PrimeraModelLoader:
    """
    Loader and manager for PRIMERA model.
    
    PRIMERA is a variant of LED (Longformer Encoder-Decoder) 
    pre-trained for multi-document summarization.
    """
    
    def __init__(
        self, 
        model_name: str = "allenai/PRIMERA",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize PRIMERA model loader.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(str(Path.home()), ".cache", "huggingface")
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"🔧 Initializing PRIMERA on device: {self.device}")
        
        # Model and tokenizer will be loaded lazily
        self._model = None
        self._tokenizer = None
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            print(f"📥 Loading PRIMERA model: {self.model_name}")
            try:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self._model.to(self.device)
                self._model.eval()  # Set to evaluation mode
                print(f"✅ Model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            print(f"📥 Loading PRIMERA tokenizer: {self.model_name}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                print(f"✅ Tokenizer loaded successfully")
            except Exception as e:
                print(f"❌ Error loading tokenizer: {e}")
                raise
        return self._tokenizer
    
    def generate(
        self,
        input_text: str,
        max_length: int = 1024,
        min_length: int = 100,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        no_repeat_ngram_size: int = 3,
        early_stopping: bool = True,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate text using PRIMERA model.
        
        Args:
            input_text: Input text to generate from
            max_length: Maximum length of generated text (in tokens)
            min_length: Minimum length of generated text
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for generation
            no_repeat_ngram_size: Size of n-grams that cannot repeat
            early_stopping: Whether to stop early
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=16384,  # PRIMERA can handle very long inputs
            padding=True,  # Only pad to longest sequence, not max_length
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                temperature=temperature,
                **kwargs
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return generated_text
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self._model is not None,
            "tokenizer_loaded": self._tokenizer is not None,
        }
        
        if self._model is not None:
            info["num_parameters"] = sum(p.numel() for p in self._model.parameters())
            info["num_trainable_parameters"] = sum(
                p.numel() for p in self._model.parameters() if p.requires_grad
            )
        
        if self._tokenizer is not None:
            info["vocab_size"] = self._tokenizer.vocab_size
            info["model_max_length"] = self._tokenizer.model_max_length
        
        return info


def set_reproducibility_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    print(f"🌱 Setting reproducibility seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Hugging Face transformers
    set_seed(seed)
    
    # Additional reproducibility settings for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("✅ Reproducibility seed set successfully")


def format_input_for_primera(
    text: str,
    task_prefix: str = "",
    max_length: int = 16384
) -> str:
    """
    Format input text for PRIMERA model.
    
    IMPORTANTE: PRIMERA é um modelo MULTI-DOCUMENT.
    Se você tem múltiplos documentos, separe-os com ' <doc-sep> '.
    
    Exemplo para MDS:
        doc1 = "Gospel of Matthew text..."
        doc2 = "Gospel of Mark text..."
        input_text = f"{doc1} <doc-sep> {doc2}"
    
    Args:
        text: Input text (pode conter múltiplos docs separados por <doc-sep>)
        task_prefix: Optional task-specific prefix
        max_length: Maximum input length (PRIMERA supports up to 16384 tokens)
        
    Returns:
        Formatted input text
    """
    if task_prefix:
        formatted_text = f"{task_prefix}: {text}"
    else:
        formatted_text = text
    
    # Note: Actual truncation happens in tokenizer
    return formatted_text


def format_multiple_documents_for_primera(
    documents: list,
    task_prefix: str = ""
) -> str:
    """
    Format multiple documents for PRIMERA model using <doc-sep> separator.
    
    Args:
        documents: List of document strings
        task_prefix: Optional task-specific prefix
        
    Returns:
        Formatted multi-document input for PRIMERA
    """
    # Join documents with PRIMERA's special separator
    multi_doc_text = " <doc-sep> ".join(doc.strip() for doc in documents)
    
    return format_input_for_primera(multi_doc_text, task_prefix)


def chunk_text(text: str, max_chunk_size: int = 10000) -> list:
    """
    Chunk text into smaller pieces for processing.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # Simple chunking by characters
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for last period before max_chunk_size
            last_period = text.rfind('.', start, end)
            if last_period > start:
                end = last_period + 1
        
        chunks.append(text[start:end].strip())
        start = end
    
    return chunks


def save_output(text: str, output_path: str, method_name: str = ""):
    """
    Save generated text to file.
    
    Args:
        text: Generated text to save
        output_path: Path to output file
        method_name: Name of method (for logging)
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"💾 Saved {method_name} output to: {output_path}")
    print(f"   Length: {len(text)} characters")


def load_reference_text(reference_path: str = "data/Golden_Sample.txt") -> str:
    """
    Load reference (Golden Sample) text.
    
    Args:
        reference_path: Path to reference file
        
    Returns:
        Reference text
    """
    reference_file = Path(reference_path)
    
    if not reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    
    with open(reference_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    print(f"📖 Loaded reference text: {len(text)} characters")
    return text


def print_generation_config(config: Dict[str, Any]):
    """
    Print generation configuration in a formatted way.
    
    Args:
        config: Configuration dictionary
    """
    print("\n" + "="*70)
    print("GENERATION CONFIGURATION")
    print("="*70)
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("="*70 + "\n")


def estimate_tokens(text: str, tokenizer=None) -> int:
    """
    Estimate number of tokens in text.
    
    Args:
        text: Input text
        tokenizer: Optional tokenizer (if None, uses rough estimation)
        
    Returns:
        Estimated token count
    """
    if tokenizer is not None:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
    else:
        # Rough estimation: ~4 characters per token for English
        return len(text) // 4


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Dictionary with memory statistics (in MB)
    """
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def print_gpu_memory():
    """Print GPU memory usage in a formatted way."""
    memory = get_gpu_memory_usage()
    
    if memory["available"]:
        print("\n🎮 GPU Memory Usage:")
        print(f"   Allocated: {memory['allocated_mb']:.1f} MB")
        print(f"   Reserved:  {memory['reserved_mb']:.1f} MB")
        print(f"   Max Alloc: {memory['max_allocated_mb']:.1f} MB\n")
    else:
        print("\n💻 Running on CPU (no GPU available)\n")


def main():
    """Test utility functions."""
    print("="*70)
    print("TESTING PRIMERA UTILITIES")
    print("="*70)
    
    # Test seed setting
    set_reproducibility_seed(42)
    
    # Test model loader initialization (but don't load yet)
    print("\n" + "="*70)
    print("INITIALIZING MODEL LOADER")
    print("="*70)
    loader = PrimeraModelLoader(device="cpu")  # Use CPU for testing
    
    # Print model info (will trigger loading)
    print("\n" + "="*70)
    print("MODEL INFORMATION")
    print("="*70)
    info = loader.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test text formatting
    print("\n" + "="*70)
    print("TEXT FORMATTING")
    print("="*70)
    sample_text = "This is a sample text for testing."
    formatted = format_input_for_primera(
        sample_text, 
        task_prefix="Summarize"
    )
    print(f"Original: {sample_text}")
    print(f"Formatted: {formatted}")
    
    # Test chunking
    print("\n" + "="*70)
    print("TEXT CHUNKING")
    print("="*70)
    long_text = "This is a sentence. " * 1000  # Create long text
    chunks = chunk_text(long_text, max_chunk_size=1000)
    print(f"Original length: {len(long_text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Chunk sizes: {[len(c) for c in chunks[:5]]}... (showing first 5)")
    
    # Test GPU memory
    print_gpu_memory()
    
    print("\n✅ All utility tests completed!")


if __name__ == "__main__":
    main()
