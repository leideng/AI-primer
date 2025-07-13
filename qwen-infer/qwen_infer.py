#!/usr/bin/env python3
"""
Qwen3 Inference Program

This program provides inference capabilities for Qwen3 models with support for:
- Local model loading
- Hugging Face model loading
- Batch inference
- Streaming generation
- Various generation parameters
- Memory optimization
"""

import torch
import argparse
import time
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import warnings
warnings.filterwarnings("ignore")

__lei_model_path__ = r"D:\models\qwen3-0.6B"

class Qwen3Inference:
    """Qwen3 inference class with various optimization options."""
    
    def __init__(
        self,
        model_name_or_path: str = __lei_model_path__,
        device: str = "auto",
        torch_dtype: str = "auto",
        use_flash_attention: bool = True,
        trust_remote_code: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        max_memory: Optional[Dict[str, str]] = None,
        use_compile: bool = False,
        compile_mode: str = "default"
    ):
        """
        Initialize Qwen3 inference.
        
        Args:
            model_name_or_path: Model name or path
            device: Device to load model on ("auto", "cpu", "cuda", "mps")
            torch_dtype: Torch dtype ("auto", "float16", "bfloat16", "float32")
            use_flash_attention: Whether to use flash attention
            trust_remote_code: Whether to trust remote code
            load_in_8bit: Whether to load model in 8-bit
            load_in_4bit: Whether to load model in 4-bit
            max_memory: Maximum memory allocation per device
        """
        self.model_name_or_path = model_name_or_path
        self.device = self._get_device(device)
        self.torch_dtype = self._get_torch_dtype(torch_dtype)
        
        print(f"Loading model: {model_name_or_path}")
        print(f"Device: {self.device}")
        print(f"Torch dtype: {self.torch_dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": self.torch_dtype,
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        
        if max_memory:
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_memory
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if not max_memory:
            self.model = self.model.to(self.device)
        
        # Enable flash attention if requested
        if use_flash_attention and hasattr(self.model, "enable_flash_attention"):
            self.model.enable_flash_attention()
        
        # Apply torch.compile if requested
        if use_compile:
            print(f"Applying torch.compile with mode: {compile_mode}")
            try:
                self.model = torch.compile(self.model, mode=compile_mode)
                print("torch.compile applied successfully!")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")
                print("Continuing without compilation...")
        
        print("Model loaded successfully!")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _get_torch_dtype(self, torch_dtype: str) -> torch.dtype:
        """Convert torch_dtype string to torch.dtype."""
        if torch_dtype == "auto":
            if torch.cuda.is_available():
                return torch.float16
            else:
                return torch.float32
        elif torch_dtype == "float16":
            return torch.float16
        elif torch_dtype == "bfloat16":
            return torch.bfloat16
        elif torch_dtype == "float32":
            return torch.float32
        else:
            return torch.float32
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling
            stream: Whether to stream the output
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        if stream:
            return self._generate_stream(inputs, generation_config)
        else:
            return self._generate_batch(inputs, generation_config)
    
    def _generate_batch(self, inputs: Dict[str, torch.Tensor], generation_config: GenerationConfig) -> str:
        """Generate text in batch mode."""
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode the generated tokens
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output
        input_length = inputs["input_ids"].shape[1]
        generated_text = generated_text[input_length:]
        
        return generated_text
    
    def _generate_stream(self, inputs: Dict[str, torch.Tensor], generation_config: GenerationConfig) -> str:
        """Generate text in streaming mode using a simple token-by-token approach."""
        generated_text = ""
        current_input_ids = inputs["input_ids"].clone()
        
        with torch.no_grad():
            for _ in range(generation_config.max_new_tokens):
                # Generate next token
                outputs = self.model.generate(
                    current_input_ids,
                    max_new_tokens=1,
                    do_sample=generation_config.do_sample,
                    temperature=generation_config.temperature,
                    top_p=generation_config.top_p,
                    top_k=generation_config.top_k,
                    repetition_penalty=generation_config.repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=False
                )
                
                # Get the new token
                new_token_id = outputs[0][-1].unsqueeze(0)
                
                # Check if we've reached the end
                if new_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                # Decode and print the new token
                new_token = self.tokenizer.decode(new_token_id, skip_special_tokens=True)
                print(new_token, end="", flush=True)
                generated_text += new_token
                
                # Add to input for next iteration
                current_input_ids = torch.cat([current_input_ids, new_token_id.unsqueeze(0)], dim=1)
        
        print()  # New line after streaming
        return generated_text
    
    def _generate_stream_simple(self, inputs: Dict[str, torch.Tensor], generation_config: GenerationConfig) -> str:
        """Generate text in streaming mode using a simpler approach."""
        generated_text = ""
        input_length = inputs["input_ids"].shape[1]
        
        # Generate one token at a time
        current_input_ids = inputs["input_ids"].clone()
        
        with torch.no_grad():
            for _ in range(generation_config.max_new_tokens):
                # Generate next token
                outputs = self.model.generate(
                    current_input_ids,
                    max_new_tokens=1,
                    generation_config=generation_config,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=False
                )
                
                # Get the new token
                new_token_id = outputs[0][-1].unsqueeze(0)
                
                # Check if we've reached the end
                if new_token_id.item() == self.tokenizer.eos_token_id:
                    break
                
                # Decode and print the new token
                new_token = self.tokenizer.decode(new_token_id, skip_special_tokens=True)
                print(new_token, end="", flush=True)
                generated_text += new_token
                
                # Add to input for next iteration
                current_input_ids = torch.cat([current_input_ids, new_token_id.unsqueeze(0)], dim=1)
        
        print()  # New line after streaming
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            **kwargs: Generation parameters
            
        Returns:
            List of generated texts
        """
        # Tokenize all inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode all outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            input_length = inputs["input_ids"][i].shape[0]
            generated_text = generated_text[input_length:]
            generated_texts.append(generated_text)
        
        return generated_texts


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Qwen3 Inference")
    parser.add_argument("--model", type=str, default=r"D:\models\qwen3-0.6B", 
                       help="Model name or path")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", 
                       help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=512, 
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, 
                       help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, 
                       help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, 
                       help="Repetition penalty")
    parser.add_argument("--device", type=str, default="auto", 
                       help="Device to use (auto, cpu, cuda, mps)")
    parser.add_argument("--torch_dtype", type=str, default="auto", 
                       help="Torch dtype (auto, float16, bfloat16, float32)")
    parser.add_argument("--stream", action="store_true", 
                       help="Enable streaming generation")
    parser.add_argument("--batch", action="store_true", 
                       help="Enable batch mode (read prompts from stdin)")
    parser.add_argument("--load_in_8bit", action="store_true", 
                       help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true", 
                       help="Load model in 4-bit precision")
    parser.add_argument("--no_flash_attention", action="store_true", 
                       help="Disable flash attention")
    parser.add_argument("--use_compile", action="store_true", 
                       help="Enable torch.compile for faster inference")
    parser.add_argument("--compile_mode", type=str, default="default", 
                       choices=["default", "reduce-overhead", "max-autotune"],
                       help="torch.compile mode (default, reduce-overhead, max-autotune)")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = Qwen3Inference(
        model_name_or_path=args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        use_flash_attention=not args.no_flash_attention,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode
    )
    
    if args.batch:
        # Batch mode
        print("Enter prompts (one per line, Ctrl+D to finish):")
        prompts = []
        try:
            while True:
                prompt = input()
                prompts.append(prompt)
        except EOFError:
            pass
        
        if prompts:
            print(f"\nGenerating for {len(prompts)} prompts...")
            start_time = time.time()
            results = inference.batch_generate(
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty
            )
            end_time = time.time()
            
            print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")
            print("\nResults:")
            for i, (prompt, result) in enumerate(zip(prompts, results)):
                print(f"\n--- Prompt {i+1} ---")
                print(f"Input: {prompt}")
                print(f"Output: {result}")
    else:
        # Single prompt mode
        print(f"Input: {args.prompt}")
        print("Generating...")
        
        start_time = time.time()
        result = inference.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stream=args.stream
        )
        end_time = time.time()
        
        if not args.stream:
            print(f"Output: {result}")
        print(f"\nGeneration completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
