"""
Inference module for generating text using trained LoRA models.
Supports both LoRA adapters and merged models.
"""

import gc
import torch
from typing import List, Dict, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class LoRAInference:
    """Handles inference with LoRA fine-tuned models."""
    
    def __init__(
        self,
        base_model_id: str,
        adapter_path: Optional[str] = None,
        use_4bit: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize inference engine.
        
        Args:
            base_model_id: Hugging Face model ID or path to base model
            adapter_path: Path to LoRA adapter (None for merged models)
            use_4bit: Use 4-bit quantization
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.use_4bit = use_4bit
        self.device = device
        
        self.tokenizer = None
        self.model = None
        
        print(f"🔧 Initializing inference engine...")
        print(f"  - Base model: {base_model_id}")
        if adapter_path:
            print(f"  - LoRA adapter: {adapter_path}")
        print(f"  - 4-bit quantization: {use_4bit}")
        
        self._load_model()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup

    def cleanup(self):
        """Clean up resources and free GPU memory."""
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            self.model = None
        
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    def _load_model(self):
        """Load model and tokenizer."""
        # Configure quantization
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        
        # Load tokenizer from adapter path or base model
        tokenizer_path = self.adapter_path if self.adapter_path else self.base_model_id
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        except (OSError, ValueError):
            # Tokenizer not found in adapter path, use base model
            if self.adapter_path:
                print(f"   ℹ️  Tokenizer not found in adapter, using base model tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map=self.device or "auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        
        # Load LoRA adapter if provided
        if self.adapter_path:
            print("  - Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            self.model = base_model
        
        self.model.eval()
        print("  ✓ Model loaded and ready")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.1,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling (False = greedy)
            repetition_penalty: Penalty for repeating tokens
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text (full output including prompt)
        """
        # Tokenize input
        inputs = self.tokenizer([prompt], return_tensors="pt", truncation=False).to(
            self.model.device
        )
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )
        
        # Decode only the newly generated tokens (skip the input prompt)
        generated_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate a chat response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        # Filter messages to handle models that don't support system role
        filtered_messages = []
        system_content = None
        
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content", "")
            
            if role == "system":
                # Store system message content to prepend to first user message
                if system_content:
                    system_content += "\n\n" + content
                else:
                    system_content = content
            else:
                # If we have system content and this is the first user message, prepend it
                if system_content and role == "user" and not filtered_messages:
                    filtered_messages.append({
                        "role": "user",
                        "content": system_content + "\n\n" + content
                    })
                    system_content = None  # Clear so we don't prepend again
                else:
                    filtered_messages.append(msg)
        
        # If we still have system content but no user messages, create one
        if system_content and not any(m.get("role", "").lower() == "user" for m in filtered_messages):
            filtered_messages.insert(0, {
                "role": "user",
                "content": system_content
            })
        
        # Use filtered messages (without system role)
        try:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                filtered_messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            # If applying template fails, try without system messages at all
            if "system" in str(e).lower() or "System role" in str(e):
                # Remove any remaining system messages
                filtered_messages = [m for m in filtered_messages if m.get("role", "").lower() != "system"]
                prompt = self.tokenizer.apply_chat_template(
                    filtered_messages, tokenize=False, add_generation_prompt=True
                )
            else:
                raise
        
        # Generate response - the generate method now returns only new tokens
        response = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        
        # Clean up the response - remove any template artifacts
        response = response.strip()
        
        # Remove any leading/trailing whitespace and template markers if present
        # Gemma uses <start_of_turn>model and <end_of_turn> markers
        if "<start_of_turn>model" in response:
            # Remove the marker and any newline after it
            response = response.replace("<start_of_turn>model", "").replace("<start_of_turn>model\n", "").strip()
        if "<end_of_turn>" in response:
            # Remove the marker and any newline before it
            response = response.replace("<end_of_turn>", "").replace("\n<end_of_turn>", "").strip()
        
        # Remove any extra newlines at the start/end
        response = response.strip()
        
        return response
    
    def chat_simple(
        self, user_message: str, system_message: str = "You are a helpful assistant.", **kwargs
    ) -> str:
        """
        Simple chat interface with a single user message.
        
        Args:
            user_message: User's message
            system_message: System prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Assistant's response
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        
        return self.chat(messages, **kwargs)


class InteractiveChatSession:
    """Interactive chat session with conversation history."""
    
    def __init__(
        self,
        inference_engine: LoRAInference,
        system_message: str = "You are a helpful assistant.",
        max_history: int = 10,
    ):
        """
        Initialize chat session.
        
        Args:
            inference_engine: LoRAInference instance
            system_message: System prompt
            max_history: Maximum number of message pairs to keep in history
        """
        self.engine = inference_engine
        self.system_message = system_message
        self.max_history = max_history
        self.messages = [{"role": "system", "content": system_message}]
    
    def send(self, user_message: str, **kwargs) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: User's message
            **kwargs: Generation parameters
            
        Returns:
            Assistant's response
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})
        
        # Trim history if needed (keep system message + last N pairs)
        if len(self.messages) > (self.max_history * 2 + 1):
            self.messages = [self.messages[0]] + self.messages[-(self.max_history * 2) :]
        
        # Generate response
        response = self.engine.chat(self.messages, **kwargs)
        
        # Add assistant response to history
        self.messages.append({"role": "assistant", "content": response})
        
        return response
    
    def reset(self, system_message: Optional[str] = None):
        """
        Reset the conversation history.
        
        Args:
            system_message: New system message (uses existing if None)
        """
        if system_message:
            self.system_message = system_message
        self.messages = [{"role": "system", "content": self.system_message}]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.messages.copy()


def run_interactive_chat(
    base_model_id: str,
    adapter_path: Optional[str] = None,
    system_message: str = "You are a helpful assistant.",
    use_4bit: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs,
):
    """
    Run an interactive chat session in the terminal.
    
    Args:
        base_model_id: Base model ID
        adapter_path: Path to LoRA adapter (optional)
        system_message: System prompt
        use_4bit: Use 4-bit quantization
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        **kwargs: Additional generation parameters
    """
    print("\n" + "=" * 60)
    print("Interactive Chat Session")
    print("=" * 60)
    print("Commands: 'quit' to exit, 'reset' to clear history, 'history' to view conversation")
    print("=" * 60 + "\n")
    
    # Initialize inference engine
    engine = LoRAInference(base_model_id, adapter_path, use_4bit)
    
    # Create chat session
    session = InteractiveChatSession(engine, system_message)
    
    # Generation parameters
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        **kwargs,
    }
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break
            elif user_input.lower() == "reset":
                session.reset()
                print("🔄 Conversation history cleared.")
                continue
            elif user_input.lower() == "history":
                print("\n📜 Conversation History:")
                for msg in session.get_history():
                    role = msg["role"].capitalize()
                    content = (
                        msg["content"][:100] + "..."
                        if len(msg["content"]) > 100
                        else msg["content"]
                    )
                    print(f"  [{role}] {content}")
                continue
            
            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)
            response = session.send(user_input, **gen_kwargs)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue
