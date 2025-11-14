"""
Dataset Generator module for creating fine-tuning datasets using multiple AI models.
Generates dialogue-based examples with Split Personality format.
"""

import os
import json
import random
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import DatasetConfig, ModelConfig
from .api_client import APIClient
from .utils import write_jsonl, ensure_dir, expand_topics

# Setup logger
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generates training datasets using configured AI models."""

    def __init__(self, config: DatasetConfig, models: List[ModelConfig]):
        """
        Initialize dataset generator.

        Args:
            config: Dataset generation configuration
            models: List of model configurations to use

        Raises:
            ValueError: If no models are provided
        """
        if not models:
            raise ValueError("At least one model must be configured")

        self.config = config
        self.models = models
        self.topics = expand_topics(config.base_topics, config.n_topic_variants)

        # Set random seed if specified
        if config.random_seed is not None:
            random.seed(config.random_seed)

        # Ensure output directory exists
        ensure_dir(config.output_dir)
        self.output_path = os.path.join(config.output_dir, "dataset.jsonl")

        print(f"Initialized DatasetGenerator:")
        print(f"  - Models: {len(self.models)}")
        print(f"  - Topics: {len(self.topics)}")
        print(f"  - Output: {self.output_path}")
        print(f"  - Format: {config.output_format_type}")

    def build_prompt(self, topic: str, seed: int, batch_size: int = 1) -> str:
        """
        Build generation prompt using configured template.

        Args:
            topic: Topic to generate content about
            seed: Random seed for variation
            batch_size: Number of examples to generate in this batch

        Returns:
            Formatted prompt string
        """
        logger.debug(
            f"Building prompt for topic: {topic[:50]}... (seed: {seed}, batch: {batch_size})"
        )

        if self.config.prompt_template:
            # Use custom template from config
            prompt = self.config.prompt_template.format(
                batch_size=batch_size, topic=topic, seed=seed
            )
            logger.debug(f"Using custom prompt template ({len(prompt)} chars)")
            return prompt
        else:
            # Default template (backwards compatibility)
            return f"""You are a data generator for a fine-tuning dataset called "Split Personality LoRA".
Generate {batch_size} diverse examples in JSONL format.

Each JSON object must contain:
{{
  "instruction": "<a concise question or task>",
  "input": "<optional context or empty string>",
  "output": "<dialogue between Analyst 🧠 and Creative 💡, ending with Consensus 🤝>"
}}

Follow this structure:
[Analyst 🧠]: ...
[Creative 💡]: ...
[Analyst 🧠]: ...
[Creative 💡]: ...
[Consensus 🤝]: ...

Topic focus for this batch: {topic}
Avoid any topic or wording repetition from previous batches.
Seed: {seed}

70% of examples have empty 'input'.
30% include a small context snippet or short paragraph.
Output only valid JSONL, no Markdown or code fences."""

    def parse_jsonl_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse JSONL response from model.

        Args:
            response: Raw response text from model

        Returns:
            List of parsed JSON objects
        """
        logger.debug(f"Parsing response ({len(response)} chars)")
        records = []

        # Clean response - remove markdown code blocks if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            logger.debug("Removing markdown code fences from response")
            # Remove opening ```jsonl or ```json
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        # Parse each line as JSON
        line_count = 0
        for line in cleaned.split("\n"):
            line = line.strip()
            if not line:
                continue

            line_count += 1
            try:
                obj = json.loads(line)
                # Validate required fields
                if all(key in obj for key in ["instruction", "input", "output"]):
                    records.append(obj)
                    logger.debug(
                        f"  ✓ Parsed record {len(records)}: instruction length={len(obj['instruction'])}, output length={len(obj['output'])}"
                    )
                else:
                    logger.warning(
                        f"  ⚠️  Skipping record without required fields: {list(obj.keys())}"
                    )
                    print(f"  ⚠️  Skipping record without required fields: {list(obj.keys())}")
            except json.JSONDecodeError as e:
                logger.warning(
                    f"  ⚠️  Failed to parse JSON line {line_count}: {line[:50]}... Error: {e}"
                )
                print(f"  ⚠️  Failed to parse JSON: {line[:50]}... Error: {e}")
                continue

        logger.info(f"Parsed {len(records)} valid records from {line_count} lines")
        return records

    def generate_batch(
        self, model_config: ModelConfig, topic: str, seed: int, batch_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate a batch of training examples.

        Args:
            model_config: Model configuration to use
            topic: Topic to generate content about
            seed: Random seed for this batch
            batch_size: Number of examples to generate

        Returns:
            List of generated examples with metadata

        Raises:
            RuntimeError: If generation fails
        """
        logger.info(f"Generating batch with {model_config.name}, topic: {topic[:40]}...")

        # Create client for this model
        client = APIClient(model_config)

        # Build prompt
        prompt = self.build_prompt(topic, seed, batch_size)

        # Build messages for API
        messages = [
            {
                "role": "system",
                "content": "You are a helpful dataset generator. Output only valid JSONL.",
            },
            {"role": "user", "content": prompt},
        ]

        # Apply slight temperature variation
        random.seed(seed)
        temperature = self.config.temperature + random.uniform(-0.1, 0.1)
        temperature = max(0.1, min(2.0, temperature))

        logger.debug(
            f"API parameters: temp={temperature:.2f}, top_p={self.config.top_p}, max_tokens={self.config.max_tokens * batch_size}"
        )

        # Generate response
        logger.info(f"Calling API {model_config.name}...")
        response = client.chat_complete(
            messages=messages,
            max_tokens=self.config.max_tokens * batch_size,  # Scale with batch size
            temperature=temperature,
            top_p=self.config.top_p,
        )
        logger.info(f"Received response from {model_config.name} ({len(response)} chars)")

        # Parse JSONL response
        records = self.parse_jsonl_response(response)

        # Add metadata to each record
        for record in records:
            record["meta"] = {
                "model_name": model_config.name,
                "model_id": model_config.model,
                "api_url": model_config.api_url,
                "topic": topic,
                "seed": seed,
                "temperature": temperature,
                "created_utc": datetime.utcnow().isoformat() + "Z",
            }

        return records

    def run_batch(self, total_examples: Optional[int] = None) -> int:
        """
        Generate multiple batches of training examples.

        Args:
            total_examples: Total number of examples to generate (uses config default if None)

        Returns:
            Number of examples successfully generated
        """
        if total_examples is None:
            total_examples = self.config.batch_size

        print(f"\nGenerating {total_examples} examples...")
        print(f"Output: {self.output_path}\n")

        produced = 0
        failed_batches = 0

        # Generate in small batches (1-3 examples per API call)
        examples_per_call = min(3, total_examples)

        while produced < total_examples:
            # Calculate how many examples to generate in this call
            remaining = total_examples - produced
            batch_size = min(examples_per_call, remaining)

            # Randomly select model and topic
            model_config = random.choice(self.models)
            topic = random.choice(self.topics)
            seed = random.randint(1, 10_000_000)

            try:
                # Generate batch
                records = self.generate_batch(model_config, topic, seed, batch_size)

                if records:
                    # Write to file immediately (resume-safe)
                    write_jsonl(self.output_path, records)

                    produced += len(records)

                    # Display progress
                    topic_display = topic[:50] + "..." if len(topic) > 50 else topic
                    print(
                        f"[{produced}/{total_examples}] ✓ Generated {len(records)} example(s) · {model_config.name} · {topic_display}"
                    )
                else:
                    print(f"[{produced}/{total_examples}] ⚠️  No valid examples generated")
                    failed_batches += 1

            except Exception as e:
                failed_batches += 1
                print(f"[{produced}/{total_examples}] ✗ Error: {str(e)[:120]}")

                # Stop if too many failures
                if failed_batches > total_examples * 0.5:
                    print(f"\n⚠️  Too many failures ({failed_batches}). Stopping.")
                    break

        print(f"\n✅ Generated {produced} examples")
        if failed_batches > 0:
            print(f"⚠️  {failed_batches} failed batches")
        print(f"📁 Output: {self.output_path}")

        return produced

    def validate_output(self, sample_size: int = 3) -> Dict[str, Any]:
        """
        Validate the generated dataset.

        Args:
            sample_size: Number of sample records to include

        Returns:
            Dictionary with validation results
        """
        from .utils import validate_jsonl

        if not os.path.exists(self.output_path):
            return {"exists": False, "error": "Output file not found"}

        total, invalid, samples = validate_jsonl(self.output_path, sample_size)

        # Additional validation for dialogue format
        dialogue_count = 0
        empty_input_count = 0

        with open(self.output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)

                    # Check if output contains dialogue markers
                    output = record.get("output", "")
                    if "[Analyst" in output and "[Creative" in output and "[Consensus" in output:
                        dialogue_count += 1

                    # Check if input is empty
                    if not record.get("input", "").strip():
                        empty_input_count += 1

                except:
                    pass

        valid_count = total - invalid

        return {
            "exists": True,
            "total_records": total,
            "invalid_records": invalid,
            "valid_records": valid_count,
            "dialogue_format_count": dialogue_count,
            "empty_input_count": empty_input_count,
            "empty_input_percentage": (
                round(empty_input_count / valid_count * 100, 1) if valid_count > 0 else 0
            ),
            "samples": samples,
        }


def clean_dataset(input_path: str, output_path: Optional[str] = None) -> int:
    """
    Clean a dataset by removing invalid entries and duplicates.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output file (defaults to input_path with '_cleaned' suffix)

    Returns:
        Number of valid records written
    """
    import json
    from .utils import read_jsonl, write_jsonl

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cleaned{ext}"

    print(f"Cleaning dataset: {input_path}")

    valid_records = []
    seen_outputs = set()

    try:
        records = read_jsonl(input_path)

        for record in records:
            # Check for required fields
            if not all(key in record for key in ["instruction", "output"]):
                continue

            # Check for duplicates (by output content)
            output = record.get("output", "")
            if output in seen_outputs:
                continue

            seen_outputs.add(output)
            valid_records.append(record)

        # Write cleaned dataset
        write_jsonl(output_path, valid_records, mode="w")

        print(f"✅ Cleaned {len(valid_records)} valid records")
        print(f"📁 Output: {output_path}")

        return len(valid_records)

    except Exception as e:
        print(f"❌ Error cleaning dataset: {e}")
        return 0
