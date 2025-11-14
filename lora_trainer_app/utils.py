"""
Utility functions for dataset generation and file operations.
"""

import os
import json
import glob
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


def find_latest_jsonl(base_dir: str) -> Optional[str]:
    """
    Find the most recently modified JSONL file in a directory tree.

    Args:
        base_dir: Base directory to search

    Returns:
        Path to the most recent JSONL file, or None if not found
    """
    candidates = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".jsonl"):
                candidates.append(os.path.join(root, file))

    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def write_jsonl(file_path: str, records: List[Dict[str, Any]], mode: str = "a"):
    """
    Write records to a JSONL file.

    Args:
        file_path: Path to the output file
        records: List of dictionaries to write
        mode: File mode ('a' for append, 'w' for write)
    """
    with open(file_path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read all records from a JSONL file.

    Args:
        file_path: Path to the input file

    Returns:
        List of dictionaries
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_jsonl(file_path: str, sample_size: int = 3) -> tuple[int, int, List[Dict[str, Any]]]:
    """
    Validate a JSONL file and return statistics.

    Args:
        file_path: Path to the file to validate
        sample_size: Number of sample records to return

    Returns:
        Tuple of (total_lines, invalid_lines, sample_records)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    total_lines = 0
    invalid_lines = 0
    samples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            try:
                obj = json.loads(line)
                if len(samples) < sample_size:
                    samples.append(obj)
            except json.JSONDecodeError:
                invalid_lines += 1

    return total_lines, invalid_lines, samples


def expand_topics(base_topics: Dict[str, List[str]], n_extras: int = 20) -> List[str]:
    """
    Expand base topics with generated variants.

    Args:
        base_topics: Dictionary of topic categories
        n_extras: Number of additional variants to generate

    Returns:
        List of all topics (base + generated)
    """
    variants = []
    templates = [
        "compare {a} with {b} for {goal}",
        "design a minimal system for {a} under constraints: {c1}, {c2}",
        "explain {a} using a metaphor from {b}",
        "pitfalls when scaling {a}; propose mitigations",
        "write a debate between two experts about {a}",
        "create a checklist to implement {a} in production",
    ]

    goals = [
        "latency",
        "cost efficiency",
        "resilience",
        "maintainability",
        "security",
        "developer velocity",
    ]

    metaphor_domains = [
        "cooking",
        "orchestra",
        "urban planning",
        "gardening",
        "sports tactics",
        "architecture",
    ]

    constraints = [
        "limited GPU",
        "no internet",
        "strict change windows",
        "hard SLOs",
        "edge nodes",
        "legacy dependencies",
    ]

    # Flatten all base topics
    pool = [topic for category in base_topics.values() for topic in category]

    # Generate variants
    for _ in range(n_extras):
        a = random.choice(pool)
        b = random.choice(metaphor_domains + pool)
        template = random.choice(templates)
        variant = template.format(
            a=a,
            b=b,
            goal=random.choice(goals),
            c1=random.choice(constraints),
            c2=random.choice(constraints),
        )
        variants.append(variant)

    # Combine and deduplicate while preserving order
    all_topics = []
    seen = set()
    for topic in pool + variants:
        if topic not in seen:
            seen.add(topic)
            all_topics.append(topic)

    return all_topics


def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory

    Returns:
        The directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def count_lines(file_path: str) -> int:
    """
    Count the number of lines in a file.

    Args:
        file_path: Path to the file

    Returns:
        Number of lines
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def print_progress(current: int, total: int, prefix: str = "", suffix: str = "", width: int = 50):
    """
    Print a progress bar.

    Args:
        current: Current progress value
        total: Total value
        prefix: Prefix string
        suffix: Suffix string
        width: Width of the progress bar in characters
    """
    percent = 100 * (current / float(total))
    filled_width = int(width * current // total)
    bar = "█" * filled_width + "-" * (width - filled_width)
    print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="", flush=True)
    if current == total:
        print()  # New line when complete


def is_gguf_file(model_path: str) -> bool:
    """
    Check if a path points to a GGUF file.
    
    Args:
        model_path: Path to check
        
    Returns:
        True if the path is a GGUF file
    """
    path = Path(model_path)
    return path.is_file() and path.suffix.lower() == ".gguf"


def read_gguf_metadata(gguf_path: str) -> Dict[str, Any]:
    """
    Read metadata from a GGUF file to identify the base model.
    
    Args:
        gguf_path: Path to the GGUF file
        
    Returns:
        Dictionary with metadata (name, architecture, etc.)
    """
    try:
        import gguf
        
        reader = gguf.GGUFReader(gguf_path)
        metadata = {}
        
        # Extract common metadata keys
        metadata_keys = [
            "general.name",
            "general.architecture",
            "general.description",
            "general.source.url",
            "general.source.repo_url",
            "general.repo_url",
            "general.basename",
        ]
        
        for key in metadata_keys:
            try:
                field = reader.get_field(key)
                if field is not None:
                    value = field.contents()
                    if value is not None:
                        # Convert key to simple name (remove "general." prefix)
                        simple_key = key.replace("general.", "").replace(".", "_")
                        metadata[simple_key] = value
            except (KeyError, AttributeError, Exception):
                pass
        
        # Also try to get architecture directly
        try:
            arch_field = reader.get_field("general.architecture")
            if arch_field is not None:
                arch = arch_field.contents()
                if arch:
                    metadata["architecture"] = arch
        except (KeyError, AttributeError, Exception):
            pass
        
        return metadata
    
    except ImportError:
        raise ImportError(
            "gguf library is required to read GGUF files. "
            "Install it with: pip install gguf"
        )
    except Exception as e:
        raise ValueError(f"Failed to read GGUF metadata from {gguf_path}: {e}")


def resolve_model_from_gguf(gguf_path: str) -> Tuple[str, Dict[str, Any]]:
    """
    Resolve a Hugging Face model ID from a GGUF file by reading its metadata.
    
    This function attempts to identify the original Hugging Face model that was
    converted to GGUF by reading metadata from the GGUF file.
    
    Args:
        gguf_path: Path to the GGUF file
        
    Returns:
        Tuple of (model_id, metadata_dict)
        
    Raises:
        ValueError: If cannot resolve the model ID from GGUF metadata
    """
    metadata = read_gguf_metadata(gguf_path)
    
    # Try to extract model ID from various metadata fields
    model_id = None
    
    # Strategy 1: Check general.repo_url or general.source.repo_url
    repo_url = metadata.get("repo_url") or metadata.get("source_repo_url")
    if repo_url:
        # Extract repo ID from Hugging Face URL
        # Example: https://huggingface.co/user/model-name -> user/model-name
        if "huggingface.co" in repo_url:
            parts = repo_url.split("huggingface.co/")
            if len(parts) > 1:
                repo_id = parts[1].split("/")[0:2]  # Take user and model name
                if len(repo_id) == 2:
                    model_id = "/".join(repo_id)
    
    # Strategy 2: Use general.name or general.basename
    if not model_id:
        name = metadata.get("name") or metadata.get("basename")
        if name:
            # Try common Hugging Face model ID patterns
            # Remove common suffixes like "-GGUF", "-Q4_K_M", etc.
            clean_name = name.replace("-GGUF", "").replace("-gguf", "")
            # Remove quantization suffixes
            for suffix in ["-Q4", "-Q5", "-Q6", "-Q8", "-F16", "-F32"]:
                if clean_name.endswith(suffix):
                    clean_name = clean_name.rsplit("-", 1)[0]
            
            # If it contains a slash, it might already be a model ID
            if "/" in clean_name:
                model_id = clean_name
            else:
                # Try common organizations
                # This is a heuristic - might not always work
                common_orgs = ["microsoft", "meta-llama", "mistralai", "google", "Qwen"]
                for org in common_orgs:
                    if org.lower() in clean_name.lower():
                        model_id = f"{org}/{clean_name}"
                        break
    
    if not model_id:
        # Strategy 3: Use architecture + name to guess
        architecture = metadata.get("architecture", "")
        name = metadata.get("name", "") or metadata.get("basename", "")
        
        if architecture and name:
            # Common mappings
            arch_to_org = {
                "llama": "meta-llama",
                "mistral": "mistralai",
                "gemma": "google",
                "qwen": "Qwen",
            }
            
            for arch_key, org in arch_to_org.items():
                if arch_key.lower() in architecture.lower():
                    clean_name = name.replace("-GGUF", "").replace("-gguf", "")
                    # Remove quantization suffixes
                    for suffix in ["-Q4", "-Q5", "-Q6", "-Q8", "-F16", "-F32"]:
                        if clean_name.endswith(suffix):
                            clean_name = clean_name.rsplit("-", 1)[0]
                    model_id = f"{org}/{clean_name}"
                    break
    
    if not model_id:
        raise ValueError(
            f"Cannot resolve Hugging Face model ID from GGUF metadata. "
            f"Found metadata: {metadata}. "
            f"Please specify the original Hugging Face model ID manually."
        )
    
    return model_id, metadata
