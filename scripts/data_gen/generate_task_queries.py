import h5py
import os
import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse
from collections import defaultdict
from openai import OpenAI

from halo.util.misc import get_dataset_domain_name, get_task_name_from_hdf5_path, get_task_language_from_hdf5


def extract_json_from_response(response: str) -> dict:
    """
    Extract JSON from GPT response.
    
    Args:
        response: The response string from GPT
        
    Returns:
        Parsed JSON dictionary, or empty dict if no JSON found
    """
    json_part = re.search(r"\{.*\}", response, re.DOTALL)
    parsed_json = {}
    if json_part:
        json_data = json_part.group()
        try:
            parsed_json = json.loads(json_data)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {json_data[:200]}...")
    else:
        print("No JSON data found in response")
    return parsed_json


def generate_task_instructions_with_gpt(
    language: str,
    client: OpenAI,
    model: str = "gpt-4o",
    num_instructions: int = 20,
    max_retries: int = 3
) -> list:
    """
    Generate related task instructions using GPT for a given language instruction.
    
    Args:
        language: The original task language instruction
        client: OpenAI client
        model: Model name to use
        num_instructions: Number of related instructions to generate
        max_retries: Maximum number of retries if JSON parsing fails
        
    Returns:
        List of generated task instructions
    """
    prompt = f"""Given the following robot task instruction, generate {num_instructions} related task instructions that are similar in nature but with variations.
The variations should involve the same movement, subtasks, timeframe for the robot. It can simply be different paraphrases of the original instruction. Do NOT

Original task instruction: "{language}"

Requirements:
- Generate exactly {num_instructions} related task instructions
- Keep instructions SHORT and concise (not very long)
- Instructions should be related to the original task but with variations
- Each instruction should be a valid robot manipulation task
- Involve the same robot behavior, subtasks, and timeframe as the original instruction.
- Do not change the object or the goal of the task.
- Some changes that are encouraged are:
    - Paraphrasing
    - Using synonyms,
    - Different sentence structure
    - Changing voices
    - Synonymous instructions (e.g., "pick up the dinosaur" --> "pick up the extinct animal")
    - Using words instead of numerical values as well as other way around
- 

Return the response in JSON format with the following structure:
{{
    "instructions": [
        "instruction 1",
        "instruction 2",
        ...
        "instruction {num_instructions}"
    ]
}}

Important: Return ONLY valid JSON, no additional text or explanation."""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates robot task instructions in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            response_text = response.choices[0].message.content
            parsed_json = extract_json_from_response(response_text)
            
            if "instructions" in parsed_json and isinstance(parsed_json["instructions"], list):
                instructions = parsed_json["instructions"]
                if len(instructions) == num_instructions:
                    return instructions
                else:
                    print(f"Warning: Expected {num_instructions} instructions, got {len(instructions)}. Retrying...")
            else:
                print(f"Warning: Invalid JSON structure. Retrying... (attempt {attempt + 1}/{max_retries})")
                
        except Exception as e:
            print(f"Error calling GPT API (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                print(f"Failed to generate instructions for language: {language}")
                return []
    
    return []

def edit_hdf5_file(hdf5_path: str, episode_results: dict, update_flag: bool = False):
    """
    Edit the HDF5 file with the generated instructions.
    """
    mode = 'r' if not update_flag else 'r+'
    with h5py.File(hdf5_path, mode) as f:
        for episode_key, episode_data in episode_results.items():
            og_instructions: list[str] = episode_data["original_instructions"]
            generated_instructions: list[str] = episode_data["generated_instructions"]
            all_instructions: list[str] = og_instructions + generated_instructions
            # convert to json and save it to the attributes of the episode
            all_instructions_json: str = json.dumps(all_instructions)
            if update_flag:
                f['data'][episode_key].attrs['ep_languages'] = all_instructions_json
            else:
                print("="*100)
                print(f"{episode_key=}")
                print(all_instructions_json)

def save_results_to_json(results: dict, output_file: str):
    """
    Save the results to a json file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
    return output_file

def process_hdf5_file(
    hdf5_path: str,
    client: OpenAI,
    model: str,
    num_instructions: int,
    output_dir: str,
    edit_hdf5_file_flag: bool = False
) -> dict:
    """
    Process a single HDF5 file and generate task queries for each episode.
    
    Args:
        hdf5_path: Path to the HDF5 file
        client: OpenAI client
        model: GPT model name
        num_instructions: Number of instructions to generate per language
        output_dir: Directory to save JSON files
        
    Returns:
        Dictionary mapping episode keys to generated instructions
    """
    print(f"Processing HDF5 file: {hdf5_path}")
    
    # Extract languages from HDF5
    task_languages = get_task_language_from_hdf5(hdf5_path)
    
    # Group episodes by language to avoid duplicate GPT calls
    language_to_episodes = defaultdict(list)
    for episode_key, languages in task_languages.items():
        for lang in languages:
            language_to_episodes[lang].append(episode_key)
    
    print(f"Found {len(language_to_episodes)} unique languages across {len(task_languages)} episodes")
    
    # Generate instructions for each unique language
    language_to_instructions = {}
    for language, episode_keys in tqdm(language_to_episodes.items(), desc="Generating instructions"):
        print(f"\nGenerating {num_instructions} instructions for language: '{language}'")
        print(f"  Affects {len(episode_keys)} episodes")
        n_retries = 3
        for retry in range(n_retries):
            instructions = generate_task_instructions_with_gpt(
                language, client, model, num_instructions
            )
            if instructions:
                language_to_instructions[language] = instructions
                print(f"  Successfully generated {len(instructions)} instructions")
                break
            else:
                print(f"  Failed to generate instructions")
    
    # Create output structure: one JSON per episode
    episode_results = {}
    
    for episode_key, languages in task_languages.items():
        episode_data = {
            "hdf5_file": hdf5_path,
            "episode_key": episode_key,
            "original_instructions": languages,
            "generated_instructions": []
        }
        
        # Get generated instructions for this episode's language
        for lang in languages:
            if lang in language_to_instructions:
                episode_data["generated_instructions"].extend(
                    language_to_instructions[lang]
                )
        
        # Remove duplicates while preserving order
        episode_results[episode_key] = episode_data

    edit_hdf5_file(hdf5_path, episode_results, update_flag=edit_hdf5_file_flag)
    filename = "all_queries_v1.json"
    filepath = os.path.join(output_dir, filename)
    save_results_to_json(episode_results, filepath)
    return episode_results

def process_robocasa_benchmark(
    args,
    client: OpenAI
):
    """Process RoboCasa benchmark dataset."""
    import robocasa.macros as macros
    dataset_base_path = macros.DATASET_BASE_PATH
    if dataset_base_path is None:
        dataset_base_path = os.environ["CASAPLAY_DATAROOT"]
    
    all_hdf5_files = list(Path(dataset_base_path).glob(f"{args.benchmark}/*/*/*/demo_im128_notp.hdf5"))
    if len(all_hdf5_files) == 0:
        all_hdf5_files = list(Path(dataset_base_path).glob(f"{args.benchmark}/*/*/demo_im128_notp.hdf5"))
    
    assert len(all_hdf5_files) > 0, f"No HDF5 files found in {dataset_base_path}/{args.benchmark}"
    
    print(f"Found {len(all_hdf5_files)} HDF5 files")
    
    
    all_results = {}
    for hdf5_file in tqdm(all_hdf5_files, desc="Processing HDF5 files"):
        output_dir = args.output_dir or os.path.join(os.path.dirname(hdf5_file), "task_queries")
        results = process_hdf5_file(
            str(hdf5_file),
            client,
            args.model,
            args.num_instructions,
            output_dir=output_dir,
        )
        all_results[str(hdf5_file)] = results
    

def process_mutex_benchmark(
    args,
    client: OpenAI
):
    """Process MUTEX benchmark dataset."""
    dataset_base_path = os.environ["CASAPLAY_DATAROOT"]
    all_hdf5_files = list(Path(dataset_base_path).glob(f"{args.benchmark}/*.hdf5"))

    assert len(all_hdf5_files) > 0, f"No HDF5 files found in {dataset_base_path}/{args.benchmark}"

    print(f"Found {len(all_hdf5_files)} HDF5 files")


    all_results = {}
    for hdf5_file in tqdm(all_hdf5_files, desc="Processing HDF5 files"):
        output_dir = args.output_dir or os.path.join(os.path.dirname(hdf5_file), "task_queries")
        results = process_hdf5_file(
            str(hdf5_file),
            client,
            args.model,
            args.num_instructions,
            output_dir=output_dir,
        )
        all_results[str(hdf5_file)] = results


def main():
    parser = argparse.ArgumentParser(
        description="Generate related task instructions using GPT for each episode in HDF5 files"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="memory",
        choices=[
            "v0.1/single_stage", "memory", "v0.1/multi_stage", "mutex/RW8", "mutex/MEM1",
        ],
        help="Benchmark to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (e.g., 'gpt-4o', 'gpt-4-turbo')"
    )
    parser.add_argument(
        "--num_instructions",
        type=int,
        default=20,
        help="Number of related task instructions to generate per language"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for JSON files (default: benchmark_path/task_queries)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (default: from OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--hdf5_file",
        type=str,
        default=None,
        help="Process a single HDF5 file instead of a benchmark"
    )
    parser.add_argument(
        "-efile",
        "--edit_hdf5_file_flag",
        action='store_true',
        default=False,
        help="Edit the HDF5 file with the generated instructions"
    )
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError(
            "OpenAI API key must be provided either as --api_key argument "
            "or OPENAI_API_KEY environment variable"
        )
    
    client = OpenAI(api_key=api_key)
    
    # Process single file or benchmark
    if args.hdf5_file:
        if not os.path.exists(args.hdf5_file):
            print(f"Error: HDF5 file {args.hdf5_file} does not exist")
            return
        
        output_dir = args.output_dir or os.path.join(os.path.dirname(args.hdf5_file), "task_queries")
        process_hdf5_file(
            args.hdf5_file,
            client,
            args.model,
            args.num_instructions,
            output_dir=output_dir,
            edit_hdf5_file_flag=args.edit_hdf5_file_flag
        )
    else:
        # Process benchmark
        domain_name = get_dataset_domain_name(args.benchmark)

        if domain_name == 'robocasa':
            process_robocasa_benchmark(args, client)
        elif 'mutex' in args.benchmark:
            process_mutex_benchmark(args, client)
        else:
            raise ValueError(f"Unknown benchmark: {args.benchmark}")


if __name__ == "__main__":
    main()
