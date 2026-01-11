import os
import glob


def load_prompts():
    """
    Reads all .txt and .md files in the prompts directory and loads them into variables
    with the same name as the files without the file extension.

    Returns:
        dict: A dictionary where keys are variable names (filename without extension)
              and values are the file contents as strings.
    """
    # Get the directory where this file is located
    prompts_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all .txt and .md files in the prompts directory
    txt_files = glob.glob(os.path.join(prompts_dir, "*.txt"))
    md_files = glob.glob(os.path.join(prompts_dir, "*.md"))

    # Combine both file types
    all_files = txt_files + md_files

    prompts = {}

    for file_path in all_files:
        # Get the filename without path and extension
        filename = os.path.basename(file_path)
        variable_name = os.path.splitext(filename)[0]

        # Read the file content
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                prompts[variable_name] = content
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            prompts[variable_name] = None

    return prompts


# Explicit variable declarations for type checking
# These will be overwritten by the dynamic loading above
gorbit_system_prompt: str = ""

# Load all prompts when the module is imported
_loaded_prompts = load_prompts()

# Create module-level variables for each prompt
for _prompt_name, _prompt_content in _loaded_prompts.items():
    globals()[_prompt_name] = _prompt_content