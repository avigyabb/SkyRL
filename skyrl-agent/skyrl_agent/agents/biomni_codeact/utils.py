from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.utils.interactive_env import is_interactive_env
from langchain_core.messages.base import get_msg_title_repr
from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import StructuredTool
from pandas import DataFrame
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Literal, Union, Set
import importlib
import json
import pickle
import ast
import enum
import requests
import os
import pandas as pd
import pandas  # Add direct import for pandas

# Add these new functions for running R code and CLI commands
def run_r_code(code: str) -> str:
    """
    Run R code using subprocess.
    
    Args:
        code: R code to run
        
    Returns:
        Output of the R code
    """
    try:
        # Create a temporary file to store the R code
        with tempfile.NamedTemporaryFile(suffix='.R', mode='w', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Run the R code using Rscript
        result = subprocess.run(['Rscript', temp_file], 
                               capture_output=True, 
                               text=True, 
                               check=False)
        
        # Clean up the temporary file
        os.unlink(temp_file)
        
        # Return the output
        if result.returncode != 0:
            return f"Error running R code:\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        return f"Error running R code: {str(e)}"

def run_bash_script(script: str) -> str:
    """
    Run a Bash script using subprocess.
    
    Args:
        script: Bash script to run
        
    Returns:
        Output of the Bash script
        
    Example:
        ```
        # Example of a complex Bash script
        script = '''
        #!/bin/bash
        
        # Define variables
        DATA_DIR="/path/to/data"
        OUTPUT_FILE="results.txt"
        
        # Create output directory if it doesn't exist
        mkdir -p $(dirname $OUTPUT_FILE)
        
        # Loop through files
        for file in $DATA_DIR/*.txt; do
            echo "Processing $file..."
            # Count lines in each file
            line_count=$(wc -l < $file)
            echo "$file: $line_count lines" >> $OUTPUT_FILE
        done
        
        echo "Processing complete. Results saved to $OUTPUT_FILE"
        '''
        result = run_bash_script(script)
        print(result)
        ```
    """
    try:
        # Trim any leading/trailing whitespace
        script = script.strip()
        
        # If the script is empty, return an error
        if not script:
            return "Error: Empty script"
        
        # Create a temporary file to store the Bash script
        with tempfile.NamedTemporaryFile(suffix='.sh', mode='w', delete=False) as f:
            # Add shebang if not present
            if not script.startswith("#!/"):
                f.write("#!/bin/bash\n")
            # Add set -e to exit on error
            if not "set -e" in script:
                f.write("set -e\n")
            f.write(script)
            temp_file = f.name
        
        # Make the script executable
        os.chmod(temp_file, 0o755)
        
        # Get current environment variables and working directory
        env = os.environ.copy()
        cwd = os.getcwd()
        
        # Run the Bash script with the current environment and working directory
        result = subprocess.run([temp_file], 
                               shell=True,
                               capture_output=True, 
                               text=True, 
                               check=False,
                               env=env,
                               cwd=cwd)
        
        # Clean up the temporary file
        os.unlink(temp_file)
        
        # Return the output
        if result.returncode != 0:
            return f"Error running Bash script (exit code {result.returncode}):\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        return f"Error running Bash script: {str(e)}"

# Keep the run_cli_command for backward compatibility
def run_cli_command(command: str) -> str:
    """
    Run a CLI command using subprocess.
    
    Args:
        command: CLI command to run
        
    Returns:
        Output of the CLI command
    """
    try:
        # Trim any leading/trailing whitespace
        command = command.strip()
        
        # If the command is empty, return an error
        if not command:
            return "Error: Empty command"
            
        # Split the command into a list of arguments, handling quoted arguments correctly
        import shlex
        args = shlex.split(command)
        
        # Run the command
        result = subprocess.run(args, 
                               capture_output=True, 
                               text=True, 
                               check=False)
        
        # Return the output
        if result.returncode != 0:
            return f"Error running command '{command}':\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        return f"Error running command '{command}': {str(e)}"

def run_with_timeout(func, args=None, kwargs=None, timeout=600):
    """
    Run a function with a timeout using threading instead of multiprocessing.
    This allows variables to persist in the global namespace between function calls.
    Returns the function result or a timeout error message.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    
    import threading
    import queue
    import ctypes
    
    result_queue = queue.Queue()
    
    def thread_func(func, args, kwargs, result_queue):
        """Function to run in a separate thread"""
        try:
            result = func(*args, **kwargs)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", str(e)))
    
    # Start a separate thread
    thread = threading.Thread(
        target=thread_func,
        args=(func, args, kwargs, result_queue)
    )
    thread.daemon = True  # Set as daemon so it will be killed when main thread exits
    thread.start()
    
    # Wait for the specified timeout
    thread.join(timeout)
    
    # Check if the thread is still running after timeout
    if thread.is_alive():
        print(f"TIMEOUT: Code execution timed out after {timeout} seconds")
        
        # Unfortunately, there's no clean way to force terminate a thread in Python
        # The recommended approach is to use daemon threads and let them be killed when main thread exits
        # Here, we'll try to raise an exception in the thread to make it stop
        try:
            # Get thread ID and try to terminate it
            thread_id = thread.ident
            if thread_id:
                # This is a bit dangerous and not 100% reliable
                # It attempts to raise a SystemExit exception in the thread
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(thread_id),
                    ctypes.py_object(SystemExit)
                )
                if res > 1:
                    # Oops, we raised too many exceptions
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        except Exception as e:
            print(f"Error trying to terminate thread: {e}")
            
        return f"ERROR: Code execution timed out after {timeout} seconds. Please try with simpler inputs or break your task into smaller steps."
    
    # Get the result from the queue if available
    try:
        status, result = result_queue.get(block=False)
        return result if status == "success" else f"Error in execution: {result}"
    except queue.Empty:
        return "Error: Execution completed but no result was returned"


class api_schema(BaseModel):
    """api schema specification"""
    api_schema: Optional[str] = Field(description="The api schema as a dictionary")

def function_to_api_schema(function_string, llm):
    prompt = """
    Based on a code snippet and help me write an API docstring in the format like this:

    {{'name': 'get_gene_set_enrichment',
    'description': 'Given a list of genes, identify a pathway that is enriched for this gene set. Return a list of pathway name, p-value, z-scores.',
    'required_parameters': [{{'name': 'genes',
    'type': 'List[str]',
    'description': 'List of g`ene symbols to analyze',
    'default': None}}],
    'optional_parameters': [{{'name': 'top_k',
    'type': 'int',
    'description': 'Top K pathways to return',
    'default': 10}},  {{'name': 'database',
    'type': 'str',
    'description': 'Name of the database to use for enrichment analysis',
    'default': "gene_ontology"}}]}}

    Strictly follow the input from the function - don't create fake optional parameters.
    For variable without default values, set them as None, not null.
    For variable with boolean values, use capitalized True or False, not true or false.
    Do not add any return type in the docstring.
    Be as clear and succint as possible for the descriptions. Please do not make it overly verbose.
    Here is the code snippet: 
    {code}
    """
    llm = llm.with_structured_output(api_schema)

    for _ in range(7):
        try:
            api = llm.invoke(prompt.format(code=function_string)).dict()['api_schema']
            return ast.literal_eval(api) # -> prefer "default": None 
            # return json.loads(api) # -> prefer "default": null 
        except Exception as e:
            print("API string:", api)
            print("Error parsing the API string:", e)
            continue

    return "Error: Could not parse the API schema"
    # return 

def get_all_functions_from_file(file_path):
    with open(file_path, "r") as file:
        file_content = file.read()

    # Parse the file content into an AST (Abstract Syntax Tree)
    tree = ast.parse(file_content)

    # List to hold the top-level functions as strings
    functions = []

    # Walk through the AST nodes
    for node in tree.body:  # Only consider top-level nodes in the body
        if isinstance(node, ast.FunctionDef):  # Check if the node is a function definition
            
            # Skip if function name starts with underscore
            if node.name.startswith('_'):
                continue
            
            start_line = node.lineno - 1  # Get the starting line of the function
            end_line = node.end_lineno  # Get the ending line of the function (only available in Python 3.8+)
            func_code = file_content.splitlines()[start_line:end_line]
            functions.append("\n".join(func_code))  # Join lines of the function and add to the list

    return functions

def write_python_code(request: str):
    from langchain_anthropic import ChatAnthropic
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    model = ChatAnthropic(model='claude-3-5-sonnet-20240620')
    template = """Write some python code to solve the user's problem. 

    Return only python code in Markdown format, e.g.:

    ```python
    ....
    ```"""
    prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])
    def _sanitize_output(text: str):
        _, after = text.split("```python")
        return after.split("```")[0]
    chain = prompt | model | StrOutputParser() | _sanitize_output
    return chain.invoke({"input": "write a code that " + request})


def execute_graphql_query(query: str, variables: dict, api_address: str = "https://api.genetics.opentargets.org/graphql") -> dict:
    """Executes a GraphQL query with variables and returns the data as a dictionary."""
    headers = {'Content-Type': 'application/json'}
    response = requests.post(api_address, json={'query': query, 'variables': variables}, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(response.text) 
        response.raise_for_status()


def get_tool_decorated_functions(relative_path):
    import ast, os
    import importlib.util
    # Get the directory of the current file (__init__.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the absolute path from the relative path
    file_path = os.path.join(current_dir, relative_path)
    
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    tool_function_names = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == 'tool':
                    tool_function_names.append(node.name)
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == 'tool':
                    tool_function_names.append(node.name)
    
    # Calculate the module name from the relative path
    package_path = os.path.relpath(file_path, start=current_dir)
    module_name = package_path.replace(os.path.sep, '.').rsplit('.', 1)[0]
    
    # Import the module and get the function objects
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    tool_functions = [getattr(module, name) for name in tool_function_names]
    
    return tool_functions

def process_bio_retrieval_ducoment(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in documents_df.itertuples():
        doc = row.document_content
        ir_corpus[row.docid] = (doc.get('name', '') or '') + ', ' + \
        (doc.get('description', '') or '') + ', ' + \
        (doc.get('url', '') or '') + ', ' + \
        ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
        ', optional_params: ' + json.dumps(doc.get('optional_parameters', '')) 

        corpus2tool[(doc.get('name', '') or '') + ', ' + \
        (doc.get('description', '') or '') + ', ' + \
        (doc.get('url', '') or '') + ', ' + \
        ', required_params: ' + json.dumps(doc.get('required_parameters', '')) + \
        ', optional_params: ' + json.dumps(doc.get('optional_parameters', ''))] = doc['name']
    return ir_corpus, corpus2tool


def load_pickle(file):
    import pickle
    with open(file, 'rb') as f:
        return pickle.load(f)


def pretty_print(message, printout = True):
    if isinstance(message, tuple):
        title = message
    else:
        if isinstance(message, list):
            title = json.dumps(message, indent=2)
            print(title)
        elif isinstance(message.content, list):
            title = get_msg_title_repr(message.type.title().upper() + " Message", bold=is_interactive_env())
            if message.name is not None:
                title += f"\nName: {message.name}"

            for i in message.content:
                if i['type'] == 'text':
                    title += f"\n{i['text']}\n"
                elif i['type'] == 'tool_use':
                    title += f"\nTool: {i['name']}"
                    title += f"\nInput: {i['input']}"
                elif i['type'] == 'thinking':
                    title += f"\nThinking: {i['thinking']}"
            if printout:
                print(f"{title}")
        else:
            title = get_msg_title_repr(message.type.title() + " Message", bold=is_interactive_env())
            if message.name is not None:
                title += f"\nName: {message.name}"
            title += f"\n\n{message.content}"
            if printout:
                print(f"{title}")
    return title

class CustomBaseModel(BaseModel):
    api_schema: ClassVar[dict] = None  # Class variable to store api_schema
    
    # Add model_config with arbitrary_types_allowed=True
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def set_api_schema(cls, schema: dict):
        cls.api_schema = schema

    @classmethod
    def model_validate(cls, obj):
        try:
            return super().model_validate(obj)
        except (ValidationError, AttributeError) as e:
            if not cls.api_schema:
                raise e  # If no api_schema is set, raise original error

            error_msg = "Required Parameters:\n"
            for param in cls.api_schema['required_parameters']:
                error_msg += f"- {param['name']} ({param['type']}): {param['description']}\n"
            
            error_msg += "\nErrors:\n"
            for err in e.errors():
                field = err["loc"][0] if err["loc"] else "input"
                error_msg += f"- {field}: {err['msg']}\n"
            
            if not obj:
                error_msg += "\nNo input provided"
            else:
                error_msg += "\nProvided Input:\n"
                for key, value in obj.items():
                    error_msg += f"- {key}: {value}\n"
                
                missing_params = set(param['name'] for param in cls.api_schema['required_parameters']) - set(obj.keys())
                if missing_params:
                    error_msg += "\nMissing Parameters:\n"
                    for param in missing_params:
                        error_msg += f"- {param}\n"
            
            # # Create proper validation error structure
            raise ValidationError.from_exception_data(
                title="Validation Error",
                line_errors=[{
                    'type': 'value_error',
                    'loc': ('input',),
                    'input': obj,
                    "ctx": {
                        "error": error_msg,
                    },
                }]
            )

def safe_execute_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return str(e)
    return wrapper

def api_schema_to_langchain_tool(api_schema, mode = 'generated_tool', module_name = None):
    if mode == 'generated_tool':
        module = importlib.import_module('biomni.tool.generated_tool.' + api_schema['tool_name'] + '.api')
    elif mode == 'custom_tool':
        module = importlib.import_module(module_name)

    api_function = getattr(module, api_schema['name'])
    api_function = safe_execute_decorator(api_function)

    # Define a mapping from string type names to actual Python type objects
    type_mapping = {
        'string': str,
        'integer': int,
        'boolean': bool,
        'pandas': pd.DataFrame,  # Use the imported pandas.DataFrame directly
        'str': str,
        'int': int,
        'bool': bool,
        'List[str]': List[str],
        'List[int]': List[int],
        'Dict': Dict,
        'Any': Any
    }

    # Create the fields and annotations
    annotations = {}
    for param in api_schema['required_parameters']:
        param_type = param['type']
        if param_type in type_mapping:
            annotations[param['name']] = type_mapping[param_type]
        else:
            # For types not in the mapping, try a safer approach than direct eval
            try:
                annotations[param['name']] = eval(param_type)
            except (NameError, SyntaxError):
                # Default to Any for unknown types
                annotations[param['name']] = Any

    fields = {
        param['name']: Field(description=param['description'])
        for param in api_schema['required_parameters']
    }

    # Create the ApiInput class dynamically
    ApiInput = type(
        "Input",
        (CustomBaseModel,),
        {
            '__annotations__': annotations,
            **fields
        }
    )
    # Set the api_schema
    ApiInput.set_api_schema(api_schema)

    # Create the StructuredTool
    api_tool = StructuredTool.from_function(
        func=api_function,
        name=api_schema['name'],
        description=api_schema['description'],
        args_schema=ApiInput,
        return_direct=True 
    )

    return api_tool

class ID(enum.Enum):
    ENTREZ = "Entrez"
    ENSEMBL = "Ensembl without version" # e.g. ENSG00000123374
    ENSEMBL_W_VERSION = "Ensembl with version" # e.g. ENSG00000123374.10 (needed for GTEx)


def get_gene_id(gene_symbol: str, id_type: ID):
    '''
    Get the ID for a gene symbol. If no match found, returns None
    '''
    if id_type == ID.ENTREZ:
        return _get_gene_id_entrez(gene_symbol)
    elif id_type == ID.ENSEMBL:
        return _get_gene_id_ensembl(gene_symbol)
    elif id_type == ID.ENSEMBL_W_VERSION:
        return _get_gene_id_ensembl_with_version(gene_symbol)
    else:
        raise ValueError(f"ID type {id_type} not supported")


def _get_gene_id_entrez(gene_symbol: str):
    '''
    Get the Entrez ID for a gene symbol. If no match found, returns None
    e.g. 1017 (CDK2)
    '''
    api_call = f'https://mygene.info/v3/query?species=human&q=symbol:{gene_symbol}'
    response = requests.get(api_call)
    response_json = response.json()

    if len(response_json["hits"]) == 0:
        return None
    else:
        return response_json["hits"][0]["entrezgene"]

def _get_gene_id_ensembl(gene_symbol):
    '''
    Get the Ensembl ID for a gene symbol. If no match found, returns None
    e.g. ENSG00000123374
    '''
    api_call = f'https://mygene.info/v3/query?species=human&fields=ensembl&q=symbol:{gene_symbol}'
    response = requests.get(api_call)
    response_json = response.json()

    if len(response_json["hits"]) == 0:
        return None
    else:
        ensembl = response_json["hits"][0]["ensembl"]
        if isinstance(ensembl, list):
            return ensembl[0]["gene"] # Sometimes returns a list, for example RNH1 (first elem is on chr11, second is on scaffold_hschr11)
        else:
            return ensembl["gene"]
    
def _get_gene_id_ensembl_with_version(gene_symbol):
    '''
    Get the Ensembl ID for a gene symbol. If no match found, returns None
    e.g. ENSG00000123374.10
    '''
    api_base = f"https://gtexportal.org/api/v2/reference/gene"
    params = {
        "geneId": gene_symbol
    }
    response_json = requests.get(api_base, params=params).json()

    if len(response_json["data"]) == 0:
        return None
    else:
        return response_json["data"][0]["gencodeId"]


def save_pkl(f, filename):
    with open(filename, 'wb') as file:
        pickle.dump(f, file)

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

_TEXT_COLOR_MAPPING = {
    "blue": "36;1",
    "yellow": "33;1",
    "pink": "38;5;200",
    "green": "32;1",
    "red": "31;1",
}

def color_print(text, color="blue"):
    color_str = _TEXT_COLOR_MAPPING[color]
    print(f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m")


class PromptLogger(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        for message in messages[0]:
            color_print(message.pretty_repr(), color="green")

class NodeLogger(BaseCallbackHandler):

    def on_llm_end(self, response, **kwargs): # response of type LLMResult
        for generations in response.generations: # response.generations of type List[List[Generations]] becuase "each input could have multiple candidate generations"
            for generation in generations:
                generated_text = generation.message.content
                #token_usage = generation.message.response_metadata["token_usage"]
                color_print(generated_text, color="yellow")

    def on_agent_action(self, action, **kwargs):
        color_print(action.log, color="pink")

    def on_agent_finish(self, finish, **kwargs):
        color_print(finish, color="red")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name")
        color_print(f"Calling {tool_name} with inputs: {input_str}", color="pink")

    def on_tool_end(self, output, **kwargs):
        output = str(output)
        color_print(output, color="blue")

def check_or_create_path(path=None):
    # Set a default path if none is provided
    if path is None:
        path = os.path.join(os.getcwd(), 'tmp_directory')
    
    # Check if the path exists
    if not os.path.exists(path):
        # If it doesn't exist, create the directory
        os.makedirs(path)
        print(f"Directory created at: {path}")
    else:
        print(f"Directory already exists at: {path}")

    return path


def langchain_to_gradio_message(message):
    
    # Build the title and content based on the message type
    if isinstance(message.content, list):
        # For a message with multiple content items (like text and tool use)
        gradio_messages = []
        for item in message.content:
            gradio_message = {
                "role": "user" if message.type == "human" else "assistant",
                "content": "",
                "metadata": {}
                }

            if item['type'] == 'text':
                item['text'] = item['text'].replace('<think>', '\n')
                item['text'] = item['text'].replace('</think>', '\n')
                gradio_message["content"] += f"{item['text']}\n"
                gradio_messages.append(gradio_message)
            elif item['type'] == 'tool_use':
                if item['name'] == 'run_python_repl':
                    gradio_message["metadata"]["title"] = f"🛠️ Writing code..."
                    #input = "```python {code_block}```\n".format(code_block=item['input']["command"])
                    gradio_message["metadata"]['log'] = f"Executing Code block..."
                    gradio_message['content'] = f"##### Code: \n ```python \n {item['input']['command']} \n``` \n"
                else:
                    gradio_message["metadata"]["title"] = f"🛠️ Used tool ```{item['name']}```"
                    to_print = ';'.join([i + ': ' + str(j) for i,j in item['input'].items()])
                    gradio_message["metadata"]['log'] = f"🔍 Input -- {to_print}\n"
                gradio_message["metadata"]['status'] = "pending"
                gradio_messages.append(gradio_message)

    else:
        gradio_message = {
        "role": "user" if message.type == "human" else "assistant",
        "content": "",
        "metadata": {}
        }
        print(message)
        content = message.content
        content = content.replace('<think>', '\n')
        content = content.replace('</think>', '\n')
        content = content.replace('<solution>', '\n')
        content = content.replace('</solution>', '\n')
        
        gradio_message["content"] = content
        gradio_messages = [gradio_message]
    return gradio_messages


def parse_hpo_obo(file_path):
    """
    Parse the HPO OBO file and create a dictionary mapping HP IDs to phenotype descriptions.

    Args:
        file_path (str): Path to the HPO OBO file.

    Returns:
        dict: A dictionary where keys are HP IDs and values are phenotype descriptions.
    """
    hp_dict = {}
    current_id = None
    current_name = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("[Term]"):
                # If a new term block starts, save the previous term
                if current_id and current_name:
                    hp_dict[current_id] = current_name
                current_id = None
                current_name = None
            elif line.startswith("id: HP:"):
                current_id = line.split(": ")[1]
            elif line.startswith("name:"):
                current_name = line.split(": ", 1)[1]

        # Add the last term to the dictionary
        if current_id and current_name:
            hp_dict[current_id] = current_name

    return hp_dict

# Example usage
file_path = "/mnt/biomni_filestore/biomni/biomni_data/data_lake/hp.obo"  # Replace with the path to your hp.obo file
# hp_dict = parse_hpo_obo(file_path)

# Updated library_content as a dictionary with detailed descriptions
library_content_dict = {
    # === PYTHON PACKAGES ===
    
    # Core Bioinformatics Libraries (Python)
    "biopython": "[Python Package] A set of tools for biological computation including parsers for bioinformatics files, access to online services, and interfaces to common bioinformatics programs.",
    #"biom-format": "[Python Package] The Biological Observation Matrix (BIOM) format is designed for representing biological sample by observation contingency tables with associated metadata.",
    "scanpy": "[Python Package] A scalable toolkit for analyzing single-cell gene expression data, specifically designed for large datasets using AnnData.",
    "scikit-bio": "[Python Package] Data structures, algorithms, and educational resources for bioinformatics, including sequence analysis, phylogenetics, and ordination methods.",
    #"anndata": "[Python Package] A Python package for handling annotated data matrices in memory and on disk, primarily used for single-cell genomics data.",
    #"mudata": "[Python Package] A Python package for multimodal data storage and manipulation, extending AnnData to handle multiple modalities.",
    #"pyliftover": "[Python Package] A Python implementation of UCSC liftOver tool for converting genomic coordinates between genome assemblies.",
    #"biopandas": "[Python Package] A package that provides pandas DataFrames for working with molecular structures and biological data.",
    #"biotite": "[Python Package] A comprehensive library for computational molecular biology, providing tools for sequence analysis, structure analysis, and more.",
    
    # Genomics & Variant Analysis (Python)
    "gget": "[Python Package] A toolkit for accessing genomic databases and retrieving sequences, annotations, and other genomic data.",
    #"lifelines": "[Python Package] A complete survival analysis library for fitting models, plotting, and statistical tests.",
    #"scvi-tools": "[Python Package] A package for probabilistic modeling of single-cell omics data, including deep generative models.",
    "gseapy": "[Python Package] A Python wrapper for Gene Set Enrichment Analysis (GSEA) and visualization.",
    "cellxgene-census": "[Python Package] A tool for accessing and analyzing the CellxGene Census, a collection of single-cell datasets.",

    # Data Science & Statistical Analysis (Python)
    "pandas": "[Python Package] A fast, powerful, and flexible data analysis and manipulation library for Python.",
    "numpy": "[Python Package] The fundamental package for scientific computing with Python, providing support for arrays, matrices, and mathematical functions.",
    "scipy": "[Python Package] A Python library for scientific and technical computing, including modules for optimization, linear algebra, integration, and statistics.",
    "scikit-learn": "[Python Package] A machine learning library featuring various classification, regression, and clustering algorithms.",
    "matplotlib": "[Python Package] A comprehensive library for creating static, animated, and interactive visualizations in Python.",
    "seaborn": "[Python Package] A statistical data visualization library based on matplotlib with a high-level interface for drawing attractive statistical graphics.",
    "statsmodels": "[Python Package] A Python module for statistical modeling and econometrics, including descriptive statistics and estimation of statistical models.",
    #"umap-learn": "[Python Package] Uniform Manifold Approximation and Projection, a dimension reduction technique.",
    #"faiss-cpu": "[Python Package] A library for efficient similarity search and clustering of dense vectors.",
    
    # General Bioinformatics & Computational Utilities (Python)
    #"tiledb": "[Python Package] A powerful engine for storing and analyzing large-scale genomic data.",
    #"tiledbsoma": "[Python Package] A library for working with the SOMA (Stack of Matrices) format using TileDB.",
    #"h5py": "[Python Package] A Python interface to the HDF5 binary data format, allowing storage of large amounts of numerical data.",
    #"tqdm": "[Python Package] A fast, extensible progress bar for loops and CLI applications.",
    #"joblib": "[Python Package] A set of tools to provide lightweight pipelining in Python, including transparent disk-caching and parallel computing.",
    "PyPDF2": "[Python Package] A library for working with PDF files, useful for extracting text from scientific papers.",
    "googlesearch-python": "[Python Package] A library for performing Google searches programmatically.",
    #"scikit-image": "[Python Package] A collection of algorithms for image processing in Python.",
    "pymed": "[Python Package] A Python library for accessing PubMed articles.",
    "arxiv": "[Python Package] A Python wrapper for the arXiv API, allowing access to scientific papers.",
    "scholarly": "[Python Package] A module to retrieve author and publication information from Google Scholar.",
    
    #"mageck": "[Python Package] Analysis of CRISPR screen data.",
    #"igraph": "[Python Package] Network analysis and visualization.",
    #"pyscenic": "[Python Package] Analysis of single-cell RNA-seq data and gene regulatory networks.",
}


def textify_api_dict(api_dict):
    """Convert a nested API dictionary to a nicely formatted string."""
    lines = []
    for category, methods in api_dict.items():
        lines.append(f"Import file: {category}")
        lines.append("=" * (len("Import file: ") + len(category)))
        for method in methods:
            lines.append(f"Method: {method.get('name', 'N/A')}")
            lines.append(f"  Description: {method.get('description', 'No description provided.')}")
            
            # Process required parameters
            req_params = method.get('required_parameters', [])
            if req_params:
                lines.append("  Required Parameters:")
                for param in req_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")
            
            # Process optional parameters
            opt_params = method.get('optional_parameters', [])
            if opt_params:
                lines.append("  Optional Parameters:")
                for param in opt_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = param.get("description", "No description")
                    param_default = param.get("default", "None")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")
            
            
            lines.append("")  # Empty line between methods
        lines.append("")  # Extra empty line after each category

    return "\n".join(lines)

# Data lake dictionary with detailed descriptions
data_lake_dict = {
    "affinity_capture-ms.parquet": "Protein-protein interactions detected via affinity capture and mass spectrometry.",
    "affinity_capture-rna.parquet": "Protein-RNA interactions detected by affinity capture.",
    #"BindingDB_All_202409.tsv": "Measured binding affinities between proteins and small molecules for drug discovery.",
    #"broad_repurposing_hub_molecule_with_smiles.parquet": "Molecules from Broad Institute's Drug Repurposing Hub with SMILES annotations.",
    #"broad_repurposing_hub_phase_moa_target_info.parquet": "Drug phases, mechanisms of action, and target information from Broad Institute.",
    "co-fractionation.parquet": "Protein-protein interactions from co-fractionation experiments.",
    #"Cosmic_Breakpoints_v101_GRCh38.csv": "Genomic breakpoints associated with cancers from COSMIC database.",
    # "Cosmic_CancerGeneCensusHallmarksOfCancer_v101_GRCh38.parquet": "Hallmarks of cancer genes from COSMIC.",
    #"Cosmic_CancerGeneCensus_v101_GRCh38.parquet": "Census of cancer-related genes from COSMIC.",
    #"Cosmic_ClassificationPaper_v101_GRCh38.parquet": "Cancer classifications and annotations from COSMIC.",
    #"Cosmic_Classification_v101_GRCh38.parquet": "Classification of cancer types from COSMIC.",
    #"Cosmic_CompleteCNA_v101_GRCh38.tsv.gz": "Complete copy number alterations data from COSMIC.",
    #"Cosmic_CompleteDifferentialMethylation_v101_GRCh38.tsv.gz": "Differential methylation patterns from COSMIC.",
    #"Cosmic_CompleteGeneExpression_v101_GRCh38.tsv.gz": "Gene expression data across cancers from COSMIC.",
    #"Cosmic_Fusion_v101_GRCh38.csv": "Gene fusion events from COSMIC.",
    # "Cosmic_Genes_v101_GRCh38.parquet": "List of genes associated with cancer from COSMIC.",
    #"Cosmic_GenomeScreensMutant_v101_GRCh38.tsv.gz": "Genome screening mutations from COSMIC.",
    #"Cosmic_MutantCensus_v101_GRCh38.csv": "Catalog of cancer-related mutations from COSMIC.",
    #"Cosmic_ResistanceMutations_v101_GRCh38.parquet": "Resistance mutations related to therapeutic interventions from COSMIC.",
    #"czi_census_datasets_v4.parquet": "Datasets from the Chan Zuckerberg Initiative's Cell Census.",
    "DisGeNET.parquet": "Gene-disease associations from multiple sources.",
    "dosage_growth_defect.parquet": "Gene dosage changes affecting growth.",
    #"enamine_cloud_library_smiles.pkl": "Compounds from Enamine REAL library with SMILES annotations.",
    "genebass_missense_LC_filtered.pkl": "Filtered missense variants from GeneBass.",
    "genebass_pLoF_filtered.pkl": "Predicted loss-of-function variants from GeneBass.",
    "genebass_synonymous_filtered.pkl": "Filtered synonymous variants from GeneBass.",
    "gene_info.parquet": "Comprehensive gene information.",
    "genetic_interaction.parquet": "Genetic interactions between genes.",
    "go-plus.json": "Gene ontology data for functional gene annotations.",
    "gtex_tissue_gene_tpm.parquet": "Gene expression (TPM) across human tissues from GTEx.",
    "gwas_catalog.pkl": "Genome-wide association studies (GWAS) results.",
    #"hp.obo": "Official HPO release in obographs format",
    "marker_celltype.parquet": "Cell type marker genes for identification.",
    #"McPAS-TCR.parquet": "T-cell receptor sequences and specificity data from McPAS database.",
    "miRDB_v6.0_results.parquet": "Predicted microRNA targets from miRDB.",
    "miRTarBase_microRNA_target_interaction.parquet": "Experimentally validated microRNA-target interactions from miRTarBase.",
    "miRTarBase_microRNA_target_interaction_pubmed_abtract.txt": "PubMed abstracts for microRNA-target interactions in miRTarBase.",
    "miRTarBase_MicroRNA_Target_Sites.parquet": "Binding sites of microRNAs on target genes from miRTarBase.",
    "mousemine_m1_positional_geneset.parquet": "Positional gene sets from MouseMine.",
    "mousemine_m2_curated_geneset.parquet": "Curated gene sets from MouseMine.",
    "mousemine_m3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MouseMine.",
    "mousemine_m5_ontology_geneset.parquet": "Ontology-based gene sets from MouseMine.",
    "mousemine_m8_celltype_signature_geneset.parquet": "Cell type signature gene sets from MouseMine.",
    "mousemine_mh_hallmark_geneset.parquet": "Hallmark gene sets from MouseMine.",
    "msigdb_human_c1_positional_geneset.parquet": "Human positional gene sets from MSigDB.",
    "msigdb_human_c2_curated_geneset.parquet": "Curated human gene sets from MSigDB.",
    "msigdb_human_c3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MSigDB.",
    "msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet": "Transcription factor targets from GTRD/MSigDB.",
    "msigdb_human_c4_computational_geneset.parquet": "Computationally derived gene sets from MSigDB.",
    "msigdb_human_c5_ontology_geneset.parquet": "Ontology-based gene sets from MSigDB.",
    "msigdb_human_c6_oncogenic_signature_geneset.parquet": "Oncogenic signatures from MSigDB.",
    "msigdb_human_c7_immunologic_signature_geneset.parquet": "Immunologic signatures from MSigDB.",
    "msigdb_human_c8_celltype_signature_geneset.parquet": "Cell type signatures from MSigDB.",
    "msigdb_human_h_hallmark_geneset.parquet": "Hallmark gene sets from MSigDB.",
    "omim.parquet": "Genetic disorders and associated genes from OMIM.",
    "proteinatlas.tsv": "Protein expression data from Human Protein Atlas.",
    "proximity_label-ms.parquet": "Protein interactions via proximity labeling and mass spectrometry.",
    "reconstituted_complex.parquet": "Protein complexes reconstituted in vitro.",
    "synthetic_growth_defect.parquet": "Synthetic growth defects from genetic interactions.",
    "synthetic_lethality.parquet": "Synthetic lethal interactions.",
    "synthetic_rescue.parquet": "Genetic interactions rescuing phenotypes.",
    "two-hybrid.parquet": "Protein-protein interactions detected by yeast two-hybrid assays.",
    #"variant_table.parquet": "Annotated genetic variants table.",
    #"Virus-Host_PPI_P-HIPSTER_2020.parquet": "Virus-host protein-protein interactions from P-HIPSTER.",
    #"txgnn_name_mapping.pkl": "Name mapping for TXGNN.",
    #"txgnn_prediction.pkl": "Prediction data for TXGNN."
}



data_lake_dict_tasks = {
    "screen_gene_retrieval": {
        "affinity_capture-ms.parquet": "Protein-protein interactions detected via affinity capture and mass spectrometry.",
        "affinity_capture-rna.parquet": "Protein-RNA interactions detected by affinity capture.",
        "co-fractionation.parquet": "Protein-protein interactions from co-fractionation experiments.",
        # "Cosmic_CancerGeneCensusHallmarksOfCancer_v101_GRCh38.parquet": "Hallmarks of cancer genes from COSMIC.",
        # "Cosmic_Genes_v101_GRCh38.parquet": "List of genes associated with cancer from COSMIC.",
        "DisGeNET.parquet": "Gene-disease associations from multiple sources.",
        "dosage_growth_defect.parquet": "Gene dosage changes affecting growth.",
        "genebass_missense_LC_filtered.pkl": "Filtered missense variants from GeneBass.",
        "genebass_pLoF_filtered.pkl": "Predicted loss-of-function variants from GeneBass.",
        "genebass_synonymous_filtered.pkl": "Filtered synonymous variants from GeneBass.",
        "gene_info.parquet": "Comprehensive gene information.",
        "genetic_interaction.parquet": "Genetic interactions between genes.",
        "go-plus.json": "Gene ontology data for functional gene annotations.",
        "gtex_tissue_gene_tpm.parquet": "Gene expression (TPM) across human tissues from GTEx.",
        "gwas_catalog.pkl": "Genome-wide association studies (GWAS) results.",
        "marker_celltype.parquet": "Cell type marker genes for identification.",
        "msigdb_human_c1_positional_geneset.parquet": "Human positional gene sets from MSigDB.",
        "msigdb_human_c2_curated_geneset.parquet": "Curated human gene sets from MSigDB.",
        "msigdb_human_c3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MSigDB.",
        "msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet": "Transcription factor targets from GTRD/MSigDB.",
        "msigdb_human_c4_computational_geneset.parquet": "Computationally derived gene sets from MSigDB.",
        "msigdb_human_c5_ontology_geneset.parquet": "Ontology-based gene sets from MSigDB.",
        "msigdb_human_c6_oncogenic_signature_geneset.parquet": "Oncogenic signatures from MSigDB.",
        "msigdb_human_c7_immunologic_signature_geneset.parquet": "Immunologic signatures from MSigDB.",
        "msigdb_human_c8_celltype_signature_geneset.parquet": "Cell type signatures from MSigDB.",
        "msigdb_human_h_hallmark_geneset.parquet": "Hallmark gene sets from MSigDB.",
        "omim.parquet": "Genetic disorders and associated genes from OMIM.",
        "proteinatlas.tsv": "Protein expression data from Human Protein Atlas.",
        "proximity_label-ms.parquet": "Protein interactions via proximity labeling and mass spectrometry.",
        "reconstituted_complex.parquet": "Protein complexes reconstituted in vitro.",
        "synthetic_growth_defect.parquet": "Synthetic growth defects from genetic interactions.",
        "synthetic_lethality.parquet": "Synthetic lethal interactions.",
        "synthetic_rescue.parquet": "Genetic interactions rescuing phenotypes.",
        "two-hybrid.parquet": "Protein-protein interactions detected by yeast two-hybrid assays.",
    },
    "rare_disease_diagnosis":{
        # "Cosmic_CancerGeneCensusHallmarksOfCancer_v101_GRCh38.parquet": "Hallmarks of cancer genes from COSMIC.",
        # "Cosmic_Genes_v101_GRCh38.parquet": "List of genes associated with cancer from COSMIC.",
        "DisGeNET.parquet": "Gene-disease associations from multiple sources.",
        "dosage_growth_defect.parquet": "Gene dosage changes affecting growth.",
        "genebass_missense_LC_filtered.pkl": "Filtered missense variants from GeneBass.",
        "genebass_pLoF_filtered.pkl": "Predicted loss-of-function variants from GeneBass.",
        "genebass_synonymous_filtered.pkl": "Filtered synonymous variants from GeneBass.",
        "gene_info.parquet": "Comprehensive gene information.",
        "genetic_interaction.parquet": "Genetic interactions between genes.",
        "go-plus.json": "Gene ontology data for functional gene annotations.",
        "gtex_tissue_gene_tpm.parquet": "Gene expression (TPM) across human tissues from GTEx.",
        "gwas_catalog.pkl": "Genome-wide association studies (GWAS) results.",
        "hp.obo": "Official HPO release in obographs format",
        "marker_celltype.parquet": "Cell type marker genes for identification.",
        "mousemine_m1_positional_geneset.parquet": "Positional gene sets from MouseMine.",
        "mousemine_m2_curated_geneset.parquet": "Curated gene sets from MouseMine.",
        "mousemine_m3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MouseMine.",
        "mousemine_m5_ontology_geneset.parquet": "Ontology-based gene sets from MouseMine.",
        "mousemine_m8_celltype_signature_geneset.parquet": "Cell type signature gene sets from MouseMine.",
        "mousemine_mh_hallmark_geneset.parquet": "Hallmark gene sets from MouseMine.",
        "msigdb_human_c1_positional_geneset.parquet": "Human positional gene sets from MSigDB.",
        "msigdb_human_c2_curated_geneset.parquet": "Curated human gene sets from MSigDB.",
        "msigdb_human_c3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MSigDB.",
        "msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet": "Transcription factor targets from GTRD/MSigDB.",
        "msigdb_human_c4_computational_geneset.parquet": "Computationally derived gene sets from MSigDB.",
        "msigdb_human_c5_ontology_geneset.parquet": "Ontology-based gene sets from MSigDB.",
        "msigdb_human_c6_oncogenic_signature_geneset.parquet": "Oncogenic signatures from MSigDB.",
        "msigdb_human_c7_immunologic_signature_geneset.parquet": "Immunologic signatures from MSigDB.",
        "msigdb_human_c8_celltype_signature_geneset.parquet": "Cell type signatures from MSigDB.",
        "msigdb_human_h_hallmark_geneset.parquet": "Hallmark gene sets from MSigDB.",
        "omim.parquet": "Genetic disorders and associated genes from OMIM.",
    },
    "gwas_variant_prioritization":{
        "affinity_capture-ms.parquet": "Protein-protein interactions detected via affinity capture and mass spectrometry.",
        "affinity_capture-rna.parquet": "Protein-RNA interactions detected by affinity capture.",
        "DisGeNET.parquet": "Gene-disease associations from multiple sources.",
        "dosage_growth_defect.parquet": "Gene dosage changes affecting growth.",
        "genebass_missense_LC_filtered.pkl": "Filtered missense variants from GeneBass.",
        "genebass_pLoF_filtered.pkl": "Predicted loss-of-function variants from GeneBass.",
        "genebass_synonymous_filtered.pkl": "Filtered synonymous variants from GeneBass.",
        "gene_info.parquet": "Comprehensive gene information.",
        "genetic_interaction.parquet": "Genetic interactions between genes.",
        "go-plus.json": "Gene ontology data for functional gene annotations.",
        "gtex_tissue_gene_tpm.parquet": "Gene expression (TPM) across human tissues from GTEx.",
        "gwas_catalog.pkl": "Genome-wide association studies (GWAS) results.",
        "hp.obo": "Official HPO release in obographs format",
        "marker_celltype.parquet": "Cell type marker genes for identification.",
        "mousemine_m1_positional_geneset.parquet": "Positional gene sets from MouseMine.",
        "mousemine_m2_curated_geneset.parquet": "Curated gene sets from MouseMine.",
        "mousemine_m3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MouseMine.",
        "mousemine_m5_ontology_geneset.parquet": "Ontology-based gene sets from MouseMine.",
        "mousemine_m8_celltype_signature_geneset.parquet": "Cell type signature gene sets from MouseMine.",
        "mousemine_mh_hallmark_geneset.parquet": "Hallmark gene sets from MouseMine.",
        "msigdb_human_c1_positional_geneset.parquet": "Human positional gene sets from MSigDB.",
        "msigdb_human_c2_curated_geneset.parquet": "Curated human gene sets from MSigDB.",
        "msigdb_human_c3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MSigDB.",
        "msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet": "Transcription factor targets from GTRD/MSigDB.",
        "msigdb_human_c4_computational_geneset.parquet": "Computationally derived gene sets from MSigDB.",
        "msigdb_human_c5_ontology_geneset.parquet": "Ontology-based gene sets from MSigDB.",
        "msigdb_human_c6_oncogenic_signature_geneset.parquet": "Oncogenic signatures from MSigDB.",
        "msigdb_human_c7_immunologic_signature_geneset.parquet": "Immunologic signatures from MSigDB.",
        "msigdb_human_c8_celltype_signature_geneset.parquet": "Cell type signatures from MSigDB.",
        "msigdb_human_h_hallmark_geneset.parquet": "Hallmark gene sets from MSigDB.",
        "omim.parquet": "Genetic disorders and associated genes from OMIM.",
        "proteinatlas.tsv": "Protein expression data from Human Protein Atlas.",
        "proximity_label-ms.parquet": "Protein interactions via proximity labeling and mass spectrometry.",
        "reconstituted_complex.parquet": "Protein complexes reconstituted in vitro.",
        "synthetic_growth_defect.parquet": "Synthetic growth defects from genetic interactions.",
        "synthetic_lethality.parquet": "Synthetic lethal interactions.",
        "synthetic_rescue.parquet": "Genetic interactions rescuing phenotypes.",
        "two-hybrid.parquet": "Protein-protein interactions detected by yeast two-hybrid assays.",
        "variant_table.parquet": "Annotated genetic variants table.",

    },
    "lab_bench_dbqa":{
        "DisGeNET.parquet": "Gene-disease associations from multiple sources.",
        "gene_info.parquet": "Comprehensive gene information.",
        "genetic_interaction.parquet": "Genetic interactions between genes.",
        "go-plus.json": "Gene ontology data for functional gene annotations.",
        "gtex_tissue_gene_tpm.parquet": "Gene expression (TPM) across human tissues from GTEx.",
        "gwas_catalog.pkl": "Genome-wide association studies (GWAS) results.",
        "hp.obo": "Official HPO release in obographs format",
        "miRDB_v6.0_results.parquet": "Predicted microRNA targets from miRDB.",
        "miRTarBase_microRNA_target_interaction.parquet": "Experimentally validated microRNA-target interactions from miRTarBase.",
        "miRTarBase_microRNA_target_interaction_pubmed_abtract.txt": "PubMed abstracts for microRNA-target interactions in miRTarBase.",
        "miRTarBase_MicroRNA_Target_Sites.parquet": "Binding sites of microRNAs on target genes from miRTarBase.",
        "mousemine_m1_positional_geneset.parquet": "Positional gene sets from MouseMine.",
        "mousemine_m2_curated_geneset.parquet": "Curated gene sets from MouseMine.",
        "mousemine_m3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MouseMine.",
        "mousemine_m5_ontology_geneset.parquet": "Ontology-based gene sets from MouseMine.",
        "mousemine_m8_celltype_signature_geneset.parquet": "Cell type signature gene sets from MouseMine.",
        "mousemine_mh_hallmark_geneset.parquet": "Hallmark gene sets from MouseMine.",
        "msigdb_human_c1_positional_geneset.parquet": "Human positional gene sets from MSigDB.",
        "msigdb_human_c2_curated_geneset.parquet": "Curated human gene sets from MSigDB.",
        "msigdb_human_c3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MSigDB.",
        "msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet": "Transcription factor targets from GTRD/MSigDB.",
        "msigdb_human_c4_computational_geneset.parquet": "Computationally derived gene sets from MSigDB.",
        "msigdb_human_c5_ontology_geneset.parquet": "Ontology-based gene sets from MSigDB.",
        "msigdb_human_c6_oncogenic_signature_geneset.parquet": "Oncogenic signatures from MSigDB.",
        "msigdb_human_c7_immunologic_signature_geneset.parquet": "Immunologic signatures from MSigDB.",
        "msigdb_human_c8_celltype_signature_geneset.parquet": "Cell type signatures from MSigDB.",
        "msigdb_human_h_hallmark_geneset.parquet": "Hallmark gene sets from MSigDB.",
        "omim.parquet": "Genetic disorders and associated genes from OMIM.",
        "Virus-Host_PPI_P-HIPSTER_2020.parquet": "Virus-host protein-protein interactions from P-HIPSTER.",
    },
    "lab_bench_seqqa":{},
    "hle": {
        "affinity_capture-ms.parquet": "Protein-protein interactions detected via affinity capture and mass spectrometry.",
        "affinity_capture-rna.parquet": "Protein-RNA interactions detected by affinity capture.",
        "BindingDB_All_202409.tsv": "Measured binding affinities between proteins and small molecules for drug discovery.",
        "broad_repurposing_hub_molecule_with_smiles.parquet": "Molecules from Broad Institute's Drug Repurposing Hub with SMILES annotations.",
        "broad_repurposing_hub_phase_moa_target_info.parquet": "Drug phases, mechanisms of action, and target information from Broad Institute.",
        "co-fractionation.parquet": "Protein-protein interactions from co-fractionation experiments.",
        # "Cosmic_Breakpoints_v101_GRCh38.csv": "Genomic breakpoints associated with cancers from COSMIC database.",
        # "Cosmic_CancerGeneCensusHallmarksOfCancer_v101_GRCh38.parquet": "Hallmarks of cancer genes from COSMIC.",
        # "Cosmic_CancerGeneCensus_v101_GRCh38.parquet": "Census of cancer-related genes from COSMIC.",
        # "Cosmic_ClassificationPaper_v101_GRCh38.parquet": "Cancer classifications and annotations from COSMIC.",
        # "Cosmic_Classification_v101_GRCh38.parquet": "Classification of cancer types from COSMIC.",
        # "Cosmic_CompleteCNA_v101_GRCh38.tsv.gz": "Complete copy number alterations data from COSMIC.",
        # "Cosmic_CompleteDifferentialMethylation_v101_GRCh38.tsv.gz": "Differential methylation patterns from COSMIC.",
        # "Cosmic_CompleteGeneExpression_v101_GRCh38.tsv.gz": "Gene expression data across cancers from COSMIC.",
        # "Cosmic_Fusion_v101_GRCh38.csv": "Gene fusion events from COSMIC.",
        # "Cosmic_Genes_v101_GRCh38.parquet": "List of genes associated with cancer from COSMIC.",
        # "Cosmic_GenomeScreensMutant_v101_GRCh38.tsv.gz": "Genome screening mutations from COSMIC.",
        # "Cosmic_MutantCensus_v101_GRCh38.csv": "Catalog of cancer-related mutations from COSMIC.",
        # "Cosmic_ResistanceMutations_v101_GRCh38.parquet": "Resistance mutations related to therapeutic interventions from COSMIC.",
        "czi_census_datasets_v4.parquet": "Datasets from the Chan Zuckerberg Initiative's Cell Census.",
        "DisGeNET.parquet": "Gene-disease associations from multiple sources.",
        "dosage_growth_defect.parquet": "Gene dosage changes affecting growth.",
        "enamine_cloud_library_smiles.pkl": "Compounds from Enamine REAL library with SMILES annotations.",
        "genebass_missense_LC_filtered.pkl": "Filtered missense variants from GeneBass.",
        "genebass_pLoF_filtered.pkl": "Predicted loss-of-function variants from GeneBass.",
        "genebass_synonymous_filtered.pkl": "Filtered synonymous variants from GeneBass.",
        "gene_info.parquet": "Comprehensive gene information.",
        "genetic_interaction.parquet": "Genetic interactions between genes.",
        "go-plus.json": "Gene ontology data for functional gene annotations.",
        "gtex_tissue_gene_tpm.parquet": "Gene expression (TPM) across human tissues from GTEx.",
        "gwas_catalog.pkl": "Genome-wide association studies (GWAS) results.",
        "hp.obo": "Official HPO release in obographs format",
        "marker_celltype.parquet": "Cell type marker genes for identification.",
        "McPAS-TCR.parquet": "T-cell receptor sequences and specificity data from McPAS database.",
        "miRDB_v6.0_results.parquet": "Predicted microRNA targets from miRDB.",
        "miRTarBase_microRNA_target_interaction.parquet": "Experimentally validated microRNA-target interactions from miRTarBase.",
        "miRTarBase_microRNA_target_interaction_pubmed_abtract.txt": "PubMed abstracts for microRNA-target interactions in miRTarBase.",
        "miRTarBase_MicroRNA_Target_Sites.parquet": "Binding sites of microRNAs on target genes from miRTarBase.",
        "mousemine_m1_positional_geneset.parquet": "Positional gene sets from MouseMine.",
        "mousemine_m2_curated_geneset.parquet": "Curated gene sets from MouseMine.",
        "mousemine_m3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MouseMine.",
        "mousemine_m5_ontology_geneset.parquet": "Ontology-based gene sets from MouseMine.",
        "mousemine_m8_celltype_signature_geneset.parquet": "Cell type signature gene sets from MouseMine.",
        "mousemine_mh_hallmark_geneset.parquet": "Hallmark gene sets from MouseMine.",
        "msigdb_human_c1_positional_geneset.parquet": "Human positional gene sets from MSigDB.",
        "msigdb_human_c2_curated_geneset.parquet": "Curated human gene sets from MSigDB.",
        "msigdb_human_c3_regulatory_target_geneset.parquet": "Regulatory target gene sets from MSigDB.",
        "msigdb_human_c3_subset_transcription_factor_targets_from_GTRD.parquet": "Transcription factor targets from GTRD/MSigDB.",
        "msigdb_human_c4_computational_geneset.parquet": "Computationally derived gene sets from MSigDB.",
        "msigdb_human_c5_ontology_geneset.parquet": "Ontology-based gene sets from MSigDB.",
        "msigdb_human_c6_oncogenic_signature_geneset.parquet": "Oncogenic signatures from MSigDB.",
        "msigdb_human_c7_immunologic_signature_geneset.parquet": "Immunologic signatures from MSigDB.",
        "msigdb_human_c8_celltype_signature_geneset.parquet": "Cell type signatures from MSigDB.",
        "msigdb_human_h_hallmark_geneset.parquet": "Hallmark gene sets from MSigDB.",
        "omim.parquet": "Genetic disorders and associated genes from OMIM.",
        "proteinatlas.tsv": "Protein expression data from Human Protein Atlas.",
        "proximity_label-ms.parquet": "Protein interactions via proximity labeling and mass spectrometry.",
        "reconstituted_complex.parquet": "Protein complexes reconstituted in vitro.",
        "synthetic_growth_defect.parquet": "Synthetic growth defects from genetic interactions.",
        "synthetic_lethality.parquet": "Synthetic lethal interactions.",
        "synthetic_rescue.parquet": "Genetic interactions rescuing phenotypes.",
        "two-hybrid.parquet": "Protein-protein interactions detected by yeast two-hybrid assays.",
        "variant_table.parquet": "Annotated genetic variants table.",
        "Virus-Host_PPI_P-HIPSTER_2020.parquet": "Virus-host protein-protein interactions from P-HIPSTER.",
        "txgnn_name_mapping.pkl": "Name mapping for TXGNN.",
        "txgnn_prediction.pkl": "Prediction data for TXGNN."
    },
    "crispr_delivery":{}
}

data_lake_dict_tasks['gwas_causal_gene_opentargets'] = data_lake_dict_tasks['screen_gene_retrieval']
data_lake_dict_tasks['gwas_causal_gene_pharmaprojects'] = data_lake_dict_tasks['screen_gene_retrieval']
data_lake_dict_tasks['gwas_causal_gene_gwas_catalog'] = data_lake_dict_tasks['screen_gene_retrieval']
data_lake_dict_tasks['patient_gene_detection'] = data_lake_dict_tasks['rare_disease_diagnosis']




library_content_dict_tasks = {
    "screen_gene_retrieval": {
        "biopython": "[Python Package] A set of tools for biological computation including parsers for bioinformatics files, access to online services, and interfaces to common bioinformatics programs.",
        "scanpy": "[Python Package] A scalable toolkit for analyzing single-cell gene expression data, specifically designed for large datasets using AnnData.",
        "scikit-bio": "[Python Package] Data structures, algorithms, and educational resources for bioinformatics, including sequence analysis, phylogenetics, and ordination methods.",
        "gget": "[Python Package] A toolkit for accessing genomic databases and retrieving sequences, annotations, and other genomic data.",
        "gseapy": "[Python Package] A Python wrapper for Gene Set Enrichment Analysis (GSEA) and visualization.",
        "cellxgene-census": "[Python Package] A tool for accessing and analyzing the CellxGene Census, a collection of single-cell datasets.",
        "pandas": "[Python Package] A fast, powerful, and flexible data analysis and manipulation library for Python.",
        "numpy": "[Python Package] The fundamental package for scientific computing with Python, providing support for arrays, matrices, and mathematical functions.",
        "scipy": "[Python Package] A Python library for scientific and technical computing, including modules for optimization, linear algebra, integration, and statistics.",
        "scikit-learn": "[Python Package] A machine learning library featuring various classification, regression, and clustering algorithms.",
        "matplotlib": "[Python Package] A comprehensive library for creating static, animated, and interactive visualizations in Python.",
        "seaborn": "[Python Package] A statistical data visualization library based on matplotlib with a high-level interface for drawing attractive statistical graphics.",
        "statsmodels": "[Python Package] A Python module for statistical modeling and econometrics, including descriptive statistics and estimation of statistical models.",
        "PyPDF2": "[Python Package] A library for working with PDF files, useful for extracting text from scientific papers.",
        # "googlesearch-python": "[Python Package] A library for performing Google searches programmatically.",
        # "pymed": "[Python Package] A Python library for accessing PubMed articles.",
        # "arxiv": "[Python Package] A Python wrapper for the arXiv API, allowing access to scientific papers.",
        # "scholarly": "[Python Package] A module to retrieve author and publication information from Google Scholar.",
        "mageck": "[Python Package] Analysis of CRISPR screen data.",
        "igraph": "[Python Package] Network analysis and visualization.",
        "pyscenic": "[Python Package] Analysis of single-cell RNA-seq data and gene regulatory networks.",
    },
    "gwas_variant_prioritization": {
        "biopython": "[Python Package] A set of tools for biological computation including parsers for bioinformatics files, access to online services, and interfaces to common bioinformatics programs.",
        "scikit-bio": "[Python Package] Data structures, algorithms, and educational resources for bioinformatics, including sequence analysis, phylogenetics, and ordination methods.",
        "pyliftover": "[Python Package] A Python implementation of UCSC liftOver tool for converting genomic coordinates between genome assemblies.",
        "gget": "[Python Package] A toolkit for accessing genomic databases and retrieving sequences, annotations, and other genomic data.",
        "gseapy": "[Python Package] A Python wrapper for Gene Set Enrichment Analysis (GSEA) and visualization.",
        "pysam": "[Python Package] A Python module for reading, manipulating and writing genomic data sets in SAM/BAM/VCF/BCF formats.",
        "pyfaidx": "[Python Package] A Python package for efficient random access to FASTA files.",
        "pyranges": "[Python Package] A Python package for interval manipulation with a pandas-like interface.",
        "pybedtools": "[Python Package] A Python wrapper for Aaron Quinlan's BEDTools programs.",
        "pandas": "[Python Package] A fast, powerful, and flexible data analysis and manipulation library for Python.",
        "numpy": "[Python Package] The fundamental package for scientific computing with Python, providing support for arrays, matrices, and mathematical functions.",
        "scipy": "[Python Package] A Python library for scientific and technical computing, including modules for optimization, linear algebra, integration, and statistics.",
        "scikit-learn": "[Python Package] A machine learning library featuring various classification, regression, and clustering algorithms.",
        "matplotlib": "[Python Package] A comprehensive library for creating static, animated, and interactive visualizations in Python.",
        "seaborn": "[Python Package] A statistical data visualization library based on matplotlib with a high-level interface for drawing attractive statistical graphics.",
        "statsmodels": "[Python Package] A Python module for statistical modeling and econometrics, including descriptive statistics and estimation of statistical models.",
        "PyPDF2": "[Python Package] A library for working with PDF files, useful for extracting text from scientific papers.",
        # "googlesearch-python": "[Python Package] A library for performing Google searches programmatically.",
        # "pymed": "[Python Package] A Python library for accessing PubMed articles.",
        # "arxiv": "[Python Package] A Python wrapper for the arXiv API, allowing access to scientific papers.",
        # "scholarly": "[Python Package] A module to retrieve author and publication information from Google Scholar.",
        "mageck": "[Python Package] Analysis of CRISPR screen data.",
        "igraph": "[Python Package] Network analysis and visualization.",
        "pyscenic": "[Python Package] Analysis of single-cell RNA-seq data and gene regulatory networks.",
    },
    "crispr_delivery":{
        "PyPDF2": "[Python Package] A library for working with PDF files, useful for extracting text from scientific papers.",
        # "googlesearch-python": "[Python Package] A library for performing Google searches programmatically.",
        # "pymed": "[Python Package] A Python library for accessing PubMed articles.",
        # "arxiv": "[Python Package] A Python wrapper for the arXiv API, allowing access to scientific papers.",
        # "scholarly": "[Python Package] A module to retrieve author and publication information from Google Scholar."
    },
    "hle": {
        "biopython": "[Python Package] A set of tools for biological computation including parsers for bioinformatics files, access to online services, and interfaces to common bioinformatics programs.",
        "biom-format": "[Python Package] The Biological Observation Matrix (BIOM) format is designed for representing biological sample by observation contingency tables with associated metadata.",
        "scanpy": "[Python Package] A scalable toolkit for analyzing single-cell gene expression data, specifically designed for large datasets using AnnData.",
        "scikit-bio": "[Python Package] Data structures, algorithms, and educational resources for bioinformatics, including sequence analysis, phylogenetics, and ordination methods.",
        "anndata": "[Python Package] A Python package for handling annotated data matrices in memory and on disk, primarily used for single-cell genomics data.",
        "mudata": "[Python Package] A Python package for multimodal data storage and manipulation, extending AnnData to handle multiple modalities.",
        "pyliftover": "[Python Package] A Python implementation of UCSC liftOver tool for converting genomic coordinates between genome assemblies.",
        "biopandas": "[Python Package] A package that provides pandas DataFrames for working with molecular structures and biological data.",
        "biotite": "[Python Package] A comprehensive library for computational molecular biology, providing tools for sequence analysis, structure analysis, and more.",
        
        # Genomics & Variant Analysis (Python)
        "gget": "[Python Package] A toolkit for accessing genomic databases and retrieving sequences, annotations, and other genomic data.",
        "lifelines": "[Python Package] A complete survival analysis library for fitting models, plotting, and statistical tests.",
        #"scvi-tools": "[Python Package] A package for probabilistic modeling of single-cell omics data, including deep generative models.",
        "gseapy": "[Python Package] A Python wrapper for Gene Set Enrichment Analysis (GSEA) and visualization.",
        "scrublet": "[Python Package] A tool for detecting doublets in single-cell RNA-seq data.",
        "cellxgene-census": "[Python Package] A tool for accessing and analyzing the CellxGene Census, a collection of single-cell datasets. To download a dataset, use the download_source_h5ad function with the dataset id as the argument (856c1b98-5727-49da-bf0f-151bdb8cb056, no .h5ad extension).",
        "hyperopt": "[Python Package] A Python library for optimizing hyperparameters of machine learning algorithms.",
        "scvelo": "[Python Package] A tool for RNA velocity analysis in single cells using dynamical models.",
        "pysam": "[Python Package] A Python module for reading, manipulating and writing genomic data sets in SAM/BAM/VCF/BCF formats.",
        "pyfaidx": "[Python Package] A Python package for efficient random access to FASTA files.",
        "pyranges": "[Python Package] A Python package for interval manipulation with a pandas-like interface.",
        "pybedtools": "[Python Package] A Python wrapper for Aaron Quinlan's BEDTools programs.",
        
        # Structural Biology & Drug Discovery (Python)
        "rdkit": "[Python Package] A collection of cheminformatics and machine learning tools for working with chemical structures and drug discovery.",
        "deeppurpose": "[Python Package] A deep learning library for drug-target interaction prediction and virtual screening.",
        "pyscreener": "[Python Package] A Python package for virtual screening of chemical compounds.",
        "openbabel": "[Python Package] A chemical toolbox designed to speak the many languages of chemical data, supporting file format conversion and molecular modeling.",
        "descriptastorus": "[Python Package] A library for computing molecular descriptors for machine learning applications in drug discovery.",
        #"pymol": "[Python Package] A molecular visualization system for rendering and animating 3D molecular structures.",
        "openmm": "[Python Package] A toolkit for molecular simulation using high-performance GPU computing.",
        "pytdc": "[Python Package] A Python package for Therapeutics Data Commons, providing access to machine learning datasets for drug discovery.",
        
        # Data Science & Statistical Analysis (Python)
        "pandas": "[Python Package] A fast, powerful, and flexible data analysis and manipulation library for Python.",
        "numpy": "[Python Package] The fundamental package for scientific computing with Python, providing support for arrays, matrices, and mathematical functions.",
        "scipy": "[Python Package] A Python library for scientific and technical computing, including modules for optimization, linear algebra, integration, and statistics.",
        "scikit-learn": "[Python Package] A machine learning library featuring various classification, regression, and clustering algorithms.",
        "matplotlib": "[Python Package] A comprehensive library for creating static, animated, and interactive visualizations in Python.",
        "seaborn": "[Python Package] A statistical data visualization library based on matplotlib with a high-level interface for drawing attractive statistical graphics.",
        "statsmodels": "[Python Package] A Python module for statistical modeling and econometrics, including descriptive statistics and estimation of statistical models.",
        "pymc3": "[Python Package] A Python package for Bayesian statistical modeling and probabilistic machine learning.",
        # "pystan": "[Python Package] A Python interface to Stan, a platform for statistical modeling and high-performance statistical computation.",
        "umap-learn": "[Python Package] Uniform Manifold Approximation and Projection, a dimension reduction technique.",
        "faiss-cpu": "[Python Package] A library for efficient similarity search and clustering of dense vectors.",
        "harmony-pytorch": "[Python Package] A PyTorch implementation of the Harmony algorithm for integrating single-cell data.",
        
        # General Bioinformatics & Computational Utilities (Python)
        "tiledb": "[Python Package] A powerful engine for storing and analyzing large-scale genomic data.",
        "tiledbsoma": "[Python Package] A library for working with the SOMA (Stack of Matrices) format using TileDB.",
        "h5py": "[Python Package] A Python interface to the HDF5 binary data format, allowing storage of large amounts of numerical data.",
        "tqdm": "[Python Package] A fast, extensible progress bar for loops and CLI applications.",
        "joblib": "[Python Package] A set of tools to provide lightweight pipelining in Python, including transparent disk-caching and parallel computing.",
        "opencv-python": "[Python Package] OpenCV library for computer vision tasks, useful for image analysis in biological contexts.",
        "PyPDF2": "[Python Package] A library for working with PDF files, useful for extracting text from scientific papers.",
        # "googlesearch-python": "[Python Package] A library for performing Google searches programmatically.",
        "scikit-image": "[Python Package] A collection of algorithms for image processing in Python.",
        # "pymed": "[Python Package] A Python library for accessing PubMed articles.",
        # "arxiv": "[Python Package] A Python wrapper for the arXiv API, allowing access to scientific papers.",
        # "scholarly": "[Python Package] A module to retrieve author and publication information from Google Scholar.",
        "cryosparc-tools": "[Python Package] Tools for working with cryoSPARC, a platform for cryo-EM data processing.",
        
        "mageck": "[Python Package] Analysis of CRISPR screen data.",
        "igraph": "[Python Package] Network analysis and visualization.",
        "pyscenic": "[Python Package] Analysis of single-cell RNA-seq data and gene regulatory networks.",
        "cooler": "[Python Package] Storage and analysis of Hi-C data.",
        "trackpy": "[Python Package] Particle tracking in images and video.",
        #"flowcytometrytools": "[Python Package] Analysis and visualization of flow cytometry data.",
        "cellpose": "[Python Package] Cell segmentation in microscopy images.",
        "viennarna": "[Python Package] RNA secondary structure prediction.",
        "PyMassSpec": "[Python Package] Mass spectrometry data analysis.",
        "python-libsbml": "[Python Package] Working with SBML files for computational biology.",
        "cobra": "[Python Package] Constraint-based modeling of metabolic networks.",
        "reportlab": "[Python Package] Creation of PDF documents.",
        "flowkit": "[Python Package] Toolkit for processing flow cytometry data.",
        "hmmlearn": "[Python Package] Hidden Markov model analysis.",
        "msprime": "[Python Package] Simulation of genetic variation.",
        "tskit": "[Python Package] Handling tree sequences and population genetics data.",
        "cyvcf2": "[Python Package] Fast parsing of VCF files.",
        "pykalman": "[Python Package] Kalman filter and smoother implementation.",
        "fanc": "[Python Package] Analysis of chromatin conformation data.",
        "loompy": "A Python implementation of the Loom file format for efficiently storing and working with large omics datasets.",
        "pyBigWig": "A Python library for accessing bigWig and bigBed files for genome browser track data.",
        "pymzml": "A Python module for high-throughput bioinformatics analysis of mass spectrometry data.",
        "optlang": "A Python package for modeling optimization problems symbolically.",
        "FlowIO": "A Python package for reading and writing flow cytometry data files.",
        "FlowUtils": "Utilities for processing and analyzing flow cytometry data.",
        "arboreto": "A Python package for inferring gene regulatory networks from single-cell RNA-seq data.",
        "pdbfixer": "A Python package for fixing problems in PDB files in preparation for molecular simulations.",
    }
}

library_content_dict_tasks['gwas_causal_gene_opentargets'] = library_content_dict_tasks['screen_gene_retrieval']
library_content_dict_tasks['gwas_causal_gene_pharmaprojects'] = library_content_dict_tasks['screen_gene_retrieval']
library_content_dict_tasks['gwas_causal_gene_gwas_catalog'] = library_content_dict_tasks['screen_gene_retrieval']
library_content_dict_tasks['patient_gene_detection'] = library_content_dict_tasks['screen_gene_retrieval']
library_content_dict_tasks['rare_disease_diagnosis'] = library_content_dict_tasks['screen_gene_retrieval']
library_content_dict_tasks['lab_bench_dbqa'] = library_content_dict_tasks['screen_gene_retrieval']
library_content_dict_tasks['lab_bench_seqqa'] = library_content_dict_tasks['gwas_variant_prioritization']
