import os
# from verl.workers.agentic.biomni.utils import (
#     api_schema_to_langchain_tool,
#     textify_api_dict,
#     library_content_dict,
#     data_lake_dict,
#     data_lake_dict_tasks,
#     library_content_dict_tasks,
#     load_pkl
# )
from skyrl_agent.agents.biomni_codeact.utils import (
    api_schema_to_langchain_tool,
    textify_api_dict,
    library_content_dict,
    data_lake_dict,
    data_lake_dict_tasks,
    library_content_dict_tasks,
    load_pkl
)

class PromptManager:
    def __init__(self, tool_path = "./tool", s3_datalake_uri="s3://biomni-datalake/"):
        module2api = load_pkl(os.path.join(tool_path, 'screen_api.pkl'))
        self.default_tool_dict = module2api
        self.s3_datalake_uri = s3_datalake_uri
        self.tool_path = tool_path
        self.configure()
    
    def configure(self, self_critic=False):
        self.self_critic = self_critic
        self.data_lake_dict_tasks = data_lake_dict_tasks
        self.library_content_dict_tasks = library_content_dict_tasks
        
        self.default_data_lake_dict = data_lake_dict
        self.default_library_content_dict = library_content_dict
        
        self.tools_pkl_map = {
            'rare_disease_diagnosis': os.path.join(self.tool_path, 'rare_disease_diagnosis_api_filter.pkl'),
            'gwas_variant_prioritization': os.path.join(self.tool_path, 'gwas_variant_prioritization_api_filter.pkl'),
            'patient_gene_detection': os.path.join(self.tool_path, 'patient_gene_detection_api_filter.pkl'),
            'lab_bench_dbqa': os.path.join(self.tool_path, 'lab_bench_dbqa_api_filter.pkl'),
            'lab_bench_seqqa': os.path.join(self.tool_path, 'lab_bench_seqqa_api_filter.pkl'),
            'screen_gene_retrieval': os.path.join(self.tool_path, 'screen_gene_retrieval_api_filter.pkl'),
            'screen_design': os.path.join(self.tool_path, 'screen_api_filter.pkl'),
            'crispr_delivery': os.path.join(self.tool_path, 'crispr_delivery_api_filter.pkl'),
            'gwas_causal_gene_opentargets': os.path.join(self.tool_path, 'gwas_causal_gene_opentargets_api_filter.pkl'),
            'gwas_causal_gene_pharmaprojects': os.path.join(self.tool_path, 'gwas_causal_gene_pharmaprojects_api_filter.pkl'),
            'gwas_causal_gene_gwas_catalog': os.path.join(self.tool_path, 'gwas_causal_gene_gwas_catalog_api_filter.pkl'),
        }

    
    def get_initial_messages(self, user_prompt, task_name = None):
        if task_name is None:
            raise ValueError("Task name is required")
        
        tool_dict = load_pkl(self.tools_pkl_map[task_name]) if task_name in self.tools_pkl_map else self.default_tool_dict
        data_lake_dict = self.data_lake_dict_tasks[task_name] if task_name in self.data_lake_dict_tasks else self.default_data_lake_dict
        library_content_dict = self.library_content_dict_tasks[task_name] if task_name in self.library_content_dict_tasks else self.default_library_content_dict

        tool_desc = {i: [x for x in j if x['name']!='run_python_repl'] for i, j in tool_dict.items()}
        
        # Get all data lake items from the data_lake_dict keys
        data_lake_items = list(data_lake_dict.keys())
        
        data_lake_with_desc = []
        for item in data_lake_items:
            description = data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})
        
        system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=list(library_content_dict.keys()),
            library_content_dict=library_content_dict,
            data_lake_dict=data_lake_dict,
            self_critic=self.self_critic,
            is_retrieval=False
        )
    
    
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    
    def _generate_system_prompt(self, tool_desc, data_lake_content, library_content_list, library_content_dict, data_lake_dict, self_critic=False, is_retrieval=False):
        """
        Generate the system prompt based on the provided resources.
        
        Args:
            tool_desc: Dictionary of tool descriptions or formatted string
            data_lake_content: List of data lake items (names or dicts with name/description)
            library_content_list: List of libraries (names or dicts with name/description)
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            
        Returns:
            The generated system prompt
        """
        def format_item_with_description(name, description_dict, item_type="data lake"):
            """Format an item with its description in a readable way."""
            # Get description from the provided dict, default if not found
            description = description_dict.get(name, f"Unknown {item_type} item: {name}")

            # Check if the item name itself already contains a description format
            if isinstance(name, str) and ': ' in name:
                # Assume it's pre-formatted, return as is
                return name

            # Basic formatting
            return f"{name}: {description}"

        # Format the data lake content using self.data_lake_dict
        data_lake_formatted = []
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get('name', 'Unknown_Item')
                # Use description from item if provided, otherwise lookup in self.data_lake_dict
                description = item.get('description', data_lake_dict.get(name, f"Data lake item: {name}"))
                data_lake_formatted.append(f"{name}: {description}")
            elif isinstance(item, str):
                 # If it's just a name, look up description in self.data_lake_dict
                data_lake_formatted.append(format_item_with_description(item, data_lake_dict, "data lake"))
            else:
                 data_lake_formatted.append(str(item)) # Fallback

        # Format the library content using self.library_content_dict
        libraries_formatted = []
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get('name', 'Unknown_Library')
                 # Use description from item if provided, otherwise lookup in self.library_content_dict
                description = lib.get('description', library_content_dict.get(name, f"Software library: {name}"))
                libraries_formatted.append(f"{name}: {description}")
            elif isinstance(lib, str):
                # If it's just a name, look up description in self.library_content_dict
                libraries_formatted.append(format_item_with_description(lib, library_content_dict, "library"))
            else:
                libraries_formatted.append(str(lib)) # Fallback

        # Base prompt
        prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving. 
To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions, data, and softwares to assist you throughout the process.

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.
Format your plan as a checklist with empty checkboxes like this:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox with a checkmark:
1. [✓] First step (completed)
2. [ ] Second step
3. [ ] Third step

If a step fails or needs modification, mark it with an X and explain why:
1. [✓] First step (completed)
2. [✗] Second step (failed because...)
3. [ ] Modified second step
4. [ ] Third step

At each turn, you should first provide your detailed thinking and reasoning given the conversation history, along with the updated plan (Always show the updated plan after each step so the user can track progress). 
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observe></observe>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
   - For Python code (default): <execute> print("Hello World!") </execute>
   - For R code: <execute> #!R\nlibrary(ggplot2)\nprint("Hello from R") </execute>
   - For Bash scripts and commands: <execute> #!BASH\necho "Hello from Bash"\nls -la </execute>
   - For CLI softwares, use Bash scripts.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.
    - If user does not specify the format, use a report format. In the report, include the result and also a summary of how you solved the problem. Make it concise and to the point. Be rigorous. 
    - Use numbered references like [1], [2] in the summary if applicable. Provide brief footnotes for each reference at the end of the report, explaining the rationale or evidence.

You have many chances to interact with the environment to receive the observation. So you can decompose your code into multiple steps. 
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.
When calling the existing python functions in the function dictionary, YOU MUST SAVE THE OUTPUT and PRINT OUT the result.
For example, result = understand_scRNA(XXX) print(result) 
Otherwise the system will not be able to know what has been done.
Don't overdo it. Stop when the plan is finished or the task is already solved. Be relatively simple and concise and understandable to the user.
Also, avoid faking or simulating code/data. Your user is a biomedical researcher. Thus, stay true and rigorous.
For the thinking process, put before the execute code block. Do not use print statement in the execute code block for the thinking process.
If you draw figures, make publication-ready and beautiful figures.

For R code, use the #!R marker at the beginning of your code block to indicate it's R code.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, conditionals, loops, and other Bash features.

In each response, you must include EITHER <execute> or <solution> tag. Not both at the same time. Do not respond with messages without any tags. No empty messages.
"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add environment resources with S3 instructions
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
--- 
{tool_desc}
---

{import_instruction}

- Biological Data Lake (Amazon S3):
The biological data lake is stored in an Amazon S3 bucket: {s3_datalake_uri}
{data_lake_intro}
Use appropriate libraries such as `boto3`, `s3fs` to access files from this S3 bucket. Assume valid AWS credentials are already configured in the execution environment.
----
Available Files (use these paths relative to the bucket URI):
{data_lake_content}
----
- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----
- Note on using R packages and Bash scripts:
  - R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
  - Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """
        
        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = "Based on your query, I've identified the following most relevant libraries that you can use:"
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""
        
        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = '\n'.join(libraries_formatted)
        data_lake_content_formatted = '\n'.join(data_lake_formatted)

        # Format the prompt with the appropriate values
        formatted_prompt = prompt_modifier.format(
            function_intro=function_intro,
            tool_desc=textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            import_instruction=import_instruction,
            s3_datalake_uri=self.s3_datalake_uri,
            data_lake_intro=data_lake_intro,
            data_lake_content=data_lake_content_formatted,
            library_intro=library_intro,
            library_content_formatted=library_content_formatted
        )
        
        return formatted_prompt