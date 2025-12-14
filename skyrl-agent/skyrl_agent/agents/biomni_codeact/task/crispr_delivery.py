from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np


class crispr_delivery(base_task):
    def __init__(self, num_samples=100):
        # Load the benchmark dataset
        self.df = pd.read_csv('/dfs/user/kexinh/BioAgentOS/data/crispr_delivery.csv')
        
        # Define the delivery methods and weights for scoring
        self.delivery_methods = {
            'a': 'Plasmid Transfection',
            'b': 'Lentivirus/Retrovirus',
            'c': 'RNP/mRNA electroporation',
            'd': 'RNP/mRNA microinjection',
            'e': 'mRNA LNP',
            'f': 'AAV'
        }
        
        self.weight_first_choice = 2
        self.weight_second_choice = 1
        
        # Clean and format the prompt template for better readability
        raw_prompt = self.df['Prompt'].iloc[0]
        # Strip extra quotes to make it cleaner
        if raw_prompt.startswith('"""'):
            raw_prompt = raw_prompt.strip('"')
        
        # Create a cleaner template that will hold the user's case description
        self.prompt_template = """Given the case description, identify the MOST relevant CRISPR delivery method from the options below:

a. Plasmid Transfection
b. Lentivirus/Retrovirus
c. RNP/mRNA electroporation
d. RNP/mRNA microinjection
e. mRNA LNP
f. AAV

Category: {category}
Case Description: {case_description}

Please provide your response as follows:
- Most relevant method (select one letter a-f): 
"""
        
        # Get unique categories of inputs
        self.categories = self.df['Category'].dropna().unique()
        self.inputs_by_category = {category: self.df[self.df['Category'] == category]['Input'].values 
                                  for category in self.categories}
        self.num_examples = min(num_samples, len(self.df))
        
        # Sample case descriptions from different categories
        np.random.seed(42)
        self.selected_cases = []
        categories_list = list(self.categories)
        for i in range(self.num_examples):
            category_idx = i % len(categories_list)
            category = categories_list[category_idx]
            cases = self.inputs_by_category[category]
            if len(cases) > 0:
                self.selected_cases.append(np.random.choice(cases))
            if len(self.selected_cases) >= num_samples:
                break
                
        self.num_examples = len(self.selected_cases)

    def __len__(self):
        return self.num_examples

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(self.num_examples)
        case_description = self.selected_cases[index]
        
        # Find the category for this case description
        category = None
        for cat, inputs in self.inputs_by_category.items():
            if case_description in inputs:
                category = cat
                break
        
        return {
            "prompt": self.prompt_template.format(
                case_description=case_description,
                category=category if category else "Unknown"
            ),
            "case_description": case_description,
            "category": category
        }

    def get_iterator(self):
        for i in range(self.num_examples):
            yield self.get_example(i)

    def evaluate(self, case_description, predictions):
        """Evaluate the model's predictions against the ground truth."""
        # Find the row in the dataframe corresponding to the case description
        row = self.df[self.df['Input'] == case_description]
        if row.empty:
            return {"score": 0, "explanation": "Case description not found in dataset"}
        
        # Get the ground truth scores for each delivery method
        scores = {method: row[method].values[0] for method in 'abcdef'}
        
        # Calculate the score based on the predictions
        first_choice = predictions
        
        first_choice_score = scores.get(first_choice, 0)
        
        # 2->1, 1->0.5, 0->0
        normalized_score = first_choice_score/2

        # Find the best possible score
        #best_methods = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        #best_first_choice = best_methods[0][0] if len(best_methods) > 0 else ''
        #best_score = scores.get(best_first_choice, 0)
        
        #normalized_score = first_choice_score / best_score if best_score > 0 else 0
        
        return {
            "score": normalized_score
        }

    def reward(self, input, output):
        """Calculate a reward score from 0 to 1 for the given predictions."""
        case_description = self.selected_cases[input]
        result = self.evaluate(case_description, output)
        return result["score"]

    def split(self, ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(self.num_examples)
        np.random.shuffle(indices)
        split_idx = int(ratio * self.num_examples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional

        class crispr_delivery_prediction(BaseModel):
            """Prediction of the most relevant CRISPR delivery method"""

            Answer: str = Field(
                description="""The most relevant CRISPR delivery method (a, b, c, d, e, or f)."""
            )
        return crispr_delivery_prediction 