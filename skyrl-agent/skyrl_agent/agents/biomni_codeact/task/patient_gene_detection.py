import json
from pydantic import BaseModel, Field
import sys
import pandas as pd
import numpy as np
import ast
from verl.workers.agentic.biomni.task.base_task import base_task

class patient_gene_detection(base_task):
    def __init__(self, path = './data', num_samples=100):
        self.data = pd.read_pickle(path + '/patient_gene_detection/patient_gene_detection_benchmark.pkl')
        
        self.query = []
        self.answer = []
        self.data = self.data[:num_samples]
        for idx in range(len(self.data)):
            patient = self.data.iloc[idx]

            phenotypes = patient['phenotypes']
            candidate_genes = patient['candidate_genes']
            true_genes = patient['true_genes']

            self.query.append({
                "phenotypes": phenotypes,
                "candidate_genes": candidate_genes
            })
            self.answer.append({
                "true_genes": true_genes
            })

        self.task_description = """
Task: Given a patient's phenotypes and a list of candidate genes, identify the causal gene.
Phenotypes: {phenotype_list}
Candidate genes: {candidate_genes}

Output format: {{"causal_gene": ["ENSG00000138449"]}}
You must output only the json-formatted answer inside the <solution> and </solution> tags with no other text.
The output inside the <solution> and </solution> tags must be a valid json string.
        """

    def __len__(self):
        return len(self.query)

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(len(self.query))
        
        q = self.query[index]
        a = self.answer[index]
        
        prompt = self.task_description.format(
            phenotype_list=', '.join(q['phenotypes']),
            candidate_genes=', '.join(q['candidate_genes'])
        )
        answer = a["true_genes"]

        return {"prompt": prompt, "answer": answer, "instance_id": index}

    def get_iterator(self):
        for i in range(len(self.query)):
            yield self.get_example(i)

    def split(self, ratio=0.8, seed=42):
        np.random.seed(seed)
        indices = np.arange(len(self.query))
        np.random.shuffle(indices)
        split_idx = int(ratio * len(self.query))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def reward(self, input, output):
        # Check if output is None or empty
        if output is None or output == "":
            return 0.0
            
        # parse output to dict/json
        # Try json.loads first for valid JSON, then fall back to ast.literal_eval for Python dict syntax
        try:
            output = json.loads(output)
        except Exception as e:
            print(e)
            try:
                output = ast.literal_eval(output)
            except Exception as e:
                print(e)
                # If both fail, return 0
                return 0.0
        
        print("Instance_id: ", input)
        print("Prompt: ", self.get_example(input)['prompt'])
        
        # Check if output is a valid dictionary
        if not isinstance(output, dict):
            print(f"Warning: Output is not a dictionary: {type(output)}")
            return 0.0
            
        true_genes = self.get_example(input)["answer"]
        # Use .get() for safer dictionary access
        predicted_genes = output.get('causal_gene', [])
        print("True answer:")
        print(true_genes)
        print("Predicted answer:")
        print(predicted_genes)
        if predicted_genes and len(np.intersect1d(true_genes, predicted_genes)) > 0:
            return 1.0
        else:
            return 0.0

    def output_class(self):
        class GeneDetectionOutput(BaseModel):
            """List of predicted causal genes for a patient."""

            causal_genes: list = Field(
                description="The list of causal gene(s) identified, e.g., ['ENSG00000138449']. Please Use ENSG ID."
            )
        
        return GeneDetectionOutput