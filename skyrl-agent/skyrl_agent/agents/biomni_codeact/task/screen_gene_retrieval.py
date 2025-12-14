from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np
import os
import glob
import re
import random


class screen_gene_retrieval(base_task):
    def __init__(self, num_negative_controls=10, seed=42):
        """
        Initialize the screen_gene_retrieval task class for gene retrieval from screens

        Parameters:
        -----------
        num_negative_controls : int, default=10
            Number of negative control genes to include in the candidate list
        seed : int, default=42
            Random seed for reproducibility
        """
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Data paths
        self.data_dir = '/dfs/project/bioagentos/data/screen_data'
        self.screens_dir = os.path.join(self.data_dir, 'BIOGRID-ORCS-ALL-homo_sapiens-1.1.16.screens')
        
        # Load ground truth data
        self.ground_truth = pd.read_csv(os.path.join(self.data_dir, 'top_100_genes_per_screen.csv'))
        
        # Load user requests (contexts)
        self.user_requests = pd.read_csv(os.path.join(self.data_dir, 'user_requests.csv'))
        
        # Convert ID to int for matching
        self.user_requests['ID'] = self.user_requests['ID'].astype(int)
        
        # Get unique screen IDs that have both context and ground truth
        all_screen_ids = set(self.ground_truth['SCREEN_ID'].unique())
        context_screen_ids = set(self.user_requests['ID'].unique())
        self.available_screen_ids = sorted(list(all_screen_ids.intersection(context_screen_ids)))
        self.screen_ids = self.available_screen_ids
            
        self.num_examples = len(self.screen_ids)
        self.num_negative_controls = num_negative_controls
        
        # Get all available genes for negative control selection
        self.all_genes = set(self.ground_truth['OFFICIAL_SYMBOL'].unique())
        
        # Define the prompt template
        self.prompt = (
            "Your task is to identify the gene with the strongest perturbation effect for the following research context:\n\n"
            "{context}\n\n"
            "From the following list of candidate genes, select the ONE gene that would have the strongest perturbation effect "
            "in this experimental context:\n\n"
            "Candidate genes: {gene_list}\n\n"
            "Only include the gene symbol of your choice in your final solution, e.g., <solution>BRCA1</solution>"
        )

    def __len__(self):
        return self.num_examples

    def get_screen_data(self, screen_id):
        """Load the screen data file for a given screen ID"""
        screen_file_pattern = f"BIOGRID-ORCS-SCREEN_{screen_id}-*.screen.tab.txt"
        screen_file_path = glob.glob(os.path.join(self.screens_dir, screen_file_pattern))
        
        if not screen_file_path:
            return None
        
        # Load the screen data file
        screen_data = pd.read_csv(screen_file_path[0], sep='\t', comment='#')
        return screen_data

    def _get_negative_controls(self, screen_id, target_gene):
        """Get negative control genes for a given screen - genes NOT in top 100"""
        # Get the top 100 genes for this screen (positive hits)
        screen_top_genes = set(self.ground_truth[self.ground_truth['SCREEN_ID'] == screen_id]['OFFICIAL_SYMBOL'].values)
        
        # Get all genes that are NOT in the top 100 for this screen (true negative controls)
        negative_control_candidates = list(self.all_genes - screen_top_genes)
        
        # If we don't have enough negative controls, this shouldn't happen given the large gene pool
        # but we'll handle it gracefully
        if len(negative_control_candidates) < self.num_negative_controls:
            print(f"Warning: Only {len(negative_control_candidates)} negative controls available for screen {screen_id}")
            return negative_control_candidates
        
        # Sort candidates to ensure consistent ordering across runs
        negative_control_candidates.sort()
        
        # Use screen_id as seed for deterministic selection
        screen_random = np.random.RandomState(seed=screen_id)
        
        # Sample the required number of negative controls from genes NOT in top 100
        negative_controls = screen_random.choice(
            negative_control_candidates, 
            size=self.num_negative_controls, 
            replace=False
        ).tolist()
        
        return negative_controls

    def get_example(self, index=None):
        """Get a single example from the dataset"""
        if index is None:
            index = np.random.randint(self.num_examples)
        
        screen_id = self.screen_ids[index]
        
        # Get the context for this screen
        context = self.user_requests[self.user_requests['ID'] == screen_id]['Request'].values[0]
        
        # Get the top gene (target gene with strongest perturbation effect)
        screen_genes = self.ground_truth[self.ground_truth['SCREEN_ID'] == screen_id]
        # Assuming the first gene in the sorted list is the top perturbation gene
        target_gene = screen_genes.iloc[0]['OFFICIAL_SYMBOL']
        
        # Get negative control genes
        negative_controls = self._get_negative_controls(screen_id, target_gene)
        
        # Create randomized candidate list
        candidate_genes = [target_gene] + negative_controls
        
        # Use screen_id as seed for deterministic shuffling
        screen_random = np.random.RandomState(seed=screen_id + 1000)  # Add offset to differentiate from negative control seed
        screen_random.shuffle(candidate_genes)
        
        gene_list_str = ", ".join(candidate_genes)
        
        return {
            "screen_id": screen_id,
            "index": index,
            "target_gene": target_gene,
            "candidate_genes": candidate_genes,
            "context": context,
            "prompt": self.prompt.format(
                context=context,
                gene_list=gene_list_str
            ),
            "instance_id": index
        }
    
    def get_example_from_screen_id(self, screen_id):
        """Get example by screen ID"""
        for i, sid in enumerate(self.screen_ids):
            if sid == screen_id:
                return self.get_example(i)
        return None

    def get_iterator(self):
        """Iterate through all examples in the dataset"""
        for i in range(self.num_examples):
            yield self.get_example(i)

    def evaluate(self, predictions, targets):
        """
        Evaluate the predictions against ground truth
        
        Parameters:
        -----------
        predictions : list
            List of predicted gene symbols
        targets : list
            List of target gene symbols
            
        Returns:
        --------
        dict
            Dictionary containing accuracy and other metrics
        """
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
        accuracy = correct / len(predictions)
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(predictions)
        }

    def reward(self, input_data, output):
        """Calculate reward based on whether the correct gene was selected"""
        try:
            if isinstance(input_data, dict):
                target_gene = input_data.get('target_gene')
            else:
                # If input_data is just an index, get the example
                example = self.get_example(input_data)
                target_gene = example['target_gene']
            
            if not output:
                return 0.0
            
            # Clean the output (remove any extra whitespace, punctuation)
            predicted_gene = output.strip().upper()
            target_gene = target_gene.strip().upper()
            
            # Return 1 if correct, 0 if incorrect
            return 1.0 if predicted_gene == target_gene else 0.0
            
        except Exception as e:
            print(f"Error in reward function: {e}")
            return 0.0

    def split(self, ratio=0.8, seed=42):
        """Split the dataset into train and validation sets"""
        np.random.seed(seed)
        indices = np.arange(self.num_examples)
        np.random.shuffle(indices)
        split_idx = int(ratio * self.num_examples)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        return train_indices, val_indices

    def output_class(self):
        """Define the output class for the task"""
        from pydantic import BaseModel, Field
        from typing import Optional

        class gene_selection(BaseModel):
            """Gene selection for screen gene retrieval task"""

            selected_gene: str = Field(
                description="""The gene symbol of the selected gene that has the strongest perturbation effect.
                Output should be a single gene symbol, e.g., BRCA1"""
            )
        return gene_selection

    def output_parser(self, output, llm):
        """Parse the output to extract the selected gene"""
        from pydantic import BaseModel, Field
        from typing import Optional
        
        class gene_selection(BaseModel):
            """Gene selection for screen gene retrieval task"""

            selected_gene: str = Field(
                description="""The gene symbol of the selected gene that has the strongest perturbation effect.
                Output should be a single gene symbol, e.g., BRCA1"""
            )

        output_parser = llm.with_structured_output(gene_selection)
        result = output_parser.invoke(output)
        return result.selected_gene 