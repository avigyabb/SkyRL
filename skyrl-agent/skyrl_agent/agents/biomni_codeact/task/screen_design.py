from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np
import os
import glob
import re


class screen_design(base_task):
    def __init__(self, top_k=20):
        """
        Initialize the screen_design task class for gene screening

        Parameters:
        -----------
        top_k : int, default=100
            Number of top genes to recommend
        """
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
        self.top_k = top_k
        
        # Define the prompt template
        self.prompt = (
            "Your task is to identify the top {top_k} genes that are most relevant for the following research context:\n\n"
            "{context}\n\n"
            "Based on the research context, identify the top {top_k} genes that would be most important to investigate. "
            "Output only the gene symbols as a comma-separated string: GENE1, GENE2, GENE3, ..."
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

    def get_example(self, index=None):
        """Get a single example from the dataset"""
        if index is None:
            index = np.random.randint(self.num_examples)
        
        screen_id = self.screen_ids[index]
        
        # Get the context for this screen
        context = self.user_requests[self.user_requests['ID'] == screen_id]['Request'].values[0]
        
        return {
            "screen_id": screen_id,
            "prompt": self.prompt.format(
                context=context,
                top_k=self.top_k
            )
        }

    def get_iterator(self):
        """Iterate through all examples in the dataset"""
        for i in range(self.num_examples):
            yield self.get_example(i)

    def evaluate(self, screen_id, predicted_genes):
        """
        Evaluate the predictions against ground truth
        
        Parameters:
        -----------
        screen_id : int
            Screen ID
        predicted_genes : list
            List of predicted gene symbols
            
        Returns:
        --------
        dict
            Dictionary containing precision, recall and F1 scores
        """
        # Get ground truth genes for this screen
        true_genes = self.ground_truth[self.ground_truth['SCREEN_ID'] == screen_id]['OFFICIAL_SYMBOL'].values
        
        # Convert to sets for comparison
        true_set = set(true_genes)  # use full ground truth genes for comparison
        pred_set = set(predicted_genes[:self.top_k])  # Top K predicted genes
        
        # Calculate metrics
        if len(pred_set) == 0:
            precision = 0.0
        else:
            precision = len(true_set.intersection(pred_set)) / len(pred_set)
            
        if len(true_set) == 0:
            recall = 0.0
        else:
            recall = len(true_set.intersection(pred_set)) / len(true_set)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": len(true_set.intersection(pred_set)),
            "false_positives": len(pred_set - true_set),
            "false_negatives": len(true_set - pred_set)
        }

    def reward(self, input, output):
        """Calculate reward as precision score for the predictions"""
        try:
            # screen_id = input['screen_id']
            screen_id = input
            
            if not output:
                return 0.0
            
            # Parse output to extract gene names
            # Assume output is a comma-separated list of gene symbols
            genes = [gene.strip() for gene in output.split(',')]
            
            if len(genes) != self.top_k:
                return 0.0
            
            # Evaluate and return F1 score as reward
            metrics = self.evaluate(screen_id, genes)
            return metrics['precision']
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

        class gene_list(BaseModel):
            """List of genes for the screen design task"""

            genes: str = Field(
                description="""A comma-separated list of gene symbols that are most relevant for the given research context.
                The output should be in the format: GENE1, GENE2, GENE3, ..., GENEK"""
            )
        return gene_list 