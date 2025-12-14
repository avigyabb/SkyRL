from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np
import os


class microbiome_analysis(base_task):
    def __init__(self, path = './data', num_samples=5):
        # Define data paths
        self.data_dir = path + '/microbiome/data_microbiome/'
        self.ground_truth_dir = path + '/microbiome/human_results_microbiome/'
        
        # Define datasets available for analysis
        self.datasets = {
            'data1': {
                'abundance': 'data1_abundance.csv',
                'labels': 'data1_labels.csv',
                'description': 'Colorectal cancer (CRC) vs control microbiome dataset',
                'ground_truth': 'data1_differential_results.csv'
            },
            'data2': {
                'abundance': 'data2_MicrobiomeNoMetadata.csv',
                'taxonomy': 'data2_MicrobiomeOTUtaxonomy.csv',
                'description': 'OTU microbiome dataset with taxonomy information',
                'ground_truth': 'data2_differential_results.csv'
            },
            'data3': {
                'abundance': 'data3_abundance.csv',
                'description': 'General microbiome abundance dataset',
                'ground_truth': 'data3_differential_results.csv'
            },
            'data4': {
                'otu': 'data4_BM_OTU.xlsx',
                'description': 'OTU data in Excel format',
                'ground_truth': 'data4_differential_results.csv'
            },
            'data5': {
                'abundance': 'data5_otu_table.csv',
                'metadata': 'data5_sam_data.csv',
                'description': 'Microbiome dataset with sample metadata',
                'ground_truth': 'data5_differential_results.csv'
            }
        }
        
        # Define the prompt template - focused specifically on differential abundance analysis
        self.prompt = """
Task: Perform differential abundance analysis on microbiome data.
Dataset: '{dataset}'
Files:
{data_path}

Objective: Identify microbial taxa (e.g., genus, OTUs) that are significantly differentially abundant between groups or conditions.

Expected Output:
Return a list of tuples containing the most significantly differentially abundant taxa and their p-values:
[
    (taxon_name, p_value),
    (taxon_name, p_value),
    ...
]

Examples:
- For a colorectal cancer dataset:
  [
    ("Fusobacterium", 0.00000012),
    ("Peptostreptococcus", 0.000005),
    ("Bacteroides", 0.0023)
  ]

- For an OTU dataset:
  [
    ("OTU2921", 0.01560452),
    ("OTU5788", 0.013678708),
    ("OTU836", 0.004600807)
  ]

Focus on the most significant taxa (typically 10-20 taxa with the lowest p-values).

Note: Your analysis should implement appropriate statistical methods for differential abundance testing in microbiome data.
"""
        
        # Set number of examples equal to number of datasets
        self.num_examples = min(num_samples, len(self.datasets))
        self.selected_inputs = list(self.datasets.keys())[:self.num_examples]

    def __len__(self):
        return self.num_examples

    def get_example(self, index=None):
        if index is None:
            index = np.random.randint(self.num_examples)
            
        dataset_key = self.selected_inputs[index]
        dataset_info = self.datasets[dataset_key]
        
        # Create a list of specific file paths for this dataset
        dataset_files = []
        for file_type, file_name in dataset_info.items():
            if file_type not in ['description', 'ground_truth']:
                file_path = os.path.join(self.data_dir, file_name)
                dataset_files.append(f"{file_type}: {file_path}")
        
        # Join the file paths with newlines
        specific_paths = "\n".join(dataset_files)
            
        formatted_prompt = self.prompt.format(
            dataset=dataset_info['description'],
            data_path=specific_paths
        )
        
        return {
            "prompt": formatted_prompt,
            "dataset": dataset_key,
            "instance_id": index
        }

    def get_iterator(self):
        for i in range(self.num_examples):
            yield self.get_example(i)

    def evaluate(self, dataset_key, predictions):
        """Evaluate the model's predictions for microbiome differential abundance analysis by comparing with ground truth."""
        # Initialize score components
        score = 0.0
        max_score = 10.0  # Total possible score
        
        # Load ground truth data
        ground_truth_path = os.path.join(self.ground_truth_dir, self.datasets[dataset_key]['ground_truth'])
        
        evaluation_results = {
            "significant_taxa_overlap": 0,
            "identified_taxa": [],
            "performance_metrics": {}
        }
        
        try:
            # Read the ground truth
            ground_truth = pd.read_csv(ground_truth_path)
            
            # Extract taxa from ground truth
            ground_truth_taxa_col = ground_truth.columns[0]  # First column usually contains taxa names
            ground_truth_taxa = set(ground_truth[ground_truth_taxa_col].astype(str))
            
            # Check if predictions has the required differential_taxa field (4 points)
            if "differential_taxa" in predictions and predictions["differential_taxa"]:
                # Extract taxa names from the list of tuples
                try:
                    agent_taxa = set([str(item[0]) for item in predictions["differential_taxa"]])
                    score += 4.0
                    
                    evaluation_results["identified_taxa"] = list(agent_taxa)
                    
                    # Calculate overlap between agent's significant taxa and ground truth (4 points)
                    overlap = agent_taxa.intersection(ground_truth_taxa)
                    overlap_ratio = len(overlap) / len(ground_truth_taxa) if ground_truth_taxa else 0
                    overlap_score = min(4.0, 4.0 * overlap_ratio)
                    score += overlap_score
                    evaluation_results["significant_taxa_overlap"] = overlap_ratio
                    
                    # Check if agent provided p-values (2 points)
                    try:
                        # Check if tuples have p-values that are valid numbers
                        valid_pvalues = True
                        for item in predictions["differential_taxa"]:
                            if len(item) < 2 or not isinstance(item[1], (int, float)) or item[1] < 0 or item[1] > 1:
                                valid_pvalues = False
                                break
                        
                        if valid_pvalues:
                            score += 2.0
                    except:
                        pass
                except:
                    evaluation_results["error"] = "Could not parse differential_taxa as a list of tuples"
            
        except Exception as e:
            evaluation_results["error"] = str(e)
        
        # Normalize score to be between 0 and 1
        normalized_score = score / max_score
        
        evaluation_results["score"] = normalized_score
        evaluation_results["explanation"] = f"Analysis scored {score}/{max_score} points"
        
        return evaluation_results

    def reward(self, input, output):
        """Calculate a reward score from 0 to 1 for the given predictions."""
        # Check if the agent returned a list of tuples for differential taxa
        if "differential_taxa" in output and isinstance(output["differential_taxa"], list) and len(output["differential_taxa"]) > 0:
            # Check if the list contains valid tuples
            try:
                for item in output["differential_taxa"]:
                    if not isinstance(item, tuple) or len(item) < 2:
                        return 0.3
                return 0.7
            except:
                return 0.3
        return 0.1

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
        from typing import Optional, List, Tuple, Union

        class microbiome_analysis_prediction(BaseModel):
            """Prediction results for microbiome differential abundance analysis"""

            analysis_completed: bool = Field(
                description="""Whether differential abundance analysis was successfully completed (True/False)."""
            )
            method_used: str = Field(
                description="""The statistical method used for differential abundance testing (e.g., 'DESeq2', 'LEfSe', 'Wilcoxon')."""
            )
            differential_taxa: List[Tuple[str, float]] = Field(
                description="""List of tuples with each tuple containing (taxon_name, p_value). Example: [("Bacteroides", 0.01), ("Fusobacterium", 0.0001)]"""
            )
        
        return microbiome_analysis_prediction
