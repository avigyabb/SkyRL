from verl.workers.agentic.biomni.task.base_task import base_task
import pandas as pd
import numpy as np
np.random.seed(42)

def shuffle(x):
    np.random.shuffle(x)
    return x

class lab_bench(base_task):

    def __init__(self, path = './data', dataset = 'DbQA'):
        if dataset not in ['DbQA', 'SeqQA']:
            raise ValueError("dataset must be one of 'DbQA', 'SeqQA'")
        
        self.dataset = dataset  # Store dataset type
        df = pd.read_parquet(path + '/' + dataset + '/train-00000-of-00001.parquet')
    
        self.prompt = """The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.

Question: {question}
Options:
{options}

You MUST include the letter of the correct answer within the following tags:
<solution> and </solution>. For example, '<solution>A</solution>',
where A is the correct letter. Always answer in exactly this format
of a single letter between the two tags, even if you are unsure.
We require this because we use automatic parsing.
            """
        
        np.random.seed(42)
        df['options'] = df.apply(lambda x: shuffle(x.distractors.tolist() + [x.ideal] + ['Insufficient information to answer the question.']), axis=1)
        df['options_letters'] = df.options.apply(lambda x: '\n'.join([chr(ord('A') + i) + '.' + item for i, item in enumerate(x)]))
        df['letter_answer'] = df.apply(lambda x: chr(ord('A') + np.where(np.array(x.options) == x.ideal)[0][0]), axis = 1)
        df['letter_refrain'] = df.apply(lambda x: chr(ord('A') + np.where(np.array(x.options) == 'Insufficient information to answer the question.')[0][0]), axis = 1)

        self.query = df.question.values
        self.options = df.options_letters.values
        self.answer = df.letter_answer.values
        self.refrain_label = df.letter_refrain.values
        
        # Store protocol information if available
        self.protocol = df.protocol.values if 'protocol' in df.columns else None

    def __len__(self):
        return len(self.query)

    def get_example(self, index = None):
        if index is None:
            index = np.random.randint(len(self.query))

        if self.dataset == 'ProtocolQA' and self.protocol is not None:
            return {"prompt": self.prompt.format(
                protocol = self.protocol[index],
                question = self.query[index], 
                options = self.options[index]
            ), "answer": self.answer[index],
                "instance_id": index}
        else:
            return {"prompt": self.prompt.format(
                question = self.query[index], 
                options = self.options[index]
            ), "answer": self.answer[index],
                "instance_id": index}

    def split(self, ratio = 0.8, seed = 42):
        np.random.seed(seed)
        indices = np.random.permutation(len(self.query))
        split_index = int(len(self.query) * ratio)
        return indices[:split_index], indices[split_index:]

    def reward(self, index, answer):
        return 1 if self.answer[index] == answer else 0

    def get_iterator(self): 
        for i in range(len(self.query)):
            yield self.get_example(i)

    def evaluate(self, response):
        ## expected a list/array of symbols
        from sklearn.metrics import accuracy_score
        ground_truth = self.answer
        response = np.array(response)

        return {
            'accuracy': accuracy_score(ground_truth, response),
            'coverage': np.mean(response != self.refrain_label),
            'refrain_ratio': np.mean(response == self.refrain_label),
            'precision': accuracy_score(ground_truth[np.where(response != self.refrain_label)], response[np.where(response != self.refrain_label)]),
        }

    def output_class(self):
        from pydantic import BaseModel, Field
        from typing import Optional
        class MultipleChoiceOutput(BaseModel):
            """Multiple choice output."""

            choice: Optional[str] = Field(
                description="Multiple choice answer. For example, if there is <answer>A</answer> in the prompt, the output should be 'A'."
            )
        return MultipleChoiceOutput