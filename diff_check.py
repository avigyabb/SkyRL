import yaml

log1 = """
train_batch_size: 32
policy_mini_batch_size: 32
n_samples_per_prompt: 8
"""

log2 = """
train_batch_size: 4
policy_mini_batch_size: 4
n_samples_per_prompt: 2
"""

# I am just mocking the diff check because I can see the text.
# But I want to parse the full blocks provided in the prompt to be sure.
# Let's paste the full content from the prompt into files.
