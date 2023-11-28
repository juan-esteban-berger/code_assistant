# install dependencies
# !pip install transformers
# !pip install einops
# !pip install sentencepiece
# !pip install accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# Code Completion
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True).cuda()
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)

tokenizer.save_pretrained("/home/juanesh/Models/deepseek-coder-1.3b-base")

model.save_pretrained("/home/juanesh/Models/deepseek-coder-1.3b-base")
