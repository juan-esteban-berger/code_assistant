# install dependencies
# !pip install transformers
# !pip install einops
# !pip install sentencepiece
# !pip install accelerate

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

def assist_load():
	# Code Completion
	tokenizer = AutoTokenizer.from_pretrained("/home/juanesh/Models/deepseek-coder-1.3b-base", local_files_only=True)
	# model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True).cuda()
	model = AutoModelForCausalLM.from_pretrained("/home/juanesh/Models/deepseek-coder-1.3b-base", local_files_only=True)
	return tokenizer, model

def assist(input_text, tokenizer, model, new_tokens=140):
	inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
	outputs = model.generate(**inputs, max_new_tokens=new_tokens)
	print(tokenizer.decode(outputs[0], skip_special_tokens=True))
