from code_assist import assist_load, assist

tokenizer, model = assist_load()

input = '''
df = pd.read_csv("titanic.csv")
# Group by age get sum of fare
'''

assist(input, tokenizer, model, 140)
