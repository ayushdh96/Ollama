import ollama
import os

model="llama3.2"

input_data = './data/names.txt'
output_data = './data/output.txt'

with open(input_data, 'r') as file:
    with open(input_data, 'r') as file:
        names = [line.strip() for line in file.readlines()]

prompt=f'Give me a berif information about the names of player mentioned. Group them according to their sport and also mention who is the best in their sport from players within the list.{names}'

response=ollama.generate(model=model, prompt=prompt)
print(response['response'])