import ollama

response=ollama.list()

#print(response)

res=ollama.chat(
    model="llama3.2",
    messages=[{
        "role":"user","content":"Say my name"},
        ],
        stream=True
)
#print(res["message"]["content"])
for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)