import math
import os
import json
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from safetensors.torch import save_file, load_file
#from safetensors import safe_open
from ollama import chat
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# import torchvision
import time
# from torchvision.datasets import MNIST
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
import numpy as np
import tools

print("\n")
print("--------------------------------------------------------")
print("-"*10, "Lesson 56-59 - Tool Use & exception handling", "-"*10)
print("--------------------------------------------------------")

# Note:  I have imported stuff from tools.py:
#        Tool function definitions ... called by accessing the tool_registry
#        tool_registry (dict of function_name_string -> function pairs)
#        tool_list (list of dicts containing a schema for all my tool functions)

def start_agent(tool_registry, tool_list, CONTEXTWINDOW):
    # start a fresh conversation history
    messages = []
    # Prompt loop... keep prompting the user until they say "/bye"
    while True:
        # Get user input
        userinput = input("You: ")
        if userinput == "/bye": break

        # make a prompt
        messages.append({"role":"user", "content":userinput})

        # now start another loop to process the prompt... keep looping until we stop getting tool calls
        while True:
            # call cat with messages and send it "tools"
            response = chat(model="llama3.1:8b", 
                            messages=messages, 
                            options={"num_ctx":CONTEXTWINDOW},
                            tools=tool_list,
                            )

            # check to see if the response is a tool call
            if not response.message.tool_calls:
                # just a regular chat response, print it and append it to messages 
                print("Chat response: ",response.message.content)
                messages.append({"role": "assistant", "content":response.message.content})
                # break out of this loop because I got a non-tool response.  Go get another prompt!
                break
            else:
                # Add the assistant's request to the conversation (adding the whole thing)
                messages.append(response.message)

                # Loop through all the tool_calls... the LLM can return more than one.
                for tool_call in response.message.tool_calls:
                    # check what tool it is... only process valid tool calls
                    if tool_call.function.name in tool_registry:
                        # call the tool... inside a "try-except" to catch errors
                        try: 
                            # Call the tool (use the registry  to find tools)
                            print(f"Tool call: {tool_call.function.name}: {tool_call.function.arguments}")
                            result = tool_registry[tool_call.function.name](**(tool_call.function.arguments))
                            print("Tool result: ", result)
                        except Exception as e:
                            # if the tools errors, still create a "result" to add to the message history
                            print(f"Tool error: {e}")
                            result = f"Error: {e}"

                        # Append tool call result to messages... note the role is "tool" and provide "tool_name", also
                        messages.append({
                            "role": "tool",
                            "tool_name": tool_call.function.name,
                            "content": str(result),
                        })
                    else:
                        # really shouldn't get here.  It should only call tools I defined
                        print("illegal function")
                        break


print("\n")
print("--------------------------------------------------------")
print("-"*10, "Lesson 60 - Memory", "-"*10)
print("--------------------------------------------------------")


# saves a dict "memory" into a json file
def save_memory(memory):

    with open(tools.MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
        
memory = {}
memory["name"] = "Matthew"
memory["color"] = "green"
memory["gpu"] = "RTX 5060"
memory["car"] = "BMW 530e"
save_memory(memory)


CONTEXTWINDOW = 8192
# start_agent(tools.tool_registry, tools.tool_list, CONTEXTWINDOW)


print("\n")
print("--------------------------------------------------------")
print("-"*10, "Lesson 61 - Symantic Search of Memory with Embeddings", "-"*10)
print("--------------------------------------------------------")

def save_memories(memories):
    with open(tools.MEMORIES_FILE, "w") as f:
        for memory in memories:
            f.write(f"{memory}\n")

memories = [
    "Matthew's name is Matthew.",
    "Matthew's favorite color is green.",
    "Matthew drives a BMW 530e.",
    "Matthew uses an NVIDIA RTX 5060 GPU."
]
save_memories(memories)

'''
def load_memories():
    # opens a text file with memory sentences, one per line
    if os.path.exists(tools.MEMORIES_FILE):
        with open(tools.MEMORIES_FILE, "r") as f:
            return [line.strip() for line in f]
    return []

memories = load_memories()
'''

# Now, we're going to make the memory use embeddings to do semantic match instead of keyword lookup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

memory_embeddings = embedding_model.encode(memories)
print(memory_embeddings.shape)

query = "What kind of computer hardware do I use?"
query_embeddings = embedding_model.encode(query)

similarities = memory_embeddings @ query_embeddings / (
    np.linalg.norm(memory_embeddings, axis=1) * np.linalg.norm(query_embeddings)
)

for memory, score in zip(memories, similarities):
    print(f"{score}: {memory}")

best = np.argmax(similarities)
print(f"Best = {memories[best]}")
print()
print()

start_agent(tools.tool_registry, tools.tool_list, CONTEXTWINDOW)
