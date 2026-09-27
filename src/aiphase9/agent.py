import math
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from safetensors.torch import save_file, load_file
#from safetensors import safe_open
from ollama import chat
from transformers import AutoTokenizer

# import torchvision
import time
# from torchvision.datasets import MNIST
# from torchvision import transforms
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
import numpy as np
from tools import tool_list

print("\n")
print("--------------------------------------------------------")
print("-"*10, "Lesson 56 - Tool Use", "-"*10)
print("--------------------------------------------------------")

def calculator(a, b, operation):
    if operation == "add" or operation == "+":
        return a + b
    elif operation == "subtract" or operation == "-":
        return a - b
    elif operation == "multiply" or operation == "*":
        return a * b
    elif operation == "divide" or operation == "/":
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")

tools = tool_list

messages = []
CONTEXTWINDOW = 8192

# make a prompt
messages.append({"role":"user", "content":"What is 3.8 times 4.9?"})

# call cat with messages and send it "tools"
response = chat(model="qwen3:0.6b", 
                messages=messages, 
                options={"num_ctx":CONTEXTWINDOW},
                tools=tools,
                )

# check to see if the response is a tool call
if not response.message.tool_calls:
    # just a regular chat response, print it and append it to messages 
    print(response.message.content)
    messages.append({"role": "assistant", "content":response.message.content})
else:
    # a tool call... get the tool info
    tool_response = response.message.tool_calls[0]
    print(tool_response.function.name)
    print(tool_response.function.arguments)

    # check what tool it is... only process valid tool calls
    if tool_response.function.name == "calculator":
        # call the tool
        result = calculator(**(tool_response.function.arguments))
        print(result)

        # Append tool call result to messages... note the role is "tool" and provide "tool_name", also
        messages.append({
            "role": "tool",
            "tool_name": tool_call.function.name,
            "content": str(result),
        })
    else:
        print("illegal function")

response = chat(model="qwen3:0.6b", 
                messages=messages, 
                options={"num_ctx":CONTEXTWINDOW},
                tools=tools,
                )

print(response.message.content)

