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

messages.append({"role":"user", "content":"What is 3.8 times 4.9"})

response = chat(model="qwen3:0.6b", 
                messages=messages, 
                options={"num_ctx":CONTEXTWINDOW},
                tools=tools,
                )

print(messages)
print()
print(tools)
print()
print(response)