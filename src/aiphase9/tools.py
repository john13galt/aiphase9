# This file contains all my agent tools.  It has 4 sections
#       Section 0:  Imports needed for the tools
#       Section 1:  Tool functions
#       Section 2:  Tool registry (dict of name-functions pairs)
#       Section 3:  Tool schema - list of dicts, with a dict for each tool

# ----------- Section 0:  Imports -----------
from datetime import datetime
import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# ----------- Section 1:  Tool functions -----------
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

def get_time():
    return f"The current time is {datetime.now().time()}."

# this next information is not an actual tool, but a helper function for the recall_memory tool
MEMORY_FILE = "memory.json"
MEMORIES_FILE = "memories.txt"
def load_memory():
    # opens a json file and creates a dict from it
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def load_memories():
    # opens a text file with memory sentences, one per line
    if os.path.exists(MEMORIES_FILE):
        with open(MEMORIES_FILE, "r") as f:
            return f.readlines()
    return []

# Now, we're going to make the memory use embeddings to do semantic match instead of keyword lookup
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Older version - uses a dict memory
def recall_memory_old(query):
    memory = load_memory()
    print(memory)
    results = []
    query_words = query.lower().split()
    print(query_words)
    for key, value in memory.items():
        text = f"{key} {value}".lower()
        for word in query_words:
            if word in text:
                results.append((key, value))
                break
    return results

def recall_memory(query):
    # read memories from a text file
    memories = load_memories()
    # print(memory)
    # create embeddings for all memories (use sentence embeddings)
    memory_embeddings = embedding_model.encode(memories)
    # create embeddings for the query
    query_embeddings = embedding_model.encode(query)
    # calculate similarities - use "cosine similarity"
    similarities = memory_embeddings @ query_embeddings / (
        np.linalg.norm(memory_embeddings, axis=1) * np.linalg.norm(query_embeddings)
    )
    # get the highest similarity
    best = np.argmax(similarities)
    print(memories[best])
    return memories[best]

# ----------- Section 2:  Tool registry -----------
tool_registry = {
    "calculator": calculator,
    "get_time": get_time,
    "recall_memory": recall_memory
}

# ----------- Section 3:  Tool schema -----------
# Gives me the schema/descriptions for all my tools (a list of dicts)
tool_list = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a basic arithmetic operation on two numbers.  Use this tool instead of calculating yourself. Handles diviision by zero.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number."
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number."
                    },
                    "operation": {
                        "type": "string",
                        "enum": [
                            "add", "+",
                            "subtract", "-",
                            "multiply", "*",
                            "divide", "/"
                        ],
                        "description": "The arithmetic operation to perform."
                    }
                },
                "required": ["a", "b", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current time.  Use this tool whenever the user asks for the current time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Use this function to query a persistent memory with facts about the user, such as facts from a previous conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A description of the information you want to find in memory."
                    },
                },
                "required": ["query"]
            }
        }
    },
]