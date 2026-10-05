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

def load_memories(MEMORIES_FILE):
    # opens a text file with memory sentences, one per line
    if os.path.exists(MEMORIES_FILE):
        with open(MEMORIES_FILE, "r") as f:
            return [line.strip() for line in f]
    return []

# Now, we're going to make a memory class to use embeddings to do semantic match instead of keyword lookup
class SemanticMemory:
    def __init__(self, memoryfile, embedding_model):
        self.memories = load_memories(memoryfile)
        self.embedding_model = embedding_model
        self.embeddings = self.embedding_model.encode(self.memories, normalize_embeddings=True)

    def search(self, query, topk = 3):
        query_embeddings = self.embedding_model.encode(query)
        similarities = self.embeddings @ query_embeddings # this is just like attention!
        # get the top"k" number of semantic match results... returns indeces, sorted smallest to largest
        indeces = np.argsort(similarities)[-topk:]
        # returns a list of tuples.  Each tuple is a memory and the degree of similarity (a float)
        return [(self.memories[i], float(similarities[i])) for i in indeces]


# Older version - uses a dict memory
'''
def recall_memory_old(query):
    memory = load_memory()
    # print(memory)
    results = []
    query_words = query.lower().split()
    # print(query_words)
    for key, value in memory.items():
        text = f"{key} {value}".lower()
        for word in query_words:
            if word in text:
                results.append((key, value))
                break
    return results
'''
    
# Newer version - uses the SemanticMemory class
# Get sentence tranferor
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Make a memory object
memory = SemanticMemory(MEMORIES_FILE, embedding_model)

# This function makes a "recall_memory" function with my memory object embedded in it!
def make_recall_memory(memory):
    # Create the function
    def recall_memory(query):
        return memory.search(query, topk=3)
    # and return it
    return recall_memory

# Then this creates an instance of my funciton with my specific memory
recall_memory = make_recall_memory(memory)

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