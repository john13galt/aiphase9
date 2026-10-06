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
NEW_MEMORIES_FILE = "new_memories.json"

''' # old vesion, simple dict
def load_memory():
    # opens a json file and creates a dict from it
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}
'''

''' # old version, simple list of strings
def load_memories(MEMORIES_FILE):
    # opens a text file with memory sentences, one per line
    if os.path.exists(MEMORIES_FILE):
        with open(MEMORIES_FILE, "r") as f:
            lines = [line.strip() for line in f]
            return []
    return []
'''

# Now, memories is a list of dicts.  Each dict contains two entries:
#       "text":  the raw sentence text
#       "embeddings":  The list of floats (len=384) for embeddings
def load_memories():
    # opens a json file and creates a dict from it
    if os.path.exists(NEW_MEMORIES_FILE):
        with open(NEW_MEMORIES_FILE, "r") as f:
            return json.load(f)
    return []

def write_memories(memories):
    with open(tools.NEW_MEMORIES_FILE, "w") as f:
        json.dump(memories, f, indent=2)
    
# Now, we're going to make a memory class to use embeddings to do semantic match instead of keyword lookup
class SemanticMemory:
    def __init__(self, memoryfile, embedding_model):
        self.memories = load_memories(memoryfile)
        self.embedding_model = embedding_model
        # no longer need to do this... I create embeddings when I add 
        # self.embeddings = self.embedding_model.encode(self.memories, normalize_embeddings=True)

    def __del__(self):
        print(f"Saving memories to {NEW_MEMORIES_FILE}")
        write_memories()

    def add(self, memory):
        self.memories.append({"text": memory, 
                              "embedding": self.embedding_model.encode(self.memory, normalize_embeddings=True)})

    def search(self, query, topk = 3):
        query_embeddings = self.embedding_model.encode(query, normalize_embeddings=True)
        memory_embeddings = [memory["embedding"] for memory in self.memories]
        similarities = memory_embeddings @ query_embeddings # this is just like attention!
        # get the top"k" number of semantic match results... returns indeces, sorted smallest to largest
        indeces = np.argsort(similarities)[-topk:]
        # returns a list of tuples.  Each tuple is a memory and the degree of similarity (a float)
        return [(self.memories[i]["text"], float(similarities[i])) for i in indeces]


'''  # Older version - uses a dict memory
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

# similarly, this functions makes a save_memory function with my memory objsect embedded
def make_save_memory(memory)
    def save_memory(text):
        return memory.add(text)
    # and return it
    return save_memory

# Then this creates the instance for me to call
save_memory = make_save_memory(memory)

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
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "Use this function to store important information in a long-term persistent memory that lasts between chat sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "A description of the information you want to store in memory."
                    },
                },
                "required": ["text"]
            }
        }
    },
]