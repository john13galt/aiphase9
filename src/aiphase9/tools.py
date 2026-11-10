# This file contains all my agent tools.  It has 4 sections
#       Section 0:  Imports needed for the tools
#       Section 1:  Tool functions
#       Section 2:  Tool registry (dict of name-functions pairs)
#       Section 3:  Tool schema - list of dicts, with a dict for each tool

# ----------- Section 0:  Imports -----------
from datetime import datetime
import os
# using json_tricks instead of json to handle the numpy array in my data struction
import json_tricks
import json
from ollama import chat
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
NEW_MEMORIES_FILE = "new_memories.json"

def load_memories(filename):
    """ Now, memories is a list of dicts.  Each dict contains two entries:
           "text":  the raw sentence text
           "embeddings":  The list of floats (len=384) for embeddings
    """
    # opens a json file and creates a dict from it
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json_tricks.load(f)
    return []

def write_memories(memories, memoryfile):
    """ Function to write memories to a file"""
    with open(memoryfile, "w") as f:
        json_tricks.dump(memories, f, indent=2)

# This is cool.  We use a separate ollama chat call to perform an function.
# Instead of writing code, we let the model figure it out.
# Works by creating a detailed prompt telling the model what we want
# it to do and what we want it to return.
def old_classify_memory(new_memory, existing_memories):
    """ new_memory:  a single string
        existing_memories:  a list of strings
    """
    prompt = f"""
You are evaluating a proposed long-term memory.

NEW MEMORY:
{new_memory}

EXISTING MEMORIES:
{json.dumps(existing_memories, indent=2)}

Classify the relationship between the new memory and the
existing memories.

Use exactly one classification:

DUPLICATE:
Use only when the NEW MEMORY expresses substantially the same
factual claim as an EXISTING MEMORY.  This includes when the
NEW MEMORY factual claim is a generalization of an EXISTING MEMORY

UPDATE:
Use only when the NEW MEMORY explicitly corrects, replaces, 
supersedes, or clarifies an EXISTING MEMORY.  

A related subject matter is not sufficient.  A new preference 
does not update an ownership fact.  A new interest does not
update an ownership fact.  A possible future event does not
update update a current fact.  A more generalized factual claim
does not update an EXISTING MEMORY.

DISTINCT:
Use when NEW MEMORY is the same general subject as an EXISTING MEMORY 
but expresses a separate factual claim. 

NEW:
Use when NEW MEMORY expresses a factual claim that is not substantially
related to or the same general subject as any EXISTING MEMORY.

EVIDENCE REQUIREMENTS

- Evaluate the NEW MEMORY against each and every EXISTING
  MEMORY, individually.
- Base your decision only on the supplied EXISTING MEMORY.
- Do not invent causal relationships or additional facts.
- Do not infer that one fact supersedes another unless the
  new memory provides evidence of correction or replacement.
  Instead prefer DISTINCT and explain the ambiguity.
- When uncertain whether two facts conflict, do not assum UPDATE. 
  Instead prefer DISTINCT and explain the ambiguity.
- If a new memory is unrelated to every existing memory,
  classify it as NEW.

Return only a JSON object with these fields:
- classification: exactly one of DUPLICATE, UPDATE, DISTINCT, NEW
- existing_memory: the exact EXISTING MEMORY being duplicated
  or updated, or null if none
- reason: a short explanation
"""

    # Here's where we call the chat with the big prompt I made
    response = chat(
        model="gemma4:e4b",
        messages=[{"role": "user", "content": prompt}],
        format="json"
    )
    # json.loads converts a json formatted string into a dict object
    return json.loads(response.message.content)


# Like the regular "classify_memory" funciton, but
# only compares two memories at a time and provides
# a more explicity result
def classify_memory_relationship(new_memory, existing_memory):
    """ new_memory:  a single string
        existing_memory:  a single string
        returns SAME_FACT, CONTRADICTION, RELATED, UNRELATED, UNCERTAIN
    """
    prompt = f"""
You are classifying the relationship between two memory statements.

EXISTING MEMORY:
{existing_memory}

NEW MEMORY:
{new_memory}

Classify the relationship between two statements: the NEW MEMORY and the EXISTING MEMORY.

Use exactly one of these relationships:

SAME_FACT:
Both statements express essentially the same fact.
Different wording is acceptable.

CONTRADICTION:
The statements make incompatible claims about the same
attribute of the same subject.

RELATED:
The statements concern a related topic but express
different facts that can both be true.

UNRELATED:
The statements concern different topics.

UNCERTAIN:
There is insufficient information to determine the relationship.

Also determine whether the new statement explicitly replaces
or corrects the existing fact by treturning a boolean called
explicity_replacement.

Set explicit_replacement to true ONLY when the new statement
clearly indicates that the old fact should no longer be treated
as current. Examples include explicitly correcting a previous
statement or describing a change such as selling one car and
buying another.

A contradiction alone does NOT establish an explicit replacement.

Rules:
- Do not invent relationships that are not supported by the text.
- Do not assume that related facts replace one another.
- A preference is not the same as ownership.
- A broad category and a specific item are not automatically
  equivalent.
- Do not assume a name, location, or other attribute
  has changed merely because a different value is mentioned.
- If the evidence is ambiguous, use UNCERTAIN.
- Evaluate only the two statements provided.

Return only a JSON object with exactly these fields:
    "relationship": exacly one of SAME_FACT | CONTRADICTION | RELATED | UNRELATED | UNCERTAIN
    "explicit_replacement": a JSON boolean: true | false
    "evidence": a string that is a brief explanation based only on the two statements

"""

    response = chat(
        model="gemma4:e4b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        format="json"
    )

    return json.loads(response.message.content)

# Now, we're going to make a memory class to use embeddings to do semantic match instead of keyword lookup
class SemanticMemory:
    def __init__(self, memoryfile, embedding_model):
        self.memoryfile = memoryfile
        self.memories = load_memories(memoryfile)
        print(f"Read {len(self.memories)} memories from {memoryfile}")
        self.embedding_model = embedding_model
        # no longer need to do this... I create embeddings when I add 
        # self.embeddings = self.embedding_model.encode(self.memories, normalize_embeddings=True)

    # append new memory to memories and save it to disk
    def add(self, memory):
        self.memories.append({"text": memory, 
                              "embedding": self.embedding_model.encode(memory, normalize_embeddings=True)})
        write_memories(self.memories, self.memoryfile)

    # Use sementic search (using embeddings) to find an item in memory
    def search(self, query, topk = 3):
        # Create embeddings for new query
        query_embeddings = self.embedding_model.encode(query, normalize_embeddings=True)
        # Get a list of just the embeddings out of the memory
        memory_embeddings = [memory["embedding"] for memory in self.memories]
        # If there are memories...
        if memory_embeddings:
            # Calculate the semantic similarity of each memory with the query
            similarities = memory_embeddings @ query_embeddings # this is just like attention!
            # get the top"k" number of best semantic match results... returns indeces, sorted smallest to largest
            indeces = np.argsort(similarities)[-topk:]
            # returns a list of tuples.  Each tuple is a memory and the degree of similarity (a float)
            return [(self.memories[i]["text"], float(similarities[i])) for i in indeces]
        else:
            # otherwise, return an empty list
            return []

    def find_similar(self, query, threshold = 0.70):
        # Create embeddings for new query
        query_embeddings = self.embedding_model.encode(query, normalize_embeddings=True)
        # Get a list of just the embeddings out of the memory
        memory_embeddings = [memory["embedding"] for memory in self.memories]
        # If there are memories...
        if memory_embeddings:
            # Calculate the semantic similarity of each memory with the query
            similarities = memory_embeddings @ query_embeddings # this is just like attention!
            print(similarities)
            # get the top"k" number of best semantic match results... returns indeces, sorted smallest to largest
            index = np.argmax(similarities)
            print(index)
            # if the highest similarity is greater than my threshold
            if similarities[index] > threshold:
                # return it, and the similarity score
                return (self.memories[index]["text"], float(similarities[index]))
            else:
                # otherwise, return an empty list
                return ()
    
# Newer version - uses the SemanticMemory class
# Get sentence transformer (creates [384] embeddings from a sentence)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Make a memory object
memory = SemanticMemory(NEW_MEMORIES_FILE, embedding_model)

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
def make_save_memory(memory):
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
    "recall_memory": recall_memory,
    "save_memory": save_memory,
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
                        "description": "The first number. Must be a numeric value, not a string."
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number. Must be a numeric value, not a string"
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
            "description": "Get the current time.  Use this tool whenever the user asks for the current time.  Make sure to return the time to the user and not just tell them it depends on the system clock.",
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
            "description": "Use this function to store important information in a long-term persistent memory that lasts between chat sessions.  Store a concise, self-contained statement of a durable fact about the user. Include enough context that the memory makes sense when retrieved later. Do not store conversational filler.",
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

def main():

    test_memories = [
        "The user's name is Matthew.",
        "The user's favorite color is green.",
        "The user's car is a BMW 530e.",
        "The user lives in Texas."
    ]

    print(test_memories)
    userinput = ""
    while True:
        userinput = input("Check this in memory: ")
        if userinput == "/bye":
            break
        else:
            for memory in test_memories:
                result = classify_memory_relationship(userinput, test_memories)
            print(result)

if __name__ == "__main__":
    main()