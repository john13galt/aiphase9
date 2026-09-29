# This file contains all my agent tools.  It has 3 sections
#       Section 1:  Tool functions
#       Section 2:  Tool registry (dict of name-functions pairs)
#       Section 3:  Tool schema - list of dicts, with a dict for each tool

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

# ----------- Section 2:  Tool registry -----------
tool_registry = {
    "calculator": calculator,
    "get_time": get_time,
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
    }
]