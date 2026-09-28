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