tool_list = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform a basic arithmetic operation.",
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
    }
]