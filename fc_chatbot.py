import openai
import json
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

# Menu data
MENU_ITEMS = {
    "Mini Cheeseburger": {"price": 6.99, "category": "Kids Menu", "vegan": False},
    "Loaded Potato Skins": {"price": 8.99, "category": "Appetizers", "vegan": False},
    "Bruschetta": {"price": 7.99, "category": "Appetizers", "vegan": True},
    "Grilled Chicken Caesar Salad": {"price": 12.99, "category": "Main Menu", "vegan": False},
    "Classic Cheese Pizza": {"price": 10.99, "category": "Main Menu", "vegan": False},
    "Spaghetti Bolognese": {"price": 14.99, "category": "Main Menu", "vegan": False},
    "Veggie Wrap": {"price": 9.99, "category": "Vegan Options", "vegan": True},
    "Vegan Beyond Burger": {"price": 11.99, "category": "Vegan Options", "vegan": True},
    "Chocolate Lava Cake": {"price": 6.99, "category": "Desserts", "vegan": False},
    "Fresh Berry Parfait": {"price": 5.99, "category": "Desserts", "vegan": True}
}

def calculate_total(items: List[str]) -> Dict:
    """Calculate the total price for the given items"""
    total = 0
    found_items = {}
    not_found_items = []
    
    for item in items:
        if item in MENU_ITEMS:
            total += MENU_ITEMS[item]["price"]
            found_items[item] = {
                "price": MENU_ITEMS[item]["price"],
                "category": MENU_ITEMS[item]["category"],
                "vegan": MENU_ITEMS[item]["vegan"]
            }
        else:
            not_found_items.append(item)
    
    return {
        "total": round(total, 2),
        "found_items": found_items,
        "not_found_items": not_found_items
    }

def chat_completion(messages: List[Dict]) -> str:
    """Get chat completion from OpenAI API"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculate_total",
                    "description": "Calculate the total price for a list of menu items",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of exact menu item names to calculate total for"
                            }
                        },
                        "required": ["items"]
                    }
                }
            }
        ]
    )
    return response.choices[0].message

def handle_function_calls(message, messages):
    """Handle function calls from the assistant"""
    if not message.tool_calls:
        return message.content
    
    print(message.tool_calls)
    function_response = []
    
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        if function_name == "calculate_total":
            print("Calling Tool: calculate_total()...")
            function_response.append(calculate_total(**function_args))
    
    # Add the function response to messages
    messages.append({
        "role": "function",
        "name": message.tool_calls[0].function.name,
        "content": json.dumps(function_response)
    })
    
    # Get the final response from the assistant
    final_response = chat_completion(messages)
    return final_response.content

def format_menu_for_prompt():
    """Format the menu items for inclusion in the system prompt"""
    menu_text = "MENU ITEMS:\n"
    
    categories = {}
    for item, details in MENU_ITEMS.items():
        category = details["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(
            f"- {item}: ${details['price']:.2f} {'(Vegan)' if details['vegan'] else ''}"
        )
    
    for category, items in categories.items():
        menu_text += f"\n{category}:\n"
        menu_text += "\n".join(items) + "\n"
    
    return menu_text

def main():
    # Format menu for inclusion in the prompt
    menu_text = format_menu_for_prompt()
    
    messages = [
        {
            "role": "system",
            "content": f"""You are a friendly restaurant chatbot that helps customers with their orders.
            Below is our complete menu. Please use these exact item names when calling the calculate_total function.
            
            {menu_text}
            
            Instructions for handling orders:
            1. When a customer wants to order items, identify which menu items they're referring to.
            2. Call the calculate_total function with the EXACT menu item names from above.
            3. Present the order clearly with individual items and prices, followed by the total.
            
            If a customer mentions an item not on our menu (like "hamburger" instead of "Mini Cheeseburger"), 
            use your judgment to match it to the closest menu item. Then use that exact menu item name 
            when calling the calculate_total function.
            """
        }
    ]
    
    print("🤖 Welcome to our restaurant! I can help you view our menu and place orders.")
    print("Type 'quit' to exit the chat.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            print("🤖 Thank you for visiting! Have a great day!")
            break
        
        messages.append({"role": "user", "content": user_input})
        response = chat_completion(messages)
        
        if response.tool_calls:
            bot_response = handle_function_calls(response, messages)
        else:
            bot_response = response.content
            
        messages.append({"role": "assistant", "content": bot_response})
        print(f"\n🤖: {bot_response}")

if __name__ == "__main__":
    main()
