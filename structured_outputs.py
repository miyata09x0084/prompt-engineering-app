import warnings
warnings.filterwarnings("ignore")
import sys
import os
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from utils import * 

"""
Demonstrates the implementation of structured outputs for LLMs using a food chatbot as an example.
This script shows how to:
1. Define schema structures with Pydantic
2. Use structured outputs with OpenAI
3. Process structured responses in a food recommendation context
4. Handle different structured outputs for various use cases

Workflow:
1. Input ("In") → Customer food-related queries handled by process_food_query()
2. Query Analysis → Determine query intent with analyze_query()
3. Schema Selection → Choose appropriate schema based on intent
4. LLM Call with Schema → Get structured response using the selected schema
5. Response Processing → Display and potentially use the structured data
"""

# Pydantic models for our structured outputs
class Ingredient(BaseModel):
    name: str
    quantity: str
    substitutes: Optional[List[str]] = None

class Recipe(BaseModel):
    name: str
    cuisine: str
    prep_time_minutes: int
    cook_time_minutes: int
    serving_size: int
    ingredients: List[Ingredient]
    instructions: List[str]
    dietary_info: List[str]

class Restaurant(BaseModel):
    name: str
    cuisine: str
    price_range: str  # "$", "$$", "$$$" or "$$$$"
    location: str
    rating: float  # 1.0 to 5.0
    popular_dishes: List[str]
    dietary_options: List[str]

class NutritionInfo(BaseModel):
    food_name: str
    serving_size: str
    calories: int
    protein_grams: float
    carbs_grams: float
    fat_grams: float
    fiber_grams: float
    vitamins: List[str]
    minerals: List[str]

class QueryClassification(BaseModel):
    query_type: str  # RECIPE, RESTAURANT, NUTRITION, or OTHER

def analyze_query(user_query):
    """Analyze the query to determine its intent using structured outputs"""
    system_message = """
    You are a food query analyzer. Classify the user's food-related query into one of these categories:
    - RECIPE: Request for a recipe or cooking instructions
    - RESTAURANT: Request for restaurant recommendations
    - NUTRITION: Request for nutritional information
    - OTHER: Any other food-related query
    
    Provide your classification as the query_type field.
    """

    result = get_structured_output(user_query, system_message, QueryClassification, "query classification")
    
    if result:
        return result.query_type
    else:
        # Fallback to a safe default if structured output fails
        return "OTHER"

def get_structured_recipe(query):
    """Get a structured recipe response for a recipe query"""
    system_message = """
    You are a helpful cooking assistant. Provide a detailed recipe based on the user's request.
    
    Make sure to include all required fields:
    - name: The name of the recipe
    - cuisine: The type of cuisine
    - prep_time_minutes: Preparation time in minutes
    - cook_time_minutes: Cooking time in minutes
    - serving_size: Number of servings
    - ingredients: List of ingredients with name, quantity, and optional substitutes
    - instructions: Step-by-step cooking instructions
    - dietary_info: List of dietary information (e.g., "vegetarian", "gluten-free")
    """

    return get_structured_output(query, system_message, Recipe, "recipe")

def get_structured_restaurant_recommendations(query):
    """Get structured restaurant recommendations"""
    system_message = """
    You are a restaurant recommendation assistant. Provide restaurant suggestions based on the user's request.
    
    Make sure to include all required fields:
    - name: Restaurant name
    - cuisine: Type of cuisine
    - price_range: Price level as "$", "$$", "$$$" or "$$$$"
    - location: City or neighborhood
    - rating: Rating from 1.0 to 5.0
    - popular_dishes: List of the restaurant's popular dishes
    - dietary_options: List of available dietary options (e.g., "vegetarian", "vegan", "gluten-free")
    """

    return get_structured_output(query, system_message, Restaurant, "restaurant recommendation")

def get_structured_nutrition_info(query):
    """Get structured nutritional information"""
    system_message = """
    You are a nutrition information assistant. Provide detailed nutritional data for the food item requested.
    
    Make sure to include all required fields:
    - food_name: Name of the food
    - serving_size: Standard serving size
    - calories: Calories per serving
    - protein_grams: Protein content in grams
    - carbs_grams: Carbohydrate content in grams
    - fat_grams: Fat content in grams
    - fiber_grams: Fiber content in grams
    - vitamins: List of vitamins present in significant amounts
    - minerals: List of minerals present in significant amounts
    """

    return get_structured_output(query, system_message, NutritionInfo, "nutrition information")

def get_unstructured_response(query):
    """Get a regular unstructured response for other food-related queries"""
    system_message = """
    You are a helpful food assistant. Answer the user's food-related query with useful information.
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with unstructured response: {e}")
        return "Sorry, I couldn't process your request at this time."

def process_food_query(user_query):
    """Main function to process food-related queries with structured outputs"""
    print(f"Processing query: '{user_query}'")
    
    # Analyze the query intent
    query_type = analyze_query(user_query)
    print(f"Query classified as: {query_type}")
    
    # Get appropriate structured response based on intent
    if query_type == "RECIPE":
        response = get_structured_recipe(user_query)
        # Example of how you might use the structured data
        if response:
            print("\n✓ Received structured recipe data")
            return format_recipe_display(response)
        
    elif query_type == "RESTAURANT":
        response = get_structured_restaurant_recommendations(user_query)
        if response:
            print("\n✓ Received structured restaurant data")
            return format_restaurant_display(response)
        
    elif query_type == "NUTRITION":
        response = get_structured_nutrition_info(user_query)
        if response:
            print("\n✓ Received structured nutrition data")
            return format_nutrition_display(response)
        
    else:
        # For other queries, use unstructured response
        print("\n✓ Using unstructured response for general query")
        return get_unstructured_response(user_query)
    
    # Fallback if structured parsing failed
    return get_unstructured_response(user_query)

# function to help display recipes
def format_recipe_display(recipe):
    """Format the structured recipe data for display"""
    display = f"# {recipe.name}\n\n"
    display += f"**Cuisine:** {recipe.cuisine}\n"
    display += f"**Prep Time:** {recipe.prep_time_minutes} minutes\n"
    display += f"**Cook Time:** {recipe.cook_time_minutes} minutes\n"
    display += f"**Servings:** {recipe.serving_size}\n\n"
    
    display += "## Ingredients\n\n"
    for ingredient in recipe.ingredients:
        display += f"- {ingredient.quantity} {ingredient.name}"
        if ingredient.substitutes:
            display += f" (Substitutes: {', '.join(ingredient.substitutes)})"
        display += "\n"
    
    display += "\n## Instructions\n\n"
    for i, step in enumerate(recipe.instructions, 1):
        display += f"{i}. {step}\n"
    
    display += f"\n**Dietary Information:** {', '.join(recipe.dietary_info)}\n"
    
    return display

# function to help display restaurants
def format_restaurant_display(restaurant):
    """Format the structured restaurant data for display"""
    display = f"# {restaurant.name}\n\n"
    display += f"**Cuisine:** {restaurant.cuisine}\n"
    display += f"**Price Range:** {restaurant.price_range}\n"
    display += f"**Location:** {restaurant.location}\n"
    display += f"**Rating:** {'★' * int(restaurant.rating)}{' ☆' * (5 - int(restaurant.rating))} ({restaurant.rating}/5)\n\n"
    
    display += "## Popular Dishes\n\n"
    for dish in restaurant.popular_dishes:
        display += f"- {dish}\n"
    
    display += f"\n**Dietary Options:** {', '.join(restaurant.dietary_options)}\n"
    
    return display

# function to help display nutrition information
def format_nutrition_display(nutrition):
    """Format the structured nutrition data for display"""
    display = f"# Nutritional Information: {nutrition.food_name}\n\n"
    display += f"**Serving Size:** {nutrition.serving_size}\n\n"
    
    display += "## Macronutrients\n\n"
    display += f"- **Calories:** {nutrition.calories} kcal\n"
    display += f"- **Protein:** {nutrition.protein_grams}g\n"
    display += f"- **Carbohydrates:** {nutrition.carbs_grams}g\n"
    display += f"- **Fat:** {nutrition.fat_grams}g\n"
    display += f"- **Fiber:** {nutrition.fiber_grams}g\n\n"
    
    display += "## Micronutrients\n\n"
    display += f"**Vitamins:** {', '.join(nutrition.vitamins)}\n"
    display += f"**Minerals:** {', '.join(nutrition.minerals)}\n"
    
    return display

# generate the json schemas
def generate_json_schema_docs(pydantic_model):
    """Generate JSON schema documentation for a Pydantic model
    
    Args:
        pydantic_model: The Pydantic model class to document
        
    Returns:
        A tuple containing (model_name, schema_json, example_dict)
    """
    model_name = pydantic_model.__name__
    schema_json = pydantic_model.schema_json(indent=2)
    
    # Create an example instance (this is a simplification and may not work for all models)
    # In a real implementation, you would create specific examples for each model
    example = {}
    for field_name, field in pydantic_model.__annotations__.items():
        if field_name == "ingredients" and model_name == "Recipe":
            example[field_name] = [{"name": "Flour", "quantity": "2 cups", "substitutes": ["Almond flour", "Coconut flour"]}]
        elif field_name == "popular_dishes" and model_name == "Restaurant":
            example[field_name] = ["Signature Pasta", "Chocolate Cake"]
        elif field_name == "vitamins" and model_name == "NutritionInfo":
            example[field_name] = ["Vitamin A", "Vitamin C"]
        elif field_name == "minerals" and model_name == "NutritionInfo":
            example[field_name] = ["Calcium", "Iron"]
        elif field_name == "instructions" and model_name == "Recipe":
            example[field_name] = ["Step 1: Mix ingredients", "Step 2: Bake for 20 minutes"]
        elif field_name == "dietary_info" and model_name == "Recipe":
            example[field_name] = ["Vegetarian", "Gluten-free"]
        elif field_name == "dietary_options" and model_name == "Restaurant":
            example[field_name] = ["Vegetarian", "Vegan options"]
        elif field_name == "query_type" and model_name == "QueryClassification":
            example[field_name] = "RECIPE"
        elif field_name in ["name", "cuisine", "food_name"]:
            example[field_name] = f"Example {field_name}"
        elif field_name == "price_range":
            example[field_name] = "$$$"
        elif field_name == "location":
            example[field_name] = "New York, NY"
        elif field_name == "rating":
            example[field_name] = 4.5
        elif field_name == "serving_size" and model_name == "NutritionInfo":
            example[field_name] = "100g"
        elif field_name == "serving_size" and model_name == "Recipe":
            example[field_name] = 4
        elif "_minutes" in field_name:
            example[field_name] = 30
        elif "_grams" in field_name:
            example[field_name] = 10.5
        elif field_name == "calories":
            example[field_name] = 250
    
    return model_name, schema_json, example

# function to display json schemas
def show_json_schema():
    """Display JSON schema examples for our Pydantic models"""
    print("\nJSON Schema Documentation")
    print("=" * 80)
    
    models = [Recipe, Restaurant, NutritionInfo, QueryClassification]
    
    for model in models:
        model_name, schema, example = generate_json_schema_docs(model)
        
        print(f"\n## {model_name} Schema:")
        print("\nJSON Schema:")
        print(schema)
        
        print("\nExample Object:")
        print(json.dumps(example, indent=2))
        print("-" * 60)
    
    print("=" * 80)

if __name__ == "__main__":
    # Example queries to test the system
    test_queries = [
        "How do I make a vegetarian lasagna?",  # Recipe
        "Can you recommend some good Italian restaurants in New York?",  # Restaurant
        "What's the nutritional information for an avocado?",  # Nutrition
        "What foods are in season during summer?",  # Other
    ]
    
    print("\nTesting Food Chatbot with Structured Outputs")
    print("=" * 80)
    
    for query in test_queries:
        print("\nTest Query:", query)
        print("-" * 60)
        
        response = process_food_query(query)
        print("\nResponse:")
        print(response)
        print("=" * 80)
    
    # Show JSON schema examples
    show_json_schema()
