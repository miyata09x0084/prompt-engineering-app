import warnings
warnings.filterwarnings("ignore")
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel
import sys
import os
import time
from statistics import mean
from utils import *

"""
Implements Chain of Thought with Prompt Caching for a food menu chatbot.
Scenario: "Answering questions about a restaurant menu using structured reasoning steps and caching common query patterns."

Workflow:
Input ("In") → User question about menu items, prices, or dietary options
LLM Call with Enhanced Prompting:
  - Uses detailed context and guidelines to exceed 1024 tokens for caching
  - Implements chain of thought reasoning with explicit steps
  - Tracks cache hits for performance optimization
Output ("Out") → Structured response with reasoning steps and cache statistics

The implementation combines:
1. Chain of Thought prompting for transparent reasoning
2. Prompt caching for improved response times
3. Cache hit tracking for performance monitoring
4. Structured outputs for consistent responses
"""

class MenuResponse(BaseModel):
    """Model for the chatbot's response"""
    reasoning_steps: List[str]
    final_response: str

# Delimiters for our prompts
menu_delimiter = "<menu_items>"
menu_delimiter_end = "</menu_items>"

# Get comprehensive menu data with detailed descriptions
food_items = get_menu_items()

def get_menu_response(query: str) -> Tuple[MenuResponse, Dict, float]:
    """
    Process a user query about the menu using chain of thought reasoning and caching.
    The system message and prompt are designed to exceed 1024 tokens for effective caching.
    Returns:
        Tuple containing (MenuResponse, cache_info dict, latency in seconds)
    """
    system_message = f"""
    You are an expert culinary consultant and menu (delimited by {menu_delimiter}{menu_delimiter_end}) specialist with deep knowledge of:
    
    1. Menu Organization and Structure:
       - Hierarchical menu categorization
       - Seasonal menu planning
       - Price point optimization
       - Dietary restriction labeling
       - Allergen identification
    
    2. Culinary Knowledge:
       - Ingredient combinations
       - Cooking techniques
       - Food safety guidelines
       - Dietary requirements
       - Allergen management
    
    3. Customer Service Excellence:
       - Clear communication
       - Dietary accommodation
       - Special request handling
       - Allergen awareness
       - Cultural sensitivity
    
    4. Quality Standards:
       - Portion control
       - Presentation guidelines
       - Temperature requirements
       - Storage protocols
       - Cross-contamination prevention
    
    5. Dietary Considerations:
       - Vegan/vegetarian options
       - Gluten-free alternatives
       - Allergen-free choices
       - Religious dietary laws
       - Nutritional information

    Here are the menu items:
    {menu_delimiter}
    {food_items}
    {menu_delimiter_end}
    
    Your task is to answer questions about our menu using a structured reasoning process:
    
    Step 1: Food Relevance Assessment
    - Determine if the query relates to food/menu items
    - Check query context and intent
    - Identify specific menu categories
    - Note any dietary restrictions
    - Consider allergen concerns
    
    Step 2: Menu Item Verification
    - Check item availability
    - Verify price accuracy
    - Confirm dietary status
    - Review included components
    - Note preparation methods
    
    Step 3: Response Formulation
    - Provide accurate information
    - Include relevant details
    - Suggest alternatives if needed
    - Address dietary concerns
    - Maintain professional tone

    The final response should be short and concise. 
    """

    # Detailed prompt that includes comprehensive context for better caching
    prompt = f"""
    Please analyze the following customer query about our menu offerings:
    Query: {query}
    """

    # Start timing
    start_time = time.time()
    
    # Get response with cache tracking
    response, cache_info = get_structured_output_with_cache_info(
        prompt, system_message, MenuResponse, "menu_query"
    )
    
    # Calculate latency
    latency = time.time() - start_time
    
    # Display cache information with latency
    display_cache_info("Menu Query", cache_info, latency)
    
    return response, cache_info, latency

def display_cache_info(query_type: str, cache_info: dict, latency: float = None):
    """Display cache hit information and latency in a readable format"""
    if not cache_info:
        print(f"❓ No cache information available for {query_type}")
        return
        
    total_tokens = cache_info.get('total_prompt_tokens', 0)
    cached_tokens = cache_info.get('cached_tokens', 0)
    cache_hit_ratio = cache_info.get('cache_hit_ratio', 0)
    is_cache_hit = cache_info.get('is_cache_hit', False)
    
    cache_status = "🟢 HIT" if is_cache_hit else "🔴 MISS"
    print(f"\n📊 Cache Status for {query_type}:")
    print(f"   {cache_status} | Total Tokens: {total_tokens} | Cached Tokens: {cached_tokens} | Hit Ratio: {cache_hit_ratio:.2%}")
    if latency is not None:
        print(f"   ⏱️  Response Time: {latency:.2f} seconds")

def run_demo():
    """Run a demonstration of the prompt caching feature"""
    print("🤖 Welcome to the Menu Assistant!")
    print("=" * 70)
    
    # Test questions that demonstrate both reasoning and caching
    test_questions = [
        "What is the price of the Mini Cheeseburger?",  
        "Do you have any vegan options?",               
        "What's your most popular dish?",               
        "What is the price of the Calamari?",           
        "What's the cheapest appetizer?",              
    ]
    
    # Store latencies for comparison
    first_run_latencies = []
    second_run_latencies = []
    
    print("\n🔄 FIRST RUN - Expect cache misses")
    print("=" * 70)
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        response, cache_info, latency = get_menu_response(question)
        first_run_latencies.append(latency)
        if response:
            print("\n🤔 Reasoning Steps:")
            for i, step in enumerate(response.reasoning_steps, 1):
                print(f"{i}. {step}")
            print(f"\n💬 Response: {response.final_response}")
        print("-" * 70)
    
    print("\n\n🔄 SECOND RUN - Expect cache hits")
    print("=" * 70)
    print("\n🧪 Running the same queries again to demonstrate cache hits...")
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        response, cache_info, latency = get_menu_response(question)
        second_run_latencies.append(latency)
        if response:
            print("\n🤔 Reasoning Steps:")
            for i, step in enumerate(response.reasoning_steps, 1):
                print(f"{i}. {step}")
            print(f"\n💬 Response: {response.final_response}")
        print("-" * 70)
    
    # Display latency comparison
    avg_first_run = mean(first_run_latencies)
    avg_second_run = mean(second_run_latencies)
    improvement = ((avg_first_run - avg_second_run) / avg_first_run) * 100
    
    print("\n⏱️  LATENCY COMPARISON")
    print("=" * 70)
    print(f"Average First Run (Potentially with missed cache hits): {avg_first_run:.2f} seconds")
    print(f"Average Second Run (Mostly cache hits): {avg_second_run:.2f} seconds")
    print(f"Performance Improvement: {improvement:.1f}%")

if __name__ == "__main__":
    run_demo()
