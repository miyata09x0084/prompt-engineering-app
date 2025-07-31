import warnings
warnings.filterwarnings("ignore")
from typing import List, Dict, Tuple, Optional, Any
import sys
import os
import time
import json
import pandas as pd
import ast
from statistics import mean
import csv
from utils import get_chat_completion, get_reasoning_response

"""
Implements LLM-as-a-judge workflow with metaprompting for iterative prompt improvement
This script:
1. Uses GPT-4o to generate predictions for ML paper tagging with a few-shot prompt
2. Evaluates the results using o3-mini as a judge
3. Uses metaprompting to improve the initial prompt based on evaluations
4. Iterates through this process until results regress
5. Saves the final best prompt

The evaluation uses a validation dataset of ML paper abstracts and their gold labels
"""

# Delimiters for our prompts
abstract_delimiter = "<abstract>"
abstract_delimiter_end = "</abstract>"

prediction_delimiter = "<prediction>"
prediction_delimiter_end = "</prediction>"

gold_delimiter = "<gold>"
gold_delimiter_end = "</gold>"

prompt_delimiter = "<prompt>"
prompt_delimiter_end = "</prompt>"

eval_delimiter = "<evaluation>"
eval_delimiter_end = "</evaluation>"

# Using zero-shot approach (no examples)

# Initial system prompt
INITIAL_SYSTEM_PROMPT = """Your task is to extract model names from machine learning paper abstracts. Your response is an array of the model names in the format ["model_name"]. If you don't find model names in the abstract or you are not sure, return ["NA"].
<instructions>
- Extract model names only, avoid things that are not model names like architectures and dataset names
<instructions>
"""

def load_validation_data(filepath: str) -> pd.DataFrame:
    """
    Load the validation dataset from a JSON file
    """
    print(f"Loading validation data from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    # Convert to DataFrame for compatibility
    df = pd.DataFrame(data)
    
    # Process gold_labels - they are stored as strings that look like arrays
    def process_gold_labels(labels_str):
        if isinstance(labels_str, str):
            try:
                # Use ast.literal_eval to safely convert string representation of list to actual list
                labels_list = ast.literal_eval(labels_str)
                # Clean up white spaces from items in the list
                return [item.strip() for item in labels_list]
            except (ValueError, SyntaxError):
                # If parsing fails, return empty list
                print(f"Warning: Could not parse gold_labels: {labels_str}")
                return []
        elif isinstance(labels_str, list):
            # If it's already a list, just clean it up
            return [item.strip() for item in labels_str]
        else:
            # Unknown type
            print(f"Warning: Unknown gold_labels type: {type(labels_str)}, value: {labels_str}")
            return []
    
    df['gold_labels'] = df['gold_labels'].apply(process_gold_labels)
    return df

def construct_prompt(system_prompt: str) -> List[Dict[str, Any]]:
    """
    Construct a zero-shot prompt with just the system prompt
    """
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    return messages

def generate_predictions(val_data: pd.DataFrame, messages: List[Dict[str, Any]]) -> List[Tuple[str, str, List[str]]]:
    """
    Generate predictions for paper abstracts using GPT-4o
    """
    print("Generating predictions...")
    results = []
    
    for _, row in val_data.iterrows():
        paper = row["paper"]
        abstract = row["abstract"]
        # Gold labels are already lists from JSON, just ensure they're clean
        gold_labels = row["gold_labels"]
        
        # Create copy of messages to modify for this specific example
        query_messages = messages.copy()
        
        # Add the current abstract as a user query
        query_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": f"Abstract: {abstract}"
                }
            ]
        })
        
        # Get model prediction
        prediction = get_chat_completion(query_messages, model="gpt-4o-mini", temperature=0)
        
        # Extract just the array from the prediction if it includes "Tags: "
        if "Tags: " in prediction:
            prediction = prediction.replace("Tags: ", "")
            
        # Parse prediction into an actual list
        try:
            pred_list = ast.literal_eval(prediction)
        except (SyntaxError, ValueError):
            # If the model doesn't return a valid list, try to extract it
            import re
            match = re.search(r'\[(.*?)\]', prediction)
            if match:
                pred_string = match.group(0)
                try:
                    pred_list = ast.literal_eval(pred_string)
                except:
                    pred_list = ["NA"]  # Fallback if everything fails
            else:
                pred_list = ["NA"]
                
        results.append((paper, abstract, gold_labels, pred_list))
    
    print(f"Generated predictions for {len(results)} papers")
    return results

def evaluate_prediction(abstract: str, 
                       prediction: List[str], 
                       gold_labels: List[str], 
                       model: str = "o3-mini") -> Tuple[float, str]:
    """
    Evaluate a single prediction using o3-mini as the judge
    Returns the score and explanation
    """
    judge_prompt = f"""
    {abstract_delimiter}
    {abstract}
    {abstract_delimiter_end}
    
    {prediction_delimiter}
    {prediction}
    {prediction_delimiter_end}
    
    {gold_delimiter}
    {gold_labels}
    {gold_delimiter_end}
    
    Your task is to evaluate how well the prediction matches the gold labels for extracting model names from a machine learning paper abstract.
    
    Evaluation criteria:
    1. Precision: Are all predicted model names actually present in the abstract and are they actual model names? 
    2. Recall: Did the prediction capture all model names in the abstract?
    3. Accuracy: Did the prediction correctly identify model names vs. non-model names?
    
    First, analyze the abstract to identify which model names are actually mentioned.
    Then compare the prediction to the gold labels.
    
    Give a score between 0.0 (completely wrong) and 1.0 (perfect match), with partial credit for partial matches.
    Explain your scoring with specific details about what was correct and incorrect in the prediction.
    
    Your response should be in the format:
    {eval_delimiter}
    Score: [score between 0.0 and 1.0]
    Explanation: [detailed explanation]
    {eval_delimiter_end}
    """
    
    # Get evaluation from o3-mini
    messages = [
        {"role": "user", "content": judge_prompt}
    ]
    
    response = get_reasoning_response(messages, model=model, reasoning_effort="low")
    
    # Extract score and explanation from response
    try:
        eval_text = response.split(eval_delimiter)[1].split(eval_delimiter_end)[0].strip()
        score_line = [line for line in eval_text.split('\n') if line.startswith('Score:')][0]
        score = float(score_line.replace('Score:', '').strip())
        explanation = eval_text.replace(score_line, '').strip()
        
        # Clean up explanation if it starts with "Explanation:"
        if explanation.startswith('Explanation:'):
            explanation = explanation.replace('Explanation:', '').strip()
            
    except (IndexError, ValueError) as e:
        print(f"Error parsing evaluation: {e}")
        print(f"Raw response: {response}")
        score = 0.0
        explanation = "Error parsing evaluation"
        
    return score, explanation

def evaluate_predictions(predictions: List[Tuple[str, str, List[str], List[str]]]) -> List[Dict[str, Any]]:
    """
    Evaluate all predictions using LLM-as-a-judge with o3-mini
    """
    print("Evaluating predictions with o3-mini...")
    evaluation_results = []
    
    for i, (paper, abstract, gold_labels, pred_list) in enumerate(predictions):
        print(f"Evaluating paper {i+1}/{len(predictions)}: {paper}")
        score, explanation = evaluate_prediction(abstract, pred_list, gold_labels)
        print(f"Score: {score:.4f}, gold labels: {gold_labels}, prediction: {pred_list}")

        evaluation_results.append({
            "paper": paper,
            "abstract": abstract,
            "gold_labels": gold_labels,
            "prediction": pred_list,
            "score": score,
            "explanation": explanation
        })
    
    # Calculate average score
    avg_score = mean([result["score"] for result in evaluation_results])
    print(f"Evaluation complete. Average score: {avg_score:.4f}")
    
    return evaluation_results, avg_score

def generate_metaprompt(system_prompt: str, evaluations: List[Dict[str, Any]]) -> str:
    """
    Generate a metaprompt for improving the system prompt based on evaluation results
    """
    print("Generating metaprompt for prompt improvement...")
    
    # Use all evaluations to provide complete context to the model
    # This gives the model more information to make better improvements
    
    eval_examples = ""
    for eval_data in evaluations:
        eval_examples += f"""
Paper: {eval_data['paper']}
Abstract: {eval_data['abstract']}
Gold Labels: {eval_data['gold_labels']}
Prediction: {eval_data['prediction']}
Score: {eval_data['score']}
Explanation: {eval_data['explanation']}

"""
    
    metaprompt = f"""
You are an expert prompt engineer tasked with improving a system prompt for extracting model names from machine learning paper abstracts.

Here is the current prompt to improve:
{prompt_delimiter}
{system_prompt}
{prompt_delimiter_end}

Here are evaluations of model predictions using the current prompt:
<eval_examples>
{eval_examples}
</eval_examples>

Based on these evaluations and its details like explanation and scores, please make important observations to improve the system prompt instructions that are found inside of <instructions><instructions>.
Don't change the instructions outside of <instructions><instructions>, keep those the same.
Output only the improved system prompt.
"""
    
    return metaprompt

def improve_prompt(system_prompt: str, evaluations: List[Dict[str, Any]]) -> str:
    """
    Generate an improved system prompt based on evaluation results using metaprompting
    """
    metaprompt = generate_metaprompt(system_prompt, evaluations)
    
    # Get improved prompt from o3-mini
    messages = [
        {"role": "user", "content": metaprompt}
    ]
    
    print("Getting improved prompt from o3-mini...")
    improved_prompt = get_reasoning_response(messages, model="o3-mini", reasoning_effort="high")
    
    return improved_prompt

def save_results(iteration: int, system_prompt: str, evaluations: List[Dict[str, Any]], avg_score: float, output_dir: str = "results"):
    """
    Save the results of an iteration to files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save system prompt
    with open(f"{output_dir}/system_prompt_iteration_{iteration}.txt", "w") as f:
        f.write(system_prompt)
    
    # Save evaluations
    with open(f"{output_dir}/evaluations_iteration_{iteration}.json", "w") as f:
        json.dump(evaluations, f, indent=2)
    
    # Save summary
    with open(f"{output_dir}/summary_iteration_{iteration}.txt", "w") as f:
        f.write(f"Iteration: {iteration}\n")
        f.write(f"Average Score: {avg_score:.4f}\n")
        f.write("\nSystem Prompt:\n")
        f.write(system_prompt)

def save_final_results(best_iteration: int, best_prompt: str, best_score: float, scores_history: List[float], output_dir: str = "results"):
    """
    Save the final results including the best prompt and performance history
    """
    # Save best prompt
    with open(f"{output_dir}/best_system_prompt.txt", "w") as f:
        f.write(best_prompt)
    
    # Save performance history
    with open(f"{output_dir}/performance_history.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Score"])
        for i, score in enumerate(scores_history):
            writer.writerow([i, score])
    
    # Save summary
    with open(f"{output_dir}/final_summary.txt", "w") as f:
        f.write(f"Best Iteration: {best_iteration}\n")
        f.write(f"Best Score: {best_score:.4f}\n\n")
        f.write("Performance History:\n")
        for i, score in enumerate(scores_history):
            f.write(f"Iteration {i}: {score:.4f}\n")
        f.write("\nBest System Prompt:\n")
        f.write(best_prompt)

def main():
    # Load validation data
    val_data = load_validation_data("../../data/val_data.json")

    # Set up parameters
    max_iterations = 5  # Maximum number of iterations to try
    system_prompt = INITIAL_SYSTEM_PROMPT
    output_dir = "../../results/llm_as_judge"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Iteration tracking
    best_score = 0.0
    best_prompt = system_prompt
    best_iteration = 0
    scores_history = []
    
    print("Starting LLM-as-a-judge with metaprompting process...")
    
    for iteration in range(max_iterations):
        print(f"\n==== Iteration {iteration} ====")
        
        # Construct zero-shot prompt
        messages = construct_prompt(system_prompt)
        
        # Generate predictions
        predictions = generate_predictions(val_data, messages)
        
        # Evaluate predictions
        evaluations, avg_score = evaluate_predictions(predictions)
        scores_history.append(avg_score)
        
        # Save results for this iteration
        save_results(iteration, system_prompt, evaluations, avg_score, output_dir)
        
        # Check if this is the best score
        if avg_score > best_score:
            best_score = avg_score
            best_prompt = system_prompt
            best_iteration = iteration
            print(f"New best score: {best_score:.4f} at iteration {best_iteration}")
        else:
            print(f"Score did not improve. Current: {avg_score:.4f}, Best: {best_score:.4f}")
            # Continue with the next iteration regardless of score improvement
        
        # Improve prompt using metaprompting (if not the last iteration)
        if iteration < max_iterations - 1:
            system_prompt = improve_prompt(system_prompt, evaluations)
            print(f"Updated system prompt for next iteration:\n{system_prompt}")
    
    # Save final results
    save_final_results(best_iteration, best_prompt, best_score, scores_history, output_dir)
    
    print(f"\n==== Process Complete ====")
    print(f"Best score: {best_score:.4f} at iteration {best_iteration}")
    print(f"Final results saved to {output_dir}")
    print(f"Best system prompt saved to {output_dir}/best_system_prompt.txt")

if __name__ == "__main__":
    main()
