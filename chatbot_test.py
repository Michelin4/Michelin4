# https://huggingface.co/docs/transformers/en/tasks/question_answering
import pandas as pd
from transformers import pipeline
from fuzzywuzzy import process
import re

# from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
# model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")

# Load the CSV file (Example Data)
df = pd.DataFrame({
    'Product': ['Product A', 'Product B', 'Product C', 'Product D'],
    'Company': ['Company A', 'Company B', 'Company C', 'Company C'],
    'Price': [100, 200, 150, 180]
})

# Initialize a pre-trained model for question answering
model = pipeline("question-answering", model="distilbert-base-uncased")

def handle_user_query(user_input):
    # Extract the company name using the pre-trained model
    context = " ".join(df['Company'].values)  # Using Company as the context
    response = model(question=user_input, context=context)
    company = response['answer']
    
    # Intent recognition (mean, sum, max, min, etc.)
    if "mean" in user_input.lower():
        intent = "mean"
    elif "sum" in user_input.lower():
        intent = "sum"
    elif "max" in user_input.lower():
        intent = "max"
    elif "min" in user_input.lower():
        intent = "min"
    else:
        intent = "unknown"
    
    # Filter the DataFrame for the extracted company
    filtered_df = df[df['Company'].str.contains(company, case=False)]
    
    # Perform the calculation based on the intent
    if intent == "mean":
        result = filtered_df['Price'].mean()
    elif intent == "sum":
        result = filtered_df['Price'].sum()
    elif intent == "max":
        result = filtered_df['Price'].max()
    elif intent == "min":
        result = filtered_df['Price'].min()
    else:
        result = "Sorry, I couldn't understand your query."
    
    # Format the response
    if isinstance(result, (int, float)):
        bot_response = f"The {intent} price for {company} is {result:.2f}."
    else:
        bot_response = result
    
    return bot_response

# Example user input
user_input = "What’s the mean of product price from Company C?"
response = handle_user_query(user_input)
print(response)
# print(df)