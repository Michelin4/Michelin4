import os
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from utils_function_calling import get_function_descriptions_multiple, extract_additional_kwargs

os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def main_loop():
    while True:
        # What is the most common type of POI for crashes happened in 2020? 
        # How many of the crashes of that POI type involved pedestrians?
        # Also, Please give me information about the crash ever happened that is the closest from here. My GPS is available.
        user_prompt = input("Please enter your query (type 'exit' to quit): ")
        
        if user_prompt.lower() == 'exit':
            print("Exiting the program.")
            break

        first_response = llm.predict_messages(
            [HumanMessage(content=user_prompt)], functions=get_function_descriptions_multiple()
        )

        # print("FIRST RESPONSE:", first_response, f"\n\n")

        second_response = llm.predict_messages(
            [
                HumanMessage(content=user_prompt),
                AIMessage(content=str(first_response.additional_kwargs)),
                AIMessage(
                    role="function",
                    additional_kwargs={
                        "name": first_response.additional_kwargs["function_call"]["name"]
                    },
                    content=f"Completed function {first_response.additional_kwargs['function_call']['name']}",
                ),
            ],
            functions=get_function_descriptions_multiple()
        )

        # print("SECOND RESPONSE:", second_response, f"\n\n")
        second_additional_kwargs_str, second_function_name = extract_additional_kwargs(second_response)
        if (second_additional_kwargs_str==None) | (second_function_name==None):
            print("ANSWER:", second_response.content)
            break

        third_response = llm.predict_messages(
            [
                HumanMessage(content=user_prompt),
                AIMessage(content=str(first_response.additional_kwargs)),
                AIMessage(content=second_additional_kwargs_str),
                AIMessage(
                    role="function",
                    additional_kwargs={
                        "name": second_function_name
                    },
                    content=f"Completed function {second_function_name}",
                ),
            ],
            functions=get_function_descriptions_multiple()
        )

        # print("THIRD RESPONSE:", third_response, f"\n\n")
        third_additional_kwargs_str, third_function_name = extract_additional_kwargs(third_response)
        if (third_additional_kwargs_str==None) | (third_function_name==None):
            print("ANSWER:", third_response.content)
            break

        fourth_response = llm.predict_messages(
            [
                HumanMessage(content=user_prompt),
                AIMessage(content=str(first_response.additional_kwargs)),
                AIMessage(content=second_additional_kwargs_str),
                AIMessage(content=third_additional_kwargs_str),
                AIMessage(
                    role="function",
                    additional_kwargs={
                        "name": third_function_name
                    },
                    content=f"Completed function {third_function_name}",
                ),
            ],
            functions=get_function_descriptions_multiple(),
        )

        # print("FOURTH RESPONSE:", fourth_response, f"\n\n")
        fourth_additional_kwargs_str, fourth_function_name = extract_additional_kwargs(fourth_response)
        if (fourth_additional_kwargs_str==None) | (fourth_function_name==None):
            print("ANSWER:", fourth_response.content)
            break


if __name__ == "__main__":
    main_loop()