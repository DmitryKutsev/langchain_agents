from linkedin_search.linkedin_lookup import linkedin_lookup
from langchain.agents import tool

@tool
def get_text_length(text):
    """
    Returns the length of the text
    """
    return len(text)



if "__main__" == __name__:
    print(get_text_length("Hello woooow"))

    template = """
         the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
    """
    # tools = [get_text_length]
    # linkedin_lookup("Dmitry Kutsev")
