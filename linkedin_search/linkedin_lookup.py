from loguru import logger

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.agents import AgentType, Tool, initialize_agent

from .tools import serpapi_search


def linkedin_lookup(persone_name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """
    Given a full name {persone_name}, find their linkedin profile page.
    Answer should contain only URL.
    """

    tools_for_agent = [
        Tool(
            name="Crawl linkedin profile in google",
            func=serpapi_search,
            description="Use when you need to get the linkedin URL",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    prompt_template = PromptTemplate(
        template=template, input_variables=["persone_name"]
    )
    prompt_template = prompt_template.format(persone_name=persone_name)


    logger.info(
        f"Running agent with template: {prompt_template}"
    )
    return agent.run(prompt_template)
