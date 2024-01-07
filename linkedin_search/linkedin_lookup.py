from loguru import logger

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent

from .tools import ddg_info_search, serpapi_link_search


def linkedin_lookup(persone_name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    template = """
    Given a full name {persone_name}, find their linkedin profile page and information about linkedin profile.
    Answer should contain:

    1. Linkedin URL.
    2. Short summary about person's career from linkedin.
    """

    tools_for_agent = [
        Tool(
            name="Crawl linkedin profile URL in google",
            func=serpapi_link_search,
            description="Use when you need to get the linkedin URL",
        ),
        Tool(
            name="Crawl information about linkedin profile in ddg",
            func=ddg_info_search,
            description="Use when you need to get the information about person's carrer in linkedin",
        ),
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    prompt_template = PromptTemplate(
        template=template, input_variables=["persone_name"]
    )
    prompt_template = prompt_template.format(persone_name=persone_name)

    logger.info(f"Running agent with template: {prompt_template}")
    result = agent.run(prompt_template)
    logger.info(f"Got result: {result}")

    return result
