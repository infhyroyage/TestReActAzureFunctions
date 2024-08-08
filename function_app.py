import logging
import os

import azure.functions as func
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun
from langchain_openai import AzureChatOpenAI

app = func.FunctionApp()


@app.route(route="healthcheck", auth_level=func.AuthLevel.FUNCTION, methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("OK", status_code=200)


@app.route(route="react", auth_level=func.AuthLevel.FUNCTION, methods=["POST"])
def react(req: func.HttpRequest) -> func.HttpResponse:
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        azure_endpoint="https://westus.api.cognitive.microsoft.com",
        api_version="2024-02-15-preview",
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    tools = [
        GoogleSearchRun(
            api_wrapper=GoogleSearchAPIWrapper(
                google_api_key=os.environ["GOOGLE_API_KEY"],
                google_cse_id=os.environ["GOOGLE_CSE_ID"],
            )
        )
    ]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=True
    )
    result = agent_executor.invoke({"input": "What is LangChain?"})
    logging.info(result)

    return func.HttpResponse(
        result["output"],
        status_code=200,
    )
