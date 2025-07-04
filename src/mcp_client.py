from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio
from langgraph.checkpoint.memory import MemorySaver  
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import logging
import os

if os.path.exists("checkpoints.db"):
    os.remove("checkpoints.db")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = ChatOllama(
    model="qwen2.5",
    temperature=0,
    system="""You are a database assistant. You MUST use the provided tools to complete tasks.

IMPORTANT RULES:
- NEVER write code or SQL directly
- ALWAYS use the available tools for database operations
- If a function needs a dummy argument, use "dummy" as the value
- When asked to connect to database: use the database connection tool
- When asked for table names: use the table listing tool
- If you don't have the right tool, say so - don't write code instead

Available tools will be provided to you. Use them step by step."""
)

server_params = StdioServerParameters(
    command='python',
    args=['src/mcp_server.py'],
)
memory=MemorySaver()

config = {"configurable": {"thread_id": "abc123"},"recursion_limit": 50}

async def run_agent_connection(prompt: str):
    try:
    
        logger.info("Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("MCP session initialized successfully")
                    
                    tools = await load_mcp_tools(session)
                    tools = [tool for tool in tools if tool.name in [
                        "connect_to_sql_server",
                        "get_available_tables",
                        "select_table"
                    ]]

                    logger.info(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
                    agent = create_react_agent(llm, tools, checkpointer=memory)
                    logger.info("Agent created successfully")
                    
                    logger.info(f"Running agent with prompt: {prompt}")
                    response = await agent.ainvoke({
                        "messages": [("user", prompt)]
                    }, config)
                    
                    return response["messages"][-1].content
                    
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return f"Agent failed with error: {str(e)}"
async def run_agent_preprocessing(prompt: str):
    try:
            logger.info("Connecting to MCP server...")
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("MCP session initialized successfully")
                        
                    tools = await load_mcp_tools(session)
                    tools = [tool for tool in tools if tool.name in [
                        "get_table_preview",
                        "get_table_info",
                        "get_missing_value_summary"
                    ]]

                    logger.info(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
                    agent = create_react_agent(llm, tools, checkpointer=memory)
                    logger.info("Agent created successfully")
                        
                    logger.info(f"Running agent with prompt: {prompt}")
                    try :
                        response = await agent.ainvoke({
                        "messages": [("user", prompt)]
                        }, config)
                            
                        return response["messages"][-1].content
                    except Exception as e :
                        import traceback
                        print("error running the agent")
                        traceback.print_exc()
                    
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return f"Agent failed with error: {str(e)}"    
async def run_agent_univariate_non_graphical(prompt: str):
    try:
    
        logger.info("Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("MCP session initialized successfully")
                    
                    tools = await load_mcp_tools(session)
                    tools = [tool for tool in tools if tool.name in [
                        "univariate_numeric_summary",
                        "univariate_categorical_summary"
                    ]]

                    logger.info(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
                    agent = create_react_agent(llm, tools, checkpointer=memory)
                    logger.info("Agent created successfully")
                    
                    logger.info(f"Running agent with prompt: {prompt}")
                    response = await agent.ainvoke({
                        "messages": [("user", prompt)]
                    }, config)
                    
                    return response["messages"][-1].content
                    
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return f"Agent failed with error: {str(e)}"
async def run_agent_univariate_graphical(prompt: str):
    try:
    
        logger.info("Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("MCP session initialized successfully")
                    
                    tools = await load_mcp_tools(session)
                    tools = [tool for tool in tools if tool.name in [
                        "create_histogram",
                        "create_boxplot",
                        "create_bar_chart"
                    ]]

                    logger.info(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
                    agent = create_react_agent(llm, tools, checkpointer=memory)
                    logger.info("Agent created successfully")
                    
                    logger.info(f"Running agent with prompt: {prompt}")
                    response = await agent.ainvoke({
                        "messages": [("user", prompt)]
                    }, config)
                    
                    return response["messages"][-1].content
                    
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return f"Agent failed with error: {str(e)}"
async def run_agent_multivariate_non_graphical(prompt: str):
    try:
    
        logger.info("Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("MCP session initialized successfully")
                    
                    tools = await load_mcp_tools(session)
                    tools = [tool for tool in tools if tool.name in [
                        "correlation_analysis",
                        "crosstab_analysis"
                    ]]

                    logger.info(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
                    agent = create_react_agent(llm, tools, checkpointer=memory)
                    logger.info("Agent created successfully")
                    
                    logger.info(f"Running agent with prompt: {prompt}")
                    response = await agent.ainvoke({
                        "messages": [("user", prompt)]
                    }, config)
                    
                    return response["messages"][-1].content
                    
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return f"Agent failed with error: {str(e)}"
async def run_agent_multivariate_graphical(prompt: str):
    try:
    
        logger.info("Connecting to MCP server...")
        async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("MCP session initialized successfully")
                    
                    tools = await load_mcp_tools(session)
                    tools = [tool for tool in tools if tool.name in [
                        "create_correlation_heatmap",
                        "create_scatter_plot",
                        "create_pairplot"
                    ]]

                    logger.info(f"Loaded {len(tools)} tools: {[tool.name for tool in tools]}")
                    agent = create_react_agent(llm, tools, checkpointer=memory)
                    logger.info("Agent created successfully")
                    
                    logger.info(f"Running agent with prompt: {prompt}")
                    response = await agent.ainvoke({
                        "messages": [("user", prompt)]
                    }, config)
                    
                    return response["messages"][-1].content
                    
    except Exception as e:
        logger.error(f"Error running agent: {str(e)}")
        return f"Agent failed with error: {str(e)}"            
async def main():
    prompt1 = """use the available tools to :
1.connect to sql_server
2.get available tables in the database
3.select the table model_evaluation_fact
Please be explicit about each step you take and any errors you encounter."""

    prompt2 = """use the available tools to:
1.get table preview of model_evaluation_fact
2.get table in info of model_evaluation_fact
3.get missing values of model_evaluation_fact
Please be explicit about each step you take and any errors you encounter."""

    prompt3 = """use the available tools to:
1.perform univariate numerical summary on the avg_score column of model_evaluation_fact"""

    prompt4 = """use the available tools to:
    1.create a histogram for the avg_score column of model_evaluation_fact
    2.create a boxplot for the avg_score column of model_evaluation_fact"""
    prompt5="""use the available tools to:
1.perform correlation analysis on the model_evaluation_fact table
2.perform crosstab analysis on the model_evaluation_fact table"""
    prompt6="""use the available tools to:
1.create a correlation heatmap for the model_evaluation_fact table
2.create a scatter plot for the avg_score and model_id columns of model_evaluation_fact
3.create a pairplot for the model_evaluation_fact table"""
    print("Starting MCP Client...")

    result1 = await run_agent_connection(prompt1)
    print("Agent Initial Response:")
    print(result1)

    result2 = await run_agent_preprocessing(prompt2)
    print("\nAgent Final Response:")
    print(result2)

    result3 = await run_agent_univariate_non_graphical(prompt3)
    print("\nUnivariate Non-Graphical Analysis Response:")
    print(result3)
    result4 = await run_agent_univariate_graphical(prompt4)
    print("\nUnivariate Graphical Analysis Response:")
    print(result4)
    result5 = await run_agent_multivariate_non_graphical(prompt5)
    print("\nMultivariate Non-Graphical Analysis Response:")
    print(result5)
    result6 = await run_agent_multivariate_graphical(prompt6)
    print("\nMultivariate Graphical Analysis Response:")
    print(result6)



if __name__ == "__main__":
    asyncio.run(main())
