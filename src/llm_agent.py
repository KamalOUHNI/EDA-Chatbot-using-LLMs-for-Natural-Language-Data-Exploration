from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_ollama.llms import OllamaLLM



def run_pandas_agent (prompt,df) :
    model = OllamaLLM(model="qwen2.5")
    pandas_agent = create_pandas_dataframe_agent(
        llm=model,
        df=df,
        verbose=True,
        allow_dangerous_code=True)
    result = pandas_agent.run(prompt)
    return result