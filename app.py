import pandas as pd
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import chainlit as cl
import json
import numpy as np
import plotly 
import matplotlib as plt

from plotly.offline import plot, iplot

from dotenv import load_dotenv
import os


from pandasai import Agent

# # Define your custom dependencies including Plotly
# custom_dependencies = ["plotly"]

# # Initialize the agent with your dataset and custom dependencies
# agent = Agent("Mall_Customers_Updated.csv", config={
#     "custom_whitelisted_dependencies": custom_dependencies,
#     "enable_plotly": True,
#     "enable_restricted": False,
#     "disable_matplotlib": True,
#     "language": "en-US"
# })


# Loading GROQ 
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
os.environ["PANDASAI_API_KEY"] = os.getenv("PANDASAI_API_KEY")

if not os.environ.get("PANDASAI_API_KEY"):
    raise ValueError("PANDASAI_API_KEY is not set. Please check your .env file or set it manually.")



#setting up LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    # model_name="Llama3-8b-8192"
    model_name = "Deepseek-R1-Distill-Llama-70b"
    
)

df = pd.read_csv("Mall_Customers_Updated.csv")

df = SmartDataframe(
        df,
        config={
            "llm": llm
        },
    )
# print(df.chat("give me python code of a pie chart of male and female"))





@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "You are a senior Data Analyst"}]
    )
    


@cl.on_message
async def main(message: cl.Message):
    # Retrieve message history
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    # Load data
    
    # data = input("Enter data file") 
    df = pd.read_csv("Mall_Customers_Updated.csv")
    
    df = SmartDataframe(df, config={"llm": llm})
    
    question = message.content
    response = df.chat(question)
    msg = cl.Message(content=response)
    print("Response from pandasai:", response)
    
    await msg.send()

    # Updating message history and sending final message
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
    

