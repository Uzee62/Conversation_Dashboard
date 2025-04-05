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
print(df.chat("give me python code of a pie chart of male and female"))





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

    # Update message history and send final message
    message_history.append({"role": "assistant", "content": msg.content})
    await msg.update()
    


# # # import pandas as pd
# # # import matplotlib.pyplot as plt
# # # import os
# # # from dotenv import load_dotenv
# # # from langchain_groq.chat_models import ChatGroq

# # # # Load API Keys
# # # load_dotenv()
# # # GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# # # if not GROQ_API_KEY:
# # #     raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")

# # # # Load dataset
# # # df = pd.read_csv("Mall_Customers_Updated.csv")

# # # # Set up Groq LLM
# # # llm = ChatGroq(api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")

# # # # Function to generate Python visualization code
# # # def generate_code(user_query):
# # #     response = llm.invoke(user_query)
# # #     return response  # Expecting Python code as a response

# # # # Example user query
# # # user_query = "Generate a pie chart for male and female distribution in the dataset using Matplotlib."

# # # # # Generate the code using Groq
# # # generated_code = generate_code(user_query)

# # # # print("Generated Code:\n", generated_code)

# # # # Function to safely execute and display the code
# # # def execute_code(code):
# # #     try:
# # #         exec_globals = {"plt": plt, "df": df}
# # #         exec_locals = {}
# # #         exec(code, exec_globals, exec_locals)

# # #         # Display the plot if 'fig' is defined in the generated code
# # #         if "fig" in exec_locals:
# # #             exec_locals["fig"].show()
# # #         else:
# # #             print("No figure generated.")

# # #     except Exception as e:
# # #         print("Error executing code:", str(e))

# # # # Execute the generated visualization code
# # # execute_code(generated_code)


# import pandas as pd
# from pandasai import SmartDataframe
# from langchain_groq.chat_models import ChatGroq
# import os
# from dotenv import load_dotenv
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt

# # Loading environment variables
# load_dotenv()

# GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# os.environ["PANDASAI_API_KEY"] = os.getenv("PANDASAI_API_KEY")

# if not os.environ.get("PANDASAI_API_KEY"):
#     raise ValueError("PANDASAI_API_KEY is not set. Please check your .env file or set it manually.")

# # Custom Response Parser with improved display handling
# class TerminalResponseParser:
#     def __init__(self, context=None):
#         self.context = context
#         # Force matplotlib to use 'TkAgg' backend for terminal display
#         plt.switch_backend('TkAgg')

#     def format_dataframe(self, result):
#         print("\n📊 DataFrame Result:")
#         print(result["value"])

#     def format_plot(self, result):
#         print("\n📈 Displaying Plot...")
#         fig = result["value"]
#         if isinstance(fig, go.Figure):
#             # For Plotly figures
#             fig.show(renderer="browser")  # Forces browser display
#         elif isinstance(fig, plt.Figure):
#             # For Matplotlib figures
#             plt.show(block=True)  # Forces the plot to display and block execution
#         else:
#             print("Unknown plot type:", type(fig))

#     def format_other(self, result):
#         print("\n💬 Other Result:")
#         print(result["value"])

# # Setting up LLM
# llm = ChatGroq(
#     api_key=GROQ_API_KEY,
#     model_name="Deepseek-R1-Distill-Llama-70b"
# )

# # Load Dataset
# df = pd.read_csv("Mall_Customers_Updated.csv")

# # SmartDataframe Initialization with custom configuration
# smart_df = SmartDataframe(
#     df,
#     config={
#         "llm": llm,
#         "response_parser": TerminalResponseParser(),
#         "save_charts": False,  # Prevent saving to files
#         "display_type": "terminal"  # Force terminal display
#     }
# )

# # Example Query with explicit plotting
# response = smart_df.chat("give me  a histogram chart of male and female")
# print(response)

