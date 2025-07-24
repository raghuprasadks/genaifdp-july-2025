from agno.agent import Agent
#from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

agent = Agent(
    #model=Claude(id="claude-sonnet-4-20250514"),
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True)],
    instructions="Use tables to display data. Display source of data. Don't include any other text.",
    markdown=True,
)
agent.print_response("Which stock is better among NVDIA and APPLE based on current market?", stream=True)