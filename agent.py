import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use import Browser, BrowserConfig
from browser_use.agent.task_plan import get_task_plan

load_dotenv()

from browser_use.browser.context import BrowserContextConfig

llm = ChatOpenAI(model="gpt-4o", temperature=0)

browser = Browser(
    config=BrowserConfig(
        headless=False,
        # NOTE: you need to close your chrome browser - so that this can open your browser in debug mode
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        new_context_config=BrowserContextConfig()
    )
)
controller = Controller()


async def main():
    task = "Find a one-way flight from Tokyo to NewYork on 2 April 2025 on Google Flights. Return me the cheapest option."

    # task = "Provide a recipe for vegetarian lasagna with more than 100 reviews and a rating of at least 4.5 stars suitable for 6 people."

    task_plan = await get_task_plan(task)
    # print(task_plan)

    agent = Agent(
        task=task_plan,
        llm=llm,
        controller=controller,
        browser=browser,
        max_actions_per_step=1
    )
    result = await agent.run()
    print(result)


asyncio.run(main())
