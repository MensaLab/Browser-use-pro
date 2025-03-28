import asyncio
import os

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
        disable_security=False,
        # NOTE: you need to close your chrome browser - so that this can open your browser in debug mode
        chrome_instance_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        new_context_config=BrowserContextConfig()
    )
)
controller = Controller()


async def main():
    task = "Find a popular quinoa salad recipe on Allrecipes with more than 500 reviews and a rating above 4 stars. Create a shopping list of ingredients for this recipe and include the total cooking and preparation time."
    # task = "Provide a recipe for vegetarian lasagna with more than 100 reviews and a rating of at least 4.5 stars suitable for 6 people."
    # task = f"我想给我3岁女儿在淘宝买个裤子，我想要销量最多的那个加到购物车里。如果没登录的话先登录下，账号是:{os.environ['TAOBAO_NAME']}，密码是:{os.environ['TAOBAO_PASSWORD']}"

    task_plan = await get_task_plan(task)
    # print(task_plan)

    # task_plan = "Find a recipe for a vegan pumpkin pie on Allrecipes with a minimum four-star rating and a total cook time exceeding 1 hour."

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
