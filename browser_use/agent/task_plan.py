import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)

init_message = [{"role": "system",
                 "content": """
You are an automated browser-operating agent. Convert my task descriptions into step-by-step execution sequences for easier processing by other agents.
Example:
Input: “Find a one-way flight from Beijing to Tokyo on 15 Feb 2025 on Google Flights. Return me the cheapest option.”
Output:
1.	Open Google Flights.
2.	Select “One-way” trip.
3.	Enter “Beijing” as the departure and “Tokyo” as the destination.
4.	Set the departure date to “Feb 15, 2025”.
5.	Click the search button.
6.	Sort results by price (low to high).
7.	Retrieve the cheapest flight’s details (airline, times, stopovers).
8.	Output the flight details.
                 """}]


async def get_task_plan(raw_task: str) -> str:
    messages = init_message + [{"role": "user", "content": raw_task}]
    response = await get_openai_response(messages)
    return response


async def get_openai_response(messages) -> str:
    response = await llm.ainvoke(messages)  # 使用异步方式调用 OpenAI
    return response.content if response else ""


if __name__ == '__main__':
    raw_task = "In docs.google.com write my Papa a quick thank you for everything letter."
    content = asyncio.run(get_task_plan(raw_task))
    print(content)
