import asyncio
from langchain_ollama import ChatOllama
from browser_use import Agent
from dotenv import load_dotenv
from browser_use import BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from process_exec import WebTaskPlanner

from loguru import logger
load_dotenv()


logger.add(sys.stdout, level="DEBUG", format="{time} | {level} | {message}")
logger.add("debug.log", rotation="1 MB", retention="10 days", level="DEBUG", format="{time} | {level} | {message}")



config = BrowserConfig(
    browser_binary_path="C:\Program Files\Google\Chrome\Application\chrome.exe"
)
browser = Browser(config=config)
llm = ChatOllama(model="qwen2.5:32b-instruct-q4_K_M")
prompt = """compare the stock price of tesla and microsoft for past 10 days"""

async def main(prompt):
    try:
        logger.info("Starting main task planning and execution.")
        planner = WebTaskPlanner(max_retries=5)
        result_steps = planner.plan_task(user_prompt=prompt)
        logger.debug(f"Planning steps: {result_steps}")

        agent = Agent(
            browser=browser,
            task=f"{prompt}. Use these steps {result_steps}",
            llm=llm,
        )
        logger.info("Agent created. Running the task...")

        result = await agent.run()
        logger.success(f"Task completed. Result: {result}")
    except Exception as e:
        logger.exception("An error occurred during execution.")


if __name__=="__main__":
    prompt = """compare the stock price of tesla and microsoft for past 10 days"""
    asyncio.run(main(prompt))