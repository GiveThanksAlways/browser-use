import os
import sys
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI

from browser_use.agent.views import ActionResult

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio

from langchain_openai import ChatOpenAI

from browser_use import Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext

CHROME_PATH = r'C:\Program Files\Google\Chrome\Application\chrome.exe'

browser = Browser(
	config=BrowserConfig(
		# NOTE: you need to close your chrome browser - so that this can open your browser in debug mode
		chrome_instance_path=CHROME_PATH,
	)
)


async def main():

	api_key = os.getenv("GEMINI_API_KEY")

	# Initialize the model
	llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=api_key)

	agent = Agent(
		task='Search for Gymshark mens hoodie',
		llm=llm,
		browser=browser,
	)

	await agent.run()
	await browser.close()

	input('Press Enter to close...')


if __name__ == '__main__':
	asyncio.run(main())
