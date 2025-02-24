from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from pydantic import SecretStr
import os
from dotenv import load_dotenv
import asyncio
import json
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from pathlib import Path
import sys

load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
	raise ValueError('GEMINI_API_KEY is not set')


CHROME_PATH = r'C:\Program Files\Google\Chrome\Application\chrome.exe'
WORKSPACE_ROOT = Path(__file__).parent.parent
RESUME_PATH = WORKSPACE_ROOT / 'sandbox' / 'resume.json'
CONVERSATION_PATH = WORKSPACE_ROOT / 'sandbox/debug/debug_conversation.txt'

# Ensure the paths exist
if not Path(CHROME_PATH).exists():
    raise ValueError(f'Chrome not found at {CHROME_PATH}')
if not RESUME_PATH.exists():
    raise ValueError(f'Resume not found at {RESUME_PATH}')

browser = Browser(
	config=BrowserConfig(
		# NOTE: you need to close your chrome browser - so that this can open your browser in debug mode
		chrome_instance_path=CHROME_PATH,
	)
)

# Initialize the model
llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=SecretStr(api_key))

# Load resume data from resume.json
with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
    resume_data = json.load(resume_file)



initial_actions = [
	{'open_tab': {'url': 'https://job-boards.greenhouse.io/andurilindustries/jobs/4605231007?gh_jid=4605231007&gh_src='}},
	{'scroll_down': {'amount': 4200}},
]



async def main():
    task_description = """
    You are applying for a job at Anduril Industries.
    The resume information is provided below in the context section.
    Use this information to fill out the application form. Only fill in the dropdown fields for now.
    
    Make sure to read through the resume data carefully before starting to fill out the form. Skip the attach/ submit buttons as well
    """

    agent = Agent(
        task=task_description,
        llm=llm,
        initial_actions=initial_actions,
        message_context=f"RESUME DATA:\n{json.dumps(resume_data, indent=2)}",
        max_input_tokens=128000,  # Ensure enough tokens for resume data
        browser=browser,
        save_conversation_path=str(CONVERSATION_PATH)  # Convert Path to string
    )

    await agent.run()
    await browser.close()

    input('Press Enter to close...')


if __name__ == '__main__':
	asyncio.run(main())