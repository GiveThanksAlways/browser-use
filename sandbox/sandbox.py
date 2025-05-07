import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent

# Load environment variables
load_dotenv()
api_key = os.getenv('GROK_API_KEY')

# Path configuration
SANDBOX_PATH = Path(__file__).parent
RESUME_PATH = SANDBOX_PATH / 'resume.json'
PDF_RESUME_PATH = SANDBOX_PATH / 'Resume_Spencer_Willett.pdf'

with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
	resume_data = json.load(resume_file)

# Initialize Grok model
llm = ChatOpenAI(base_url='https://api.x.ai/v1', model='grok-3-beta', api_key=SecretStr(api_key))

# Define the URL (replace with your actual URL)
url = 'https://job-boards.greenhouse.io/andurilindustries/jobs/4673813007?gh_jid=4673813007&gh_src='
initial_actions = [
	{'open_tab': {'url': url}},
	{'scroll_down': {'amount': 3000}},
]

# Define the task
task = 'Fill in the job application using my resume data. Do NOT submit the form.'

# Create agent
agent = Agent(
	task=task,
	llm=llm,
	use_vision=False,
	message_context=f'RESUME DATA:\n{json.dumps(resume_data, indent=2)}',
	initial_actions=initial_actions,
)

# Run the agent
asyncio.run(agent.run())
