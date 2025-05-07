import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import Agent


# Helper function to stream output from the subprocess
async def stream_output(stream, prefix):
	if stream is None:
		print(f'{prefix}: (No stream available)')
		return
	while True:
		line = await stream.readline()
		if not line:
			break
		print(f'{prefix}: {line.decode().rstrip()}', flush=True)


async def main():
	try:
		# Load environment variables and resume data
		load_dotenv()
		api_key = os.getenv('GROK_API_KEY')

		SANDBOX_PATH = Path(__file__).parent
		RESUME_PATH = SANDBOX_PATH / 'resume.json'
		PDF_RESUME_PATH = SANDBOX_PATH / 'Resume_Spencer_Willett.pdf'
		SCRIPT_PATH = SANDBOX_PATH / 'fill_application_script.py'

		with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
			resume_data = json.load(resume_file)

		# Initialize Grok model
		llm = ChatOpenAI(base_url='https://api.x.ai/v1', model='grok-3-beta', api_key=SecretStr(api_key))

		# Define the URL and initial actions
		url = 'https://job-boards.greenhouse.io/andurilindustries/jobs/4673813007?gh_jid=4673813007&gh_src='
		# initial_actions = [
		# 	{'open_tab': {'url': url}},
		# 	{'scroll_down': {'amount': 3000}},
		# ]

		# Define the task
		task = f"""
		1. Go to {url}
		2. scroll down three times
		3. Fill in name, email, and phone number
		4. Do not submit the form
		"""

		# Create agent with Playwright script generation
		agent = Agent(
			task=task,
			llm=llm,
			use_vision=False,
			message_context=f'RESUME DATA:\n{json.dumps(resume_data, indent=2)}',
			# initial_actions=initial_actions,
			save_playwright_script_path=str(SCRIPT_PATH),
		)

		# Run the agent to fill the form and generate the script
		print('Running the agent to fill the form and generate the Playwright script...')
		await agent.run()
		print('Agent finished running.')

		# Check if the script was generated
		if SCRIPT_PATH.exists():
			print(f'Playwright script generated at: {SCRIPT_PATH}')
			# Execute the script
			process = await asyncio.create_subprocess_exec(
				sys.executable,
				str(SCRIPT_PATH),
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)
			# Stream the script's output
			stdout_task = asyncio.create_task(stream_output(process.stdout, 'stdout'))
			stderr_task = asyncio.create_task(stream_output(process.stderr, 'stderr'))
			await asyncio.gather(stdout_task, stderr_task)
			returncode = await process.wait()
			if returncode == 0:
				print('Playwright script executed successfully.')
			else:
				print(f'Playwright script finished with exit code {returncode}.')
		else:
			print('Playwright script was not generated.')
	except Exception as e:
		print(f'An error occurred: {e}')


if __name__ == '__main__':
	asyncio.run(main())
