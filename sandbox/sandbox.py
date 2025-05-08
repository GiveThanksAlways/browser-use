import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.context import BrowserContext

SANDBOX_PATH = Path(__file__).parent
RESUME_PATH = SANDBOX_PATH / 'resume.json'
PDF_RESUME_PATH = SANDBOX_PATH / 'Resume_Spencer_Willett.pdf'
SCRIPT_PATH = SANDBOX_PATH / 'fill_application_script.py'
MODIFIED_SCRIPT_PATH = SANDBOX_PATH / 'modified_fill_application_script.py'

controller = Controller()


@controller.action('Upload resume - call this function to uploade the resume to the attach or upload button')
async def upload_resume(index: int, browser: BrowserContext):
	"""Upload resume to a file upload element at the specified index."""
	path = str(PDF_RESUME_PATH.absolute())

	# Get the DOM element at the specified index
	dom_el = await browser.get_dom_element_by_index(index)
	if dom_el is None:
		print(f'No element found at index {index}')
		return ActionResult(error=f'No element found at index {index}')

	# Get the file upload element
	file_upload_dom_el = dom_el.get_file_upload_element()
	if file_upload_dom_el is None:
		print(f'No file upload element found at index {index}')
		return ActionResult(error=f'No file upload element found at index {index}')

	# Get the locatable element
	file_upload_el = await browser.get_locate_element(file_upload_dom_el)
	if file_upload_el is None:
		print(f'Could not locate file upload element at index {index}')
		return ActionResult(error=f'Could not locate file upload element at index {index}')

	try:
		# Upload the file
		await file_upload_el.set_input_files(path)
		success_msg = f'Successfully uploaded resume from "{path}" to element at index {index}'
		print(success_msg)
		return ActionResult(extracted_content=success_msg, include_in_memory=True)
	except Exception as e:
		error_msg = f'Failed to upload resume to element at index {index}: {str(e)}'
		print(error_msg)
		return ActionResult(error=error_msg)


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

		with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
			resume_data = json.load(resume_file)

		# Initialize Grok model
		llm = ChatOpenAI(base_url='https://api.x.ai/v1', model='grok-3-beta', api_key=SecretStr(api_key))

		# Define the URL and task
		url = 'https://jobs.lever.co/palantir/81decd45-4b82-4201-a24f-25746b5d8caa/apply'
		task = f"""
        1. Go to {url}
		2. Then use the upload_resume controller action to upload the resume (Look for the resume upload field named attach)
        3. Do not submit the form
        """

		# Create agent with Playwright script generation
		agent = Agent(
			task=task,
			llm=llm,
			use_vision=False,
			# message_context=f'RESUME DATA:\n{json.dumps(resume_data, indent=2)}',
			controller=controller,
			save_playwright_script_path=str(SCRIPT_PATH),
		)

		# Run the agent to fill the form and generate the script
		print('Running the agent to fill the form and generate the Playwright script...')
		await agent.run()
		print('Agent finished running.')

		# Check if the script was generated
		if SCRIPT_PATH.exists():
			print(f'Playwright script generated at: {SCRIPT_PATH}')

			# Modify the generated script to keep the browser open
			with open(SCRIPT_PATH, 'r', encoding='utf-8') as f:
				lines = f.readlines()

			# Find the index of the except line to insert waiting code before it
			except_index = next(i for i, line in enumerate(lines) if 'except PlaywrightActionError as pae:' in line)

			# Define the code to keep the browser open until Ctrl+C
			waiting_code = [
				'            print("Browser will remain open. Press Ctrl+C to close.")\n',
				'            await asyncio.Event().wait()\n',
			]

			# Insert the waiting code before the except block
			modified_lines = lines[:except_index] + waiting_code + lines[except_index:]

			# Write the modified script to a new file
			with open(MODIFIED_SCRIPT_PATH, 'w', encoding='utf-8') as f:
				f.writelines(modified_lines)
			print(f'Modified script saved at: {MODIFIED_SCRIPT_PATH}')

			# Execute the modified script
			process = await asyncio.create_subprocess_exec(
				sys.executable,
				str(MODIFIED_SCRIPT_PATH),
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)
			# Stream the script's output
			stdout_task = asyncio.create_task(stream_output(process.stdout, 'stdout'))
			stderr_task = asyncio.create_task(stream_output(process.stderr, 'stderr'))
			await asyncio.gather(stdout_task, stderr_task)
			returncode = await process.wait()
			if returncode == 0:
				print('Modified Playwright script executed successfully.')
			else:
				print(f'Modified Playwright script finished with exit code {returncode}.')
		else:
			print('Playwright script was not generated.')
	except Exception as e:
		print(f'An error occurred: {e}')


if __name__ == '__main__':
	asyncio.run(main())
