import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

SANDBOX_PATH = Path(__file__).parent
RESUME_PATH = SANDBOX_PATH / 'resume.json'
PDF_RESUME_PATH = SANDBOX_PATH / 'Resume_Spencer_Willett.pdf'
SCRIPT_PATH = SANDBOX_PATH / 'fill_application_script.py'
MODIFIED_SCRIPT_PATH = SANDBOX_PATH / 'modified_fill_application_script.py'

controller = Controller()

# Create a single browser instance for all jobs
viewport_width, viewport_height = 1000, 1000
start_x, start_y = 1920, 0
viewport_expansion_pixels = 500  # -1
browser = Browser(
	config=BrowserConfig(
		disable_security=True,
		headless=False,
		extra_chromium_args=[
			'--force-dark-mode',
			'--enable-features=WebContentsForceDark',
			f'--window-position={start_x},{start_y}',
			f'--window-size={viewport_width},{viewport_height}',  # This sets the actual window size of Chrome
		],
		new_context_config=BrowserContextConfig(
			browser_window_size={'width': viewport_width, 'height': viewport_height},
			viewport_expansion=viewport_expansion_pixels,
			keep_alive=True,
		),
	)
)


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


async def fill_in_application(url, llm, resume_data):
	initial_actions = [
		{'go_to_url': {'url': url}},
		{'wait': {'seconds': 5}},
		# {'scroll_down': {'amount': 1000}},
	]

	task = """
	1. Fill in as much of the job application as you can (Skip location!!!!!)
	2. Use the upload_resume controller action to upload the resume (Look for the resume upload field named attach)
	3. Do not submit the form
	"""

	# Create agent with Playwright script generation
	agent = Agent(
		task=task,
		llm=llm,
		use_vision=False,
		message_context=f'RESUME DATA:\n{json.dumps(resume_data, indent=2)}',
		controller=controller,
		browser=browser,
		initial_actions=initial_actions,
		save_playwright_script_path=str(SCRIPT_PATH),
	)

	# Run the agent to fill the form and generate the script
	print('Running the agent to fill the form and generate the Playwright script...')
	await agent.run()

	print('Agent finished running.')
	input('Press Enter to continue...')
	await browser.close()


async def main():
	try:
		# Load environment variables and resume data
		load_dotenv()
		api_key = os.getenv('GROK_API_KEY')

		with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
			resume_data = json.load(resume_file)

		# Initialize Grok model
		llm = ChatOpenAI(base_url='https://api.x.ai/v1', model='grok-3-mini-beta', api_key=SecretStr(api_key))

		# Define the URL and task
		list_of_url = ['https://jobs.lever.co/palantir/81decd45-4b82-4201-a24f-25746b5d8caa/apply']

		for url in list_of_url:
			await fill_in_application(url, llm, resume_data)

	except Exception as e:
		print(f'An error occurred: {e}')


if __name__ == '__main__':
	asyncio.run(main())
