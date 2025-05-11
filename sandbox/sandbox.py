import asyncio
import json
import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig

# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User")

SANDBOX_PATH = Path(__file__).parent
RESUME_PATH = SANDBOX_PATH / 'resume.json'
PDF_RESUME_PATH = SANDBOX_PATH / 'Resume_Spencer_Willett.pdf'
SCRIPT_PATH = SANDBOX_PATH / 'fill_application_script.py'
# Browser configuration - update with your browser path
if platform.system() == 'Windows':
	CHROME_PATH = r'C:\Program Files\Google\Chrome\Application\chrome.exe'  # Windows
else:
	CHROME_PATH = '/opt/google/chrome/chrome'  # Linux

controller = Controller()

viewport_width, viewport_height = 1000, 1000
start_x, start_y = -1920, 0
viewport_expansion_pixels = -1


@controller.action('Upload resume - call this function to upload the resume to the attach or upload button')
async def upload_resume(index: int, browser: BrowserContext):
	path = str(PDF_RESUME_PATH.absolute())
	dom_el = await browser.get_dom_element_by_index(index)
	if dom_el is None:
		print(f'No element found at index {index}')
		return ActionResult(error=f'No element found at index {index}')
	file_upload_dom_el = dom_el.get_file_upload_element()
	if file_upload_dom_el is None:
		print(f'No file upload element found at index {index}')
		return ActionResult(error=f'No file upload element found at index {index}')
	file_upload_el = await browser.get_locate_element(file_upload_dom_el)
	if file_upload_el is None:
		print(f'Could not locate file upload element at index {index}')
		return ActionResult(error=f'Could not locate file upload element at index {index}')
	try:
		await file_upload_el.set_input_files(path)
		success_msg = f'Successfully uploaded resume from "{path}" to element at index {index}'
		print(success_msg)
		return ActionResult(extracted_content=success_msg, include_in_memory=True)
	except Exception as e:
		error_msg = f'Failed to upload resume to element at index {index}: {str(e)}'
		print(error_msg)
		return ActionResult(error=error_msg)


async def process_url(url, llm, resume_data, controller, index, browsers):
	x = start_x  # + index * 100  # Offset each window by 100 pixels
	port = 9223 + index  # Assign unique debugging port
	print(f'Launching browser for {url} with port {port}')
	browser = Browser(
		config=BrowserConfig(
			disable_security=True,
			headless=False,
			extra_browser_args=[
				'--force-dark-mode',
				'--enable-features=WebContentsForceDark',
				f'--window-position={x},{start_y}',
				f'--window-size={viewport_width},{viewport_height}',
				# f'--remote-debugging-port={port}',
			],
		)
	)
	browsers.append(browser)  # Add browser to the list for later closure
	try:
		context = await browser.new_context(
			config=BrowserContextConfig(
				browser_window_size={'width': viewport_width, 'height': viewport_height},
				viewport_expansion=viewport_expansion_pixels,
				keep_alive=True,
			)
		)
		await asyncio.sleep(index * 0.5)  # Stagger browser launches
		page = await context.get_current_page()
		await page.goto(url)
		await page.wait_for_load_state('networkidle')
		message_context = f'RESUME DATA:\n{json.dumps(resume_data, indent=2)}'
		agent = Agent(
			task='Fill in the job application form fields with information from my resume (skip the location field). After filling in everything, then use the upload_resume controller action to upload the resume at the end. DO NOT SUBMIT the application',
			llm=llm,
			use_vision=False,
			browser_context=context,
			controller=controller,
			message_context=message_context,
			save_playwright_script_path=str(SCRIPT_PATH),
		)
		await agent.run()
		# Check if context is still open
		return context
	except Exception as e:
		print(f'Error processing {url}: {e}')
		return None


async def main():
	browsers = []  # List to hold browser instances
	contexts = []  # List to hold browser contexts
	try:
		load_dotenv()
		api_key = os.getenv('GROK_API_KEY')
		if api_key is None:
			print('Error: GROK_API_KEY not set')
			sys.exit(1)

		with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
			resume_data = json.load(resume_file)

		llm = ChatOpenAI(base_url='https://api.x.ai/v1', model='grok-3-mini-fast-beta', api_key=SecretStr(api_key))

		urls = [
			'https://jobs.lever.co/palantir/a5fdd5ec-d1f3-4837-83af-161b003931dd/apply',
			'https://jobs.lever.co/palantir/30730c7d-d292-4dcd-a253-6b28258d186c/apply',
			'https://jobs.lever.co/palantir/34b3a697-6e22-4751-befd-0b7921abbd5f/apply',
			'https://jobs.lever.co/palantir/492a16bb-6b9f-457e-82c3-294e1a2c565d/apply',
		]

		tasks = [process_url(url, llm, resume_data, controller, i, browsers) for i, url in enumerate(urls)]
		contexts = await asyncio.gather(*tasks)

		# Wait for user input asynchronously
		await asyncio.to_thread(
			input,
			'All forms have been filled. Browser windows are open for manual review and submission.\nPlease fill in the location field if required and submit the forms manually.\nPress Enter to close all browser windows and exit...',
		)

	except Exception as e:
		print(f'An error occurred: {e}')
	finally:
		# Close all contexts and browsers after user input
		for context in contexts:
			if context:
				await context.close()
		for browser in browsers:
			await browser.close()
		print('All browsers closed.')


if __name__ == '__main__':
	asyncio.run(main())
