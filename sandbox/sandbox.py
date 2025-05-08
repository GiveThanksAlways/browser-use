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

controller = Controller()

viewport_width, viewport_height = 1000, 1000
start_x, start_y = 1920, 0
viewport_expansion_pixels = -1
browser = Browser(
	config=BrowserConfig(
		disable_security=True,
		headless=False,
		extra_chromium_args=[
			'--force-dark-mode',
			'--enable-features=WebContentsForceDark',
			f'--window-position={start_x},{start_y}',
			f'--window-size={viewport_width},{viewport_height}',
		],
		new_context_config=BrowserContextConfig(
			browser_window_size={'width': viewport_width, 'height': viewport_height},
			viewport_expansion=viewport_expansion_pixels,
			keep_alive=True,
		),
	)
)


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


async def main():
	try:
		load_dotenv()
		api_key = os.getenv('GROK_API_KEY')

		with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
			resume_data = json.load(resume_file)

		llm = ChatOpenAI(base_url='https://api.x.ai/v1', model='grok-3-mini-fast-beta', api_key=SecretStr(api_key))

		url = 'https://jobs.lever.co/palantir/81decd45-4b82-4201-a24f-25746b5d8caa/apply'

		tasks = {
			'1': 'Fill in name, email, phone number, GitHub, linkedIn, website (skip location). Do not submit the form.',
			'2': 'Use the upload_resume controller action to upload the resume (look for the resume upload field named attach). Do not submit the form.',
			'3': 'Fill in the job application with info from my resume (SKIP location, Do NOT Submit resume) and do not submit the form',
			'4': 'Fill in the voluntary self-identification section. Do not submit the form.',
		}

		async with await browser.new_context(
			config=BrowserContextConfig(
				browser_window_size={'width': viewport_width, 'height': viewport_height},
				viewport_expansion=viewport_expansion_pixels,
				keep_alive=True,
			)
		) as context:
			# Create a page and navigate once
			page = await context.get_current_page()
			await page.goto(url)
			await page.wait_for_load_state('networkidle')

			current_agent = None

			while True:
				print('\nSelect an option:')
				for key, desc in tasks.items():
					print(f'{key}. {desc}')
				print('p. Pause current task')
				print('r. Resume current task')
				print('s. Stop current task')
				print('q. Quit')

				choice = await asyncio.to_thread(input, 'Enter your choice: ')

				if choice in tasks:
					if current_agent:
						print('Stopping previous agent...')
						current_agent.stop()
						await asyncio.sleep(1)  # Give time for the agent to stop cleanly
					task = tasks[choice]
					message_context = f'RESUME DATA:\n{json.dumps(resume_data, indent=2)}'
					current_agent = Agent(
						task=task,
						llm=llm,
						use_vision=False,
						browser_context=context,
						controller=controller,
						message_context=message_context,
						save_playwright_script_path=str(SCRIPT_PATH),
					)
					print(f'Starting task: {task}')
					asyncio.create_task(current_agent.run())
					await asyncio.sleep(1)  # Allow some time to see initial logs
				elif choice == 'p':
					if current_agent:
						current_agent.pause()
						print('Agent paused')
				elif choice == 'r':
					if current_agent:
						current_agent.resume()
						print('Agent resumed')
				elif choice == 's':
					if current_agent:
						current_agent.stop()
						current_agent = None
						print('Agent stopped')
				elif choice == 'q':
					if current_agent:
						current_agent.stop()
					break
				else:
					print('Invalid choice')

	except Exception as e:
		print(f'An error occurred: {e}')

	finally:
		await browser.close()


if __name__ == '__main__':
	asyncio.run(main())
