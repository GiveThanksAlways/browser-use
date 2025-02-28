from langchain_google_genai import ChatGoogleGenerativeAI
from browser_use import Agent
from pydantic import SecretStr
import os
from dotenv import load_dotenv
import asyncio
import json
import logging
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from pathlib import Path
import sys
from browser_use.browser.context import BrowserContextConfig
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from browser_use.controller.greenhouse_helpers import GREENHOUSE_DROPDOWN_CACHE, reset_dropdown_tracking
from browser_use import Agent, Controller
from browser_use import ActionResult, Agent, Controller

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()
api_key_gemini = os.getenv('GEMINI_API_KEY')
if not api_key_gemini:
	raise ValueError('GEMINI_API_KEY is not set')

api_key_openAI = os.getenv('OPENAI_API_KEY')
if not api_key_openAI:
	raise ValueError('OPENAI_API_KEY is not set')

api_key_anthropic = os.getenv('ANTHROPIC_API_KEY')
if not api_key_anthropic:
	raise ValueError('ANTHROPIC_API_KEY is not set')

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
llm_google = ChatGoogleGenerativeAI(model='gemini-2.0-flash', api_key=SecretStr(api_key_gemini))

llm_openAI = ChatOpenAI(model='gpt-4o-mini', api_key=SecretStr(api_key_openAI))

llm_anthropic = ChatAnthropic(model_name='claude-3-5-sonnet-20241022', api_key=SecretStr(api_key_anthropic), timeout=100, stop=None)

llm = llm_google


controller_resume = Controller()

RESUME = WORKSPACE_ROOT / 'sandbox' / 'Resume_Spencer_Willett.pdf'

@controller_resume.action(
	'Attach resume: - call this function to upload if element is not found, try with different index of the same upload element',
)
async def upload_resume(index: int, browser: BrowserContext):
	"""Upload the resume to a file upload element at the specified index."""
	path = str(RESUME.absolute())
	
	# Log attempt to upload resume
	logger.info(f"Attempting to upload resume from {path} to element at index {index}")
	
	# Get the DOM element at the specified index
	dom_el = await browser.get_dom_element_by_index(index)
	if dom_el is None:
		logger.error(f'No element found at index {index}')
		return ActionResult(error=f'No element found at index {index}')
	
	# Get the file upload element from the DOM element
	file_upload_dom_el = dom_el.get_file_upload_element()
	if file_upload_dom_el is None:
		logger.error(f'No file upload element found at index {index}')
		return ActionResult(error=f'No file upload element found at index {index}')
	
	# Get the locatable element for the file upload
	file_upload_el = await browser.get_locate_element(file_upload_dom_el)
	if file_upload_el is None:
		logger.error(f'Could not locate file upload element at index {index}')
		return ActionResult(error=f'Could not locate file upload element at index {index}')
	
	try:
		# Attempt to upload the file
		await file_upload_el.set_input_files(path)
		success_msg = f'Successfully uploaded resume from "{path}" to element at index {index}'
		logger.info(success_msg)
		return ActionResult(
			extracted_content=success_msg, 
			include_in_memory=True
		)
	except Exception as e:
		error_msg = f'Failed to upload resume to element at index {index}: {str(e)}'
		logger.error(error_msg)
		return ActionResult(error=error_msg)

# Load resume data from resume.json
with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
    resume_data = json.load(resume_file)



initial_actions = [
	{'open_tab': {'url': 'https://job-boards.greenhouse.io/andurilindustries/jobs/4605231007?gh_jid=4605231007&gh_src='}},
	{'scroll_down': {'amount': 4200}},
]

extend_system_message = (
	'For Greenhouse.io applications, use these known dropdown options for faster filling:\n'
	'- Disability Status: "Yes, I have a disability, or have had one in the past", "No, I do not have a disability and have not had one in the past", "I do not want to answer"\n'
	'- Gender: "Male", "Female", "Non-binary", "I do not wish to answer"\n'
	'- Veteran Status: "I identify as one or more of the classifications of protected veteran listed above", "I am not a protected veteran", "I don\'t wish to answer"\n'
	'Always check for these exact wordings first before using other approaches.'
)

async def main():
    from browser_use.controller.greenhouse_helpers import reset_dropdown_tracking
    
    # First agent run - for dropdowns only
    task_description = """
    You are applying for a job at Anduril Industries.
    The resume information is provided below in the context section.
    Use this information to fill out the application form. Only fill in the dropdown fields for now.
    
    Make sure to read through the resume data carefully before starting to fill out the form. Skip the attach/ submit buttons as well.
    
    IMPORTANT: For each dropdown, identify it first, then select the appropriate value from the options. If you find
    yourself trying to interact with the same dropdown multiple times, try moving on to the next one.
    """

    # Reset dropdown tracking before starting
    reset_dropdown_tracking()
    logger.info("Starting first agent run - dropdown fields only")
    # Create a new browser context for the second agent
    async with await browser.new_context() as browser_context:

        agent = Agent(
            task=task_description,
            llm=llm,
            initial_actions=initial_actions,
            message_context=f"RESUME DATA:\n{json.dumps(resume_data, indent=2)}",
            max_input_tokens=128000,  # Ensure enough tokens for resume data
            browser=browser,
            browser_context=browser_context,
            save_conversation_path=str(CONVERSATION_PATH),  # Convert Path to string
            extend_system_message=extend_system_message
        )

        await agent.run()
        logger.info("First agent completed, now running second agent...")
        # First, run the resume upload agent
        agent_upload_resume = Agent(
            task="Just upload the resume by using the upload_resume contoller option. Attach resume. Don't do anything else. Just attach the resume",
            llm=llm_openAI,
            controller=controller_resume,
            initial_actions=[{'scroll_up': {'amount': 5000}}],
            browser=browser,
            browser_context=browser_context
        )
		
        await agent_upload_resume.run()
        logger.info("Resume uploaded successfully")
    
        # Reset tracking for the second run
        reset_dropdown_tracking()
        
        # Create a new task for text fields
        new_task = """
        Only fill in the text fields. Don't do anything else. Don't click on dropdowns/buttons/links.
        
        Just fill in the text fields (think Name, email, phone, etc.)
        """
        # Create a new agent with the existing browser but new context
        agent2 = Agent(
            task=new_task,
            llm=llm,
            initial_actions=[{'scroll_up': {'amount': 5000}}],
            message_context=f"RESUME DATA:\n{json.dumps(resume_data, indent=2)}",
            max_input_tokens=128000,
            browser=browser,  # Same browser
            browser_context=browser_context,  # New context
        )
        
        # Run the second agent
        await agent2.run()
        logger.info("Second agent completed successfully")

    # Now close the browser when everything is done
    input('Press Enter to close...')
    await browser.close()


if __name__ == '__main__':
	asyncio.run(main())