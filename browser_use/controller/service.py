import asyncio
import json
import logging
from typing import Dict, Generic, Optional, Type, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

# from lmnr.sdk.laminar import Laminar
from pydantic import BaseModel

from browser_use.agent.views import ActionModel, ActionResult
from browser_use.browser.context import BrowserContext
from browser_use.controller.registry.service import Registry
from browser_use.controller.views import (
	ClickElementAction,
	DoneAction,
	GoToUrlAction,
	InputTextAction,
	NoParamsAction,
	OpenTabAction,
	ScrollAction,
	SearchGoogleAction,
	SendKeysAction,
	SwitchTabAction,
)
from browser_use.utils import time_execution_sync
from browser_use.controller.greenhouse_helpers import is_greenhouse_dropdown, get_greenhouse_dropdown_options, track_dropdown_completion, get_dropdown_completion_status

logger = logging.getLogger(__name__)


Context = TypeVar('Context')


class Controller(Generic[Context]):
	def __init__(
		self,
		exclude_actions: list[str] = [],
		output_model: Optional[Type[BaseModel]] = None,
	):
		self.registry = Registry[Context](exclude_actions)

		"""Register all default browser actions"""

		if output_model is not None:
			# Create a new model that extends the output model with success parameter
			class ExtendedOutputModel(output_model):  # type: ignore
				success: bool = True

			@self.registry.action(
				'Complete task - with return text and if the task is finished (success=True) or not yet  completly finished (success=False), because last step is reached',
				param_model=ExtendedOutputModel,
			)
			async def done(params: ExtendedOutputModel):
				# Exclude success from the output JSON since it's an internal parameter
				output_dict = params.model_dump(exclude={'success'})
				return ActionResult(is_done=True, success=params.success, extracted_content=json.dumps(output_dict))
		else:

			@self.registry.action(
				'Complete task - with return text and if the task is finished (success=True) or not yet  completly finished (success=False), because last step is reached',
				param_model=DoneAction,
			)
			async def done(params: DoneAction):
				return ActionResult(is_done=True, success=params.success, extracted_content=params.text)

		# Basic Navigation Actions
		@self.registry.action(
			'Search the query in Google in the current tab, the query should be a search query like humans search in Google, concrete and not vague or super long. More the single most important items. ',
			param_model=SearchGoogleAction,
		)
		async def search_google(params: SearchGoogleAction, browser: BrowserContext):
			page = await browser.get_current_page()
			await page.goto(f'https://www.google.com/search?q={params.query}&udm=14')
			await page.wait_for_load_state()
			msg = f'🔍  Searched for "{params.query}" in Google'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		@self.registry.action('Navigate to URL in the current tab', param_model=GoToUrlAction)
		async def go_to_url(params: GoToUrlAction, browser: BrowserContext):
			page = await browser.get_current_page()
			await page.goto(params.url)
			await page.wait_for_load_state()
			msg = f'🔗  Navigated to {params.url}'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		@self.registry.action('Go back', param_model=NoParamsAction)
		async def go_back(_: NoParamsAction, browser: BrowserContext):
			await browser.go_back()
			msg = '🔙  Navigated back'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		# wait for x seconds
		@self.registry.action('Wait for x seconds default 3')
		async def wait(seconds: int = 3):
			msg = f'🕒  Waiting for {seconds} seconds'
			logger.info(msg)
			await asyncio.sleep(seconds)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		# Element Interaction Actions
		@self.registry.action('Click element', param_model=ClickElementAction)
		async def click_element(params: ClickElementAction, browser: BrowserContext):
			session = await browser.get_session()

			if params.index not in await browser.get_selector_map():
				raise Exception(f'Element with index {params.index} does not exist - retry or use alternative actions')

			element_node = await browser.get_dom_element_by_index(params.index)
			initial_pages = len(session.context.pages)

			# if element has file uploader then dont click
			if await browser.is_file_uploader(element_node):
				msg = f'Index {params.index} - has an element which opens file upload dialog. To upload files please use a specific function to upload files '
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True)

			msg = None

			try:
				download_path = await browser._click_element_node(element_node)
				if download_path:
					msg = f'💾  Downloaded file to {download_path}'
				else:
					msg = f'🖱️  Clicked button with index {params.index}: {element_node.get_all_text_till_next_clickable_element(max_depth=2)}'

				logger.info(msg)
				logger.debug(f'Element xpath: {element_node.xpath}')
				if len(session.context.pages) > initial_pages:
					new_tab_msg = 'New tab opened - switching to it'
					msg += f' - {new_tab_msg}'
					logger.info(new_tab_msg)
					await browser.switch_to_tab(-1)
				return ActionResult(extracted_content=msg, include_in_memory=True)
			except Exception as e:
				logger.warning(f'Element not clickable with index {params.index} - most likely the page changed')
				return ActionResult(error=str(e))

		@self.registry.action(
			'Input text into a input interactive element',
			param_model=InputTextAction,
		)
		async def input_text(params: InputTextAction, browser: BrowserContext, has_sensitive_data: bool = False):
			if params.index not in await browser.get_selector_map():
				raise Exception(f'Element index {params.index} does not exist - retry or use alternative actions')

			element_node = await browser.get_dom_element_by_index(params.index)
			await browser._input_text_element_node(element_node, params.text)
			if not has_sensitive_data:
				msg = f'⌨️  Input {params.text} into index {params.index}'
			else:
				msg = f'⌨️  Input sensitive data into index {params.index}'
			logger.info(msg)
			logger.debug(f'Element xpath: {element_node.xpath}')
			return ActionResult(extracted_content=msg, include_in_memory=True)

		# Tab Management Actions
		@self.registry.action('Switch tab', param_model=SwitchTabAction)
		async def switch_tab(params: SwitchTabAction, browser: BrowserContext):
			await browser.switch_to_tab(params.page_id)
			# Wait for tab to be ready
			page = await browser.get_current_page()
			await page.wait_for_load_state()
			msg = f'🔄  Switched to tab {params.page_id}'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		@self.registry.action('Open url in new tab', param_model=OpenTabAction)
		async def open_tab(params: OpenTabAction, browser: BrowserContext):
			await browser.create_new_tab(params.url)
			msg = f'🔗  Opened new tab with {params.url}'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		# Content Actions
		@self.registry.action(
			'Extract page content to retrieve specific information from the page, e.g. all company names, a specifc description, all information about, links with companies in structured format or simply links',
		)
		async def extract_content(goal: str, browser: BrowserContext, page_extraction_llm: BaseChatModel):
			page = await browser.get_current_page()
			import markdownify

			content = markdownify.markdownify(await page.content())

			prompt = 'Your task is to extract the content of the page. You will be given a page and a goal and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format. Extraction goal: {goal}, Page: {page}'
			template = PromptTemplate(input_variables=['goal', 'page'], template=prompt)
			try:
				output = page_extraction_llm.invoke(template.format(goal=goal, page=content))
				msg = f'📄  Extracted from page\n: {output.content}\n'
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True)
			except Exception as e:
				logger.debug(f'Error extracting content: {e}')
				msg = f'📄  Extracted from page\n: {content}\n'
				logger.info(msg)
				return ActionResult(extracted_content=msg)

		@self.registry.action(
			'Scroll down the page by pixel amount - if no amount is specified, scroll down one page',
			param_model=ScrollAction,
		)
		async def scroll_down(params: ScrollAction, browser: BrowserContext):
			page = await browser.get_current_page()
			if params.amount is not None:
				await page.evaluate(f'window.scrollBy(0, {params.amount});')
			else:
				await page.evaluate('window.scrollBy(0, window.innerHeight);')

			amount = f'{params.amount} pixels' if params.amount is not None else 'one page'
			msg = f'🔍  Scrolled down the page by {amount}'
			logger.info(msg)
			return ActionResult(
				extracted_content=msg,
				include_in_memory=True,
			)

		# scroll up
		@self.registry.action(
			'Scroll up the page by pixel amount - if no amount is specified, scroll up one page',
			param_model=ScrollAction,
		)
		async def scroll_up(params: ScrollAction, browser: BrowserContext):
			page = await browser.get_current_page()
			if params.amount is not None:
				await page.evaluate(f'window.scrollBy(0, -{params.amount});')
			else:
				await page.evaluate('window.scrollBy(0, -window.innerHeight);')

			amount = f'{params.amount} pixels' if params.amount is not None else 'one page'
			msg = f'🔍  Scrolled up the page by {amount}'
			logger.info(msg)
			return ActionResult(
				extracted_content=msg,
				include_in_memory=True,
			)

		# send keys
		@self.registry.action(
			'Send strings of special keys like Escape,Backspace, Insert, PageDown, Delete, Enter, Shortcuts such as `Control+o`, `Control+Shift+T` are supported as well. This gets used in keyboard.press. ',
			param_model=SendKeysAction,
		)
		async def send_keys(params: SendKeysAction, browser: BrowserContext):
			page = await browser.get_current_page()

			try:
				await page.keyboard.press(params.keys)
			except Exception as e:
				if 'Unknown key' in str(e):
					# loop over the keys and try to send each one
					for key in params.keys:
						try:
							await page.keyboard.press(key)
						except Exception as e:
							logger.debug(f'Error sending key {key}: {str(e)}')
							raise e
				else:
					raise e
			msg = f'⌨️  Sent keys: {params.keys}'
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

		@self.registry.action(
			description='If you dont find something which you want to interact with, scroll to it',
		)
		async def scroll_to_text(text: str, browser: BrowserContext):  # type: ignore
			page = await browser.get_current_page()
			try:
				# Try different locator strategies
				locators = [
					page.get_by_text(text, exact=False),
					page.locator(f'text={text}'),
					page.locator(f"//*[contains(text(), '{text}')]"),
				]

				for locator in locators:
					try:
						# First check if element exists and is visible
						if await locator.count() > 0 and await locator.first.is_visible():
							await locator.first.scroll_into_view_if_needed()
							await asyncio.sleep(0.5)  # Wait for scroll to complete
							msg = f'🔍  Scrolled to text: {text}'
							logger.info(msg)
							return ActionResult(extracted_content=msg, include_in_memory=True)
					except Exception as e:
						logger.debug(f'Locator attempt failed: {str(e)}')
						continue

				msg = f"Text '{text}' not found or not visible on page"
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True)

			except Exception as e:
				msg = f"Failed to scroll to text '{text}': {str(e)}"
				logger.error(msg)
				return ActionResult(error=msg, include_in_memory=True)

		@self.registry.action(
			description='Get all options from a native dropdown',
		)
		async def get_dropdown_options(index: int, browser: BrowserContext) -> ActionResult:
			"""Get all options from a native dropdown"""
			page = await browser.get_current_page()
			selector_map = await browser.get_selector_map()
			dom_element = selector_map[index]

			try:
				# Frame-aware approach since we know it works
				all_options = []
				frame_index = 0

				for frame in page.frames:
					try:
						options = await frame.evaluate(
							"""
							(xpath) => {
								const select = document.evaluate(xpath, document, null,
									XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
								if (!select) return null;

								return {
									options: Array.from(select.options).map(opt => ({
										text: opt.text, //do not trim, because we are doing exact match in select_dropdown_option
										value: opt.value,
										index: opt.index
									})),
									id: select.id,
									name: select.name
								};
							}
						""",
							dom_element.xpath,
						)

						if options:
							logger.debug(f'Found dropdown in frame {frame_index}')
							logger.debug(f'Dropdown ID: {options["id"]}, Name: {options["name"]}')

							formatted_options = []
							for opt in options['options']:
								# encoding ensures AI uses the exact string in select_dropdown_option
								encoded_text = json.dumps(opt['text'])
								formatted_options.append(f'{opt["index"]}: text={encoded_text}')

							all_options.extend(formatted_options)

					except Exception as frame_e:
						logger.debug(f'Frame {frame_index} evaluation failed: {str(frame_e)}')

					frame_index += 1

				if all_options:
					msg = '\n'.join(all_options)
					msg += '\nUse the exact text string in select_dropdown_option'
					logger.info(msg)
					return ActionResult(extracted_content=msg, include_in_memory=True)
				else:
					msg = 'No options found in any frame for dropdown'
					logger.info(msg)
					return ActionResult(extracted_content=msg, include_in_memory=True)

			except Exception as e:
				logger.error(f'Failed to get dropdown options: {str(e)}')
				msg = f'Error getting options: {str(e)}'
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True)

		@self.registry.action(
			description='Select dropdown option for interactive element index by the text of the option you want to select',
		)
		async def select_dropdown_option(
			index: int,
			text: str,
			browser: BrowserContext,
		) -> ActionResult:
			"""Select dropdown option by the text of the option you want to select"""
			page = await browser.get_current_page()
			selector_map = await browser.get_selector_map()
			dom_element = selector_map[index]

			# Validate that we're working with a select element
			if dom_element.tag_name != 'select':
				logger.error(f'Element is not a select! Tag: {dom_element.tag_name}, Attributes: {dom_element.attributes}')
				msg = f'Cannot select option: Element with index {index} is a {dom_element.tag_name}, not a select'
				return ActionResult(extracted_content=msg, include_in_memory=True)

			logger.debug(f"Attempting to select '{text}' using xpath: {dom_element.xpath}")
			logger.debug(f'Element attributes: {dom_element.attributes}')
			logger.debug(f'Element tag: {dom_element.tag_name}')

			xpath = '//' + dom_element.xpath

			try:
				frame_index = 0
				for frame in page.frames:
					try:
						logger.debug(f'Trying frame {frame_index} URL: {frame.url}')

						# First verify we can find the dropdown in this frame
						find_dropdown_js = """
							(xpath) => {
								try {
									const select = document.evaluate(xpath, document, null,
										XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
									if (!select) return null;
									if (select.tagName.toLowerCase() !== 'select') {
										return {
											error: `Found element but it's a ${select.tagName}, not a SELECT`,
											found: false
										};
									}
									return {
										id: select.id,
										name: select.name,
										found: true,
										tagName: select.tagName,
										optionCount: select.options.length,
										currentValue: select.value,
										availableOptions: Array.from(select.options).map(o => o.text.trim())
									};
								} catch (e) {
									return {error: e.toString(), found: false};
								}
							}
						"""

						dropdown_info = await frame.evaluate(find_dropdown_js, dom_element.xpath)

						if dropdown_info:
							if not dropdown_info.get('found'):
								logger.error(f'Frame {frame_index} error: {dropdown_info.get("error")}')
								continue

							logger.debug(f'Found dropdown in frame {frame_index}: {dropdown_info}')

							# "label" because we are selecting by text
							# nth(0) to disable error thrown by strict mode
							# timeout=1000 because we are already waiting for all network events, therefore ideally we don't need to wait a lot here (default 30s)
							selected_option_values = (
								await frame.locator('//' + dom_element.xpath).nth(0).select_option(label=text, timeout=1000)
							)

							msg = f'selected option {text} with value {selected_option_values}'
							logger.info(msg + f' in frame {frame_index}')

							return ActionResult(extracted_content=msg, include_in_memory=True)

					except Exception as frame_e:
						logger.error(f'Frame {frame_index} attempt failed: {str(frame_e)}')
						logger.error(f'Frame type: {type(frame)}')
						logger.error(f'Frame URL: {frame.url}')

					frame_index += 1

				msg = f"Could not select option '{text}' in any frame"
				logger.info(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True)

			except Exception as e:
				msg = f'Selection failed: {str(e)}'
				logger.error(msg)
				return ActionResult(error=msg, include_in_memory=True)

		@self.registry.action(
			description='Handle any type of dropdown (both native <select> and custom dropdowns) with optimization for Greenhouse.io',
		)
		async def handle_dropdown(
			index: int,
			text: str,
			browser: BrowserContext,
		) -> ActionResult:
			"""Handle both native <select> and custom dropdowns with greenhouse optimization"""
			page = await browser.get_current_page()
			selector_map = await browser.get_selector_map()
			dom_element = selector_map[index]
			
			# Check if we've already handled this dropdown to avoid loops
			is_new_completion = track_dropdown_completion(index, text)
			if not is_new_completion:
				completion_status = get_dropdown_completion_status()
				msg = f"SKIPPING: Dropdown with index {index} and value '{text}' was already completed.\n{completion_status}"
				logger.warning(msg)
				return ActionResult(extracted_content=msg, include_in_memory=True)
			
			# First check if this is a Greenhouse.io dropdown we can optimize
			is_greenhouse, field_id = is_greenhouse_dropdown(dom_element)
			
			if is_greenhouse:
				logger.info(f"Processing Greenhouse dropdown: index={index}, field_id={field_id}")
				cached_options = get_greenhouse_dropdown_options(field_id)
				
				if cached_options and text in cached_options:
					logger.info(f"Using optimized approach for Greenhouse dropdown: index={index}, field_id={field_id}, text='{text}'")
					
					try:
						# 1. Click to open the dropdown
						await browser._click_element_node(dom_element)
						await asyncio.sleep(0.5)  # Wait for dropdown to open
						
						# 2. Try to find and click the option matching our cached knowledge
						option_selectors = [
							f"//div[contains(@class, 'select__option') and contains(text(), '{text}')]",
							f"//div[contains(@class, 'select__option') and text()='{text}']",
							f"//div[contains(@id, 'react-select-{field_id}-option') and contains(text(), '{text}')]"
						]
						
						for selector in option_selectors:
							try:
								logger.debug(f"Trying selector: {selector}")
								option = await page.wait_for_selector(selector, timeout=1000)
								if option:
									await option.click()
									msg = f'SUCCESS: Selected "{text}" from Greenhouse dropdown {field_id} using cached approach'
									logger.info(msg)
									return ActionResult(extracted_content=msg, include_in_memory=True)
							except Exception as e:
								logger.debug(f"Selector {selector} failed: {str(e)}")
								continue
						
						logger.info(f"Cached approach failed, falling back to standard method")
					
					except Exception as e:
						logger.error(f"Error in Greenhouse dropdown handler: {str(e)}")
						# Fall through to standard methods if optimized approach failed
			
			# Original implementation for native select
			if dom_element.tag_name == 'select':
				logger.info(f"Processing native select dropdown: index={index}")
				# Use existing select_dropdown_option logic
				return await self.registry.execute_action('select_dropdown_option', {'index': index, 'text': text}, browser)
			else:
				# Handle custom dropdown
				try:
					logger.info(f"Processing custom dropdown: index={index}")
					# 1. Click to open the dropdown
					await browser._click_element_node(dom_element)
					await asyncio.sleep(0.5)  # Wait for dropdown to open
					
					# 2. Find and click the option with matching text
					option_found = await page.evaluate("""
						(text) => {
							// Look for elements that might be dropdown options
							const options = Array.from(document.querySelectorAll('.select__option, [role="option"], .dropdown-item, li'));
							console.log('Found ' + options.length + ' potential dropdown options');
							
							// Log what we found
							options.forEach((opt, i) => {
								console.log(`Option ${i}: ${opt.textContent.trim()}`);
							});
							
							// Try to find and click the matching option
							for (const option of options) {
								if (option.textContent.trim() === text) {
									option.click();
									return true;
								}
							}
							return false;
						}
					""", text)
					
					if option_found:
						msg = f'SUCCESS: Selected option "{text}" in custom dropdown (index={index})'
						logger.info(msg)
						return ActionResult(extracted_content=msg, include_in_memory=True)
					else:
						# Try a less strict matching approach
						option_found = await page.evaluate("""
							(text) => {
								const options = Array.from(document.querySelectorAll('.select__option, [role="option"], .dropdown-item, li'));
								for (const option of options) {
									if (option.textContent.trim().includes(text) || text.includes(option.textContent.trim())) {
										console.log('Found partial match: ' + option.textContent.trim());
										option.click();
										return true;
									}
								}
								return false;
							}
						""", text)
						
						if option_found:
							msg = f'SUCCESS: Selected option partially matching "{text}" in custom dropdown (index={index})'
							logger.info(msg)
							return ActionResult(extracted_content=msg, include_in_memory=True)
						else:
							msg = f'FAILED: Could not find option "{text}" in custom dropdown (index={index})'
							logger.warning(msg)
							return ActionResult(extracted_content=msg, include_in_memory=True)
						
				except Exception as e:
					msg = f'ERROR: Custom dropdown selection failed: {str(e)}'
					logger.error(msg)
					return ActionResult(error=msg, include_in_memory=True)

		@self.registry.action(
			description='Handle custom dropdown by clicking to open it and then clicking the option with the specified text',
		)
		async def handle_custom_dropdown(
			dropdown_index: int,
			option_text: str,
			browser: BrowserContext,
		) -> ActionResult:
			"""Handle custom dropdowns by clicking to open and then clicking the option"""
			page = await browser.get_current_page()
			selector_map = await browser.get_selector_map()
			
			try:
				# 1. Click to open the dropdown
				dropdown_element = selector_map[dropdown_index]
				logger.info(f"Clicking dropdown element with index {dropdown_index}")
				await browser._click_element_node(dropdown_element)
				await asyncio.sleep(0.5)  # Wait for dropdown to open
				
				# 2. Get updated DOM after dropdown opens
				updated_state = await browser.get_state()
				updated_selector_map = updated_state.selector_map
				
				# 3. Look for option elements that appeared after opening the dropdown
				option_found = False
				for idx, element in updated_selector_map.items():
					element_text = element.get_all_text_till_next_clickable_element()
					if (element_text and option_text in element_text) or (element.attributes.get('value') == option_text):
						logger.info(f"Found matching option with index {idx}: {element_text}")
						await browser._click_element_node(element)
						option_found = True
						break
						
				if option_found:
					msg = f'Selected option "{option_text}" in custom dropdown'
					return ActionResult(extracted_content=msg, include_in_memory=True)
				else:
					msg = f'Could not find option "{option_text}" in dropdown. Try clicking the dropdown again and looking for the option.'
					return ActionResult(extracted_content=msg, include_in_memory=True)
				
			except Exception as e:
				msg = f'Custom dropdown handling failed: {str(e)}'
				logger.error(msg)
				return ActionResult(error=msg, include_in_memory=True)

		@self.registry.action(
			description='Navigate dropdown using keyboard (Tab, Arrow keys, Enter)',
		)
		async def dropdown_keyboard_navigation(
			dropdown_index: int,
			option_position: int,  # Approximate position of the option (1 for first, 2 for second, etc.)
			browser: BrowserContext,
		) -> ActionResult:
			"""Navigate dropdown using keyboard"""
			page = await browser.get_current_page()
			selector_map = await browser.get_selector_map()
			
			try:
				# 1. Click to focus the dropdown
				dropdown_element = selector_map[dropdown_index]
				await browser._click_element_node(dropdown_element)
				
				# 2. Press down arrow to open dropdown and navigate
				for _ in range(option_position):
					await page.keyboard.press("ArrowDown")
					await asyncio.sleep(0.2)
				
				# 3. Press Enter to select
				await page.keyboard.press("Enter")
				
				msg = f'Used keyboard navigation to select option at position {option_position}'
				return ActionResult(extracted_content=msg, include_in_memory=True)
			except Exception as e:
				msg = f'Keyboard navigation failed: {str(e)}'
				logger.error(msg)
				return ActionResult(error=msg, include_in_memory=True)

		@self.registry.action(
			description='Detect the type of dropdown (native select or custom) to determine the best interaction method',
		)
		async def detect_dropdown_type(
			index: int,
			browser: BrowserContext,
		) -> ActionResult:
			"""Detect if element is a native select or custom dropdown"""
			selector_map = await browser.get_selector_map()
			dom_element = selector_map[index]
			
			if dom_element.tag_name == 'select':
				msg = 'Element is a native <select> dropdown. Use select_dropdown_option.'
			elif 'select' in dom_element.attributes.get('class', '').lower() or dom_element.attributes.get('role') == 'combobox':
				msg = 'Element appears to be a custom dropdown. Click it to open, then click the desired option.'
			else:
				msg = 'Element does not appear to be a dropdown. Consider other interaction methods.'
			
			logger.info(msg)
			return ActionResult(extracted_content=msg, include_in_memory=True)

	# Register ---------------------------------------------------------------

	def action(self, description: str, **kwargs):
		"""Decorator for registering custom actions

		@param description: Describe the LLM what the function does (better description == better function calling)
		"""
		return self.registry.action(description, **kwargs)

	# Act --------------------------------------------------------------------

	@time_execution_sync('--act')
	async def act(
		self,
		action: ActionModel,
		browser_context: BrowserContext,
		#
		page_extraction_llm: Optional[BaseChatModel] = None,
		sensitive_data: Optional[Dict[str, str]] = None,
		available_file_paths: Optional[list[str]] = None,
		#
		context: Context | None = None,
	) -> ActionResult:
		"""Execute an action"""

		try:
			for action_name, params in action.model_dump(exclude_unset=True).items():
				if params is not None:
					# with Laminar.start_as_current_span(
					# 	name=action_name,
					# 	input={
					# 		'action': action_name,
					# 		'params': params,
					# 	},
					# 	span_type='TOOL',
					# ):
					result = await self.registry.execute_action(
						action_name,
						params,
						browser=browser_context,
						page_extraction_llm=page_extraction_llm,
						sensitive_data=sensitive_data,
						available_file_paths=available_file_paths,
						context=context,
					)

					# Laminar.set_span_output(result)

					if isinstance(result, str):
						return ActionResult(extracted_content=result)
					elif isinstance(result, ActionResult):
						return result
					elif result is None:
						return ActionResult()
					else:
						raise ValueError(f'Invalid action result type: {type(result)} of {result}')
			return ActionResult()
		except Exception as e:
			raise e
