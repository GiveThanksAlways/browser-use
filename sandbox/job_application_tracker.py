"""
Job Application Tracker and Automation

This script provides a functional programming approach to:
1. Scrape job listings from a main page
2. Apply to multiple jobs in parallel
3. Track application status in CSV files
4. Allow manual review before finalizing each application

Usage:
    python job_application_tracker.py
"""

import asyncio
import csv
import json
import os
import logging
from pathlib import Path
from functools import partial
from typing import List, Dict, Any, Optional, Callable, Tuple
import concurrent.futures
from dataclasses import dataclass, asdict
import re
import platform

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic  # Optional for using Claude
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr, BaseModel, Field

from browser_use import Agent, Controller, ActionResult
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Path configuration
SANDBOX_PATH = Path(__file__).parent
RESUME_PATH = SANDBOX_PATH / 'resume.json'
PDF_RESUME_PATH = SANDBOX_PATH / 'Resume_Spencer_Willett.pdf'
CONVERSATION_PATH = SANDBOX_PATH / 'debug/debug_conversation.txt'
TO_APPLY_CSV = SANDBOX_PATH / 'jobs_to_apply.csv'
APPLIED_CSV = SANDBOX_PATH / 'jobs_applied.csv'

# Browser configuration - update with your browser path
if platform.system() == 'Windows':
    CHROME_PATH = r'C:\Program Files\Google\Chrome\Application\chrome.exe'  # Windows
else:
    CHROME_PATH = '/opt/google/chrome/chrome'  # Linux

# Define job data structure
@dataclass
class Job:
    title: str
    company: str
    location: str
    link: str
    description: str = ""
    status: str = "to_apply"  # to_apply, applied, failed
    notes: str = ""

# Define a simpler Pydantic model with fewer required fields
class JobListing(BaseModel):
    job_title: str = Field(description="The title of the job position")
    apply_link: str = Field(description="URL link to apply for the job")
    company_name: str = Field(
        default="SpaceX",
        description="The name of the company offering the job"
    )
    location: Optional[str] = Field(
        default="",
        description="The location of the job"
    )
    description: Optional[str] = Field(
        default="",
        description="Brief description of the job"
    )

class JobListingsChunk(BaseModel):
    jobs: List[JobListing] = Field(description="List of job listings found in this chunk")

class JobListings(BaseModel):
    jobs: List[JobListing]

# CSV file handling functions
def ensure_csv_exists(file_path: Path, headers: List[str]) -> None:
    """Ensure CSV file exists with proper headers."""
    if not file_path.exists():
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        logger.info(f"Created new CSV file: {file_path}")

def read_jobs_from_csv(file_path: Path) -> List[Job]:
    """Read jobs from CSV file into Job objects."""
    if not file_path.exists():
        return []
    
    jobs = []
    with open(file_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            jobs.append(Job(**row))
    
    return jobs

def write_jobs_to_csv(jobs: List[Job], file_path: Path) -> None:
    """Write job objects to CSV file."""
    if not jobs:
        return
    
    # Get field names from the Job dataclass
    fieldnames = list(asdict(jobs[0]).keys())
    
    # Write to CSV with proper headers
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            writer.writerow(asdict(job))
    
    logger.info(f"Wrote {len(jobs)} jobs to {file_path}")

def move_job_between_csvs(job: Job, source_csv: Path, target_csv: Path, 
                          new_status: str, notes: str = "") -> None:
    """Move a job from one CSV to another with updated status."""
    # Read jobs from both CSVs
    source_jobs = read_jobs_from_csv(source_csv)
    target_jobs = read_jobs_from_csv(target_csv)
    
    # Find the job in source
    job_to_move = None
    remaining_jobs = []
    
    for j in source_jobs:
        if j.link == job.link:
            job_to_move = j
        else:
            remaining_jobs.append(j)
    
    if not job_to_move:
        logger.warning(f"Job not found in source CSV: {job.title} at {job.company}")
        return
    
    # Update job status and notes
    job_to_move.status = new_status
    job_to_move.notes = notes
    
    # Add to target CSV
    target_jobs.append(job_to_move)
    
    # Write back to files
    write_jobs_to_csv(remaining_jobs, source_csv)
    write_jobs_to_csv(target_jobs, target_csv)
    
    logger.info(f"Moved job '{job_to_move.title}' from {source_csv.name} to {target_csv.name}")

# Job application controller
controller = Controller()
controller_extract = Controller(output_model=JobListings)
controller_dropdown = Controller()

@controller.action('Upload resume to application form')
async def upload_resume(index: int, browser: BrowserContext):
    """Upload resume to a file upload element at the specified index."""
    path = str(PDF_RESUME_PATH.absolute())
    
    # Get the DOM element at the specified index
    dom_el = await browser.get_dom_element_by_index(index)
    if dom_el is None:
        return ActionResult(error=f'No element found at index {index}')
    
    # Get the file upload element
    file_upload_dom_el = dom_el.get_file_upload_element()
    if file_upload_dom_el is None:
        return ActionResult(error=f'No file upload element found at index {index}')
    
    # Get the locatable element
    file_upload_el = await browser.get_locate_element(file_upload_dom_el)
    if file_upload_el is None:
        return ActionResult(error=f'Could not locate file upload element at index {index}')
    
    try:
        # Upload the file
        await file_upload_el.set_input_files(path)
        success_msg = f'Successfully uploaded resume from "{path}" to element at index {index}'
        logger.info(success_msg)
        return ActionResult(extracted_content=success_msg, include_in_memory=True)
    except Exception as e:
        error_msg = f'Failed to upload resume to element at index {index}: {str(e)}'
        logger.error(error_msg)
        return ActionResult(error=error_msg)

@controller.action('Read resume data for filling forms')
def get_resume_data():
    """Provide resume data to the agent for form filling."""
    if not RESUME_PATH.exists():
        return ActionResult(error=f"Resume data not found at {RESUME_PATH}")
    
    with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
        resume_data = json.load(resume_file)
    
    return ActionResult(
        extracted_content=f"Resume data loaded: {json.dumps(resume_data, indent=2)}", 
        include_in_memory=True
    )

@controller_extract.action('Extract job listings from page in chunks (USE THIS FOR LARGE PAGES WITH MANY JOBS)')
async def extract_job_listings_chunked(goal: str, browser: BrowserContext, page_extraction_llm: BaseChatModel):
    page = await browser.get_current_page()
    import markdownify
    import json
    
    # Get the full page content
    content = markdownify.markdownify(await page.content())
    
    # Print a sample of the content for debugging
    logger.debug(f"Sample content: {content[:500]}")
    
    # Get the company name from the page title or URL
    company_name = "Unknown Company"
    try:
        page_title = await page.title()
        page_url = page.url
        
        # Try to extract company name from title
        if "careers" in page_title.lower() or "jobs" in page_title.lower():
            title_parts = page_title.split(" - ") or page_title.split(" | ")
            if len(title_parts) > 1:
                company_name = title_parts[0].strip()
        
        # If still unknown, try from URL
        if company_name == "Unknown Company" and page_url:
            # Extract domain from URL
            from urllib.parse import urlparse
            domain = urlparse(page_url).netloc
            domain_parts = domain.split('.')
            if len(domain_parts) > 1 and domain_parts[0] != "www":
                company_name = domain_parts[0].capitalize()
            elif len(domain_parts) > 2 and domain_parts[0] == "www":
                company_name = domain_parts[1].capitalize()
    
        logger.info(f"Detected company name: {company_name}")
    except Exception as e:
        logger.warning(f"Could not detect company name: {str(e)}")
    
    # Define chunking parameters
    chunk_size = 10000  # Characters per chunk
    max_chunks = 10     # Maximum number of chunks to process
    
    # Create chunks of the content
    chunks = []
    for i in range(0, len(content), chunk_size):
        if len(chunks) >= max_chunks:
            break
        chunk = content[i:i + chunk_size]
        chunks.append(chunk)
    
    logger.info(f"Split page content into {len(chunks)} chunks")
    
    # Process each chunk with structured output
    all_jobs = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        
        # Create a structured output version of the LLM
        structured_llm = page_extraction_llm.with_structured_output(JobListingsChunk)
        
        # Use a simpler prompt that focuses on the essential fields
        prompt = '''
        Extract job listings from this chunk of a job board page.
        
        For each job listing, I need:
        1. job_title (REQUIRED)
        2. apply_link (REQUIRED)
        
        FOCUS ON LINKS:
        - Look for markdown links in the format [Apply](URL)
        - Extract the COMPLETE URL from inside the parentheses
        - Do not modify or truncate the URLs
        
        Example in the content:
        ```
        Mission Software Engineer
        Costa Mesa, California, United States
        [Apply](https://boards.greenhouse.io/andurilindustries/jobs/4608506007)
        ```
        
        This should be extracted as:
        - job_title: "Mission Software Engineer"
        - apply_link: "https://boards.greenhouse.io/andurilindustries/jobs/4608506007"
        
        Page chunk: {page}
        '''
        
        template = PromptTemplate(
            input_variables=['page'],
            template=prompt
        )
        
        try:
            # Use structured output to get properly formatted results
            chunk_result = await structured_llm.ainvoke(
                template.format(page=chunk)
            )
            
            # Add jobs from this chunk to our overall list
            all_jobs.extend(chunk_result.jobs)
            logger.info(f"Successfully extracted {len(chunk_result.jobs)} jobs from chunk {i+1}")
            
        except Exception as e:
            logger.error(f"Error extracting content from chunk {i+1}: {str(e)}")
            
            # Try fallback approach with manual JSON parsing
            try:
                # Use regular invoke instead of structured output
                regular_llm = page_extraction_llm
                fallback_prompt = '''
                Extract job listings from this chunk of a job board page.
                Return ONLY a JSON array of job objects with this structure:
                [
                  {
                    "job_title": "Required job title",
                    "apply_link": "Required link from [Apply](URL) pattern"
                  }
                ]
                
                IMPORTANT: Look for markdown links in the format [Apply](URL) and extract the complete URL.
                
                Page chunk: {page}
                '''
                
                fallback_template = PromptTemplate(
                    input_variables=['page'],
                    template=fallback_prompt
                )
                
                fallback_result = await regular_llm.ainvoke(
                    fallback_template.format(page=chunk)
                )
                
                # Try to extract JSON from the response
                import re
                json_pattern = r'(\[[\s\S]*\])'
                content_str = str(fallback_result.content) if hasattr(fallback_result, 'content') else str(fallback_result)
                json_matches = re.findall(json_pattern, content_str)
                
                if json_matches:
                    parsed_data = json.loads(json_matches[0])
                    if isinstance(parsed_data, list):
                        # Manually validate and fix each job
                        for job in parsed_data:
                            if "job_title" in job and "apply_link" in job:
                                # Create a valid job object
                                valid_job = JobListing(
                                    job_title=job.get("job_title"),
                                    apply_link=job.get("apply_link"),
                                    company_name=company_name,
                                    location=job.get("location", ""),
                                    description=job.get("description", "")
                                )
                                all_jobs.append(valid_job)
                        
                        logger.info(f"Fallback extracted {len(parsed_data)} jobs from chunk {i+1}")
            except Exception as fallback_err:
                logger.error(f"Fallback extraction also failed: {str(fallback_err)}")
    
    # Create the final combined result
    combined_result = {
        "jobs": [job.model_dump() for job in all_jobs],
        "total_jobs_found": len(all_jobs)
    }
    
    result_json_str = json.dumps(combined_result, indent=2)
    
    msg = f'📄 Extracted {len(all_jobs)} jobs from {len(chunks)} page chunks\n'
    logger.info(msg)
    
    # Make sure we're returning a non-empty result
    if len(all_jobs) == 0:
        logger.warning("No jobs were extracted. Check the page structure or extraction logic.")
        # Return a message that will help debugging
        return ActionResult(
            extracted_content=f"No jobs could be extracted from the page. Please check the page structure.",
            include_in_memory=True
        )
    
    return ActionResult(extracted_content=result_json_str, include_in_memory=True)

# Job listing scraper function
async def scrape_job_listings(url: str, llm) -> List[Job]:
    """Scrape job listings from the main job board page."""
    logger.info(f"Starting job listing scraping from: {url}")
    
    # Create browser instance
    viewport_width, viewport_height = 1000, 1000
    start_x, start_y = 1920, 0
    browser = Browser(
        config=BrowserConfig(
            disable_security=True,
            headless=False,
            extra_chromium_args=[
                "--force-dark-mode",
                "--enable-features=WebContentsForceDark",
                f"--window-position={start_x},{start_y}",
                f"--window-size={viewport_width},{viewport_height}"  # This sets the actual window size of Chrome
            ]
        )
    )

    try:
        # Define the task for the agent
        task_description = f"""
        Extract information about all available jobs.
        For each job listing, extract:
        1. Job title (REQUIRED)
        2. Company name (if available)
        3. Location (if available)
        4. Link to apply (REQUIRED - find the URL from the "Apply" button or link)
        5. Brief description (if available)
        
        Return the data in the required structured format.
        Only extract job listings - don't apply to any jobs.
        """

        extend_system_message = f"""
            IMPORTANT: THIS IS A ONE-STEP TASK.

            Extract all job listings using EXACTLY ONE CALL to the extract_job_listings_chunked action.
            Do not repeat this action. Do not scroll multiple times.

            The extract_job_listings_chunked action will:
            1. Process the entire page automatically
            2. Extract all job listings from all sections of the page
            3. Handle scrolling internally
            4. Return ALL jobs in a single operation

            After the extract_job_listings_chunked action completes successfully, use the "done" action.
            Do not try to extract additional jobs or process the page further.
            """
        
        # Initial actions for the agent
        initial_actions = [
            {'go_to_url': {'url': url}},
            {'wait': {'seconds': 5}}
        ]
        
        # Create an agent to scrape the job listings
        async with await browser.new_context() as browser_context:
            agent = Agent(
                task=task_description,
                llm=llm,
                initial_actions=initial_actions,
                browser=browser,
                browser_context=browser_context,
                controller=controller_extract,  # Use the controller with output model
                max_input_tokens=1040000,
                save_conversation_path=str(CONVERSATION_PATH),
                extend_system_message=extend_system_message,
                max_failures=1,
                max_actions_per_step=1
            )
            
            # Run the agent
            history = await agent.run()
            
            scraped_output = history.extracted_content()
            # Get the structured result
            result = history.final_result()
            jobs = []
            
            if scraped_output:
                try:
                    # Parse the structured output
                    parsed_listings = JobListings.model_validate_json(scraped_output[0])
                    
                    # Convert to Job objects
                    for job_listing in parsed_listings.jobs:
                        job = Job(
                            title=job_listing.job_title,
                            company=job_listing.company_name,
                            location=job_listing.location,
                            link=job_listing.apply_link,
                            description=job_listing.description or "",
                            status="to_apply",
                            notes=""
                        )
                        jobs.append(job)
                    
                    logger.info(f"Successfully parsed {len(jobs)} jobs from structured output")
                except Exception as e:
                    logger.error(f"Failed to parse structured output: {e}")
                    logger.debug(f"Raw result: {scraped_output}")
            
            # If we still have no jobs, log details for debugging
            if not jobs:
                logger.warning("Failed to extract any jobs from the response")
        
        if jobs:
            # Add to to_apply CSV
            existing_jobs = read_jobs_from_csv(TO_APPLY_CSV)
            
            # Filter out duplicates by link
            existing_links = {job.link for job in existing_jobs}
            new_jobs = [job for job in jobs if job.link not in existing_links]
            
            if new_jobs:
                all_jobs = existing_jobs + new_jobs
                write_jobs_to_csv(all_jobs, TO_APPLY_CSV)
                print(f"Added {len(new_jobs)} new jobs to {TO_APPLY_CSV}")
            else:
                print("No new jobs found.")
        else:
            print("No jobs scraped.")
        return jobs
    
    finally:
        # Close the browser
        await browser.close()
        logger.info("Browser closed after scraping job listings")

# Job application function
async def apply_to_job(job: Job, llm_gemini, llm_gemini_tool_use, resume_data: Dict[str, Any], browser, browser_context) -> Tuple[bool, str, BrowserContext]:
    """Apply to a single job and return success status and notes."""
    logger.info(f"Starting application for: {job.title} at {job.company}")
    
    if browser is None:
        raise ValueError("Browser instance is required for apply_to_job function.")
    
    try:
        # Define the dropdown-specific extend system message with limited retry instructions
        extend_system_message = (
            'IMPORTANT: For handling dropdowns in job applications:\n\n'
            '1. For dropdowns with visible labels, use "fill_greenhouse_dropdown" with a SHORT version of the label and option text.\n'
            '   Example: For "CLEARANCE ELIGIBILITY - This position requires...", just use "CLEARANCE ELIGIBILITY"\n\n'
            '2. For dropdowns without clear labels, use "handle_custom_dropdown" with the dropdown index and option text.\n\n'
            'Common dropdown fields and their options:\n'
            '- Disability Status: "No, I do not have a disability and have not had one in the past"\n'
            '- Gender: "Male"\n'
            '- Veteran Status: "I am not a protected veteran"\n'
            '- Clearance Eligibility: "Yes, I am eligible for a U.S. security clearance"\n'
            '- Current Clearance Level: "N/A - have never held U.S. security clearance"\n\n'
            'IMPORTANT: If a dropdown action fails after 1 attempt, move on to other fields.\n'
            'If you cannot find any dropdown fields after 2 scroll attempts, assume they are not present and continue with other tasks.\n'
            'DO NOT use the standard select_dropdown_option action as it will not work with these custom UI components.\n\n'
            'CRITICAL: NEVER CLICK ANY BUTTON LABELED "SUBMIT", "APPLY", "SEND", "CONTINUE", OR SIMILAR. DO NOT CLICK ANY BUTTONS AT THE BOTTOM OF THE FORM.'
        )
        
        # Initial actions for the agent - add a pause at the beginning
        initial_actions = [
            {'go_to_url': {'url': job.link}},
            {'scroll_down': {'amount': 3200}},  # Initial scroll to see the form
        ]
        
        # Use the provided browser context or create a new one if needed
        if browser_context is None and browser is not None:
            browser_context = await browser.new_context()
            context_created_internally = True
        else:
            context_created_internally = False
            
        # STEP 1: Combined form filling agent - handles both dropdowns and text fields
        logger.info("Starting combined form filling agent - handling all form fields")
        combined_task = f"""
        You are applying for the job: {job.title} at {job.company} in {job.location}.
        
        Use the resume information provided to fill out the application form completely.
        Fill in BOTH dropdown fields AND text fields in a single pass.
        
        IMPORTANT INSTRUCTIONS:
        
        For dropdown fields:
        1. Use SHORT versions of field labels with "fill_greenhouse_dropdown"
           Example: For "CLEARANCE ELIGIBILITY - This position requires...", just use "CLEARANCE ELIGIBILITY"
        2. If that fails, try "handle_custom_dropdown" with the dropdown index
        3. IMPORTANT: If a dropdown is giving you trouble after 1 attempt, move on to other fields
        4. If you cannot find any dropdown fields after 2 scroll attempts, assume they are not present and continue
        5. DO NOT use the standard select_dropdown_option action
        
        For text fields:
        1. Fill in all text input fields with appropriate information from the resume
        2. Common fields include name, email, phone, address, work experience, education, etc.
        3. IMPORTANT: Make sure to fill in LinkedIn profile and website fields if present
           - For LinkedIn, use the full URL from the resume
           - For website, use the personal website or GitHub URL from the resume
        
        Common dropdown fields and their typical values:
        - Disability Status: "No, I do not have a disability and have not had one in the past"
        - Gender: "Male" (adjust based on applicant)
        - Veteran Status: "I am not a protected veteran"
        - Clearance Eligibility: "Yes, I am eligible for a U.S. security clearance"
        - Current Clearance Level: "N/A - have never held U.S. security clearance"
        
        CRITICAL RESTRICTIONS - YOU MUST FOLLOW THESE:
        - DO NOT CLICK ANY BUTTON LABELED "SUBMIT", "APPLY", "SEND", "CONTINUE", OR SIMILAR
        - DO NOT CLICK ANY BUTTONS AT THE BOTTOM OF THE FORM
        - DO NOT CLICK ANY BUTTONS THAT MIGHT SUBMIT THE FORM
        - DO NOT USE THE click_element ACTION ON ANY BUTTON THAT MIGHT SUBMIT THE FORM
        - Do not upload resume/cover letter (this will be handled separately)
        - Do not click on links that navigate away from the form
        - Do NOT click "Enter manually" buttons - just fill in the fields that are available
        - Do NOT try to enter work experience in a text area if it's not already visible
        
        FIELD PRIORITY ORDER:
        1. Basic information (name, email, phone)
        2. LinkedIn profile and website (don't skip these!)
        3. Work experience fields ONLY if they are already visible text fields
        4. Dropdown fields (clearance, gender, etc.) - but don't spend too much time on these if they're not working
        
        SCROLLING INSTRUCTIONS:
        - Scroll down in small increments (300-400 pixels at a time)
        - After scrolling down 3 times, check if you've reached the bottom of the form
        - If you see a Submit button or the end of the form, STOP scrolling down
        - Maximum 4 scroll_down actions total - then assume you've seen the whole form
        
        ADAPTIVE APPROACH:
        - If you can't find dropdown fields after 2 attempts, consider the form may not have them
        - Focus on completing all text fields correctly rather than getting stuck on dropdowns
        - After 3 failed dropdown attempts total, move on and report success on the fields you did complete
        - If you can't find a field to enter work experience, just skip it - DO NOT click "Enter manually"
        
        Make sure to read through the resume data carefully before starting to fill out the form.
        
        FINAL INSTRUCTION: When done filling out the form, use the "done" action with success=true. DO NOT CLICK SUBMIT.
        """
        
        # Set a timeout for the agent to prevent getting stuck
        combined_agent = Agent(
            task=combined_task,
            llm=llm_gemini,
            initial_actions=initial_actions,
            message_context=f"RESUME DATA:\n{json.dumps(resume_data, indent=2)}",
            max_input_tokens=128000,  # Ensure enough tokens for resume data
            browser=browser,
            browser_context=browser_context,
            extend_system_message=extend_system_message,
            controller=controller_dropdown
        )
        
        await combined_agent.run()
        logger.info("Form fields completed, now uploading resume...")
        
        # STEP 2: Resume upload agent (uses OpenAI LLM)
        resume_task = """
        Your ONLY task is to upload the resume using the upload_resume controller action.
        
        IMPORTANT:
        1. First scroll up to find the resume upload section
        2. Look for the resume upload field (usually near the top of the form)
        3. Use ONLY the upload_resume controller action to upload the resume
        4. DO NOT click any buttons labeled 'Attach' or 'Browse' - just use the upload_resume action directly
        5. DO NOT click any other buttons or links
        6. Skip the cover letter
        7. DO NOT CLICK ANY BUTTON LABELED "SUBMIT", "APPLY", "SEND", "CONTINUE", OR SIMILAR
        8. NEVER use click_element on any button at the bottom of the form
        
        EXAMPLE ACTION:
        {"upload_resume": {"index": 0}}  // Use the appropriate index number
        
        Just focus on using the upload_resume controller action directly.
        
        FINAL INSTRUCTION: When done uploading the resume, use the "done" action with success=true. DO NOT CLICK SUBMIT.
        """
        
        resume_agent = Agent(
            task=resume_task,
            llm=llm_gemini_tool_use,  # Using OpenAI specifically for resume upload
            controller=controller,
            initial_actions=[{'scroll_up': {'amount': 4000}}],
            browser=browser,
            browser_context=browser_context,
            extend_system_message='CRITICAL: NEVER CLICK ANY BUTTON LABELED "SUBMIT", "APPLY", "SEND", "CONTINUE", OR SIMILAR. DO NOT CLICK ANY BUTTONS AT THE BOTTOM OF THE FORM.'
        )
        
        await resume_agent.run()
        logger.info("Resume upload attempt completed")
        
        # Simply log that the application is ready and assume success
        print(f"Application prepared for: {job.title} at {job.company}")
        
        # Return success by default - we'll handle failures at the end
        return True, "", browser_context
    
    except Exception as e:
        logger.error(f"Error during application process: {str(e)}")
        # Return browser_context even in case of error to maintain type compatibility
        return False, f"Application failed with error: {str(e)}", browser_context

# Add a new function to handle the parallel job applications with review
async def apply_to_jobs_in_parallel(jobs_list, llm_gemini, llm_gemini_tool_use, resume_data):
    """Apply to multiple jobs in parallel with manual review."""
    if not jobs_list:
        print("No jobs to apply to.")
        return
    
    # Show jobs that will be processed
    print(f"\nPreparing to apply to {len(jobs_list)} jobs:")
    for i, job in enumerate(jobs_list, 1):
        print(f"{i}. {job.title} at {job.company} - {job.location}")
    
    confirm = input("\nProceed with applications? (Press Enter to continue, or 'n' to cancel): ")
    if confirm.lower().startswith('n'):
        return
    
    # Create a single browser instance for all jobs
    viewport_width, viewport_height = 1000, 1000
    start_x, start_y = 1920, 0
    viewport_expansion_pixels = -1
    browser = Browser(
        config=BrowserConfig(
            disable_security=True,
            headless=False,
            extra_chromium_args=[
                "--force-dark-mode",
                "--enable-features=WebContentsForceDark",
                f"--window-position={start_x},{start_y}",
                f"--window-size={viewport_width},{viewport_height}"  # This sets the actual window size of Chrome
            ],
            new_context_config=BrowserContextConfig(
                browser_window_size={'width': viewport_width, 'height': viewport_height},
                save_recording_path=f'./tmp/recordings',
                viewport_expansion=viewport_expansion_pixels
            )
        )
    )
    
    try:
        # Create application tasks for each job
        application_tasks = []
        browser_contexts = []
        
        for i, job in enumerate(jobs_list, 1):
            # Create a browser context for this job
            browser_context = await browser.new_context(BrowserContextConfig(browser_window_size={'width': viewport_width, 'height': viewport_height}, save_recording_path=f'./tmp/recordings', viewport_expansion=viewport_expansion_pixels))
            browser_contexts.append(browser_context)
            
            # Create an application task using apply_to_job
            task = apply_to_job(job, llm_gemini, llm_gemini_tool_use, resume_data, browser, browser_context)
            application_tasks.append(task)
        
        print("\nStarting job applications in parallel. Please wait...")
        
        # Run all application tasks in parallel
        application_results = await asyncio.gather(*application_tasks)
        
        print("\n" + "="*80)
        print("APPLICATION REVIEW PHASE")
        print("="*80)
        
        # Display a simplified table with just job title and status
        print("\nApplications Summary:")
        print(f"{'#':<3} {'Job Title':<70} {'Status':<10}")
        print("-" * 85)
        
        for i, (job, result) in enumerate(zip(jobs_list, application_results), 1):
            success, notes, _ = result
            status = "✅ Ready" if success else "❌ Failed"
            print(f"{i:<3} {job.title[:70]:<70} {status:<10}")
        
        print("\nPlease review all applications and manually submit them.")
        # Ask once for any failed applications
        failed_apps = input("\nEnter the numbers of any applications that failed or weren't submitted (comma-separated, or press Enter if all succeeded): ")
        
        if failed_apps.strip():
            # Process failed applications
            failed_indices = [int(idx.strip()) for idx in failed_apps.split(",") if idx.strip().isdigit()]
            
            # Update all other applications as successful
            for i, (job, _) in enumerate(zip(jobs_list, application_results), 1):
                if i not in failed_indices:
                    # Move successful job to applied CSV
                    move_job_between_csvs(
                        job=job,
                        source_csv=TO_APPLY_CSV,
                        target_csv=APPLIED_CSV,
                        new_status="applied",
                        notes="Manually submitted through automated process"
                    )
                    print(f"✅ Application {i} marked as submitted successfully!")
                else:
                    # Update notes for failed jobs
                    jobs_to_apply = read_jobs_from_csv(TO_APPLY_CSV)
                    write_jobs_to_csv(jobs_to_apply, TO_APPLY_CSV)
                    print(f"❌ Application {i} marked as failed")
        else:
            # All applications were successful
            for job in jobs_list:
                move_job_between_csvs(
                    job=job,
                    source_csv=TO_APPLY_CSV,
                    target_csv=APPLIED_CSV,
                    new_status="applied",
                    notes="Manually submitted through automated process"
                )
            print(f"✅ All {len(jobs_list)} applications marked as submitted successfully!")
        
        print("\nApplication process complete!")
    
    finally:
        # Close the browser only at the end of the entire process
        await browser.close()
        print("\nBrowser closed.")

def create_job_application_agent(job, llm_gemini, llm_openai, resume_data, browser, job_index):
    """Create an agent that will handle the entire job application process."""
    
    # Create a comprehensive task for the agent
    application_task = f"""
    You are applying for the job: {job.title} at {job.company} in {job.location}.
    
    Follow these steps in order:
    
    1. Go to the job application URL: {job.link}
    2. Fill out all form fields using the resume data provided
       - For dropdown fields, select the most appropriate option
       - For text fields, enter the relevant information from the resume
    3. Upload the resume when prompted (use the upload_resume controller action)
    4. DO NOT submit the application - pause at the final step for manual review
    
    IMPORTANT NOTES:
    - Use the resume data to fill out all fields accurately
    - If you encounter a field not covered by the resume data, use reasonable defaults
    - For Greenhouse.io applications, common dropdown options include:
      * Disability Status: "No, I do not have a disability and have not had one in the past"
      * Gender: "Male" 
      * Veteran Status: "I am not a protected veteran"
    
    When you're done, clearly indicate that the application is ready for review.
    """
    
    # Create the agent
    agent = Agent(
        task=application_task,
        llm=llm_gemini,
        initial_actions=[{'open_tab': {'url': job.link}}],
        message_context=f"RESUME DATA:\n{json.dumps(resume_data, indent=2)}",
        max_input_tokens=128000,  # Ensure enough tokens for resume data
        browser=browser,
        controller=controller,  # Use the custom controller with resume upload
    )
    
    return agent

# Main application function
async def main():
    # Initialize the models
    api_key_gemini = os.getenv('GEMINI_API_KEY')
    if not api_key_gemini:
        raise ValueError('GEMINI_API_KEY is not set')
        
    api_key_openai = os.getenv('OPENAI_API_KEY')
    if not api_key_openai:
        raise ValueError('OPENAI_API_KEY is not set')
    
    # Create Gemini LLM for most tasks
    llm_gemini = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash', 
        api_key=SecretStr(api_key_gemini),
        max_tokens=1040000
    )

    llm_gemini_tool_use = ChatGoogleGenerativeAI(
        model='gemini-2.0-flash-lite', 
        api_key=SecretStr(api_key_gemini),
        max_tokens=1040000
    )
    
    # Create OpenAI LLM for resume upload only
    llm_openai = ChatOpenAI(
        model='gpt-4o-mini',
        api_key=SecretStr(api_key_openai)
    )
    
    # Ensure CSV files exist
    job_fields = ["title", "company", "location", "link", "description", "status", "notes"]
    ensure_csv_exists(TO_APPLY_CSV, job_fields)
    ensure_csv_exists(APPLIED_CSV, job_fields)
    
    # Load resume data
    if not RESUME_PATH.exists():
        raise ValueError(f"Resume data not found at {RESUME_PATH}")
    
    with open(RESUME_PATH, 'r', encoding='utf-8') as resume_file:
        resume_data = json.load(resume_file)
    
    # Menu system
    while True:
        print("\n" + "="*50)
        print("JOB APPLICATION TRACKER")
        print("="*50)
        print("1. Scrape new job listings")
        print("2. Apply to jobs in parallel")
        print("3. View jobs to apply")
        print("4. View applied jobs")
        print("5. Exit")
        choice = input("\nSelect an option (1-5): ")
        
        if choice == "1":
            # Scrape new job listings
            url = input("Enter job board URL: ")
            jobs = await scrape_job_listings(url, llm_gemini)  # Use Gemini for scraping
        
        elif choice == "2":
            # Apply to jobs in parallel with custom count
            jobs_to_apply = read_jobs_from_csv(TO_APPLY_CSV)
            
            if not jobs_to_apply:
                print("No jobs to apply to. Scrape some jobs first.")
                continue
            
            # Show total available jobs
            print(f"\nThere are {len(jobs_to_apply)} jobs available to apply to.")
            
            # Ask for number of jobs to process
            while True:
                try:
                    num_jobs = input(f"How many jobs would you like to apply to? (1-{len(jobs_to_apply)}, default=10): ")
                    if not num_jobs:
                        num_jobs = 10  # Default value
                    else:
                        num_jobs = int(num_jobs)
                    
                    if 1 <= num_jobs <= len(jobs_to_apply):
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(jobs_to_apply)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            # Get the specified number of jobs
            selected_jobs = jobs_to_apply[:num_jobs]
            
            # Apply to the selected jobs
            await apply_to_jobs_in_parallel(selected_jobs, llm_gemini, llm_gemini_tool_use, resume_data)
        
        elif choice == "3":
            # View jobs to apply
            jobs = read_jobs_from_csv(TO_APPLY_CSV)
            
            if not jobs:
                print("No jobs to apply to.")
                continue
            
            print(f"\nJOBS TO APPLY ({len(jobs)}):")
            print(f"{'ID':<4} | {'TITLE':<30} | {'COMPANY':<20} | {'LOCATION':<20}")
            print("-" * 80)
            
            for i, job in enumerate(jobs, 1):
                print(f"{i:<4} | {job.title[:28]:<30} | {job.company[:18]:<20} | {job.location[:18]:<20}")
        
        elif choice == "4":
            # View applied jobs
            jobs = read_jobs_from_csv(APPLIED_CSV)
            
            if not jobs:
                print("No applied jobs.")
                continue
            
            print(f"\nAPPLIED JOBS ({len(jobs)}):")
            print(f"{'ID':<4} | {'TITLE':<30} | {'COMPANY':<20} | {'LOCATION':<20}")
            print("-" * 80)
            
            for i, job in enumerate(jobs, 1):
                print(f"{i:<4} | {job.title[:28]:<30} | {job.company[:18]:<20} | {job.location[:18]:<20}")
        
        elif choice == "5":
            # Exit
            print("Exiting application. Goodbye!")
            break
        
        else:
            print("Invalid option. Please try again.")

@controller_dropdown.action('Fill Greenhouse dropdown field')
async def fill_greenhouse_dropdown(field_label: str, option_text: str, browser: BrowserContext):
    """
    Fill a Greenhouse.io dropdown field by its label and option text.
    
    Args:
        field_label: The label text of the field (e.g., "Gender", "Veteran Status")
        option_text: The text of the option to select
        browser: The browser context
    """
    logger.info(f"Filling Greenhouse dropdown: {field_label} with value: {option_text}")
    
    try:
        page = await browser.get_current_page()
        
        # Step 1: Find the dropdown container by its label (using partial text match)
        # Create a shorter version of the label for matching
        short_label = field_label.split(' - ')[0] if ' - ' in field_label else field_label
        if len(short_label) > 30:
            short_label = short_label[:30]  # Use first 30 chars for very long labels
        
        label_selector = f"label:text-matches('{short_label}', 'i')"
        label_elements = await page.query_selector_all(label_selector)
        
        if not label_elements:
            # Try a more general approach
            all_labels = await page.query_selector_all('label')
            label_element = None
            
            for label in all_labels:
                text = await label.text_content()
                if text and short_label.lower() in text.lower():
                    label_element = label
                    break
                    
            if not label_element:
                return ActionResult(error=f"Could not find field with label containing '{short_label}'")
        else:
            label_element = label_elements[0]
        
        # Get the ID from the label's for attribute
        field_id = await label_element.get_attribute('for')
        
        # Step 2: Click the dropdown to open it
        if field_id:
            dropdown_selector = f"#{field_id}"
            dropdown = await page.query_selector(dropdown_selector)
        else:
            # Try to find the dropdown near the label
            dropdown = await label_element.evaluate('el => el.closest(".field")?.querySelector(".select-selected, [role=combobox], select, .dropdown")')
        
        if not dropdown:
            # Try to find any clickable element near the label
            dropdown = await label_element.evaluate('el => el.closest(".field")?.querySelector("div[class*=select], div[class*=dropdown], button")')
        
        if not dropdown:
            return ActionResult(error=f"Could not find dropdown for field '{short_label}'")
        
        await dropdown.click()
        await asyncio.sleep(0.5)  # Small delay to let the dropdown open
        
        # Step 3: Find and click the option
        option_selectors = [
            f"div[role='option']:text-matches('{option_text}', 'i')",
            f"li[role='option']:text-matches('{option_text}', 'i')",
            f".select-items div:text-matches('{option_text}', 'i')",
            f".dropdown-menu li:text-matches('{option_text}', 'i')"
        ]
        
        for selector in option_selectors:
            try:
                options = await page.query_selector_all(selector)
                if options:
                    await options[0].click()
                    logger.info(f"Selected '{option_text}' for field '{short_label}'")
                    return ActionResult(
                        extracted_content=f"Selected '{option_text}' for field '{short_label}'",
                        include_in_memory=True
                    )
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {str(e)}")
        
        # If we get here, try a more general approach - find any element containing the option text
        all_elements = await page.query_selector_all('div, li, span')
        for element in all_elements:
            try:
                text = await element.text_content()
                if text and option_text.lower() in text.lower():
                    await element.click()
                    logger.info(f"Selected element with text '{text}'")
                    return ActionResult(
                        extracted_content=f"Selected element with text '{text}' from dropdown",
                        include_in_memory=True
                    )
            except Exception:
                continue
        
        return ActionResult(
            error=f"Could not find option '{option_text}' for field '{short_label}'",
            include_in_memory=True
        )
        
    except Exception as e:
        error_msg = f"Error filling Greenhouse dropdown: {str(e)}"
        logger.error(error_msg)
        return ActionResult(error=error_msg)

@controller_dropdown.action('Handle custom dropdown selection (for React/modern dropdowns)')
async def handle_custom_dropdown(dropdown_index: int, option_text: str, browser: BrowserContext):
    """
    Handle selection in modern custom dropdowns (like React Select) that aren't standard HTML select elements.
    
    Args:
        dropdown_index: The index of the dropdown trigger element
        option_text: The text of the option to select
        browser: The browser context
    """
    logger.info(f"Handling custom dropdown at index {dropdown_index}, selecting '{option_text}'")
    
    try:
        # Step 1: Click the dropdown to open it
        dom_el = await browser.get_dom_element_by_index(dropdown_index)
        if dom_el is None:
            return ActionResult(error=f"No element found at index {dropdown_index}")
        
        dropdown_el = await browser.get_locate_element(dom_el)
        if dropdown_el is None:
            return ActionResult(error=f"Could not locate dropdown element at index {dropdown_index}")
        
        # Click to open the dropdown
        await dropdown_el.click()
        await asyncio.sleep(1.0)  # Longer delay to let the dropdown open fully
        
        # Step 2: Find and click the option with matching text
        page = await browser.get_current_page()
        
        # Try different selectors that are commonly used for dropdown options
        option_selectors = [
            f"div[role='option']:text-matches('{option_text}', 'i')",
            f"li[role='option']:text-matches('{option_text}', 'i')",
            f".select__option:text-matches('{option_text}', 'i')",
            f"[id*='react-select'][id*='option']:text-matches('{option_text}', 'i')",
            f".select-items div:text-matches('{option_text}', 'i')",
            f".dropdown-menu li:text-matches('{option_text}', 'i')"
        ]
        
        for selector in option_selectors:
            try:
                options = await page.query_selector_all(selector)
                if options:
                    await options[0].click()
                    logger.info(f"Successfully selected option '{option_text}' from dropdown")
                    return ActionResult(
                        extracted_content=f"Selected '{option_text}' from dropdown at index {dropdown_index}",
                        include_in_memory=True
                    )
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {str(e)}")
        
        # If specific selectors fail, try a more general approach
        # Look for any visible element that contains the option text
        all_elements = await page.query_selector_all('div, li, span')
        for element in all_elements:
            try:
                text = await element.text_content()
                if text and option_text.lower() in text.lower():
                    is_visible = await element.is_visible()
                    if is_visible:
                        await element.click()
                        logger.info(f"Selected element with text '{text}'")
                        return ActionResult(
                            extracted_content=f"Selected element with text '{text}' from dropdown",
                            include_in_memory=True
                        )
            except Exception:
                continue
        
        # If we get here, we couldn't find the option
        return ActionResult(
            error=f"Could not find option '{option_text}' in the dropdown",
            include_in_memory=True
        )
        
    except Exception as e:
        error_msg = f"Error handling custom dropdown: {str(e)}"
        logger.error(error_msg)
        return ActionResult(error=error_msg)

@controller_dropdown.action('Fill profile URLs (LinkedIn/Website)')
async def fill_profile_urls(linkedin_url: str, website_url: str, browser: BrowserContext):
    """
    Fill LinkedIn and website fields by finding them based on their labels.
    
    Args:
        linkedin_url: The LinkedIn profile URL
        website_url: The website URL (GitHub or personal site)
        browser: The browser context
    """
    logger.info(f"Filling profile URLs: LinkedIn={linkedin_url}, Website={website_url}")
    
    try:
        page = await browser.get_current_page()
        
        # Find LinkedIn field
        linkedin_selectors = [
            "input[id*='linkedin' i]",
            "input[name*='linkedin' i]",
            "input[placeholder*='linkedin' i]",
            "label:text-matches('LinkedIn', 'i') + input",
            "label:text-matches('LinkedIn', 'i')"
        ]
        
        linkedin_field = None
        for selector in linkedin_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    linkedin_field = elements[0]
                    break
            except Exception:
                continue
                
        if linkedin_field:
            # If we found a label, get the associated input
            if await linkedin_field.evaluate('el => el.tagName.toLowerCase()') == 'label':
                field_id = await linkedin_field.get_attribute('for')
                if field_id:
                    linkedin_field = await page.query_selector(f"#{field_id}")
                else:
                    # Try to find the input near the label
                    linkedin_field = await linkedin_field.evaluate('el => el.closest(".field")?.querySelector("input")')
            
            if linkedin_field:
                await linkedin_field.fill(linkedin_url)
                logger.info(f"Successfully filled LinkedIn URL: {linkedin_url}")
        
        # Find Website field
        website_selectors = [
            "input[id*='website' i]",
            "input[name*='website' i]",
            "input[placeholder*='website' i]",
            "label:text-matches('Website', 'i') + input",
            "label:text-matches('Website', 'i')"
        ]
        
        website_field = None
        for selector in website_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    website_field = elements[0]
                    break
            except Exception:
                continue
                
        if website_field:
            # If we found a label, get the associated input
            if await website_field.evaluate('el => el.tagName.toLowerCase()') == 'label':
                field_id = await website_field.get_attribute('for')
                if field_id:
                    website_field = await page.query_selector(f"#{field_id}")
                else:
                    # Try to find the input near the label
                    website_field = await website_field.evaluate('el => el.closest(".field")?.querySelector("input")')
            
            if website_field:
                await website_field.fill(website_url)
                logger.info(f"Successfully filled Website URL: {website_url}")
        
        return ActionResult(
            extracted_content=f"Filled LinkedIn URL: {linkedin_url if linkedin_field else 'Not found'}, Website URL: {website_url if website_field else 'Not found'}",
            include_in_memory=True
        )
        
    except Exception as e:
        error_msg = f"Error filling profile URLs: {str(e)}"
        logger.error(error_msg)
        return ActionResult(error=error_msg)

if __name__ == "__main__":
    asyncio.run(main()) 