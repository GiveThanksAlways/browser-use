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
from browser_use.controller.greenhouse_helpers import GREENHOUSE_DROPDOWN_CACHE, reset_dropdown_tracking

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
CHROME_PATH = r'C:\Program Files\Google\Chrome\Application\chrome.exe'  # Windows
# CHROME_PATH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'  # macOS

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

# Define the output format as a Pydantic model
class JobListing(BaseModel):
    job_title: str = Field(description="The title of the job position")
    company_name: Optional[str] = Field(
        default="Anduril Industries",  # Default value since all jobs are from Anduril
        description="The name of the company offering the job"
    )
    location: Optional[str] = Field(
        default="",
        description="The location of the job"
    )
    apply_link: Optional[str] = Field(
        default="",
        description="URL link to apply for the job"
    )
    description: Optional[str] = Field(
        default="",
        description="Brief description of the job"
    )

class JobListingsChunk(BaseModel):
    jobs: List[JobListing] = Field(description="List of job listings found in this chunk")
    chunk_number: Optional[int] = Field(
        default=1,
        description="The number of this chunk in the sequence"
    )
    total_chunks: Optional[int] = Field(
        default=1,
        description="Total number of chunks processed"
    )

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
        
        # Use a prompt that guides the model to extract job listings
        prompt = '''
        Extract all job listings from this chunk of a job board page.
        This is chunk {chunk_num} of {total_chunks}.
        
        For each job listing you find, extract:
        1. Job title (REQUIRED)
        2. Company name (REQUIRED - if not explicitly mentioned, use "Anduril Industries")
        3. Location (if available)
        4. Link to apply (if available)
        5. Brief description (if available)
        
        IMPORTANT: Every job MUST have both a job_title and company_name.
        
        Only include job listings you can definitively identify in this chunk.
        Extraction goal: {goal}
        
        Page chunk: {page}
        '''
        
        template = PromptTemplate(
            input_variables=['chunk_num', 'total_chunks', 'goal', 'page'],
            template=prompt
        )
        
        try:
            # Use structured output to get properly formatted results
            chunk_result = await structured_llm.ainvoke(
                template.format(
                    chunk_num=i+1,
                    total_chunks=len(chunks),
                    goal=goal,
                    page=chunk
                )
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
                Return ONLY a JSON object with this exact structure:
                {
                  "jobs": [
                    {
                      "job_title": "Required job title",
                      "company_name": "Anduril Industries",
                      "location": "Optional location",
                      "apply_link": "Optional link",
                      "description": "Optional description"
                    }
                  ]
                }
                
                Extraction goal: {goal}
                Page chunk: {page}
                '''
                
                fallback_template = PromptTemplate(
                    input_variables=['goal', 'page'],
                    template=fallback_prompt
                )
                
                fallback_result = await regular_llm.ainvoke(
                    fallback_template.format(goal=goal, page=chunk)
                )
                
                # Try to extract JSON from the response
                import re
                json_pattern = r'(\{[\s\S]*\})'
                json_matches = re.findall(json_pattern, fallback_result.content)
                
                if json_matches:
                    parsed_data = json.loads(json_matches[0])
                    if "jobs" in parsed_data and isinstance(parsed_data["jobs"], list):
                        # Manually validate and fix each job
                        for job in parsed_data["jobs"]:
                            if "job_title" in job:
                                # Create a valid job object
                                valid_job = JobListing(
                                    job_title=job.get("job_title", "Unknown Position"),
                                    company_name=job.get("company_name", "Anduril Industries"),
                                    location=job.get("location", ""),
                                    apply_link=job.get("apply_link", ""),
                                    description=job.get("description", "")
                                )
                                all_jobs.append(valid_job)
                        
                        logger.info(f"Fallback extracted {len(parsed_data['jobs'])} jobs from chunk {i+1}")
            except Exception as fallback_err:
                logger.error(f"Fallback extraction also failed: {str(fallback_err)}")
    
    # Create the final combined result
    combined_result = {
        "jobs": [job.model_dump() for job in all_jobs],
        "total_jobs_found": len(all_jobs),
        "chunks_processed": len(chunks)
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
    browser = Browser(
        config=BrowserConfig(
            chrome_instance_path=CHROME_PATH,
            new_context_config=BrowserContextConfig(
                viewport_expansion=500,
            ),
        ),
    )
    # https://www.anduril.com/open-roles/?location=&department=Software&search=&gh_src=
    try:
        # Define the task for the agent
        task_description = f"""
        Extract information about all available jobs.
        For each job listing, extract:
        1. Job title
        2. Company name
        3. Location
        4. Link to apply
        5. Brief description (if available)
        
        Return the data in the required structured format.
        Only extract job listings - don't apply to any jobs.
        """

        extend_system_message = """
        IMPORTANT: Use the 'extract_job_listings_chunked' tool to handle the large page content.
        DO NOT use the regular 'extract_job_listings' or 'extract_content' tools as they may fail with large pages.

        The extract_job_listings_chunked tool will process the page in manageable chunks to avoid token limits. Only call this tool once. It will extract all of the jobs from the page.

        Return the structured data in JSON format with job listings.
        """
        
        # Initial actions for the agent
        initial_actions = [
            {'open_tab': {'url': url}},
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
                extend_system_message=extend_system_message
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
async def apply_to_job(job: Job, llm_gemini, llm_openai, resume_data: Dict[str, Any]) -> Tuple[bool, str]:
    """Apply to a single job and return success status and notes."""
    logger.info(f"Starting application for: {job.title} at {job.company}")
    
    # Create browser instance
    browser = Browser(
        config=BrowserConfig(
            chrome_instance_path=CHROME_PATH,
        )
    )
    
    try:
        # Define the dropdown-specific extend system message
        extend_system_message = (
            'For Greenhouse.io applications, use these known dropdown options for faster filling:\n'
            '- Disability Status: "Yes, I have a disability, or have had one in the past", "No, I do not have a disability and have not had one in the past", "I do not want to answer"\n'
            '- Gender: "Male", "Female", "Non-binary", "I do not wish to answer"\n'
            '- Veteran Status: "I identify as one or more of the classifications of protected veteran listed above", "I am not a protected veteran", "I don\'t wish to answer"\n'
            'Always check for these exact wordings first before using other approaches.'
        )
        
        # Initial actions for the agent
        initial_actions = [
            {'open_tab': {'url': job.link}},
            {'scroll_down': {'amount': 4200}},  # Initial scroll to see the form
        ]
        
        async with await browser.new_context() as browser_context:
            # Reset dropdown tracking before starting
            reset_dropdown_tracking()
            
            # STEP 1: First agent run - for dropdowns only
            logger.info("Starting first agent run - dropdown fields only")
            dropdown_task = f"""
            You are applying for the job: {job.title} at {job.company} in {job.location}.
            
            Use the resume information provided to fill out the application form. Only fill in the dropdown fields for now.
            
            Make sure to read through the resume data carefully before starting to fill out the form. 
            Skip the attach/submit buttons as well.
            
            IMPORTANT: For each dropdown, identify it first, then select the appropriate value from the options. 
            If you find yourself trying to interact with the same dropdown multiple times, try moving on to the next one.
            """
            
            dropdown_agent = Agent(
                task=dropdown_task,
                llm=llm_gemini,
                initial_actions=initial_actions,
                message_context=f"RESUME DATA:\n{json.dumps(resume_data, indent=2)}",
                max_input_tokens=128000,  # Ensure enough tokens for resume data
                browser=browser,
                browser_context=browser_context,
                extend_system_message=extend_system_message
            )
            
            await dropdown_agent.run()
            logger.info("Dropdown fields completed, now uploading resume...")
            
            # STEP 2: Resume upload agent (uses OpenAI LLM)
            resume_task = "Just upload the resume by using the upload_resume controller option. Attach resume. Don't do anything else. Just attach the resume."
            
            resume_agent = Agent(
                task=resume_task,
                llm=llm_openai,  # Using OpenAI specifically for resume upload
                controller=controller,
                initial_actions=[{'scroll_up': {'amount': 5000}}],
                browser=browser,
                browser_context=browser_context
            )
            
            await resume_agent.run()
            logger.info("Resume uploaded successfully")
            
            # Reset tracking for the text fields
            reset_dropdown_tracking()
            
            # STEP 3: Text fields agent
            text_task = f"""
            You are applying for the job: {job.title} at {job.company} in {job.location}.
            
            Only fill in the text fields. Don't do anything else. Don't click on dropdowns/buttons/links.
            Just fill in the text fields (think Name, email, phone, etc.)
            
            Make sure to read through the resume data carefully to fill in the appropriate information.
            DO NOT submit the application - pause at the final step for manual review.
            """
            
            text_agent = Agent(
                task=text_task,
                llm=llm_gemini,
                initial_actions=[{'scroll_up': {'amount': 5000}}],
                message_context=f"RESUME DATA:\n{json.dumps(resume_data, indent=2)}",
                max_input_tokens=128000,
                browser=browser,
                browser_context=browser_context,
            )
            
            await text_agent.run()
            logger.info("Text fields completed successfully")
            
            # Prompt for manual review
            print(f"\n\n{'='*80}")
            print(f"APPLICATION READY FOR REVIEW: {job.title} at {job.company}")
            print(f"{'='*80}")
            success = input("Was the application filled out correctly? (y/n): ").lower().startswith('y')
            notes = input("Additional notes for this application: ")
            
            return success, notes
    
    finally:
        # Close the browser
        await browser.close()
        logger.info(f"Browser closed after application to {job.title}")

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
        print("2. Apply to first 5 jobs (in parallel)")
        print("3. View jobs to apply")
        print("4. View applied jobs")
        print("5. Exit")
        choice = input("\nSelect an option (1-5): ")
        
        if choice == "1":
            # Scrape new job listings
            url = input("Enter job board URL: ")
            jobs = await scrape_job_listings(url, llm_gemini)  # Use Gemini for scraping
        
        elif choice == "2":
            # Apply to first 5 jobs in parallel
            jobs_to_apply = read_jobs_from_csv(TO_APPLY_CSV)[:5]
            
            if not jobs_to_apply:
                print("No jobs to apply to. Scrape some jobs first.")
                continue
            
            print(f"Preparing to apply to {len(jobs_to_apply)} jobs:")
            for i, job in enumerate(jobs_to_apply, 1):
                print(f"{i}. {job.title} at {job.company} - {job.location}")
            
            confirm = input("\nProceed with applications? (y/n): ")
            if not confirm.lower().startswith('y'):
                continue
            
            # Use asyncio.gather to run applications in parallel
            application_results = await asyncio.gather(
                *[apply_to_job(job, llm_gemini, llm_openai, resume_data) for job in jobs_to_apply]
            )
            
            # Process results
            for job, result_tuple in zip(jobs_to_apply, application_results):
                success, notes = result_tuple
                if success:
                    move_job_between_csvs(
                        job=job,
                        source_csv=TO_APPLY_CSV,
                        target_csv=APPLIED_CSV,
                        new_status="applied",
                        notes=notes
                    )
                    print(f"Successfully applied to: {job.title} at {job.company}")
                else:
                    # Update status for failed jobs
                    failed_jobs = read_jobs_from_csv(TO_APPLY_CSV)
                    for j in failed_jobs:
                        if j.link == job.link:
                            j.notes = notes
                            j.status = "failed"
                    
                    write_jobs_to_csv(failed_jobs, TO_APPLY_CSV)
                    print(f"Failed to apply to: {job.title} at {job.company}")
        
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

if __name__ == "__main__":
    asyncio.run(main()) 