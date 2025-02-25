"""Helper module for Greenhouse.io application forms."""
import logging
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Common Greenhouse.io dropdown options (EEOC and other common fields)
GREENHOUSE_DROPDOWN_CACHE = {
    "disability_status": [
        "Yes, I have a disability, or have had one in the past",
        "No, I do not have a disability and have not had one in the past",
        "I do not want to answer"
    ],
    "gender": [
        "Male",
        "Female", 
        "Non-binary",
        "I do not wish to answer"
    ],
    "veteran_status": [
        "I identify as one or more of the classifications of protected veteran listed above",
        "I am not a protected veteran",
        "I don't wish to answer"
    ],
    "race_ethnicity": [
        "American Indian or Alaska Native",
        "Asian",
        "Black or African American",
        "Hispanic or Latino",
        "Native Hawaiian or Other Pacific Islander",
        "White",
        "Two or More Races",
        "I do not wish to answer"
    ]
}

# Track which dropdowns have been completed to avoid loops
COMPLETED_DROPDOWNS: Set[Tuple[int, str]] = set()

def is_greenhouse_dropdown(element_node) -> Tuple[bool, Optional[str]]:
    """Detect if an element is a Greenhouse.io dropdown by its structure and attributes."""
    # Check for common Greenhouse.io dropdown identifiers
    if element_node.tag_name != 'div':
        return False, None
        
    # Look for the select__control class in this element or its children
    has_select_control = False
    select_class_found = None
    
    if 'class' in element_node.attributes:
        class_attr = element_node.attributes['class']
        if 'select__control' in class_attr:
            has_select_control = True
            select_class_found = 'select__control in main element'
        elif 'select__container' in class_attr:
            has_select_control = True
            select_class_found = 'select__container in main element'
    
    if not has_select_control:
        for child in element_node.children:
            if hasattr(child, 'attributes') and 'class' in child.attributes:
                if 'select__control' in child.attributes['class']:
                    has_select_control = True
                    select_class_found = 'select__control in child element'
                    break
                elif 'select__container' in child.attributes['class']:
                    has_select_control = True
                    select_class_found = 'select__container in child element'
                    break
    
    # Look for field identifier
    field_id = None
    
    # Check for ID in main element first
    if 'id' in element_node.attributes:
        field_id = element_node.attributes['id']
        if field_id.endswith('-label'):
            field_id = field_id[:-6]
            
    # If no ID found in main element, check children
    if not field_id:
        for child in element_node.children:
            if hasattr(child, 'attributes') and 'id' in child.attributes:
                field_id = child.attributes['id']
                # Often IDs end with "-label"
                if field_id.endswith('-label'):
                    field_id = field_id[:-6]
                break
    
    # Log the detection results
    if has_select_control:
        logger.info(f"Greenhouse dropdown detected: class={select_class_found}, field_id={field_id}")
    
    return has_select_control, field_id

def get_greenhouse_dropdown_options(field_id: Optional[str]) -> Optional[List[str]]:
    """Get cached options for a Greenhouse dropdown if available."""
    if not field_id:
        return None
        
    # Try exact match first
    options = GREENHOUSE_DROPDOWN_CACHE.get(field_id)
    if options:
        logger.info(f"Found exact match for field_id '{field_id}' with {len(options)} options")
        return options
        
    # Try partial match
    for cache_key, cache_options in GREENHOUSE_DROPDOWN_CACHE.items():
        if cache_key in field_id:
            logger.info(f"Found partial match: '{cache_key}' in '{field_id}' with {len(cache_options)} options")
            return cache_options
            
    logger.info(f"No cached options found for field_id '{field_id}'")
    return None

def track_dropdown_completion(index: int, text: str) -> bool:
    """Track completed dropdowns to avoid loops.
    
    Returns:
        bool: True if this is a new completion, False if already completed
    """
    dropdown_key = (index, text)
    if dropdown_key in COMPLETED_DROPDOWNS:
        logger.warning(f"Dropdown {index} with value '{text}' was already completed!")
        return False
        
    COMPLETED_DROPDOWNS.add(dropdown_key)
    logger.info(f"Tracking dropdown {index} with value '{text}' as completed")
    return True

def reset_dropdown_tracking():
    """Reset the dropdown completion tracking."""
    COMPLETED_DROPDOWNS.clear()
    logger.info("Reset dropdown completion tracking")
    
def get_dropdown_completion_status() -> str:
    """Get a string representation of completed dropdowns for logging."""
    if not COMPLETED_DROPDOWNS:
        return "No dropdowns completed yet"
    
    completed = [f"Index {idx}: '{text}'" for idx, text in COMPLETED_DROPDOWNS]
    return f"Completed dropdowns ({len(COMPLETED_DROPDOWNS)}): {', '.join(completed)}" 