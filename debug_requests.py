#!/usr/bin/env python3
"""Helper script to enable HTTP request logging for debugging"""

import logging
import sys

# Configure logging to show HTTP requests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

# Enable debug logging for httpx and openai
logging.getLogger('httpx').setLevel(logging.DEBUG)
logging.getLogger('openai').setLevel(logging.DEBUG)
logging.getLogger('core.llm_chat').setLevel(logging.DEBUG)

print("HTTP request logging enabled. Run your script now to see request details.", file=sys.stderr)
