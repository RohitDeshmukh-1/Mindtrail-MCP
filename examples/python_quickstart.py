"""Use the Mindtrail engine directly from Python, without MCP.

Run: python examples/python_quickstart.py
"""

import tempfile
from pathlib import Path

from mindtrail import MemoryService, MindtrailConfig

memory = MemoryService.from_config(MindtrailConfig(home=Path(tempfile.mkdtemp())))

memory.remember("The API is built with FastAPI and PostgreSQL", space_id="project:shop")
memory.remember("Run tests with `pytest -x`", space_id="project:shop")
old = memory.remember("Launch is planned for October 1", space_id="project:shop").memory
memory.remember("Launch moved to October 15", space_id="project:shop", supersedes=old.id)
memory.remember("User prefers concise answers with code first")  # personal space

print("Recall:")
for hit in memory.search("when is the launch?", space_ids=["project:shop"]):
    print(f"  {hit.score:.2f}  {hit.memory.content}")

print("\nPrompt-ready context:")
context = memory.get_context(
    "set up the test suite for the API", space_ids=["project:shop", "personal"], token_budget=400
)
print(context.text)

memory.close()
