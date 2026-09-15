import agent
from ai_router import llm_text

# Keep AriaPsi's existing validation, memory, and Moltbook logic.
# Replace only the single-provider text generation layer.
agent.llm_text = llm_text

agent.cycle()
