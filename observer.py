from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from config import GROQ_MODEL
from state import GitState

load_dotenv()

llm = ChatGroq(model=GROQ_MODEL)

OBSERVER_PROMPT = SystemMessage(
    content="""
You are a GitOps Execution Reviewer.

You receive the current TODO, the command that ran, and its output.
Decide whether that command fully completed the TODO.

Rules:
- Judge only whether THIS command achieved THIS TODO.
- TODO is complete only if the command succeeded and the goal was met.
- Finding information (hash, branch list, status) completes when output contains it.
- Creating a branch completes only on successful creation.
- Cherry-pick completes when cherry-pick succeeds (no conflict).
- If cherry-pick reports conflict, TODO is NOT complete.
- If the branch already exists, TODO is NOT complete.

Output format:

TODO_COMPLETE: YES|NO

ANALYSIS:
<one or two sentences>
"""
)


def observer_node(state: GitState):
    todos = state.get("todos", [])
    current_idx = state.get("current_todo", 0)

    if current_idx >= len(todos):
        return {"done": True}

    current_todo = todos[current_idx]
    latest_message = state["messages"][-1].content if state.get("messages") else ""

    observation_input = f"""
CURRENT TODO:
{current_todo}

LAST RETURN CODE: {state.get("last_return_code", "unknown")}

{latest_message}
"""

    response = llm.invoke(
        [OBSERVER_PROMPT, HumanMessage(content=observation_input)]
    )

    review = response.content.strip()

    print("\n===== OBSERVER =====")
    print(f"TODO: {current_todo}")
    print(review)

    todo_complete = "TODO_COMPLETE: YES" in review.upper()

    return {
        "review": review,
        "todo_complete": todo_complete,
        "done": False,
    }
