import re

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from config import GROQ_MODEL
from git_context import gather_repo_context
from state import GitState

load_dotenv()

llm = ChatGroq(model=GROQ_MODEL)


def _build_planner_prompt(repo_rules: str) -> SystemMessage:
    return SystemMessage(
        content=f"""
You are a GitOps Task Planner.

You receive a rich repository snapshot: branches, numbered commits, files changed
per commit, file-to-commit map, branch heads, and divergence info.

Use this context directly. Do NOT plan discovery todos for data already in context.

Repository rules:
{repo_rules}

Your job:
1. For git operations: break the request into atomic, ordered todos.
2. For informational questions: answer directly from context.

RULES:
- Output ONLY in one of the formats below.
- Do NOT output shell commands.
- Do NOT output markdown fences.
- Each action todo maps to exactly one git command.
- Include commit hashes in todos when they appear in context.
- Commit #N on branch X = the #N entry under branch X in the commit index.
- For selective multi-branch work: plan create-branch then cherry-picks in order.
- Cherry-pick oldest commit first when order matters.
- Do not cherry-pick commits the user did not ask for.

FORMAT FOR OPERATIONS:

TODOS:
1. <todo>
2. <todo>

FORMAT FOR INFORMATIONAL QUESTIONS:

ANSWER:
<clear, accurate answer>

EXAMPLES:

User: create branch release from main, cherry-pick commit 2 from feature-a and commit 1 from hotfix
Context shows main tip abc1234, feature-a #2 def5678, hotfix #1 ghi9012

TODOS:
1. Create branch release at abc1234 (main tip)
2. Cherry-pick def5678 (commit #2 on feature-a) onto release
3. Cherry-pick ghi9012 (commit #1 on hotfix) onto release

User: which commit on developer added config.py?

ANSWER:
Use the FILE to COMMIT MAP and commit index from context.

User: create branch rett from developer commit 3

TODOS:
1. Create branch rett at the hash for commit #3 on developer (from context)
"""
    )


def extract_todos(text: str) -> list[str]:
    todos = []
    in_todos = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("TODOS:"):
            in_todos = True
            continue
        if stripped.upper().startswith("ANSWER:"):
            break
        if in_todos and re.match(r"^\d+\.", stripped):
            todos.append(re.sub(r"^\d+\.\s*", "", stripped))

    return todos


def extract_answer(text: str) -> str | None:
    match = re.search(r"ANSWER:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def planner_node(state: GitState):
    user_request = (
        state["original_request"]
        if state.get("replan")
        else state["messages"][-1].content
    )
    context = gather_repo_context(user_query=user_request)
    repo_rules = context["repo_rules"]

    if state.get("replan"):
        user_content = f"""
The workflow was interrupted and replanning is required.

Original Request:
{state["original_request"]}

User Correction / Update:
{state["human_response"]}

Generate a new plan using the updated request.
"""
    else:
        user_content = state["messages"][-1].content

    messages = [
        _build_planner_prompt(repo_rules),
        HumanMessage(
            content=f"""
REPOSITORY CONTEXT:
{context["summary"]}

USER REQUEST:
{user_content}
"""
        ),
    ]

    response = llm.invoke(messages)

    print("\n===== PLANNER =====")
    print(response.content)

    content = response.content.strip()
    answer = extract_answer(content)
    if answer:
        print("\n===== ANSWER =====")
        print(answer)
        return {
            "messages": [response],
            "todos": [],
            "current_todo": 0,
            "done": True,
            "final_answer": answer,
            "repo_context": context["summary"],
            "default_branch": context["default_branch"],
            "replan": False,
            "human_response": "",
        }

    todos = extract_todos(content)

    print("\n===== TODOS =====")
    print(todos)

    if not todos:
        raise ValueError(f"Planner produced invalid output:\n{content}")

    return {
        "messages": [response],
        "todos": todos,
        "current_todo": 0,
        "done": False,
        "final_answer": "",
        "repo_context": context["summary"],
        "default_branch": context["default_branch"],
        "replan": False,
        "human_response": "",
        "execution_history": state.get("execution_history", []),
    }
