"""Basic usage examples for langchain-tessai.

Requires a .env file with TESSAI_API_KEY, TESSAI_AGENT_ID, TESSAI_WORKSPACE_ID.
"""

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_tessai import ChatTessAI

load_dotenv()


def main() -> None:
    api_key = os.environ["TESSAI_API_KEY"]
    agent_id = int(os.environ["TESSAI_AGENT_ID"])
    workspace_id = int(os.environ["TESSAI_WORKSPACE_ID"])

    llm = ChatTessAI(
        api_key=api_key,
        agent_id=agent_id,
        workspace_id=workspace_id,
        model="tess-5",
        temperature=0.5,
    )

    # --- Simple invoke ---
    print("=== Invoke ===")
    response = llm.invoke("Hello, how can you help me today?")
    print(response.content)
    print()

    # --- With message history ---
    print("=== Message History ===")
    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant that speaks Portuguese."),
        HumanMessage(content="Ola!"),
        AIMessage(content="Oi! Como posso ajudar?"),
        HumanMessage(content="Qual a capital do Brasil?"),
    ])
    print(response.content)
    print()

    # --- Streaming ---
    print("=== Streaming ===")
    for chunk in llm.stream("Tell me a very short story in 3 sentences."):
        print(chunk.content, end="", flush=True)
    print("\n")

    # --- Batch ---
    print("=== Batch ===")
    responses = llm.batch(["Hello", "Goodbye"])
    for r in responses:
        print(f"  -> {r.content[:80]}...")
    print()

    # --- With Tess tools (internet search) ---
    print("=== With Internet Tool ===")
    llm_web = ChatTessAI(
        api_key=api_key,
        agent_id=agent_id,
        workspace_id=workspace_id,
        model="tess-5",
        tools="internet",
    )
    response = llm_web.invoke("What are the latest AI news?")
    print(response.content)


if __name__ == "__main__":
    main()
