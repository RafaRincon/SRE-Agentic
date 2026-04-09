"""
SRE Agent — Epistemic Falsifier Node (Track B)

Implements Popperian Falsification using the NATIVE Gemini SDK (google-genai),
with a transparent multi-turn agentic loop.

Architecture:
- Custom Tools (Function Calling): search_code, search_file, search_safeguards
  → RAG against the indexed eShop codebase (Cosmos DB DiskANN)
- Built-in Tool: code_execution
  → Gemini runs Python to do fuzzy matching / regex on retrieved code
- Tool combination via include_server_side_tool_invocations=True

We manually drive the agentic loop (turn by turn) to:
  1. Maintain full observability (no black-box)
  2. Eliminate deadlocks (we control iteration count)
  3. Properly circulate thought_signatures and tool IDs

Verdicts:
- CORROBORATED: Agent tried to falsify but found no counter-evidence
- FALSIFIED: Agent found concrete counter-evidence (cites it)
- INSUFFICIENT_EVIDENCE: File/function not found in index
"""

from __future__ import annotations

import asyncio
import logging
from pydantic import BaseModel, Field
from typing import Literal

from google import genai
from google.genai import types

from app.config import get_settings
from app.providers import llm_provider, db_provider

logger = logging.getLogger(__name__)

# Max turns in the agentic loop per hypothesis (a turn = model call)
MAX_FALSIFIER_TURNS = 6


# ---------------------------------------------------------------------------
# Verdict Schema
# ---------------------------------------------------------------------------


class FalsificationVerdict(BaseModel):
    hypothesis_id: str = Field(description="Summary of the hypothesis under test")
    verdict: Literal["CORROBORATED", "FALSIFIED", "INSUFFICIENT_EVIDENCE"]
    reasoning: str = Field(description="Which axes were checked and what was found")
    counter_evidence: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5)


# ---------------------------------------------------------------------------
# Function Declarations (custom RAG tools for Gemini)
# ---------------------------------------------------------------------------

SEARCH_CODE_DECL = {
    "name": "search_code",
    "description": (
        "Search the eShop codebase using semantic similarity. "
        "Use to find specific functions, null checks, error handling, "
        "DI registrations, or any code relevant to the hypothesis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Semantic search query, e.g. 'null check OrderItems CreateOrderCommandHandler'",
            }
        },
        "required": ["query"],
    },
}

SEARCH_FILE_DECL = {
    "name": "search_file",
    "description": (
        "Verify that a specific file exists in the indexed eShop codebase. "
        "Returns the file's content chunks if found, or 'NOT FOUND' as counter-evidence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Partial or full file path, e.g. 'CreateOrderCommandHandler.cs'",
            }
        },
        "required": ["file_path"],
    },
}

SEARCH_SAFEGUARDS_DECL = {
    "name": "search_safeguards",
    "description": (
        "Search for defensive code patterns (null checks, try/catch, "
        "validation, guard clauses) that could FALSIFY the hypothesis "
        "by preventing the claimed failure mode."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "context": {
                "type": "string",
                "description": "The function or code area to search for safeguards, e.g. 'Handle method OrderItems'",
            }
        },
        "required": ["context"],
    },
}

ALL_TOOL_DECLS = [SEARCH_CODE_DECL, SEARCH_FILE_DECL, SEARCH_SAFEGUARDS_DECL]


# ---------------------------------------------------------------------------
# Tool Executors (our side of the function calling contract)
# ---------------------------------------------------------------------------


async def _exec_search_code(query: str) -> str:
    try:
        embedding = await llm_provider.generate_embedding(query, task_type="RETRIEVAL_QUERY")
        results = db_provider.vector_search(query_vector=embedding, top_k=5)
        if not results:
            return "No matching code found in the indexed codebase."
        return "\n\n".join(
            f"FILE: {r.get('file_path', '?')} (L{r.get('start_line','?')}-{r.get('end_line','?')})\n"
            f"```csharp\n{r.get('chunk_text', '')}\n```"
            for r in results
        )
    except Exception as e:
        return f"search_code failed: {e}"


async def _exec_search_file(file_path: str) -> str:
    try:
        embedding = await llm_provider.generate_embedding(
            f"file {file_path}", task_type="RETRIEVAL_QUERY"
        )
        results = db_provider.vector_search(query_vector=embedding, top_k=8)
        matching = [r for r in results if file_path.lower() in r.get("file_path", "").lower()]
        if not matching:
            return f"NOT FOUND: '{file_path}' is not in the indexed codebase. This falsifies any hypothesis citing this file."
        return "\n\n".join(
            f"FOUND: {r.get('file_path', '')} (L{r.get('start_line','?')}-{r.get('end_line','?')})\n"
            f"```csharp\n{r.get('chunk_text', '')}\n```"
            for r in matching[:4]
        )
    except Exception as e:
        return f"search_file failed: {e}"


async def _exec_search_safeguards(context: str) -> str:
    try:
        query = f"null check guard clause validation try catch exception {context}"
        embedding = await llm_provider.generate_embedding(query, task_type="RETRIEVAL_QUERY")
        results = db_provider.vector_search(query_vector=embedding, top_k=5)
        if not results:
            return "No safeguard patterns found — hypothesis remains unfalsified on this axis."
        parts = []
        for r in results:
            text = r.get("chunk_text", "")
            safeguard_patterns = ["?? ", "is null", "!= null", "== null", "?.",
                                  "ArgumentNullException", "throw new", "if (", "guard"]
            has_guard = any(p in text.lower() for p in safeguard_patterns)
            tag = "⚠️ SAFEGUARD" if has_guard else "Code"
            parts.append(
                f"{tag} in {r.get('file_path', '?')}:\n```csharp\n{text}\n```"
            )
        return "\n\n".join(parts)
    except Exception as e:
        return f"search_safeguards failed: {e}"


async def _dispatch_tool(name: str, args: dict) -> str:
    """Route a function_call from the model to the correct executor."""
    if name == "search_code":
        return await _exec_search_code(args.get("query", ""))
    elif name == "search_file":
        return await _exec_search_file(args.get("file_path", ""))
    elif name == "search_safeguards":
        return await _exec_search_safeguards(args.get("context", ""))
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Epistemic Falsifier Prompt
# ---------------------------------------------------------------------------

_CODE_EXEC_EXAMPLE = (
    "```python\n"
    "# Example: Count null checks in retrieved C# code\n"
    "import re\n"
    "code = '...paste C# code here as a string...'\n"
    "null_checks = re.findall(r'[?][?]|is null|is not null|!= null|== null|[?][.]', code)\n"
    "try_catches = re.findall(r'try\\s*[{]', code)\n"
    "print(f'Null checks found: {len(null_checks)}')\n"
    "print(f'Try-catch blocks: {len(try_catches)}')\n"
    "for nc in null_checks:\n"
    "    print(f'  found: {nc}')\n"
    "```"
)

FALSIFIER_SYSTEM_PROMPT = (
    "You are an EPISTEMIC FALSIFIER operating under Karl Popper's philosophy of science.\n\n"
    "## Your Ontology\n\n"
    "A hypothesis H makes a CLAIM about the codebase:\n"
    '  "In file F, function G, condition C causes failure M"\n\n'
    "This generates TESTABLE PREDICTIONS. Your job is to check those predictions\n"
    "against the real indexed codebase using your tools.\n\n"
    "## Your Tools\n\n"
    "You have TWO categories of tools:\n\n"
    "### 1. RAG Tools (function calling — you call, we execute, we return results)\n"
    "- search_file(file_path): Verify a file exists. NOT FOUND = fabricated reference.\n"
    "- search_code(query): Semantic search for functions, DI registrations, patterns.\n"
    "- search_safeguards(context): Search for null checks, try/catch, validation.\n\n"
    "### 2. Code Execution (built-in — you write Python, it runs automatically)\n"
    "Use this to PROGRAMMATICALLY VERIFY retrieved C# code. Example:\n\n"
    + _CODE_EXEC_EXAMPLE + "\n\n"
    "CRITICAL: do not just read code with your eyes — COMPUTE over it.\n"
    "Regex is more reliable than neural pattern-matching for counting safeguards.\n\n"
    "## 4 Falsification Axes\n\n"
    "Axis 1 EXISTENCE: search_file -> does the file exist? NOT FOUND = FALSIFIED.\n"
    "Axis 2 MECHANISM: search_code -> does the function behave as claimed?\n"
    "Axis 3 DEFENSES: search_safeguards -> use code_execution to COUNT defensive\n"
    "  patterns. If count > 0 for the specific failure mode -> FALSIFIED\n"
    "Axis 4 CONTEXT: search_code for DI registration if H claims a wiring issue\n\n"
    "## Workflow\n\n"
    "1. Call search_file to check EXISTENCE\n"
    "2. Call search_code or search_safeguards to get real C# code\n"
    "3. Use code_execution to programmatically count/verify patterns\n"
    "4. Output your verdict\n\n"
    "## Decision Criteria\n\n"
    "- FALSIFIED: Found specific code that CONTRADICTS H. Cite the exact code.\n"
    "- CORROBORATED: Checked relevant axes, found NO counter-evidence.\n"
    "- INSUFFICIENT_EVIDENCE: File/function not in the index.\n\n"
    "## Output Format\n\n"
    "When done, output EXACTLY this block (no more tool calls after this):\n\n"
    "VERDICT: [CORROBORATED|FALSIFIED|INSUFFICIENT_EVIDENCE]\n"
    "REASONING: [which axes you checked and what code_execution computed]\n"
    "COUNTER_EVIDENCE: [exact code that falsifies, or none found]\n"
    "CONFIDENCE: [0.0-1.0]"
)



def _build_falsification_prompt(hypothesis: dict) -> str:
    h_desc = hypothesis.get("description", "No description")
    h_file = hypothesis.get("suspected_file", "unknown")
    h_func = hypothesis.get("suspected_function", "unknown")
    h_span = hypothesis.get("exact_span", "")
    h_conf = hypothesis.get("confidence", 0.5)

    return f"""## Hypothesis Under Test

**Claim**: {h_desc}
**Suspected File**: {h_file}
**Suspected Function**: {h_func}
**Cited Code Span**: ```{h_span}```
**Prior Confidence**: {h_conf}

## Pre-derived Falsification Plan

Axis 1 — EXISTENCE: call search_file("{h_file}")
  → NOT FOUND → FALSIFIED

Axis 2 — MECHANISM: call search_code("{h_func} {h_file}")
  → Does the code path described actually behave as claimed?

Axis 3 — DEFENSES: call search_safeguards("{h_func}")
  → Any guard/null-check that prevents the described failure?

Axis 4 — CONTEXT (if H claims DI/wiring issue):
  → call search_code("DI registration {h_file.split('/')[-1]}")

Execute the relevant axes. When you have enough evidence → output your verdict block."""


# ---------------------------------------------------------------------------
# Agentic Loop (manual, transparent)
# ---------------------------------------------------------------------------


async def falsify_hypothesis(hypothesis: dict) -> FalsificationVerdict:
    """
    Run the Popperian falsification loop for a single hypothesis.

    Uses the native Gemini SDK with:
    - Custom function declarations (RAG tools) — we execute them
    - Built-in code_execution — Gemini runs Python server-side
    - include_server_side_tool_invocations=True for tool context circulation

    The loop is driven manually (turn-by-turn), giving us full control
    and eliminating any possibility of an opaque deadlock.
    """
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)
    model = settings.gemini_model

    h_desc = hypothesis.get("description", "No description")

    # Build the tool config: custom declarations + built-in code_execution
    # include_server_side_tool_invocations lives in ToolConfig, not GenerateContentConfig
    tools = [
        types.Tool(
            function_declarations=ALL_TOOL_DECLS,
            code_execution=types.ToolCodeExecution(),  # built-in server-side
        )
    ]

    tool_config = types.ToolConfig(
        include_server_side_tool_invocations=True,  # enables tool context circulation
    )

    config = types.GenerateContentConfig(
        system_instruction=FALSIFIER_SYSTEM_PROMPT,
        tools=tools,
        tool_config=tool_config,
        temperature=0.1,
    )

    # Conversation history — grows each turn
    history: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part(text=_build_falsification_prompt(hypothesis))],
        )
    ]

    logger.info(f"[falsifier] Starting loop for: '{h_desc[:60]}'")

    for turn in range(MAX_FALSIFIER_TURNS):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=history,
                config=config,
            )
        except Exception as e:
            logger.error(f"[falsifier] Gemini API error on turn {turn}: {e}")
            break

        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content:
            logger.warning(f"[falsifier] Empty candidate on turn {turn}")
            break

        model_content = candidate.content
        # Add model response to history (preserves thought_signatures, IDs)
        history.append(model_content)

        # --- Classify what the response contains ---
        function_calls = []
        has_code_execution = False
        text_parts = []

        for p in model_content.parts:
            if p.function_call is not None:
                function_calls.append(p.function_call)
            if p.executable_code is not None:
                has_code_execution = True
                logger.info(
                    f"[falsifier] Turn {turn}: code_execution → "
                    f"{p.executable_code.code[:80]}..."
                )
            if p.code_execution_result is not None:
                has_code_execution = True
                logger.info(
                    f"[falsifier] Turn {turn}: code_execution result → "
                    f"{p.code_execution_result.output[:100] if p.code_execution_result.output else 'empty'}"
                )
            if p.text:
                text_parts.append(p.text)

        full_text = " ".join(text_parts)

        # --- Decision: is the model done? ---
        # The model is done if:
        #   1. No function_calls pending (our tools are not needed)
        #   2. The text contains a VERDICT line (model has decided)
        # Code execution parts are server-side (already ran), so they don't
        # require us to send anything back. But if the model used code_execution
        # WITHOUT producing a verdict, it may want to do more → continue loop.

        if function_calls:
            # Model wants to use our RAG tools — dispatch and return results
            logger.info(
                f"[falsifier] Turn {turn}: dispatching {len(function_calls)} tool(s): "
                f"{[fc.name for fc in function_calls]}"
            )
            function_response_parts = []
            for fc in function_calls:
                result = await _dispatch_tool(fc.name, dict(fc.args or {}))
                function_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": result},
                            id=fc.id,
                        )
                    )
                )
            history.append(
                types.Content(role="user", parts=function_response_parts)
            )
            continue  # next turn

        # No function calls — check if model has a verdict
        has_verdict = any(
            marker in full_text.lower()
            for marker in ["verdict: corroborated", "verdict: falsified",
                           "verdict: insufficient", "verdict:**"]
        )

        if has_verdict:
            logger.info(f"[falsifier] Turn {turn}: ✅ verdict reached")
            return _parse_verdict(full_text, hypothesis)

        if has_code_execution:
            # Model ran code but hasn't given a verdict yet.
            # It might need another turn to reason about the results.
            # Send a nudge to produce the verdict.
            logger.info(f"[falsifier] Turn {turn}: code ran, nudging for verdict")
            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(
                        text="Code execution complete. Now deliver your VERDICT block."
                    )],
                )
            )
            continue

        # Text without verdict and no tools — model might be thinking out loud
        # Give it one more chance
        if full_text:
            logger.info(f"[falsifier] Turn {turn}: text without verdict, nudging")
            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text="Now output your final VERDICT block.")],
                )
            )
            continue

        # Empty response — bail
        logger.warning(f"[falsifier] Turn {turn}: empty response, stopping")
        break

    # Loop exhausted
    # Try to parse whatever text we accumulated
    all_text = " ".join(
        p.text for c in history if c.role == "model"
        for p in c.parts if p.text
    )
    if all_text and "verdict:" in all_text.lower():
        return _parse_verdict(all_text, hypothesis)

    logger.warning(f"[falsifier] Max turns ({MAX_FALSIFIER_TURNS}) reached: '{h_desc[:60]}'")
    return FalsificationVerdict(
        hypothesis_id=h_desc[:80],
        verdict="INSUFFICIENT_EVIDENCE",
        reasoning=f"Falsification loop exhausted after {MAX_FALSIFIER_TURNS} turns.",
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Verdict Parser
# ---------------------------------------------------------------------------


def _parse_verdict(agent_output: str, hypothesis: dict) -> FalsificationVerdict:
    text = str(agent_output)
    lower = text.lower()

    if "verdict: falsified" in lower:
        verdict = "FALSIFIED"
    elif "verdict: corroborated" in lower:
        verdict = "CORROBORATED"
    else:
        verdict = "INSUFFICIENT_EVIDENCE"

    reasoning = text
    if "REASONING:" in text:
        reasoning = text.split("REASONING:")[-1].split("COUNTER_EVIDENCE:")[0].strip()

    counter_evidence = []
    if "COUNTER_EVIDENCE:" in text:
        ce = text.split("COUNTER_EVIDENCE:")[-1].split("CONFIDENCE:")[0].strip()
        if ce and "none" not in ce.lower():
            counter_evidence = [ce[:500]]

    confidence = 0.5
    if "CONFIDENCE:" in text:
        try:
            conf_str = text.split("CONFIDENCE:")[-1].strip().split()[0]
            confidence = float(conf_str.strip(".,:"))
            if confidence > 1:
                confidence /= 100
        except (ValueError, IndexError):
            pass

    return FalsificationVerdict(
        hypothesis_id=hypothesis.get("description", "")[:80],
        verdict=verdict,
        reasoning=reasoning[:600],
        counter_evidence=counter_evidence,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Main Node
# ---------------------------------------------------------------------------


async def falsifier_node(state: dict) -> dict:
    """
    Epistemic Falsifier node: spawns a Popperian falsification loop
    for each hypothesis, using native Gemini SDK with tool combination
    (function calling + code_execution).

    Runs all hypotheses in parallel.
    """
    hypotheses = state.get("hypotheses", [])

    if not hypotheses:
        logger.info("[falsifier] No hypotheses to falsify")
        return {"falsifier_verdicts": []}

    logger.info(f"[falsifier] Spawning {len(hypotheses)} falsification loops (parallel)...")

    tasks = [falsify_hypothesis(h) for h in hypotheses]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verdicts = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"[falsifier] Coroutine exception: {r}")
            verdicts.append(FalsificationVerdict(
                hypothesis_id="unknown",
                verdict="INSUFFICIENT_EVIDENCE",
                reasoning=f"Exception: {r}",
                confidence=0.0,
            ).model_dump())
        else:
            verdicts.append(r.model_dump())

    corroborated = sum(1 for v in verdicts if v["verdict"] == "CORROBORATED")
    falsified = sum(1 for v in verdicts if v["verdict"] == "FALSIFIED")
    insufficient = sum(1 for v in verdicts if v["verdict"] == "INSUFFICIENT_EVIDENCE")

    logger.info(
        f"[falsifier] ✅ {corroborated} corroborated | "
        f"❌ {falsified} falsified | "
        f"⚠️ {insufficient} insufficient_evidence"
    )

    return {"falsifier_verdicts": verdicts}
