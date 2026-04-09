"""
SRE Agent — Epistemic Falsifier Node (Track B)

Popperian Falsification with native Gemini SDK (google-genai).

Architecture:
  - 2 focused tools: lookup_code (existence check) + check_defenses (safeguard scan)
  - Built-in code_execution for programmatic verification
  - Manual turn-by-turn loop for full observability
  - Tool results include CONCLUSIONS, not just raw code

Design principles (from Google FC best practices):
  - Few tools with clear, distinct purposes (not 4 overlapping search tools)
  - Rich, self-interpreting tool results (the model reads conclusions, not C# walls)
  - temperature=1, thinking_level=MEDIUM for Gemini 3 Flash
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

MAX_FALSIFIER_TURNS = 4


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
# Tool Declarations — only 2 tools, clearly distinct
# ---------------------------------------------------------------------------

LOOKUP_CODE_DECL = {
    "name": "lookup_code",
    "description": (
        "Look up a file and function in the indexed codebase. "
        "Returns the actual source code if found, or 'FILE_NOT_FOUND' if the file "
        "does not exist. Use this FIRST to verify the hypothesis references real code. "
        "The result includes a CONCLUSION telling you whether the file and span exist."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "The filename to look up, e.g. 'CreateOrderCommandHandler.cs'",
            },
            "function_name": {
                "type": "string",
                "description": "The function or method name to find, e.g. 'Handle'",
            },
        },
        "required": ["filename"],
    },
}

CHECK_DEFENSES_DECL = {
    "name": "check_defenses",
    "description": (
        "Search for defensive code (null checks, try/catch, validation, guard clauses) "
        "that would PREVENT the failure described in the hypothesis. "
        "Returns matching code with a CONCLUSION: how many safeguards were found "
        "and whether they cover the specific failure mode. "
        "Call this AFTER lookup_code confirms the file exists."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "failure_description": {
                "type": "string",
                "description": "What failure the hypothesis claims, e.g. 'NullReferenceException when iterating OrderItems'",
            },
            "function_name": {
                "type": "string",
                "description": "The function to check for defenses, e.g. 'Handle'",
            },
        },
        "required": ["failure_description", "function_name"],
    },
}

ALL_TOOL_DECLS = [LOOKUP_CODE_DECL, CHECK_DEFENSES_DECL]


# ---------------------------------------------------------------------------
# Tool Executors — results include CONCLUSIONS
# ---------------------------------------------------------------------------


async def _exec_lookup_code(filename: str, function_name: str = "") -> str:
    """Direct Cosmos lookup by filename. Returns code + a clear conclusion."""
    try:
        settings = get_settings()
        container = db_provider.get_container(settings.cosmos_container_chunks)

        # Direct SQL — no vector search needed for existence check
        fn_lower = filename.split("/")[-1].lower()
        query = """
        SELECT TOP 3
            c.file_path, c.chunk_text, c.start_line, c.end_line,
            c.class_name, c.method_name
        FROM c
        WHERE CONTAINS(LOWER(c.file_path), @fn)
        ORDER BY c.start_line
        """
        params = [{"name": "@fn", "value": fn_lower}]
        results = list(
            container.query_items(query=query, parameters=params,
                                  enable_cross_partition_query=True)
        )

        if not results:
            return (
                f"FILE_NOT_FOUND: '{filename}' does not exist in the indexed codebase.\n\n"
                "CONCLUSION: The hypothesis references a file that is NOT in the codebase. "
                "This is enough to FALSIFY the hypothesis with confidence 1.0."
            )

        # Build response with code
        chunks = []
        method_found = False
        for r in results:
            mname = r.get("method_name", "") or ""
            if function_name and function_name.lower() in mname.lower():
                method_found = True
            chunks.append(
                f"— {r['file_path']} L{r.get('start_line','?')}-{r.get('end_line','?')} "
                f"class={r.get('class_name','?')} method={mname}\n"
                f"```csharp\n{r.get('chunk_text', '')}\n```"
            )

        code_block = "\n\n".join(chunks)

        # Conclusion
        if function_name and not method_found:
            conclusion = (
                f"CONCLUSION: File '{filename}' EXISTS in the codebase, but the method "
                f"'{function_name}' was not found in the returned chunks. "
                "The method may exist in a different chunk or may be named differently."
            )
        elif function_name and method_found:
            conclusion = (
                f"CONCLUSION: File '{filename}' EXISTS and method '{function_name}' FOUND. "
                "The hypothesis references real code. Now call check_defenses to see "
                "if any safeguards prevent the claimed failure."
            )
        else:
            conclusion = (
                f"CONCLUSION: File '{filename}' EXISTS in the codebase. "
                "Now call check_defenses to verify the claimed vulnerability."
            )

        return f"{code_block}\n\n{conclusion}"

    except Exception as e:
        return f"ERROR: lookup_code failed: {e}"


async def _exec_check_defenses(failure_description: str, function_name: str) -> str:
    """Hybrid search for safeguards. Returns code + a conclusion with counts."""
    try:
        query = f"null check guard try catch exception handling {function_name} {failure_description}"
        embedding = await llm_provider.generate_embedding(query, task_type="RETRIEVAL_QUERY")
        results = db_provider.vector_search(
            query_vector=embedding,
            query_text=query,
            top_k=3,
        )

        if not results:
            return (
                "No defensive code found in the codebase for this area.\n\n"
                f"CONCLUSION: No safeguards found for '{failure_description}'. "
                "The hypothesis CANNOT be falsified — the claimed vulnerability has no defenses. "
                "Verdict should be CORROBORATED."
            )

        # Scan for safeguard patterns
        safeguard_patterns = [
            "?? ", "is null", "!= null", "== null", "?.",
            "argumentnullexception", "throw new", "try {", "catch ("
        ]

        parts = []
        total_guards = 0
        relevant_guards = 0

        for r in results:
            text = r.get("chunk_text", "")
            text_lower = text.lower()
            found_patterns = [p for p in safeguard_patterns if p in text_lower]
            count = len(found_patterns)
            total_guards += count

            # Check if this guard is in the right method
            method_match = function_name.lower() in (r.get("method_name", "") or "").lower()
            if method_match and count > 0:
                relevant_guards += count

            tag = f"⚠️ {count} SAFEGUARD(S)" if count > 0 else "no safeguards"
            location = "IN TARGET METHOD" if method_match else "in nearby code"
            parts.append(
                f"— {r.get('file_path', '?')} L{r.get('start_line','?')}-{r.get('end_line','?')} "
                f"method={r.get('method_name','?')} [{tag}] [{location}]\n"
                f"```csharp\n{text}\n```"
            )

        code_block = "\n\n".join(parts)

        # Build conclusion
        if relevant_guards > 0:
            conclusion = (
                f"CONCLUSION: Found {relevant_guards} safeguard(s) DIRECTLY in method "
                f"'{function_name}' that may prevent '{failure_description}'. "
                "If these safeguards cover the exact failure mode, the hypothesis is FALSIFIED. "
                "Examine the code above to confirm."
            )
        elif total_guards > 0:
            conclusion = (
                f"CONCLUSION: Found {total_guards} safeguard(s) in nearby code, but NONE "
                f"directly in method '{function_name}'. The hypothesis may still hold "
                "if the safeguards don't cover this specific failure path."
            )
        else:
            conclusion = (
                f"CONCLUSION: NO safeguards found anywhere near '{function_name}'. "
                f"The claimed vulnerability '{failure_description}' has no defenses. "
                "Verdict should be CORROBORATED."
            )

        return f"{code_block}\n\n{conclusion}"

    except Exception as e:
        return f"ERROR: check_defenses failed: {e}"


async def _dispatch_tool(name: str, args: dict) -> str:
    """Route a function_call to the correct executor."""
    if name == "lookup_code":
        return await _exec_lookup_code(
            args.get("filename", ""), args.get("function_name", "")
        )
    elif name == "check_defenses":
        return await _exec_check_defenses(
            args.get("failure_description", ""), args.get("function_name", "")
        )
    return (
        f"Unknown tool: '{name}'. You only have 2 tools: lookup_code and check_defenses. "
        "Output your VERDICT now with whatever information you have."
    )


# ---------------------------------------------------------------------------
# System Prompt — practical, not philosophical
# ---------------------------------------------------------------------------

FALSIFIER_SYSTEM_PROMPT = (
    "You verify whether a code-level hypothesis is true or false by checking it "
    "against the actual indexed codebase.\n\n"
    "## Your 2 tools\n\n"
    "1. lookup_code(filename, function_name) → checks if the file/function exists. "
    "Read the CONCLUSION at the bottom of the result.\n"
    "2. check_defenses(failure_description, function_name) → searches for null checks, "
    "try/catch, guards that would prevent the claimed bug. "
    "Read the CONCLUSION at the bottom of the result.\n\n"
    "## Workflow\n\n"
    "1. Call lookup_code to see if the file exists.\n"
    "   - If CONCLUSION says FILE_NOT_FOUND → immediately output VERDICT: FALSIFIED\n"
    "2. Call check_defenses to see if safeguards prevent the bug.\n"
    "   - If safeguards found that cover the failure → VERDICT: FALSIFIED\n"
    "   - If no safeguards found → VERDICT: CORROBORATED\n"
    "3. Output your verdict. Do NOT call more tools after step 2.\n\n"
    "## Verdict format\n\n"
    "VERDICT: CORROBORATED | FALSIFIED | INSUFFICIENT_EVIDENCE\n"
    "REASONING: what lookup_code and check_defenses told you\n"
    "COUNTER_EVIDENCE: exact code that falsifies, or 'none'\n"
    "CONFIDENCE: 0.0 to 1.0"
)


def _build_falsification_prompt(hypothesis: dict) -> str:
    h_desc = hypothesis.get("description", "No description")
    h_file = hypothesis.get("suspected_file", "unknown")
    h_func = hypothesis.get("suspected_function", "unknown")
    h_span = hypothesis.get("exact_span", "") or ""
    h_conf = hypothesis.get("confidence", 0.5)
    fname = h_file.split("/")[-1]

    return (
        f"Hypothesis: {h_desc}\n"
        f"File: {h_file} (lookup as: {fname})\n"
        f"Function: {h_func}\n"
        f"Claimed code: {h_span}\n"
        f"Prior confidence: {h_conf}\n\n"
        f"Start with: lookup_code(filename=\"{fname}\", function_name=\"{h_func}\")"
    )


# ---------------------------------------------------------------------------
# Agentic Loop
# ---------------------------------------------------------------------------


async def falsify_hypothesis(hypothesis: dict) -> FalsificationVerdict:
    """
    Run the falsification loop for a single hypothesis.
    2 tools, ≤4 turns, temp=1, thinking=MEDIUM.
    """
    settings = get_settings()
    client = genai.Client(api_key=settings.gemini_api_key)
    model = "gemini-3.1-pro-preview"  # Explicitly use 3.1 Pro for falsifier

    h_desc = hypothesis.get("description", "No description")

    tools = [
        types.Tool(
            function_declarations=ALL_TOOL_DECLS,
            code_execution=types.ToolCodeExecution(),
        )
    ]

    tool_config = types.ToolConfig(
        include_server_side_tool_invocations=True,
    )

    config = types.GenerateContentConfig(
        system_instruction=FALSIFIER_SYSTEM_PROMPT,
        tools=tools,
        tool_config=tool_config,
        temperature=1,
        thinking_config=types.ThinkingConfig(
            thinking_level="MEDIUM",
        ),
    )

    history: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part(text=_build_falsification_prompt(hypothesis))],
        )
    ]

    logger.info(f"[falsifier] Starting: '{h_desc[:60]}'")

    for turn in range(MAX_FALSIFIER_TURNS):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=history,
                config=config,
            )
        except Exception as e:
            logger.error(f"[falsifier] Turn {turn} API error: {e}")
            break

        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content:
            logger.warning(f"[falsifier] Turn {turn}: empty response")
            break

        model_content = candidate.content
        history.append(model_content)

        # Classify response parts
        function_calls = []
        has_code_execution = False
        text_parts = []

        for p in model_content.parts:
            if p.function_call is not None:
                function_calls.append(p.function_call)
            if p.executable_code is not None:
                has_code_execution = True
            if p.code_execution_result is not None:
                has_code_execution = True
            if p.text:
                text_parts.append(p.text)

        full_text = " ".join(text_parts)

        # --- Function calls: dispatch and log ---
        if function_calls:
            names = [fc.name for fc in function_calls]
            logger.info(f"[falsifier] Turn {turn}: calling {names}")

            function_response_parts = []
            for fc in function_calls:
                result = await _dispatch_tool(fc.name, dict(fc.args or {}))
                logger.debug(
                    f"[falsifier] {fc.name} → "
                    f"\n{result[:400]}\n{'...' if len(result) > 400 else ''}"
                )
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
            continue

        # --- Check for verdict in text ---
        has_verdict = any(
            marker in full_text.lower()
            for marker in ["verdict: corroborated", "verdict: falsified",
                           "verdict: insufficient"]
        )

        if has_verdict:
            logger.info(f"[falsifier] Turn {turn}: ✅ verdict")
            return _parse_verdict(full_text, hypothesis)

        if has_code_execution:
            logger.info(f"[falsifier] Turn {turn}: code ran, nudging for verdict")
            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text="Now output your VERDICT.")],
                )
            )
            continue

        if full_text:
            logger.info(f"[falsifier] Turn {turn}: text without verdict, nudging")
            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text="Output your VERDICT now.")],
                )
            )
            continue

        break

    # Try to salvage a verdict from accumulated text
    all_text = " ".join(
        p.text for c in history if c.role == "model"
        for p in c.parts if p.text
    )
    if all_text and "verdict:" in all_text.lower():
        return _parse_verdict(all_text, hypothesis)

    logger.warning(f"[falsifier] Max turns ({MAX_FALSIFIER_TURNS}): '{h_desc[:60]}'")
    return FalsificationVerdict(
        hypothesis_id=h_desc[:80],
        verdict="INSUFFICIENT_EVIDENCE",
        reasoning=f"Loop exhausted after {MAX_FALSIFIER_TURNS} turns without verdict.",
        confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Verdict Parser
# ---------------------------------------------------------------------------


def _parse_verdict(agent_output: str, hypothesis: dict) -> FalsificationVerdict:
    text = str(agent_output)
    lower = text.lower()

    if "verdict: falsified" in lower or "verdict:**falsified" in lower:
        verdict = "FALSIFIED"
    elif "verdict: corroborated" in lower or "verdict:**corroborated" in lower:
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
            confidence = float(conf_str.strip(".,: "))
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
# Main Node (LangGraph integration)
# ---------------------------------------------------------------------------


async def falsifier_node(state: dict) -> dict:
    """
    Epistemic Falsifier node: runs Popperian falsification for each hypothesis.
    Hypotheses processed in parallel.
    """
    hypotheses = state.get("hypotheses", [])

    if not hypotheses:
        logger.info("[falsifier] No hypotheses to falsify")
        return {"falsifier_verdicts": []}

    logger.info(f"[falsifier] Spawning {len(hypotheses)} loops (parallel)...")

    tasks = [falsify_hypothesis(h) for h in hypotheses]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    verdicts = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"[falsifier] Exception: {r}")
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
        f"⚠️ {insufficient} insufficient"
    )

    return {"falsifier_verdicts": verdicts}
