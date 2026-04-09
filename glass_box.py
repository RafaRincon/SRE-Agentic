import asyncio
import logging
import os
import uuid
import json
from datetime import datetime, timezone

# Hide all regular logging to keep the console clean for the Glass Box
logging.getLogger().setLevel(logging.CRITICAL)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.CRITICAL)

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich import box
from rich.markdown import Markdown

from app.agents.graph import get_stateless_graph
from app.agents.state import IncidentStatus
from app.config import get_settings

console = Console()

# A sample incident written in plain user language — no SRE jargon, just what the user observed
SAMPLE_REPORT = """
Hi, the product search is completely broken. When I type anything in the search bar the page
just spins forever and then shows an empty list. Even searching for "shirt" returns nothing.
This started after the deployment we did like 30 minutes ago.
Some colleagues say the catalog page loads fine but search is dead.
Could be the database? We changed something in the catalog service config today.
"""

async def run_glassbox():
    console.print(Panel(Text("SRE Agent: The Glass Box", style="bold cyan", justify="center"), border_style="cyan"))
    console.print()

    # Use stateless graph for demos — no CosmosDB checkpointer needed locally.
    # The checkpointer causes 65s timeout on large checkpoint payloads (>70KB).
    with console.status("[bold green]Waking up the SRE Agent...[/bold green]", spinner="dots"):
        graph = get_stateless_graph()
    
    incident_id = uuid.uuid4().hex
    initial_state = {
        "incident_id": incident_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_report": SAMPLE_REPORT.strip(),
        "has_image": False,
        "image_mime_type": "",
        "image_data_b64": "",
        "status": IncidentStatus.RECEIVED.value,
        "hypotheses": [],
        "span_verdicts": [],
        "falsifier_verdicts": [],
        "verified_root_causes": [],
        "epistemic_context": {},
        "suggested_runbooks": [],
        "errors": [],
    }

    thread_config = {"configurable": {"thread_id": incident_id}}

    console.print(Panel(
        f"[bold yellow]New Incident Received[/bold yellow]\n\n"
        f"[dim]ID:[/dim] {incident_id}\n"
        f"[dim]Report:[/dim]\n{SAMPLE_REPORT.strip()}",
        title="📥 Alert Intake",
        border_style="yellow"
    ))
    console.print()

    # We will stream the graph execution to show exactly what each node does
    with console.status("[bold magenta]Agent is thinking...[/bold magenta]", spinner="bouncingBar") as status:
        try:
            async for chunk in graph.astream(initial_state, config=thread_config, stream_mode="updates"):
                for node_name, node_state in chunk.items():
                
                # Format output nicely for the SME
                if node_name == "n_intake":
                    status.update("[bold magenta]Checking for duplicates...[/bold magenta]")
                    console.print(Panel(
                        "🔍 [bold]Intake Node[/bold]: Read the raw incident and parsed basic details.",
                        border_style="blue"
                    ))

                elif node_name == "n_dedup":
                    is_dup = node_state.get("is_duplicate", False)
                    if is_dup:
                        parent = node_state.get("duplicate_of", "?")
                        sim = node_state.get("duplicate_similarity", 0)
                        console.print(Panel(
                            f"🔗 [bold red]DUPLICATE DETECTED[/bold red]\n\n"
                            f"This incident matches [bold]{parent}[/bold] "
                            f"(similarity: [bold]{sim:.0%}[/bold]).\n"
                            f"Skipping full triage — see existing ticket.",
                            title="🔗 Dedup Gate",
                            border_style="red"
                        ))
                    else:
                        status.update("[bold magenta]Building World Model...[/bold magenta]")
                        console.print(Panel(
                            "🔗 [bold green]No duplicate found[/bold green] — proceeding with full triage.",
                            title="🔗 Dedup Gate",
                            border_style="green"
                        ))
                
                elif node_name == "n_world_model":
                    wm = node_state.get("world_model", {})
                    svc = wm.get("affected_service", "Unknown")
                    console.print(Panel(
                        f"🌍 [bold]World Model Analysis[/bold]:\n"
                        f"Identified Service: [bold cyan]{svc}[/bold cyan]",
                        border_style="blue"
                    ))
                    
                elif node_name == "n_slot_filler":
                    status.update("[bold magenta]Formulating Hypotheses...[/bold magenta]")
                    ents = node_state.get("entities", {})
                    err = ents.get("error_message", "N/A")
                    console.print(Panel(
                        f"🧩 [bold]Slot Filler[/bold]:\n"
                        f"Extracted Error: [red]{err}[/red]",
                        border_style="blue"
                    ))

                elif node_name == "n_risk_hypothesizer":
                    status.update("[bold magenta]Reviewing Evidences...[/bold magenta]")
                    hypotheses = node_state.get("hypotheses", [])
                    historical_ctx = node_state.get("historical_context", {})
                    
                    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
                    table.add_column("Hypothesis")
                    table.add_column("Exact Span (Evidence)")
                    
                    for h in hypotheses:
                        desc = h.get("description", "")
                        span = h.get("exact_span", "")[:100] + "..." if h.get("exact_span") else ""
                        table.add_row(desc, span)
                        
                    console.print(Panel(
                        table,
                        title="🧠 Risk Hypothesizer (RAG & Expansion)",
                        border_style="magenta"
                    ))
                    
                    if historical_ctx.get("recurrence_count", 0) > 0:
                        console.print(f"   [dim]ℹ️ Found {historical_ctx['recurrence_count']} similar past incidents in Knowledge Base[/dim]")

                elif node_name == "n_span_arbiter":
                    status.update("[bold magenta]Falsifying Hypotheses...[/bold magenta]")
                    verdicts = node_state.get("span_verdicts", [])

                    VERDICT_COLOR = {
                        "VERIFIED": "green",
                        "PARTIAL_MATCH": "yellow",
                        "HALLUCINATION": "red",
                        "ERROR": "red",
                    }
                    VERDICT_ICON = {
                        "VERIFIED": "✅",
                        "PARTIAL_MATCH": "🔶",
                        "HALLUCINATION": "❌",
                        "ERROR": "⚠️",
                    }

                    lines = []
                    for v in verdicts:
                        vrd = v.get("verdict", "UNVERIFIED")
                        score = v.get("similarity_score", 0.0)
                        matched = v.get("matched_file") or "no file matched"
                        hyp_id = v.get("hypothesis_id", "?")
                        c = VERDICT_COLOR.get(vrd, "white")
                        icon = VERDICT_ICON.get(vrd, "❓")
                        lines.append(
                            f"[{c}]{icon} [{vrd}][/{c}] "
                            f"[dim]hyp={hyp_id[:8]} | score={score:.2f} | {matched}[/dim]"
                        )

                    if lines:
                        console.print(Panel(
                            "\n".join(lines),
                            title="⚖️ Span Arbiter (Citation Verification)",
                            border_style="cyan"
                        ))

                elif node_name == "n_falsifier":
                    status.update("[bold magenta]Consolidating Final Ticket...[/bold magenta]")
                    f_verdicts = node_state.get("falsifier_verdicts", [])

                    F_COLOR = {
                        "CORROBORATED": "green",
                        "FALSIFIED": "red",
                        "INSUFFICIENT_EVIDENCE": "yellow",
                    }
                    F_ICON = {
                        "CORROBORATED": "✅",
                        "FALSIFIED": "❌",
                        "INSUFFICIENT_EVIDENCE": "⚠️",
                    }

                    lines = []
                    for v in f_verdicts:
                        vrd = v.get("verdict", "INSUFFICIENT_EVIDENCE")
                        conf = v.get("confidence", 0.0)
                        reasoning = (v.get("reasoning") or "")[:120]
                        c = F_COLOR.get(vrd, "white")
                        icon = F_ICON.get(vrd, "⚠️")
                        lines.append(
                            f"[{c}]{icon} {vrd}[/{c}] [dim](conf={conf:.0%})[/dim]\n"
                            f"   [dim]{reasoning}[/dim]"
                        )

                    if lines:
                        # Also build a detailed reasoning panel for auditing the agent's rationale
                        reasoning_lines = []
                        for v in f_verdicts:
                            vrd = v.get("verdict", "INSUFFICIENT_EVIDENCE")
                            hyp_id = v.get("hypothesis_id", "?")[:40]
                            conf = v.get("confidence", 0.0)
                            reasoning = (v.get("reasoning") or "(no reasoning)").strip()
                            counter = v.get("counter_evidence", [])
                            counter_str = ("\n      → " + counter[0][:200]) if counter else ""
                            c = F_COLOR.get(vrd, "white")
                            icon = F_ICON.get(vrd, "⚠️")
                            reasoning_lines.append(
                                f"[bold][{c}]{icon} {vrd}[/{c}][/bold] [dim](conf={conf:.0%})[/dim]\n"
                                f"   [italic dim]hyp: {hyp_id}[/italic dim]\n"
                                f"   {reasoning}{counter_str}\n"
                            )

                        console.print(Panel(
                            "\n".join(lines),
                            title="🧪 Falsifier — Verdicts",
                            border_style="red"
                        ))
                        console.print(Panel(
                            "\n".join(reasoning_lines),
                            title="🔬 Falsifier — Agent Rationale (full)",
                            border_style="dim"
                        ))

                elif node_name == "n_consolidator":
                    summary = node_state.get("triage_summary", "")
                    sev = node_state.get("final_severity", "")
                    causes = node_state.get("verified_root_causes", [])

                    # Color the UX-friendly confidence labels
                    cause_lines = []
                    for cause in causes:
                        styled = cause
                        for label, color in [
                            ("✅ Confirmed", "green"), ("🔶 Probable", "yellow"),
                            ("⚠️ Needs Review", "cyan"), ("❌ Ruled Out", "red"),
                            ("❌ Unverified", "red"),
                        ]:
                            styled = styled.replace(label, f"[{color}]{label}[/{color}]")
                        cause_lines.append(f"- {styled}")

                    causes_block = "\n".join(cause_lines) if cause_lines else "- (none)"

                    console.print(Panel(
                        f"[bold]Severity:[/bold] [red]{sev}[/red]\n\n"
                        f"[bold]Root Causes:[/bold]\n{causes_block}\n\n"
                        f"[bold]Architect's Diagnosis:[/bold]",
                        title="✅ Synthesis & Consolidation",
                        border_style="green"
                    ))
                    console.print(Markdown(summary))

                elif node_name == "n_create_ticket":
                    ticket = node_state.get("ticket", {})
                    t_id = ticket.get("ticket_id", "")
                    team = ticket.get("assigned_team", "")
                    runbooks = ticket.get("suggested_runbooks", [])
                    
                    rb_str = "\n".join([f"- {rb.get('title')}" for rb in runbooks]) if runbooks else "None found"
                    
                    console.print(Panel(
                        f"[bold]Ticket ID:[/bold] {t_id}\n"
                        f"[bold]Assigned Team:[/bold] {team}\n\n"
                        f"[bold]Runbook Suggestions:[/bold]\n{rb_str}",
                        title="🎫 Ticket Created",
                        border_style="blue"
                    ))

                elif node_name == "n_notify_team":
                    console.print("[bold green]🚀 Agent workflow completed successfully! Team Notified.[/bold green]")

        except Exception as e:
            err_str = str(e)
            # CosmosDB checkpointer timeout happens AFTER the agent has already
            # produced all output — the triage is complete, only persistence failed.
            if "ServiceResponseTimeoutError" in type(e).__name__ or "Read timed out" in err_str or "cosmosdb" in err_str.lower():
                console.print(Panel(
                    "[yellow]⚠️ Checkpoint save timed out (Cosmos DB)[/yellow]\n"
                    "The triage above is complete and correct.\n"
                    "The incident state was not persisted to the checkpointer for this run.",
                    title="⚠️ Persistence Warning",
                    border_style="yellow"
                ))
            else:
                console.print(Panel(
                    f"[red]❌ Pipeline error:[/red]\n{err_str[:400]}",
                    title="❌ Error",
                    border_style="red"
                ))
                raise


if __name__ == "__main__":
    asyncio.run(run_glassbox())
