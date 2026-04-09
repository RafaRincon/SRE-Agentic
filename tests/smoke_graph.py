import asyncio
import uuid
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from app.agents.graph import get_graph

console = Console()

async def main():
    graph = get_graph()
    
    incident_id = f"smoke_INC_{uuid.uuid4().hex[:8]}"
    
    # Simulating a real incident report
    raw_report = (
        "We are seeing a spike of NullReferenceException when creating order drafts. "
        "The Basket API is returning 500s because the BasketRepository is failing to save."
    )
    
    initial_state = {
        "incident_id": incident_id,
        "raw_report": raw_report,
        "status": "NEW"
    }
    
    config = {"configurable": {"thread_id": incident_id}}
    
    console.print(Panel(f"🚀 [bold green]Starting SRE Pipeline Smoke Test[/bold green]\nIncident ID: {incident_id}\n\n[dim]{raw_report}[/dim]"))

    # We will stream the events to show the "crystal box" effect
    async for event in graph.astream(initial_state, config):
        for node_name, node_state in event.items():
            
            # Format node state
            state_text = Text()
            
            if "world_model" in node_state:
                state_text.append(f"Affected Service: {node_state['world_model'].get('affected_service')}\n", style="cyan")
                state_text.append(f"Incident Category: {node_state['world_model'].get('incident_category')}\n", style="blue")
                state_text.append(f"Severity: {node_state['world_model'].get('estimated_severity')}", style="red")

            if "entities" in node_state:
                state_text.append(f"\nEntities extracted: {node_state['entities']}\n", style="magenta")

            if "hypotheses" in node_state:
                state_text.append(f"\nGenerated {len(node_state['hypotheses'])} hypotheses:\n", style="yellow")
                for idx, h in enumerate(node_state['hypotheses']):
                    state_text.append(f"  {idx+1}. {h.get('description', '')}\n", style="yellow")

            if "span_verdicts" in node_state:
                state_text.append(f"Span Arbiter finished checking {len(node_state['span_verdicts'])} spans.\n", style="green")
                
            if "falsifier_verdicts" in node_state:
                state_text.append(f"Falsifier completed. Verdicts:\n", style="bold red")
                for v in node_state['falsifier_verdicts']:
                    state_text.append(f"  [{v.get('verdict')}] {v.get('hypothesis_id')[:60]}...\n", style="bold white")
                    
            if "triage_summary" in node_state:
                state_text.append(f"\nConsolidator finished. Final Severity: {node_state.get('final_severity')}\n", style="bold green")
                
            if "ticket" in node_state:
                state_text.append(f"Ticket Created: {node_state['ticket'].get('ticket_url')}", style="bold blue")

            console.print(Panel(state_text, title=f"Node 🎯 [bold yellow]{node_name}[/bold yellow]"))

    console.print("[bold green]✅ Pipeline execution completed successfully![/bold green]")

if __name__ == "__main__":
    asyncio.run(main())
