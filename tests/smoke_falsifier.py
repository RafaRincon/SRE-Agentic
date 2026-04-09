"""
Smoke test: Epistemic Falsifier (REAL — no mocks)

Runs the complete falsification loop against actual:
- Gemini API (function calling + code_execution)
- Cosmos DB DiskANN (RAG tools)

Usage:
    cd hackaton
    source .venv/bin/activate
    python tests/smoke_falsifier.py
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# --- Make app importable from tests/ ---
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up logging so we can see every turn in the loop
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# Also show the falsifier's internal loop
logging.getLogger("app.agents.nodes.falsifier").setLevel(logging.DEBUG)

from app.agents.nodes.falsifier import falsify_hypothesis, FalsificationVerdict


# ---------------------------------------------------------------------------
# Test hypotheses — one we expect to be CORROBORATED (no defenses found)
# and one we expect to be FALSIFIED (strong defenses exist)
# ---------------------------------------------------------------------------

HYPOTHESIS_ORDERING_NULL = {
    "hypothesis_id": "h-ordering-01",
    "description": (
        "The 'message.OrderItems' collection is null when "
        "CreateOrderCommandHandler.Handle() iterates over it, "
        "causing a NullReferenceException. No null guard exists."
    ),
    "suspected_file": "src/Ordering.API/Application/Commands/CreateOrderCommandHandler.cs",
    "suspected_function": "Handle",
    "exact_span": "foreach (var item in message.OrderItems)",
    "confidence": 0.82,
}

HYPOTHESIS_BASKET_REDIS = {
    "hypothesis_id": "h-basket-redis-01",
    "description": (
        "Redis basket keys are stored without TTL in RedisBasketRepository, "
        "meaning stale cart data persists indefinitely across user sessions."
    ),
    "suspected_file": "src/Basket.API/Repositories/RedisBasketRepository.cs",
    "suspected_function": "UpdateBasketAsync",
    "exact_span": "await _database.StringSetAsync(key, JsonSerializer.Serialize(basket))",
    "confidence": 0.75,
}

# A deliberately fabricated hypothesis — should be FALSIFIED (file doesn't exist)
HYPOTHESIS_FABRICATED = {
    "hypothesis_id": "h-fabricated-01",
    "description": (
        "The PaymentGatewayAdapterV2.cs file has a missing null check "
        "on the transaction ID that causes silent payment failures."
    ),
    "suspected_file": "src/Payment.API/Adapters/PaymentGatewayAdapterV2.cs",
    "suspected_function": "ProcessTransaction",
    "exact_span": "var result = gateway.Charge(transactionId.Value)",
    "confidence": 0.5,
}


async def run_smoke():
    hypotheses = [
        ("Ordering NullRef (expect CORROBORATED)", HYPOTHESIS_ORDERING_NULL),
        ("Basket Redis TTL (expect CORROBORATED)", HYPOTHESIS_BASKET_REDIS),
        ("Fabricated file (expect FALSIFIED/INSUFFICIENT)", HYPOTHESIS_FABRICATED),
    ]

    results = []
    total_start = time.perf_counter()

    for label, hyp in hypotheses:
        print(f"\n{'='*70}")
        print(f"TESTING: {label}")
        print(f"Hypothesis: {hyp['description'][:80]}...")
        print(f"{'='*70}")

        start = time.perf_counter()
        try:
            verdict: FalsificationVerdict = await falsify_hypothesis(hyp)
            elapsed = time.perf_counter() - start

            print(f"\n✅ Verdict      : {verdict.verdict}")
            print(f"   Confidence   : {verdict.confidence}")
            print(f"   Elapsed      : {elapsed:.1f}s")
            print(f"   Reasoning    : {verdict.reasoning[:200]}")
            if verdict.counter_evidence:
                print(f"   Counter-ev.  : {verdict.counter_evidence[0][:150]}")

            results.append({
                "label": label,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "elapsed_s": round(elapsed, 2),
                "ok": True,
            })

        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"\n❌ EXCEPTION after {elapsed:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "label": label,
                "verdict": "ERROR",
                "error": str(e),
                "elapsed_s": round(elapsed, 2),
                "ok": False,
            })

    total_elapsed = time.perf_counter() - total_start

    print(f"\n{'='*70}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    for r in results:
        status = "✅" if r["ok"] else "❌"
        print(f"{status} {r['label']}")
        print(f"   Verdict: {r['verdict']} | {r['elapsed_s']}s")
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # Fail if any errored
    errors = [r for r in results if not r["ok"]]
    if errors:
        print(f"\n❌ {len(errors)} test(s) ERRORED — smoke test FAILED")
        sys.exit(1)
    else:
        print("\n✅ All falsifier smoke tests completed (no exceptions)")


if __name__ == "__main__":
    asyncio.run(run_smoke())
