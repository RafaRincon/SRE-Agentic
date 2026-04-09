"""
Smoke test: Epistemic Falsifier — muestra razonamiento y trazas del agente

Usage:
    cd hackaton
    source .venv/bin/activate
    python tests/smoke_falsifier.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Logging: silenciar todo excepto el falsifier y la app
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,  # default: callado
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# DEBUG en el falsifier para ver tool results
logging.getLogger("app.agents.nodes.falsifier").setLevel(logging.DEBUG)
# Silenciar Azure, httpx, cosmos
for noisy in ("azure", "httpx", "urllib3", "asyncio", "azure.cosmos"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)


from app.agents.nodes.falsifier import falsify_hypothesis, FalsificationVerdict  # noqa: E402

# ---------------------------------------------------------------------------
# Hipótesis de prueba
# ---------------------------------------------------------------------------

HYPOTHESES = [
    {
        "label": "Ordering — NullRef en OrderItems",
        "expect": "CORROBORATED o INSUFFICIENT_EVIDENCE",
        "hypothesis": {
            "description": (
                "El método CreateOrderCommandHandler.Handle() itera sobre "
                "message.OrderItems sin verificar si la colección es null, "
                "causando NullReferenceException en pedidos vacíos."
            ),
            "suspected_file": "src/Ordering.API/Application/Commands/CreateOrderCommandHandler.cs",
            "suspected_function": "Handle",
            "exact_span": "foreach (var item in message.OrderItems)",
            "confidence": 0.82,
        },
    },
    {
        "label": "Basket — Redis sin TTL",
        "expect": "CORROBORATED o INSUFFICIENT_EVIDENCE",
        "hypothesis": {
            "description": (
                "RedisBasketRepository.UpdateBasketAsync guarda la cesta en Redis "
                "sin TTL, dejando datos obsoletos indefinidamente entre sesiones."
            ),
            "suspected_file": "src/Basket.API/Repositories/RedisBasketRepository.cs",
            "suspected_function": "UpdateBasketAsync",
            "exact_span": "await _database.StringSetAsync(key, JsonSerializer.Serialize(basket))",
            "confidence": 0.75,
        },
    },
    {
        "label": "Archivo fabricado — debe FALSIFICAR",
        "expect": "FALSIFIED",
        "hypothesis": {
            "description": (
                "PaymentGatewayAdapterV2.ProcessTransaction tiene un null check "
                "faltante en el transaction ID que causa fallos silenciosos."
            ),
            "suspected_file": "src/Payment.API/Adapters/PaymentGatewayAdapterV2.cs",
            "suspected_function": "ProcessTransaction",
            "exact_span": "var result = gateway.Charge(transactionId.Value)",
            "confidence": 0.5,
        },
    },
]


# ---------------------------------------------------------------------------
# Traza visual del agente
# ---------------------------------------------------------------------------

def print_trace(verdict: FalsificationVerdict):
    """Imprime el razonamiento completo del agente de forma legible."""
    color = {
        "FALSIFIED": "\033[91m",       # rojo
        "CORROBORATED": "\033[92m",    # verde
        "INSUFFICIENT_EVIDENCE": "\033[93m",  # amarillo
    }.get(verdict.verdict, "\033[0m")
    reset = "\033[0m"

    print(f"\n  Veredicto   : {color}{verdict.verdict}{reset}  (confianza: {verdict.confidence})")
    print(f"\n  Razonamiento:\n")
    # Imprime el razonamiento con indentación
    for line in verdict.reasoning.strip().splitlines():
        print(f"    {line}")

    if verdict.counter_evidence:
        print(f"\n  Contra-evidencia encontrada:")
        for ev in verdict.counter_evidence:
            print(f"    → {ev.strip()[:200]}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run():
    total_start = time.perf_counter()
    results = []

    for item in HYPOTHESES:
        label = item["label"]
        hyp = item["hypothesis"]
        expect = item["expect"]

        sep = "─" * 65
        print(f"\n┌{sep}┐")
        print(f"│  {label}")
        print(f"│  Espera: {expect}")
        print(f"│  Hipótesis: {hyp['description'][:70]}...")
        print(f"└{sep}┘")

        start = time.perf_counter()
        try:
            verdict = await falsify_hypothesis(hyp)
            elapsed = time.perf_counter() - start
            print_trace(verdict)
            print(f"\n  ⏱  {elapsed:.1f}s")
            results.append({"label": label, "verdict": verdict.verdict,
                            "confidence": verdict.confidence, "ok": True,
                            "elapsed": elapsed})
        except Exception as e:
            elapsed = time.perf_counter() - start
            import traceback
            print(f"\n  ❌ EXCEPCIÓN tras {elapsed:.1f}s: {e}")
            traceback.print_exc()
            results.append({"label": label, "verdict": "ERROR",
                            "ok": False, "elapsed": elapsed})

    # -----------------------------------------------------------------------
    # Resumen
    # -----------------------------------------------------------------------
    total = time.perf_counter() - total_start
    print(f"\n{'═' * 67}")
    print("  RESUMEN SMOKE TEST")
    print(f"{'═' * 67}")
    for r in results:
        icon = "✅" if r["ok"] else "❌"
        conf = f"  conf={r.get('confidence', '?')}" if r["ok"] else ""
        print(f"  {icon}  [{r['verdict']:<25}]{conf}  {r['elapsed']:.1f}s  —  {r['label']}")
    print(f"\n  Total: {total:.1f}s")

    errors = [r for r in results if not r["ok"]]
    if errors:
        print(f"\n  ❌ {len(errors)} test(s) con errores")
        sys.exit(1)
    else:
        print("\n  ✅ Todos los tests completados sin excepciones\n")


if __name__ == "__main__":
    asyncio.run(run())
