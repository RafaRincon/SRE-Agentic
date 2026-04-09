"""
Smoke test riguroso: Epistemic Falsifier — 6 ejes de validación

Cada caso tiene un veredicto ESPERADO. Si el LLM falla, el test falla.

Ejes cubiertos:
  1. Archivo existe, método existe, sin safeguards  → CORROBORATED
  2. Archivo existe, método existe, con guards       → FALSIFIED
  3. Archivo existe, método NO existe               → INSUFFICIENT_EVIDENCE
  4. Archivo inventado (alucinación pura)           → FALSIFIED inmediato
  5. Guard existe pero a nivel superior/externo     → FALSIFIED (razonamiento contextual)
  6. Sin suspected_file (hipótesis difusa)          → manejo graceful, no crash

Usage:
    cd hackaton
    source .venv/bin/activate
    PYTHONPATH=. python tests/smoke_falsifier.py
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Logging: solo el falsifier en DEBUG para ver cada turno y tool result
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("app.agents.nodes.falsifier").setLevel(logging.DEBUG)
for noisy in ("azure", "httpx", "urllib3", "asyncio", "azure.cosmos"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

from app.agents.nodes.falsifier import falsify_hypothesis, FalsificationVerdict  # noqa: E402

# ---------------------------------------------------------------------------
# 6 hipótesis de prueba — cubren los ejes de falsificación
# ---------------------------------------------------------------------------

CASES = [
    # ─── EJE 1: Archivo existe + método existe + sin guards ──────────────────
    {
        "label": "[EJE-1] Archivo real, método real, sin null-check",
        "expected_verdict": "CORROBORATED",
        "expected_verdict_alt": None,   # no hay alternativa aceptable
        "rationale": "El foreach sobre OrderItems existe en el código y no hay guard previo",
        "hypothesis": {
            "description": (
                "CreateOrderCommandHandler.Handle() itera sobre message.OrderItems "
                "sin verificar si la colección es null, causando NullReferenceException "
                "en pedidos vacíos."
            ),
            "suspected_file": "src/Ordering.API/Application/Commands/CreateOrderCommandHandler.cs",
            "suspected_function": "Handle",
            "exact_span": "foreach (var item in message.OrderItems)",
            "confidence": 0.85,
        },
    },

    # ─── EJE 2: Archivo existe + método existe + CON guard explícito ─────────
    {
        "label": "[EJE-2] Código real con validación — debe FALSIFICAR",
        "expected_verdict": "FALSIFIED",
        "expected_verdict_alt": "CORROBORATED",  # si el model razona que el guard es insuficiente
        "rationale": "CreateOrderCommandValidator usa MustAsync para validar items — hay guard",
        "hypothesis": {
            "description": (
                "CreateOrderCommandValidator no valida en ningún momento si la lista de "
                "OrderItems está vacía, permitiendo comandos inválidos llegar al handler."
            ),
            "suspected_file": "src/Ordering.API/Application/Validations/CreateOrderCommandValidator.cs",
            "suspected_function": "ContainOrderItems",
            "exact_span": "orderItems.Any()",
            "confidence": 0.4,
        },
    },

    # ─── EJE 3: Archivo existe + método NO existe ────────────────────────────
    {
        "label": "[EJE-3] Método inventado en archivo real",
        "expected_verdict": "INSUFFICIENT_EVIDENCE",
        "expected_verdict_alt": "FALSIFIED",  # si el model concluye que el método no existe
        "rationale": "El archivo RedisBasketRepository existe pero ProcessPaymentAsync es inventado",
        "hypothesis": {
            "description": (
                "RedisBasketRepository.ProcessPaymentAsync falla silenciosamente cuando "
                "la conexión a Redis está degradada, sin retry lógic."
            ),
            "suspected_file": "src/Basket.API/Repositories/RedisBasketRepository.cs",
            "suspected_function": "ProcessPaymentAsync",
            "exact_span": "await _paymentClient.ProcessAsync(basket.BuyerId)",
            "confidence": 0.3,
        },
    },

    # ─── EJE 4: Archivo completamente inventado (alucinación pura) ───────────
    {
        "label": "[EJE-4] Archivo fabricado — FALSIFICAR inmediato en turno 1",
        "expected_verdict": "FALSIFIED",
        "expected_verdict_alt": None,
        "rationale": "PaymentGatewayAdapterV2.cs no existe en el índice",
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

    # ─── EJE 5: Guard existe pero a nivel de validación externa ─────────────
    {
        "label": "[EJE-5] Guard real a nivel superior — modelo debe razonar contexto",
        "expected_verdict": "FALSIFIED",
        "expected_verdict_alt": "CORROBORATED",
        "rationale": "UpdateBasketAsync tiene el guard en el flujo de serialización previo",
        "hypothesis": {
            "description": (
                "RedisBasketRepository.UpdateBasketAsync no serializa correctamente la "
                "cesta cuando el objeto CustomerBasket es null, causando una excepción "
                "no controlada al invocar StringSetAsync."
            ),
            "suspected_file": "src/Basket.API/Repositories/RedisBasketRepository.cs",
            "suspected_function": "UpdateBasketAsync",
            "exact_span": "await _database.StringSetAsync(key, json)",
            "confidence": 0.6,
        },
    },

    # ─── EJE 6: Sin suspected_file (hipótesis difusa) ────────────────────────
    {
        "label": "[EJE-6] Sin archivo sospechoso — manejo graceful",
        "expected_verdict": "INSUFFICIENT_EVIDENCE",
        "expected_verdict_alt": "FALSIFIED",
        "rationale": "Sin anchor de archivo, el modelo no puede buscar; debe admitir evidencia insuficiente",
        "hypothesis": {
            "description": (
                "En algún lugar del pipeline de checkout, un objeto de pago es "
                "desreferenciado sin validar su nullabilidad, causando un crash esporádico."
            ),
            # Sin `suspected_file`; el modelo debe manejar el caso sin fallar.
            "confidence": 0.2,
        },
    },
]

# ---------------------------------------------------------------------------
# Colores para impresión
# ---------------------------------------------------------------------------
COLORS = {
    "FALSIFIED": "\033[91m",           # rojo
    "CORROBORATED": "\033[92m",        # verde
    "INSUFFICIENT_EVIDENCE": "\033[93m",  # amarillo
    "ERROR": "\033[95m",               # magenta
    "PASS": "\033[92m",
    "FAIL": "\033[91m",
    "RESET": "\033[0m",
}


def color(text: str, key: str) -> str:
    return f"{COLORS.get(key, '')}{text}{COLORS['RESET']}"


def print_trace(verdict: FalsificationVerdict):
    print(f"\n  Veredicto   : {color(verdict.verdict, verdict.verdict)}  (confianza: {verdict.confidence:.2f})")
    print(f"\n  Razonamiento:")
    for line in verdict.reasoning.strip().splitlines():
        print(f"    {line}")
    if verdict.counter_evidence:
        print(f"\n  Counter-evidence:")
        for ev in verdict.counter_evidence:
            print(f"    → {ev.strip()[:200]}")
    if verdict.supporting_evidence:
        print(f"\n  Supporting-evidence:")
        for ev in verdict.supporting_evidence:
            print(f"    + {ev.strip()[:200]}")


# ---------------------------------------------------------------------------
# Runner principal
# ---------------------------------------------------------------------------
async def run():
    total_start = time.perf_counter()
    results = []

    for case in CASES:
        label = case["label"]
        hyp = case["hypothesis"]
        expected = case["expected_verdict"]
        alt = case["expected_verdict_alt"]
        rationale = case["rationale"]

        sep = "─" * 68
        print(f"\n┌{sep}┐")
        print(f"│  {label}")
        print(f"│  Espera : {color(expected, expected)}{f'  o  {color(alt, alt)}' if alt else ''}")
        print(f"│  Razón  : {rationale}")
        desc = hyp.get("description", "")[:75]
        print(f"│  Hipótesis: {desc}...")
        print(f"└{sep}┘")

        start = time.perf_counter()
        try:
            verdict = await falsify_hypothesis(hyp)
            elapsed = time.perf_counter() - start
            print_trace(verdict)
            print(f"\n  ⏱  {elapsed:.1f}s")

            # ─── Evaluación del veredicto ────────────────────────────────────
            acceptable = {expected}
            if alt:
                acceptable.add(alt)
            passed = verdict.verdict in acceptable

            status_str = color("PASS ✅", "PASS") if passed else color("FAIL ❌", "FAIL")
            print(f"\n  [{status_str}]  Obtenido: {verdict.verdict}  |  Aceptable: {', '.join(acceptable)}")

            results.append({
                "label": label,
                "verdict": verdict.verdict,
                "expected": expected,
                "acceptable": acceptable,
                "passed": passed,
                "confidence": verdict.confidence,
                "elapsed": elapsed,
                "ok": True,
            })

        except Exception as e:
            elapsed = time.perf_counter() - start
            import traceback
            print(f"\n  ❌ EXCEPCIÓN tras {elapsed:.1f}s: {e}")
            traceback.print_exc()
            results.append({
                "label": label,
                "verdict": "ERROR",
                "passed": False,
                "ok": False,
                "elapsed": elapsed,
            })

    # ---------------------------------------------------------------------------
    # Resumen final
    # ---------------------------------------------------------------------------
    total = time.perf_counter() - total_start
    passed_count = sum(1 for r in results if r.get("passed"))
    failed_count = len(results) - passed_count

    print(f"\n{'═' * 70}")
    print("  RESUMEN SMOKE TEST — 6 EJES DE FALSIFICACIÓN")
    print(f"{'═' * 70}")
    for r in results:
        icon = color("✅ PASS", "PASS") if r.get("passed") else color("❌ FAIL", "FAIL")
        conf = f"conf={r.get('confidence', '?'):.2f}" if r.get("ok") else "ERROR"
        verdict_str = r["verdict"].ljust(25)
        print(f"  {icon}  [{verdict_str}] {conf}  {r['elapsed']:.1f}s  │  {r['label']}")

    print(f"\n  Total: {total:.1f}s  |  {passed_count}/{len(results)} pasaron")

    if failed_count > 0:
        print(color(f"\n  ❌ {failed_count} caso(s) con veredicto inesperado\n", "FAIL"))
        sys.exit(1)
    else:
        print(color(f"\n  ✅ Todos los ejes validados correctamente\n", "PASS"))


if __name__ == "__main__":
    asyncio.run(run())
