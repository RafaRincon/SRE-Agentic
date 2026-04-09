# ADR-005: Alert Storm Protection y Deduplicación Temprana (Short-Circuit)

## Estado
Aceptado

## Contexto y Problema
En entornos de Site Reliability Engineering (SRE), las interrupciones graves de servicios a menudo causan "Tormentas de Alertas" (Alert Storms). En estas situaciones, una sola falla raíz (ej. caída de base de datos) detona cientos de alertas periféricas, logs o reportes de usuarios casi en simultáneo. 

Pasar cada una de estas alertas idénticas o colaterales a través de toda nuestra arquitectura Neuro-Simbólica de Múltiples Agentes (p. ej. `world_model`, `risk_hypothesizer`, `falsifier`) generaría **problemas críticos:**
1. **Caída por Límites de Cuota (Rate Limits):** Colapso de comunicación con el LLM (Gemini/Azure OpenAI) debido a errores `HTTP 429 RESOURCE_EXHAUSTED` por superar concurrentemente los Tokens por Minuto (TPM) o Requests por Minuto (RPM).
2. **Alert Fatigue:** Inundar a los ingenieros de guardia (On-Call) con decenas de tickets P0 en JIRA apuntando exactamente a la misma eventualidad obstaculiza la visualización y entorpece las métricas MTTA/MTTR.
3. **Costo Financiero Innecesario:** Pago por cómputo y tokens LLM generados totalmente reiterativos.

## Decisión Técnica
Se decide estructurar la estrategia de Deduplicación como una **barrera preventiva temprana ("Short-Circuit")** ejecutada sincrónicamente en la de capa de API (`main.py`, enrutador FastAPI), antes de despachar tareas complejas hacia `LangGraph`. 

La lógica implementada realiza lo siguiente:
1. **Vincular sin descartar (Flagging & Linkage):** Si la similitud del vector semántico (`Cosine Similarity`) del nuevo incidente es anormalmente alta respecto a algún incidente actualmente abierto (`status != RESOLVED`), el nuevo reporte es catalogado de inmediato. La información **se conserva (no se descarta)** en la base de datos (Cosmos DB), designando explícitamente su estatus como `DUPLICATE` y registrando el enlace causal mediante la clave `duplicate_of`.
2. **Incrementalidad (Tallying):** Se actualiza en el incidente "Padre" el contador recursivo (`occurrence_count += 1`), otorgándole implícitamente mayor señal de urgencia.
3. **Short-Circuit de IA Automática:** El sistema interrumpe el ciclo de Triage (se previene invocar el DAG en LangGraph), librando la CPU y elásticos de inferencia; inmediatamente devolviendo a la UI la confirmación de la falla junto con el `ticket_url` del ticket originador, sin duplicar notificaciones en el sistema de tickets de la empresa.

### Alternativas Consideradas: Flaggeo Post-Análisis
El equipo consideró dejar que el `consolidator_node.py` en `LangGraph` agrupara los incidentes que fuesen similares *después* de analizar su código AST fuente (RAG). 
**Rechazada.** Pese a que esto proveería a los técnicos de guardia una gran variedad de correlaciones de sub-fallas contextuales (y unificaba el ticket JIRA único), la arquitectura sucumbiría ante el tráfico de una Alert Storm. El "Post-Flagging" es óptimo a escala analítica, pero prohibitivo en disponibilidad y facturación para infraestructura LLM por incidentes reactivos del entorno de producción.

## Consecuencias

### Positivas
* **Alta Disponibilidad en Caos:** Resiliencia y supervivencia garantizada de la API en el peor de los escenarios de degradación de red (Alert Storms). 
* **Ahorro masivo:** Reducción probabilística exponencial de los gastos en métricas de tokens Gemini.
* **On-Call Experience:** Mejora exponencial en el clima SRE reduciendo severamente la fricción cognitiva del *Alert Fatigue* a 1 "Ticket Padre" por crisis con conteos certeros de repitencia.

### Negativas / Riesgos Conocidos
* **Falso Positivo en Deduplicación Temprana:** Si dos incidentes comparten un léxico semántico inicial casi idéntico (superando el threshold duro del `0.80` de distancia cosena) pero radican en librerías distintas u orígenes totalmente remotos que no han sido especificados en la descripción de Triage inicial; el Agente podría enclaustrar erróneamente un incidente único como un incidente subyacente. Para contrarrestar, el Umbral (Threshold) debe mantenerse elevado.
