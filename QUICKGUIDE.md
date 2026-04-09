# Neuro-Symbolic Agent Quick Guide

Follow these sequential steps to run, test, and audit the SRE Agent locally. This guide will walk you through setting up the environment, compiling the stack, triggering an incident, and observing the "Glass Box" features (Anti-Alucination and Deduplication).

---

## 📌 Prerequisites
- Docker and Docker Compose installed.
- Valid API Keys (Google Gemini API & Azure Cosmos DB).

## Step 1: Clone and Configure
Clone the repo and configure your secure `.env` file. Do NOT upload this file.
```bash
git clone <your-repo-url>
cd <repo-folder>
cp .env.example .env
```
Inside `.env`, verify you have placed:
- `GEMINI_API_KEY`: Used for embeddings and LLM reasoning.
- `COSMOS_ENDPOINT` & `COSMOS_KEY`: Points to your serverless NoSQL db.
- `APP_ADMIN_API_KEY`: Required for admin-only endpoints such as indexing, ledger audit, and incident resolution.

## Step 2: Build and Detach
Turn on the machine. This command spins up the backend and configures the Cosmos DB databases in the background.
```bash
docker compose up --build -d
```
You can view the startup logs with `docker compose logs -f sre-agent`. Wait until it acknowledges it is serving on port 8000.

## Step 3: Populate the RAG Index (AST Chunking)
Before the Agent can fact-check itself, it needs to slice the eShop codebase into Abstract Syntax Trees (AST) and embed them into Cosmos DB vector search.
```bash
curl -X POST http://localhost:8000/index \
  -H "X-Admin-Api-Key: $APP_ADMIN_API_KEY"
```
*(Note: Depending on API Rate Limits, this might take 1–3 minutes as it embeds hundreds of C# methods).*

---

## Step 4: The 3-Phase Simulation Test

We will simulate a production incident scenario to see the Agent execute its Dual-Track validation.

### Phase 4.1: Submit the Incident
Simulate a failed user checkout process throwing a 500 error:
```bash
curl -X POST http://localhost:8000/incident \
  -F "report=URGENT: Everything is broken! Checkout throws HTTP 500 on endpoint /api/orders with NullReferenceException. Customers can't buy."
```
*Note the returned `{"incident_id": "xxxx"}`.*

### Phase 4.2: Audit the Ledger (Glass Box)
Instead of waiting blindly, check the AI's internal reasoning. Watch how the `Span Arbiter` forces code citations or drops hallucinated text.
```bash
curl http://localhost:8000/incident/{incident_id}/ledger \
  -H "X-Admin-Api-Key: $APP_ADMIN_API_KEY"
```
*Observe the `VERDICT: HALLUCINATED` vs `VERDICT: VERIFIED` tags as the agent reads the code.*

### Phase 4.3: Test Alert Storm Deduplication (Optional)
Fire the exact same report again.
```bash
curl -X POST http://localhost:8000/incident \
  -F "report=URGENT: Checkout throws HTTP 500 on endpoint /api/orders!"
```
You will notice the response takes under **100ms**. The Agent successfully caught the similarity via In-Memory `Math.cos` distance, short-circuited the LLM to save quota, and returned a `duplicate_of` ID!

---

## Step 5: Incident Resolution
Once the issue is tracked and fixed, the SRE team can close the ticket. In this hackathon build, ticketing and notifications are mocked but the end-to-end flow is fully demoable.
```bash
curl -X POST http://localhost:8000/incident/{incident_id}/resolve \
  -H "X-Admin-Api-Key: $APP_ADMIN_API_KEY"
```

🎉 **Testing Complete.** Ensure you use Docker Desktop to spin down the local image gracefully when finished to clear dangling tasks.
