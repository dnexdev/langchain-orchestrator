# Streaming LangChain Orchestrator

FastAPI service with one endpoint, `POST /ask`. Uses LangChain to decide whether a question is about math and sends it to a matching chain. The answer is then streamed token by token as Server-Sent Events.
