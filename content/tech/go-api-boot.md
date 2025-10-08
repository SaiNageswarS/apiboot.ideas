---
title: "Designing Next-Gen Agentic RAG Systems with go-api-boot and agent-boot"
date: "2025-10-08T02:49:06Z"
draft: false
tags: ["golang", "Micro-Services", "Agentic RAG", "streaming RAG", "temporal"]
description: "Using go-api-boot and agent-boot to enable production-ready Agentic RAG systems by combining Go's performance with Python's ML power."
comments: true
---

**Agentic RAG** takes Retrieval-Augmented Generation one step further.
In traditional RAG, retrieval is a fixed step — fetching context from a knowledge base before generating a response.
In Agentic RAG, however, the **data source itself becomes a tool** in the agent's toolkit.
The agent autonomously decides when and how to query it based on the **conversation history, user intent, and reasoning chain**, alongside other tools like summarizers or planners.
This transforms RAG from a static retrieval process into a dynamic, decision-driven system.

Building such systems requires both **intelligence** and **infrastructure**.
Python continues to lead in ML research, but scaling agentic workloads demands the concurrency, reliability, and type safety that Go provides.
This is where [go-api-boot](https://github.com/SaiNageswarS/go-api-boot) and [agent-boot](https://github.com/SaiNageswarS/agent-boot) come together — frameworks that unite Go's high-performance microservice architecture with Python's ML ecosystem, enabling developers to build fully-agentic RAG systems that think, retrieve, and respond in real time.

## go-api-boot: The Muscle Behind Agentic AI Microservices

[go-api-boot](https://github.com/SaiNageswarS/go-api-boot) is a batteries-included Go framework built for AI and microservice developers who want production readiness from day one.
It hides the boilerplate — yet keeps Go's explicitness intact.

Highlights:
- **gRPC + HTTP gateway** auto-generated from proto files (no manual routing).
- **Temporal workflow** integration for orchestrating long-running ML jobs.
- **MongoDB ODM** with vector search support for hybrid retrieval.
- **Dependency Injection** without code-gen, designed specifically for Go’s type system.
- **Zero-config SSL** and cloud abstraction across Azure and GCP.

Example minimal bootstrap:
```go
boot, _ := server.New().
    GRPCPort(":50051").
    HTTPPort(":8080").
    WithTemporal("ML_TASK_QUEUE", &client.Options{HostPort: "temporal:7233"}).
    RegisterTemporalWorkflow(workflows.RAGWorkflow).
    RegisterTemporalActivity(activities.PythonTasks).
    Build()
```

In less than ten lines, we have a gRPC + REST API server with Temporal orchestration, metrics, and structured logging — ready for AI workloads.

## agent-boot: Bringing Agents and LLMs into the Mix

[agent-boot](https://github.com/SaiNageswarS/agent-boot) is a framework for building intelligent agents capable of tool selection using GPT-OSS, tool invocation, optional tool results summarization, and answer generation with reasoning.

It complements [go-api-boot](https://github.com/SaiNageswarS/go-api-boot) perfectly — together, they form a cohesive foundation for building end-to-end Agentic RAG systems that combine structured orchestration with autonomous decision-making.

Using its MCP-style tool definition, developers can bind domain-specific actions to LLM agents:
```go
mcp := agentboot.NewMCPToolBuilder("medicine-rag", "Search and retrieve medical info.").
  StringParam("query", "Search Query", true).
  WithHandler(func(ctx context.Context, params api.ToolCallFunctionArguments) <-chan *schema.ToolResultChunk {
      query := params["query"].(string)
      return search.Run(ctx, query)
  }).
  Summarize(true).
  Build()
```

Each tool runs asynchronously in Go, but can also invoke Python ML tasks through Temporal (through go-api-boot's Temporal integration) — achieving concurrency and ML richness in one pipeline.

## Building a Real Agentic RAG System: medicine-rag

To see these frameworks in action, let’s look at [medicine-rag](https://github.com/SaiNageswarS/medicine-rag), a RAG application that augments Claude with domain-specific medical knowledge.