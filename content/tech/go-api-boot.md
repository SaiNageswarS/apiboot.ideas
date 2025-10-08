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

## agent-boot: Building Agents with Reasoning and Tool Use

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

[medicine-rag](https://github.com/SaiNageswarS/medicine-rag) has following components:
1. **[Ingestion Workers](https://github.com/SaiNageswarS/medicine-rag/tree/master/core/workers/workflows):** Convert PDF to markdown, chunk and embed into MongoDB. Built on temporal integrated in go-api-boot.
2. **[Agentic RAG service](https://github.com/SaiNageswarS/medicine-rag/blob/6942c97b593319e5c76047e82e8f44ff6328df5e/core/services/agent_service.go#L43):** A service that uses agent-boot to use the tools to answer questions and stream the answer real time to the frontend.
3. **[Frontend](https://github.com/SaiNageswarS/medicine-rag/tree/master/web):** A frontend that uses the agentic rag service to answer questions.

Ingestion Workers and Agentic RAG Service are powered by [go-api-boot](https://github.com/SaiNageswarS/go-api-boot).

## go-api-boot ODM with Vector Search and Text Search

The **Object Document Mapper (ODM)** in go-api-boot comes with first-class support for vector and text search, allowing developers to unify CRUD and hybrid retrieval operations — a critical requirement for knowledge-driven agentic systems.

```go
// genric multi-tenant CRUD operations
type Article struct {
    ID        string      `bson:"_id"`
    Title     string      `bson:"title"`
    Content   string      `bson:"content"`
    Embedding bson.Vector `bson:"embedding"` // 768-dim vector
}

func (a Article) Id() string { return a.ID }
func (a Article) CollectionName() string { return "articles" }

// Query
client, err := odm.GetClient(ccfg)
repo := odm.CollectionOf[Article](client, tenant)
article, err := async.Await(repo.FindOneById(grpcCtx, id))

// Vector Search
params := odm.VectorSearchParams{
  IndexName:     "contentVecIdx",
  Path:          "embedding",    // vector column name
  K:             5,              // number of results to return
  NumCandidates: 20,
}
results, _ := async.Await(repo.VectorSearch(ctx, embedding, params))

// Text Search
params = odm.TermSearchParams{
  IndexName: "contentTextIdx",
  Path:      []string {"content", "title"}, // text columns to search
  Limit:     10,
}
results, _ := async.Await(repo.TermSearch(ctx, "golang guides", params))
```

This unified ODM design makes hybrid search and retrieval a native feature of your Go services.

## go-api-boot Temporal Integration

[go-api-boot](https://github.com/SaiNageswarS/go-api-boot) integrates with [Temporal](https://temporal.io/) to provide a seamless way to orchestrate long-running ML jobs/workers. It allows to run Python ML tasks in Temporal workers and invoke them from Go.

```go
boot, _ := server.New().
    WithTemporal("ML_TASK_QUEUE", &client.Options{
        HostPort: "temporal:7233",
    }).
    RegisterTemporalActivity(Activities).
    RegisterTemporalWorkflow(GoPythonCombinedWorkflow).
    Build()
```

## go-api-boot Microservice APIs & Dependency Injection

[go-api-boot](https://github.com/SaiNageswarS/go-api-boot) supports building first-class gRpc APIs with out-of-the-box support for authentication, authorization, metrics, tracing, logging, and more. Further, it also creates REST proxy APIs for the gRpc APIs without any additional code.

Here, we are registering a gRpc service for the search service and the agent service.

```go
boot, err := server.New().
    GRPCPort(":50051").
    HTTPPort(":8081").
    // Dependency injection
    Provide(config).
    Provide(azureClient).
    Provide(claudeClient).
    Provide(mongoClient).
    // Temporal workflow registration
    WithTemporal(config.TemporalGoTaskQueue, &temporalClient.Options{
        HostPort: config.TemporalHostPort,
    }).
    RegisterTemporalActivity(activities.ProvideActivities).
    RegisterTemporalWorkflow(workflows.ChunkMarkdownWorkflow).
    RegisterTemporalWorkflow(workflows.PdfHandlerWorkflow).
    // gRPC service registration
    RegisterService(server.Adapt(pb.RegisterSearchServer), services.ProvideSearchService).
    RegisterService(server.Adapt(pb.RegisterAgentServer), services.ProvideAgentService).
    Build()
```

The above dependency injection is build group up, customized for gRpc, temporal, cloud and more. Unlike Wire, it doesn't generate any code and is more natural for Go.

## go-api-boot Streaming for real-time Agentic RAG

go-api-boot provides gRpc streaming support for the microservices. This can be used to stream the results from the agentic rag service to the frontend in real-time.

Below is screen grab of [medicine-rag](https://github.com/SaiNageswarS/medicine-rag) in action.

![Medicine RAG Demo](/images/medicine-rag.gif)

Further, agent-boot supports gRpc stream to provide updates of internal agent state like tool selection, tool execution, and answer generation with reasoning.

This helps to build real-time Agentic RAG systems with minimal code.

Below is medicine-rag code snippet for streaming the results to the frontend. Here, gRpc server stream is being passed to agent.Execute.

[https://github.com/SaiNageswarS/medicine-rag/blob/6942c97b593319e5c76047e82e8f44ff6328df5e/core/services/agent_service.go#L61](https://github.com/SaiNageswarS/medicine-rag/blob/6942c97b593319e5c76047e82e8f44ff6328df5e/core/services/agent_service.go#L61)

## Conclusion

**go-api-boot** and **agent-boot** form a powerful foundation for building **production-ready Agentic RAG systems**.
They enable developers to create intelligent agents that can think, retrieve, reason, and stream results in real time — backed by Go’s stability and Python’s flexibility.

> The next generation of AI systems won’t just generate answers — they’ll reason, plan, and act.
> And when they do, go-api-boot will be the muscle that powers them.

Thanks for reading!
Explore the source code and examples here:

- [go-api-boot on GitHub](https://github.com/SaiNageswarS/go-api-boot)
- [agent-boot on GitHub](https://github.com/SaiNageswarS/agent-boot)
- [medicine-rag example](https://github.com/SaiNageswarS/medicine-rag)
