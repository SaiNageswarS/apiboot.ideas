---
date: '2025-10-06T13:24:25Z'
draft: false
title: 'Streaming Map-Filter-Reduce in Go: Building Cancellable Pipelines for Real-Time AI'
description: 'Build streaming, cancellable map-filter-reduce pipelines in Go for real-time AI applications.'
tags: ['go', 'streaming', 'map-reduce', 'concurrency']
categories: ['tech']
comments: true
---

Classic map-reduce-filter pattern process data step-by-step. For example, a map-filter-collect on an array of numbers would first transform each number, then filter out the one's that meet a condition, and finally collect the results. 

However, in real-time applications, we often need data to be processed as it becomes available. Imagine an LLM agent that summarizes paragraphs from search one by one. Instead of waiting for all sections to finish, it can start streaming summaries to the frontend immediately, giving users faster, progressive feedback.

This inspired me to build [go-collection-boot/linq](https://github.com/SaiNageswarS/go-collection-boot), a Go library that brings streaming map-filter-reduce pipelines to life — allowing results to stream live, cancel unnecessary work, and run safely across goroutines.

## From Batches to Streams

Let’s start with something every developer knows — the map–filter–reduce flow

```js
const numbers = [1, 2, 3, 4, 5];

numbers.map(n => n * 2).
    filter(n => n % 4 === 0).
    map(n => n/2).
    foreach(n => console.log(n)); // Output after all steps finish
```

In this example, the array goes through three complete transformations before any output appears.
The console prints [2, 4] only after every stage (map → filter → map) has processed the entire array.

This works fine for small datasets — but in real-time or I/O-bound systems, it creates unnecessary latency.
Imagine waiting for a language model to finish generating all tokens before showing the first word to a user — that’s exactly what a batch pipeline does.

A streaming pipeline, on the other hand, starts producing results as soon as they’re ready.
The first item (2) would flow through immediately, followed by the next (4) once it’s processed.
Downstream consumers can begin work instantly, making this approach ideal for:
- LLM output streaming (tokens rendered as they generate)
- Real-time data ingestion
- Concurrent transformations in Go routines

## Introducing go-collection-boot/linq

[go-collection-boot/linq](https://github.com/SaiNageswarS/go-collection-boot) brings a streaming, cancel-aware flavor of map–filter–reduce to Go. It keeps the API familiar (Select, Where, ForEach, ToSlice, …) while running each stage in its own goroutine and wiring context propagation under the hood.

Core ideas:
- **Streaming**: stages push items as soon as they’re ready (no full-batch wait).
- **Early termination**: sinks like First, Any, All short-circuit and cancel upstream.
- **Parallel map with order**: SelectPar fans out work per element and still preserves original order.
- **Safety**: every send/receive checks ctx.Done(); no goroutine leaks.

## Quick Start: Streamed Map → Filter → Collect

```go
ctx := context.Background()
data := []int{1, 2, 3, 4, 5}

squaresOfOdds, err := linq.Pipe3(
    linq.FromSlice(ctx, data),                        // source
    linq.Where(func(n int) bool { return n%2 == 1 }), // filter (keep odds)
    linq.Select(func(n int) int { return n * n }),    // map (square)
    linq.ToSlice[int](),                              // sink (collect)
)
```

- **Nothing exotic**: readable like batch map/filter/reduce.
- **Behavior**: streaming—Select emits to the sink as soon as each odd number is squared.

## Early Termination (and Why It Matters)

Compute-heavy pipelines should stop the moment we have enough. Sinks like First, Any, and All trigger a cancel that ripples upstream.

```go
firstEven, err := linq.Pipe2(
    linq.FromSlice(ctx, []int{1, 3, 6, 8}),
    linq.Where(func(n int) bool { return n%2 == 0 }),
    linq.First[int](), // emits 6 → cancels everything upstream
)
```

This pattern is perfect for:
- Top-K scenarios (find first/any matching result).
- Heuristic cutoffs (stop when confidence threshold is reached).
- LLM gateways (stop summarizing once the answer is “good enough”).

We can also short-circuit with predicates:
```go
hasLarge, err := linq.Pipe1(
    linq.FromSlice(ctx, []int{1,2,3,10,5}),
    linq.Any(func(n int) bool { return n >= 10 }), // true, cancels upstream
)
```

## Parallel Map That Keeps Order (SelectPar) 

We often want one goroutine per element (e.g., LLM calls per section) but we still want results in the original order. SelectPar does both:

```go
ctx := context.Background()
nums := []int{1,2,3,4,5}

squared, err := linq.Pipe2(
    linq.FromSlice(ctx, nums),
    linq.SelectPar(func(n int) int {
        // pretend this is expensive
        return n * n
    }),
    linq.ToSlice[int](),
)
// => [1, 4, 9, 16, 25] (order preserved)
```

Internally, it fans out work, buffers out-of-order completions, and releases them in sequence.

## Real-Time LLM Example: Summarize → Filter → Stream to UI

Here’s a minimal, production-shaped flow for section-wise summarization. It streams as each summary is ready, filters bad/irrelevant chunks, and pushes updates to the frontend immediately.

```go
func SummarizeAndStream(ctx context.Context, sections <-chan *Section) error {
    // NewStream accepts an existing producer channel
    _, err := linq.Pipe4(
        linq.NewStream(ctx, sections, func(){}, 10),

        // parallel LLM calls while preserving order
        linq.SelectPar(func(s *Section) *Summary {
            return summarizeWithLLM(ctx, s)
        }),

        // drop empty/noisy summaries
        linq.Where(validSummary),

        // stream out as soon as each summary is ready
        linq.ForEach(func(s *Summary) {
            streamToClient(s) // UI updates incrementally
        }),
    )
    return err
}
```

I used above pattern in [Agent-Boot](https://github.com/SaiNageswarS/agent-boot/blob/19583f1d8c3ee7bdc831aed8bb465049d1f9dae0/agentboot/run_tool.go#L36) for tool execution with summarization support. The performance improvement was significant, from 80 seconds for complete tool result output to front-end to 6 seconds for first summary to be streamed to front-end.

**Before SelectPar (sequential):**

20 sections × 4 seconds each = 80 seconds
**After SelectPar (parallel streaming):**

- 20 sections in parallel: ~6 seconds total
- Results stream to user starting at ~4 seconds

**85% improvement with real-time feedback!**

## Conclusion

Streaming map-filter-reduce in Go is a powerful tool for building real-time applications. It allows us to process data as it becomes available, cancel unnecessary work, and run safely across goroutines.

I hope this article has given you a good introduction to the concept and how you can use it in your projects.

Thanks for reading!

### Resources
[1] [https://github.com/SaiNageswarS/go-collection-boot/blob/master/linq/linq.go](https://github.com/SaiNageswarS/go-collection-boot/blob/master/linq/linq.go)

```sh
go get github.com/SaiNageswarS/go-collection-boot
```

[2] Integrations: [Agent-Boot](https://github.com/SaiNageswarS/agent-boot), [Medicine-RAG](https://github.com/SaiNageswarS/medicine-rag)
