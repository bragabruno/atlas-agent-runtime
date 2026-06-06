# RegDoc Q&A Agent — Sequence Diagram

End-to-end sequence for the RegDoc Q&A demo agent: LLM call → doc_search → compose answer → citation verification → persist → repeat until done. Cap-breach abort branch shown.

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant AgentRunner
    participant CapsEnforcer
    participant GatewayClient
    participant Gateway as "Atlas Gateway\n/v1/chat/completions"
    participant McpClient
    participant DocSearch as "mcp-doc-search\ndoc_search"
    participant Citations as "mcp-citations\nverify_citation"
    participant Persistence

    User->>AgentRunner: run(input)
    AgentRunner->>Persistence: insert agent_run (status=running)
    AgentRunner->>CapsEnforcer: start timer

    loop Each iteration
        AgentRunner->>CapsEnforcer: tick_iteration()
        CapsEnforcer-->>AgentRunner: ok (or CapBreachError: max_iterations)

        AgentRunner->>GatewayClient: chat(model_alias, messages, tools)
        GatewayClient->>Gateway: POST /v1/chat/completions
        Gateway-->>GatewayClient: ChatResponse (tool_call or message)
        GatewayClient-->>AgentRunner: ChatResponse

        AgentRunner->>CapsEnforcer: add_tokens(response.usage.total_tokens)
        CapsEnforcer-->>AgentRunner: ok (or CapBreachError: token_budget)

        AgentRunner->>Persistence: insert agent_step (type=llm_call)

        alt LLM requests doc_search
            AgentRunner->>McpClient: doc_search(query, k)
            McpClient->>DocSearch: doc_search(query, k)
            DocSearch-->>McpClient: chunks + source_ids
            McpClient-->>AgentRunner: sanitized result
            AgentRunner->>Persistence: insert agent_step (type=tool_call)
            AgentRunner->>CapsEnforcer: check_wall_time()
            CapsEnforcer-->>AgentRunner: ok (or CapBreachError: timeout_s)

        else LLM emits final answer
            AgentRunner->>McpClient: verify_citation(source_id, claim) [per claim]
            McpClient->>Citations: verify_citation(source_id, claim)
            Citations-->>McpClient: {exists, snippet}
            McpClient-->>AgentRunner: CitationResult

            alt Citation valid
                AgentRunner->>Persistence: insert agent_step (type=tool_call)
                AgentRunner->>Persistence: update agent_run (status=completed)
                AgentRunner-->>User: answer
            else Citation invalid
                AgentRunner->>Persistence: update agent_run (status=failed)
                AgentRunner-->>User: CitationError (claim not backed by source_id)
            end
        end
    end

    opt Cap breach at any point
        CapsEnforcer-->>AgentRunner: CapBreachError (explicit message)
        AgentRunner->>Persistence: update agent_run (status=capped)
        AgentRunner-->>User: CapBreachError
    end
```
