# Agent UML

2026-09-18時点の実装を示す。全体方針は[agent_overview.md](agent_overview.md)を参照。
読みやすさのため補助型と一部引数は省略している。

## 1. Agent制御、tool、限定Reasoning

```mermaid
classDiagram
    class Agent {
        +process(input) AgentResponse
        -finish_turn(parsed, response)
        -compress_for_chat(parsed, memories)
        -_recent ConversationTurn[]
        -_summary string
        -_discussion_state json
    }
    class IInputParser { <<interface>>
        +parse(input) ParsedInput
    }
    class IRouter { <<interface>>
        +route(parsed) RequestType
    }
    class IPlanner { <<interface>>
        +create(parsed, memories) Plan
        +replan(parsed, plan, result, evaluation) Plan
    }
    class IExecutor { <<interface>>
        +execute(task, previousResults) ToolResult
    }
    class IReasoningTaskExecutor { <<interface>>
        +execute(task, previousResults) ToolResult
    }
    class IEvaluator { <<interface>>
        +evaluate(task, result) EvaluationResult
    }
    class IAggregator { <<interface>>
        +aggregate(results) json
    }
    class IMemoryManager { <<interface>>
        +retrieve(query, type, limit) MemoryRecord[]
        +store(candidate)
        +store_batch(candidates)
    }
    class IReasoner { <<interface>>
        +parse(input) ParsedInput
        +plan(parsed, memories) Plan
        +replan(...) Plan
        +evaluate(task, result) EvaluationResult
        +chat(...) string
        +summarize(...) string
        +memory_candidates(...) MemoryCandidate[]
        +final_response(...) string
    }

    Agent o-- IInputParser
    Agent o-- IRouter
    Agent o-- IPlanner
    Agent o-- IExecutor
    Agent o-- IEvaluator
    Agent o-- IAggregator
    Agent o-- IMemoryManager
    Agent o-- IReasoner
    Agent o-- IEntityExtractor

    IInputParser <|.. DefaultInputParser
    IRouter <|.. DefaultRouter
    IPlanner <|.. DefaultPlanner
    IExecutor <|.. DefaultExecutor
    IEvaluator <|.. DefaultEvaluator
    IAggregator <|.. DefaultAggregator
    IMemoryManager <|.. SqliteMemory

    DefaultInputParser o-- IReasoner
    DefaultPlanner o-- IReasoner
    DefaultEvaluator o-- IReasoner
    DefaultExecutor o-- ToolRegistry
    DefaultExecutor o-- IReasoningTaskExecutor
    IReasoningTaskExecutor <|.. DiscussionEngine

    class ITool { <<interface>>
        +execute(arguments) ToolResult
    }
    ToolRegistry o-- ITool
    ITool <|.. CalculatorTool
    ITool <|.. FileReadTool
    ITool <|.. MemoryRetrieveTool
    MemoryRetrieveTool o-- IMemoryManager
```

## 2. 議論状態

```mermaid
classDiagram
    class DiscussionEngine {
        +execute(task, previousResults) ToolResult
        +state(scenarioId) json
        -_states map
    }
    class DiscussionState {
        +scenario_id string
        +topic string
        +objective string
        +option_ids string[]
        +priorities json
        +requested_attributes string[]
        +applied_update_ids set
        +revisions json
    }
    class Evidence {
        +id string
        +text string
        +source_id string
        +start size_t
        +end size_t
        +origin EvidenceOrigin
        +content_verified bool
    }
    class Claim {
        +id string
        +speaker string
        +subject string
        +attribute string
        +time string
        +condition string
        +value double
        +unit string
        +stance ClaimStance
        +evidence_ids string[]
    }
    class DiscussionConstraint {
        +id string
        +attribute string
        +operation string
        +value double
        +unit string
        +required bool
        +evidence_ids string[]
    }
    class Decision {
        +status DecisionStatus
        +conclusion string
        +selected_option optional~string~
        +evidence_ids string[]
        +open_questions string[]
    }
    class StateUpdate {
        +id string
        +kind string
        +target_id string
        +replacement json
        +evidence optional~Evidence~
    }

    DiscussionEngine *-- DiscussionState
    DiscussionState *-- Evidence
    DiscussionState *-- Claim
    DiscussionState *-- DiscussionConstraint
    DiscussionState *-- Decision
    Claim --> Evidence : evidence_ids
    DiscussionConstraint --> Evidence : evidence_ids
    StateUpdate o-- Evidence
```

`StateUpdate`相当のJSONは`correct_constraint`、`correct_claim`、`set_priorities`、
`retract_claim`として検証後に適用する。update IDで重複適用を防ぐ。

## 3. Reasonerと生成モデル

```mermaid
classDiagram
    IReasoner <|.. RuleReasoner
    IReasoner <|.. HybridReasoner
    IReasoner <|.. ModelReasoner
    HybridReasoner *-- RuleReasoner : _rule
    HybridReasoner o-- ILanguageModel : chat・summary
    ModelReasoner o-- ILanguageModel : multi-mode
    HybridReasoner ..> ContextBuilder
    ModelReasoner ..> ContextBuilder

    class ILanguageModel { <<interface>>
        +complete(mode, prompt) string
        +token_count(text) size_t
    }
    ILanguageModel <|.. ModelLanguageModel
    ModelLanguageModel *-- AgentBundle
    AgentBundle *-- AgentTokenizer
    AgentBundle *-- AgentTransformer

    class AgentTransformer {
        +forward(ids) TensorPtr
        +loss(ids, targets) TensorPtr
        +config() AgentTransformerConfig
    }
    Model <|-- AgentTransformer
    AgentTransformer *-- AgentTransformerConfig
    AgentTransformer "1" *-- "1..*" AgentTransformerBlock
    Module <|-- AgentTransformerBlock
    AgentTransformerBlock *-- AgentAttention
    AgentTransformerBlock *-- AgentFeedForward
```

明示的な`/compare JSON`では`DefaultInputParser`と`DefaultPlanner`が決定的な経路を
作り、`DefaultEvaluator`もモデルによる自己評価を使わない。`ModelReasoner`の最終文も
検証済みdecisionを短く整形するだけで、判断内容を書き換えない。

## 4. NER

```mermaid
classDiagram
    class IEntityExtractor { <<interface>>
        +extract(text, utteranceId) EntityMention[]
    }
    IEntityExtractor <|.. RuleEntityExtractor
    IEntityExtractor <|.. ModelEntityExtractor
    IEntityExtractor <|.. HybridEntityExtractor
    HybridEntityExtractor o-- IEntityExtractor : rules
    HybridEntityExtractor o-- IEntityExtractor : model

    class EntityMention {
        +type EntityType
        +start size_t
        +end size_t
        +surface string
        +normalized optional~string~
        +source EntitySource
        +utterance_id string
        +score optional~float~
    }
    IEntityExtractor ..> EntityMention
    ModelEntityExtractor *-- NERBundle
    RuleEntityExtractor ..> NumericRules
```

`EntityMention.start/end`はUTF-8 byte offsetの半開区間で、model内部のUnicode code
point位置から原文位置へ戻す。scoreは未校正logitであり正解確率ではない。

## 5. 通常会話シーケンス（hybrid）

```mermaid
sequenceDiagram
    actor U as User
    participant A as Agent
    participant N as IEntityExtractor
    participant P as DefaultInputParser
    participant R as DefaultRouter
    participant DB as SqliteMemory
    participant H as HybridReasoner
    participant C as ContextBuilder
    participant LM as ModelLanguageModel

    U->>A: process(input)
    opt NER enabled
        A->>N: extract(input, utteranceId)
        N-->>A: EntityMention candidates
    end
    A->>P: parse(input)
    P-->>A: ParsedInput
    A->>R: route(parsed)
    R-->>A: SimpleConversation
    A->>DB: retrieve(raw, nullopt, 8)
    DB-->>A: memories
    A->>A: compress_for_chat if budget requires
    A->>H: chat(parsed, memories, recent, summary)
    H->>C: build(context within 766-token prompt budget)
    C-->>H: prompt
    H->>LM: complete(Chat, prompt)
    LM-->>H: response
    H-->>A: response
    A->>A: finish_turn
    A->>DB: store_batch(accepted candidates)
    A-->>U: AgentResponse
```

## 6. 限定比較シーケンス

```mermaid
sequenceDiagram
    actor U as User
    participant A as Agent
    participant P as DefaultInputParser
    participant PL as DefaultPlanner
    participant EX as DefaultExecutor
    participant D as DiscussionEngine
    participant EV as DefaultEvaluator
    participant AG as DefaultAggregator
    participant RR as Rule final formatter

    U->>A: /compare JSON
    A->>P: parse
    Note over P: explicit comparisonはmodel parseを迂回
    P-->>A: ParsedInput(intent=comparison)
    A->>PL: create
    PL-->>A: discussion.compare task
    A->>EX: execute(task)
    EX->>D: execute(task)
    D->>D: validate evidence/update
    D->>D: conflict・missing・constraint・priority判定
    D-->>EX: decision + DiscussionState
    A->>EV: evaluate
    Note over EV: ToolStatusと検証済みdecisionを使用
    A->>AG: aggregate
    AG-->>A: explicit semantic conflicts only
    A->>RR: format conclusion/evidence/open questions
    A->>A: retain DiscussionState separately from summary
    A-->>U: short grounded response
```

## 7. 学習とbundle

```mermaid
flowchart LR
    J[Jawiki shards] --> PT[pretrain_sharded]
    PT --> LM[Language model bundle]
    BASE[既存agent SFT] --> MIX[mix_discussion_sft.py]
    DISC[discussion FINAL examples] --> MIX
    MIX --> SFT[agent_model_cli sft]
    LM --> SFT
    SFT --> LMF[新しいlanguage model bundle]

    NW[NER Wikipedia data] --> NC[convert_ner_wikipedia.py]
    NC --> NT[ner_cli train]
    NT --> NB[独立NER bundle]

    LMF --> RT[Agent runtime]
    NB --> RT
```

言語モデルとNERは別bundleである。既存事前学習をやり直さず、完了checkpointから
混合SFTを別出力へ作る。実行時に学習やdownloadはしない。
