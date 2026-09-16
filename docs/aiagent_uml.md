# AIAgent UML別紙

2026-09-16時点の作業ツリーに基づく主要クラスの抜粋。[概要・コード対応表](aiagent_overview.md)と併読してください。型・引数・補助メンバーは読みやすさのため一部省略しています。

凡例：白三角の破線はinterface実装、白三角の実線は継承、黒菱形は所有、白菱形は共有参照、点線矢印は利用です。

## 別紙A：Agent制御層のクラス図

```mermaid
classDiagram
    class Agent {
        +process(input) AgentResponse
        -finish_turn(parsed, response)
        -_recent
        -_summary
    }
    class IInputParser { <<interface>>
        +parse(input) ParsedInput
    }
    class IRouter { <<interface>>
        +route(input) RequestType
    }
    class IPlanner { <<interface>>
        +create(input, memories) Plan
        +replan(input, plan, result, evaluation) Plan
    }
    class IExecutor { <<interface>>
        +execute(task, results) ToolResult
    }
    class IEvaluator { <<interface>>
        +evaluate(task, result) EvaluationResult
    }
    class IAggregator { <<interface>>
        +aggregate(results) json
    }
    class IMemoryManager { <<interface>>
        +retrieve(query) MemoryRecord[]
        +store(candidate)
        +store_batch(candidates)
    }
    class IReasoner { <<interface>>
        +parse()
        +plan()
        +replan()
        +evaluate()
        +chat()
        +summarize()
        +memory_candidates()
        +final_response()
    }
    Agent o-- IInputParser
    Agent o-- IRouter
    Agent o-- IPlanner
    Agent o-- IExecutor
    Agent o-- IEvaluator
    Agent o-- IAggregator
    Agent o-- IMemoryManager
    Agent o-- IReasoner
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
    ToolRegistry o-- ITool
    class ITool { <<interface>>
        +execute(arguments) ToolResult
    }
    ITool <|.. CalculatorTool
    ITool <|.. FileReadTool
    ITool <|.. MemoryRetrieveTool
    MemoryRetrieveTool o-- IMemoryManager
```

## 別紙B：推論器とニューラルモデルのクラス図

```mermaid
classDiagram
    IReasoner <|.. RuleReasoner
    IReasoner <|.. HybridReasoner
    IReasoner <|.. ModelReasoner
    HybridReasoner *-- RuleReasoner : _rule
    HybridReasoner o-- ILanguageModel : chatのみ
    ModelReasoner o-- ILanguageModel : 7 mode
    HybridReasoner ..> ContextBuilder
    ModelReasoner ..> ContextBuilder
    class ModelReasoner {
        -structured(mode, prompt, schema) json
        -build_prompt(input) string
    }
    class ContextBuilder {
        +build(input) string
    }
    class ILanguageModel { <<interface>>
        +complete(mode, prompt) string
        +token_count(text) size_t
    }
    ILanguageModel <|.. ModelLanguageModel
    class ModelLanguageModel {
        +complete(mode, prompt) string
        +token_count(text) size_t
        +model_config()
    }
    ModelLanguageModel *-- AgentBundle
    AgentBundle *-- AgentTokenizer
    AgentBundle *-- AgentTransformer
    class AgentTokenizer {
        +train(path, vocabulary)$ AgentTokenizer
        +load(path)$ AgentTokenizer
        +encode(text) IDs
        +decode(ids) string
        +mode_id(mode) int
    }
    class AgentTransformer {
        +forward(ids) TensorPtr
        +loss(ids, targets) TensorPtr
        +config() AgentTransformerConfig
    }
    Model <|-- AgentTransformer
    AgentTransformer *-- AgentTransformerConfig
    AgentTransformer "1" *-- "1..*" AgentTransformerBlock : 既定4層
    Module <|-- AgentTransformerBlock
    Module <|-- AgentAttention
    Module <|-- AgentFeedForward
    AgentTransformerBlock *-- AgentAttention
    AgentTransformerBlock *-- AgentFeedForward
    class AgentTransformerBlock {
        +forward(inputs) TensorPtr
    }
    class AgentAttention {
        +forward(inputs) TensorPtr
    }
    class AgentFeedForward {
        +forward(inputs) TensorPtr
    }
```

`generate()`・`train()`・`validate()` はクラスではなく `ai::model` の自由関数です。`build_model_prompt()` も自由関数で、内部でContextBuilderを利用します。

## 別紙C：通常会話のシーケンス（hybrid）

```mermaid
sequenceDiagram
    actor U as ユーザー
    participant A as Agent
    participant P as DefaultInputParser
    participant R as DefaultRouter
    participant H as HybridReasoner
    participant Rule as RuleReasoner
    participant DB as SqliteMemory
    participant LM as ModelLanguageModel
    participant T as AgentTokenizer
    participant NN as AgentTransformer
    U->>A: process(input)
    A->>P: parse(input)
    P->>H: parse(input)
    H->>Rule: parse(input)
    Rule-->>A: ParsedInput（Parser経由）
    A->>R: route(parsed)
    R-->>A: SimpleConversation
    A->>DB: retrieve(raw, nullopt, 8)
    DB-->>A: memories
    A->>H: chat(parsed, memories, recent, summary)
    Note over H: build_model_prompt()で文脈を構築
    H->>LM: complete(Chat, prompt)
    LM->>T: encode(prompt)
    T-->>LM: token IDs
    Note over LM: BOSとmode IDを付加してgenerate()
    loop EOSまたは生成上限まで
        LM->>NN: forward(ids)（generate経由）
        NN-->>LM: logits
        Note over LM: 次のIDを選択・履歴に追加
    end
    LM->>T: decode(generated IDs)
    T-->>LM: 応答文
    LM-->>H: 応答文
    H-->>A: 応答文
    A->>A: finish_turn(parsed, response)
    A->>H: memory_candidates(parsed, response)
    H->>Rule: memory_candidates(parsed, response)
    Rule-->>A: candidates（Hybrid経由）
    opt 保存条件を満たす候補あり
        A->>DB: store_batch(accepted)
    end
    Note over A: recentに追加、4ターン超なら古い分を要約
    A-->>U: AgentResponse
```

## 別紙D：ツール実行のシーケンス

```mermaid
sequenceDiagram
    participant A as Agent
    participant P as DefaultPlanner
    participant R as IReasoner
    participant E as DefaultExecutor
    participant TR as ToolRegistry
    participant T as ITool
    participant V as DefaultEvaluator
    participant G as DefaultAggregator
    Note over A: parse・route・retrieve完了後、通常会話以外の経路
    A->>P: create(parsed, memories)
    P->>R: plan(parsed, memories)
    R-->>P: Plan
    P-->>A: Plan
    A->>A: validate_plan(plan)
    loop 計画を依存順に実行、再計画時は新計画で再開
        A->>A: topological_order(plan)
        A->>E: execute(task, results)
        E->>TR: execute(operation, arguments)
        TR->>T: execute(arguments)
        T-->>A: ToolResult（Registry・Executor経由）
        A->>V: evaluate(task, result)
        V->>R: evaluate(task, result)
        R-->>A: EvaluationResult（Evaluator経由）
        alt Retry
            Note over A: 同一タスクを追加2回まで実行・評価
        else Replan
            A->>P: replan(parsed, plan, result, evaluation)
            P->>R: replan(...)
            R-->>A: 新Plan（Planner経由）
            A->>A: validate_plan(newPlan)
            Note over A: 再計画は3回まで
        else Failed または上限到達
            Note over A: 例外を捕捉しエラー応答、以降の集約へ進まない
        else Success
            Note over A: 次のタスクへ
        end
    end
    Note over A: 全タスクが正常に完了した場合
    A->>G: aggregate(results)
    G-->>A: 集約JSON
    A->>A: contradictionsが空であることを確認
    A->>R: final_response(parsed, aggregate)
    R-->>A: 最終文
    A->>A: finish_turn(parsed, response)
```

この図はTool型タスクの正常系を中心に示しています。`hybrid` では図中の `IReasoner` の計画・評価・最終文はすべてRuleReasonerへ委譲されます。`model` でも実際のツール実行はC++です。
