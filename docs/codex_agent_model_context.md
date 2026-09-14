# Project Context

## Goal

小型Transformerを中心に、以下を同時に実現するAI Agentを構築する。

- 自然な日本語会話
- 長い入力・会話履歴の圧縮
- タスク分解とプランニング
- ツールや外部機能の逐次実行
- 実行結果の評価と再計画
- 長期記憶・短期記憶の管理
- 将来的なロボット制御システムとの統合

巨大なコンテキストを毎回Transformerへ入力する方式ではなく、

```text
Small Model + Short Context + External Memory + Iterative Reasoning
```

を基本設計とする。

モデルそのものに全履歴や全知識を保持させず、必要な情報だけをその都度取得して入力する。

---

# High-Level Architecture

全体構成は以下とする。

```text
User Input
    |
    v
Input Parser
    |
    v
Router
    |
    +--------------------+
    |                    |
    v                    v
Simple Chat          Complex Task
    |                    |
    v                    v
Language Model        Retriever
                         |
                         v
                      Planner
                         |
                         v
                      Executor
                         |
                         v
                      Evaluator
                         |
               +---------+---------+
               |                   |
             Success             Failure
               |                   |
               v                   |
           Aggregator <--------- Replanner
               |
               v
          Response Generator
               |
               v
          Memory Manager
               |
               v
          Final Response
```

---

# Core Design Principles

## 1. Transformerに全処理を担当させない

Transformerは高レベル判断に使用する。

Transformerが担当するもの:

- 会話生成
- 意図理解
- タスク分解
- プラン生成
- ツール選択
- 実行結果の意味理解
- 要約
- Memoryに保存する内容の判断

Transformerに担当させないもの:

- モータ制御
- PID / MPC
- 数値計算
- 画像前処理
- DB処理
- ファイルIO
- 通信処理
- 安全制御
- 決定論的に実装可能な処理

これらは通常のC++コードや専用モデルへ任せる。

---

# 2. Input Parser

自然言語入力をそのまま長時間保持せず、意味を構造化する。

単なるkeyword extractionにはしない。

例:

Input:

```text
赤いボールを取って。ただし人が近くにいたら動かないで。
```

Internal representation:

```json
{
  "intent": "object_pickup",
  "goal": "pickup",
  "target": {
    "type": "ball",
    "color": "red"
  },
  "constraints": [
    {
      "condition": "human_nearby",
      "action": "stop"
    }
  ]
}
```

否定、条件、因果、対象、数量などの意味を失わないこと。

---

# 3. Router

入力を以下に分類する。

```cpp
enum class RequestType
{
    SimpleConversation,
    QuestionAnswer,
    MemoryRecall,
    ComplexReasoning,
    ToolTask,
    RobotTask
};
```

Routerは可能な限り軽量にする。

単純な会話ではPlannerを起動しない。

例:

```text
「今日は疲れた」
    -> SimpleConversation

「TransformerのAttentionを説明して」
    -> QuestionAnswer

「昨日話していた設計を教えて」
    -> MemoryRecall

「この問題を3つに分解して解いて」
    -> ComplexReasoning

「ファイルを開いて解析して」
    -> ToolTask

「ボールを探して近づいて」
    -> RobotTask
```

---

# 4. Memory Architecture

Memoryは最低3層に分離する。

## Working Memory

現在のタスクでのみ使用する短期状態。

例:

```cpp
struct WorkingMemory
{
    Goal goal;
    std::vector<Task> tasks;
    std::vector<TaskResult> results;
    std::vector<Constraint> constraints;
};
```

Transformerへ渡す量は小さく保つ。

---

## Conversation Summary

長い会話履歴を圧縮した状態。

保持するもの:

- 現在の話題
- 過去の重要な結論
- 未解決事項
- ユーザーの直近の意図
- 会話上必要な前提

全文保存ではなくsummaryを使用する。

ただし直近数ターンはraw textのまま保持する。

推奨Context:

```text
Recent Conversation
+
Conversation Summary
+
Retrieved Memory
+
Current Input
```

---

## Long-Term Memory

長期的に再利用する情報。

分類:

```text
Semantic Memory
    知識・設定・事実

Episodic Memory
    過去に起きたイベント

User / Project Memory
    継続プロジェクトやユーザー指定事項
```

Memoryは毎回全部入力しない。

```text
retrieve(query)
```

により関連情報のみ取得する。

---

# 5. Planner

PlannerはGoalをTaskへ分割する。

自然言語ではなく可能な限り構造化データを出力する。

例:

```json
{
  "goal": "locate_and_approach_ball",
  "tasks": [
    {
      "id": 0,
      "type": "tool",
      "tool": "camera.detect",
      "args": {
        "class": "ball"
      }
    },
    {
      "id": 1,
      "type": "reasoning",
      "operation": "calculate_target_direction",
      "depends_on": [0]
    },
    {
      "id": 2,
      "type": "tool",
      "tool": "robot.move",
      "depends_on": [1]
    }
  ]
}
```

Taskは依存関係を持てるようにする。

将来的には単純なvectorではなくTask Graphへ拡張できる設計にする。

---

# 6. Executor

ExecutorはTaskを一件ずつ処理する。

Executor自身はなるべく決定論的にする。

interface例:

```cpp
class ITool
{
public:
    virtual ToolResult execute(
        const ToolArguments& args
    ) = 0;

    virtual ~ITool() = default;
};
```

ToolRegistryを用意する。

例:

```text
camera.detect
robot.move
database.search
memory.retrieve
calculator.calculate
file.read
```

Plannerは直接C++関数を呼ばず、Tool名とargumentsを生成する。

---

# 7. Evaluator

Executorの結果をそのまま次へ渡さない。

必ずEvaluatorを通す。

Evaluator output例:

```cpp
enum class EvaluationStatus
{
    Success,
    Retry,
    Replan,
    Failed
};

struct EvaluationResult
{
    EvaluationStatus status;
    std::string reason;
    std::optional<StateUpdate> update;
};
```

例:

```text
camera.detect("ball")

Result:
    object_found = false

Evaluator:
    status = Replan
    reason = "Target ball was not detected"

Planner:
    camera direction change
    -> detect again
```

基本ループ:

```text
Plan
-> Execute
-> Evaluate
-> Replan if required
```

---

# 8. Aggregator

複数TaskのResultを統合する。

Aggregatorの目的は、

- 必要な情報だけ残す
- 重複削除
- 矛盾検出
- 最終回答用Context生成

を行うこと。

Executorの生ログをすべてLLMへ渡さない。

例:

```text
Raw results:
Task 1: 2000 tokens
Task 2: 1500 tokens
Task 3: 1000 tokens
```

Aggregator:

```json
{
  "target_found": true,
  "target_direction": "right",
  "distance": 1.4,
  "execution_status": "success"
}
```

のように圧縮する。

---

# 9. Conversation Mode

通常会話では以下の軽量経路を使用する。

```text
Current Input
+
Recent Conversation
+
Conversation Summary
+
Relevant Long-Term Memory
    |
    v
Language Model
    |
    v
Response
```

毎回Planner / Executorを使用しない。

---

# 10. Model Mode

当初は1つのDecoder-only Transformerを共有する。

用途はSpecial Tokenで切り替える。

Special Tokens:

```text
<CHAT>
<PARSE>
<PLAN>
<EVALUATE>
<SUMMARIZE>
<MEMORY_WRITE>
<MEMORY_QUERY>
<FINAL>
<TOOL>
```

例:

```text
<CHAT>
今日は疲れた

<PLAN>
Goal: 人工衛星の軌道変更方法を調べる

<MEMORY_WRITE>
以下の会話から長期保存すべき情報を抽出する
```

将来的に各機能で性能不足が発生した場合のみモデルを分離する。

---

# 11. Model Responsibilities

Shared Transformer:

```text
Input:
    token sequence

Output:
    token sequence
```

用途ごとにprompt / special tokenで挙動を切り替える。

モデル内部で特殊なAgentロジックをハードコードしない。

Agentのstate machineはC++側に持つ。

重要:

```text
LLM = Reasoning Component

Agent = Application Architecture
```

と分離する。

---

# 12. Suggested C++ Structure

```text
src/
    agent/
        Agent.cpp
        Agent.h

        Router.cpp
        Router.h

        Planner.cpp
        Planner.h

        Executor.cpp
        Executor.h

        Evaluator.cpp
        Evaluator.h

        Aggregator.cpp
        Aggregator.h

    memory/
        WorkingMemory.cpp
        WorkingMemory.h

        ConversationMemory.cpp
        ConversationMemory.h

        LongTermMemory.cpp
        LongTermMemory.h

        MemoryRetriever.cpp
        MemoryRetriever.h

    model/
        LanguageModel.cpp
        LanguageModel.h

        ModelPrompt.cpp
        ModelPrompt.h

    tools/
        Tool.cpp
        Tool.h

        ToolRegistry.cpp
        ToolRegistry.h

    state/
        AgentState.cpp
        AgentState.h

    tokenizer/
        Tokenizer.cpp
        Tokenizer.h
```

---

# 13. Main Agent State

```cpp
struct AgentState
{
    RequestType requestType;

    ParsedInput input;

    WorkingMemory workingMemory;

    ConversationState conversation;

    std::vector<RetrievedMemory> retrievedMemories;

    std::optional<Plan> currentPlan;

    std::vector<TaskResult> taskResults;
};
```

AgentStateはLLM内部には持たせず、C++側で管理する。

---

# 14. Agent Execution

基本フロー:

```cpp
AgentResponse Agent::process(
    const std::string& input
)
{
    ParsedInput parsed = parser.parse(input);

    RequestType type = router.route(parsed);

    if (type == RequestType::SimpleConversation)
    {
        return conversation.respond(
            parsed,
            memory.retrieve(parsed)
        );
    }

    auto memories = memory.retrieve(parsed);

    Plan plan = planner.createPlan(
        parsed,
        memories
    );

    while (!plan.finished())
    {
        Task task = plan.nextTask();

        TaskResult result =
            executor.execute(task);

        EvaluationResult evaluation =
            evaluator.evaluate(
                task,
                result
            );

        if (evaluation.status ==
            EvaluationStatus::Replan)
        {
            plan = planner.replan(
                parsed,
                plan,
                result,
                evaluation
            );
        }
        else
        {
            plan.apply(result);
        }
    }

    auto aggregate =
        aggregator.aggregate(
            plan.results()
        );

    auto response =
        languageModel.generateResponse(
            parsed,
            aggregate
        );

    memory.update(
        parsed,
        aggregate,
        response
    );

    return response;
}
```

---

# 15. Context Management

モデルへの最大入力token数を固定する。

ContextBuilderが入力を生成する。

優先順位:

1. Current Input
2. Current Goal
3. Current Task
4. Critical Constraints
5. Recent Conversation
6. Retrieved Memory
7. Conversation Summary
8. Previous Task Results

token budgetを超えた場合、下位priorityから削減する。

ただしConstraintは削除禁止。

否定条件、安全条件、数値制限などは常に保持する。

---

# 16. Memory Compression

Memoryへ保存する際に全文を保存しない。

MemoryCandidate:

```cpp
struct MemoryCandidate
{
    std::string content;

    float importance;
    float confidence;

    MemoryType type;
};
```

importanceが低い一時的情報は保存しない。

例:

```text
保存:
「ユーザーのプロジェクトではROS2を使用しない」

保存しない:
「今ちょっと眠い」
```

ただし会話継続に必要な短期情報はConversation Memoryに一時保存可能。

---

# 17. Training Strategy

Phase 1:

```text
Tokenizer training
```

Phase 2:

```text
Japanese pretraining using Jawiki etc.
```

Phase 3:

```text
Instruction tuning
```

学習対象:

```text
<CHAT>
自然な会話

<PARSE>
自然文 -> structured state

<PLAN>
Goal -> task plan

<EVALUATE>
Execution result -> evaluation

<SUMMARIZE>
Conversation -> compressed summary

<MEMORY_WRITE>
Conversation -> important memory

<FINAL>
Task results -> final natural language response
```

同じDecoder TransformerへMulti-task SFTする。

---

# 18. Important Constraints

以下を守る。

- 全会話履歴を毎回モデルへ入力しない
- Keywordだけに圧縮しない
- 意味・否定・条件を保持する
- Planner outputは構造化する
- Tool executionはLLMから分離する
- Agent stateはC++側で保持する
- Memory検索後、関連情報のみContextへ入れる
- 単純な会話でPlanning loopを回さない
- Execution resultには必ずevaluation stepを入れる
- Failure時にReplanning可能にする
- Safety constraintはmemory圧縮時にも削除しない
- 将来Robot / Vision / Controlへ拡張可能なinterfaceにする

---

# Final Objective

最終的な設計思想は、

```text
Large Context + Large Model
```

ではなく、

```text
Small Model
+
Short Context
+
Structured State
+
External Memory
+
Retrieval
+
Planning
+
Execution
+
Evaluation
+
Iterative Processing
```

とする。

モデル単体の性能だけに依存せず、Agent architecture全体で知能を構成する。

---

# MVP Implementation Profile

初回実装はCPUで完結するrule backendとし、以下を固定する。

- 公開namespaceは`ai::agent`、主APIは`Agent::process(std::string_view)`とする。
- Task graphは最大32件とし、重複ID、欠落依存、cycleを実行前に拒否する。
- 同一Toolのretryは2回、replanは3回を上限とする。
- Long-Term MemoryはSQLite schema version 1、WAL、5秒busy timeoutを使用する。
- 日本語検索はFTS5 trigram、3 code point未満はparameter bindingした完全一致・部分一致を使用する。
- raw conversationは直近4 turnを保持し、それ以前をReasonerでsummary化する。
- context budgetは既定1024とし、完全な項目単位で採否を決める。critical領域自体の超過はerrorとする。
- `calculator.calculate`、`file.read`、`memory.retrieve`だけを組み込みToolとして提供する。
- Transformer学習、実model backend、旧bundle互換、実Robot/Vision、外部通信はこのMVPに含めない。

SQLite 3.50.4とnlohmann/json 3.12.0はrepositoryへ固定して同梱し、通常build時のnetwork取得を禁止する。
既存`lib/`は変更せず、`AI_CPP_BUILD_CUDA_LIB=ON`の場合だけ`ai_cpp` targetとしてbuildする。
