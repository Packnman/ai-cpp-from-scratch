# Qwen Context Recognition 設計・実行手順

この文書は [`order_01.md`](order_01.md) に対する実装計画、現在の設計、実行手順をまとめる。
対象は Brain の Context Recognition のみで、Planner は既存の `RuleBasedPlanner`、既定動作は
既存の `RuleContextRecognizer` のままとする。Agent v3 bundle・学習コード・形式には変更を加えない。

## 実装計画と判断

1. 既存の `IContextRecognizer` / `ContextRecognitionResult` を再利用し、既定のRule経路を維持する。
2. `BrainSystem` に recognizer の単一所有DIを追加する。
3. 推論通信を `IContextModelBackend` に分離し、決定的FakeとQwen HTTP実装を用意する。
4. Qwenにはsystem/userを分離したpromptとJSON Schemaを渡す。
5. C++で再度厳格にparse・schema相当検証し、既存DTOへ変換する。
6. 通信・parse・validationの全失敗をRuleへfallbackし、理由を `ILogManager` へ記録する。
7. Fakeを使う通常テストと、稼働中の実モデルを要求する任意テストを分離する。

専用の設定基盤は既存コードに存在しないため、新しいグローバル設定方式は増やしていない。
選択と設定はDIおよび `QwenContextBackendConfig` で行う。

## 構成

```text
BrainSystem
  -> IContextRecognizer
       +-> RuleContextRecognizer                 (既定・fallback)
       `-> ModelContextRecognizer
             -> IContextModelBackend
                  +-> FakeContextModelBackend    (CI/単体テスト)
                  `-> QwenContextBackend         (HTTP adapter)
                        -> Qwen/Qwen3-4B-Instruct-2507 server
```

`ModelContextRecognizer` はHTTPを知らず、`QwenContextBackend` はBrainのGoalやConstraintを知らない。
したがって将来のbackendは `IContextModelBackend::infer()` を実装するだけで差し替えられる。

## JSON契約と検証

必須トップレベル項目は `intent`, `goal`, `conditions`, `constraints`, `confidence`。
未知のトップレベル・入れ子fieldも拒否する。現在許可する値は次の通り。

- intent: `statement`, `command`, `question`, `correction`
- goal.type: `AcquireObject`, `FindObject`, `MoveObject`, `AnswerQuestion`
- constraint scope: `global`, `goal`, `plan`, `action`, `resource`, `entity`
- constraint source: `safety`, `system_rule`, `human_explicit`, `environment`, `policy`
- constraint expression: `deny_all`, `deny_action_type`, `max_action_priority`
- confidence: 有限な `[0, 1]`

文字列長、応答byte数、配列数、属性数にも上限がある。属性値は文字列・数値・整数・真偽値だけを
許可し、オブジェクトや配列を実行命令として解釈しない。targetは既存DTOが数値IDを要求するため、
ASCII識別子を安定なFNV-1a値へ写像し、元の識別子もGoalの `target_name` 属性に残す。
この値は認識用の局所IDであり、外部DBの永続IDとはみなさない。

Qwen側のStructured Outputが成功しても信頼境界は越えない。C++側でJSON parse、必須field、型、
enum、範囲、個数、サイズ、既存 `valid_constraint()` を検証して初めてBrainへ渡す。

## Promptと失敗時動作

system promptはContext Recognition専用で、ユーザー文を別のuser messageに格納する。
ユーザー文中の命令はデータとして扱い、自然文、Markdown、tool/action出力を禁止する。
`temperature=0`, `seed=42`, `stream=false` が既定値である。実runtime差を含むbitwise一致は保証しない。

接続不能、timeout、HTTP error、空応答、OpenAI envelope不正、JSON不正、schema違反、未知enum、
backend例外はすべてRule認識へfallbackする。fallback自体は例外を隠さず、モデル側の失敗理由だけを
warningとしてloggerへ残す。HTTP処理は非blocking socketと単一deadlineを使い、応答サイズを制限する。
現在のadapterは `http://` のIPv4/localhost用で、TLSやretryは行わない。

## BrainSystemでの選択

既定は後方互換なRule認識である。

```cpp
ai::brain::BrainSystem ruleBrain;
```

Qwen利用時だけ明示的に構築する。

```cpp
using namespace ai::brain;

QwenContextBackendConfig qwen;
qwen.endpoint = "http://127.0.0.1:8000/v1/chat/completions";
qwen.model = "Qwen/Qwen3-4B-Instruct-2507";
qwen.timeout = Duration{5'000};
qwen.temperature = 0.0;

auto backend = std::make_unique<QwenContextBackend>(qwen);
auto recognizer = std::make_unique<ModelContextRecognizer>(std::move(backend));
BrainSystem modelBrain(std::move(recognizer));
```

`ContextRecognitionResult` のgoalはGoal Managerへ、conditionはWorld Stateへ、constraintはConstraint
Managerへ既存Semantic DTOを通して反映される。PlannerはRuleBasedのままである。

## Qwen serverの起動

モデル重みとPython runtimeはこのrepositoryやBrain executableに含めない。GPU容量を確認し、別環境で
vLLM serverを起動する。Qwen 2507系の公式例に合わせ、vLLM 0.9.0以降を使用する。

```sh
python3 -m venv .venv-qwen
. .venv-qwen/bin/activate
python -m pip install "vllm>=0.9.0"
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 --port 8000
```

モデル情報とlicenseは
[Hugging Faceの公式model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)、
serverの詳細は[Qwen公式ドキュメント](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)と
[vLLM OpenAI-compatible server](https://docs.vllm.ai/en/v0.18.0/serving/openai_compatible_server/)を参照する。

## テスト

通常CIはQwenを使わない。Fake正常応答、mapping、validation、timeout/接続/HTTP失敗、例外、fallback、
決定性、およびTextからRuleBasedPlannerまでを次で実行する。

```sh
./scripts/brain/run_tests.sh all
./scripts/brain/run_tests.sh sanitizer
```

実モデルテストはserver起動後だけ別buildで有効にする。

```sh
cmake -S . -B build/brain-qwen -G Ninja \
  -DBUILD_TESTING=ON \
  -DAI_CPP_BUILD_CUDA_LIB=OFF -DAI_CPP_BUILD_MODEL=OFF \
  -DAI_CPP_ENABLE_QWEN_INTEGRATION_TESTS=ON
cmake --build build/brain-qwen --target brain_qwen_integration_test
AI_CPP_QWEN_ENDPOINT=http://127.0.0.1:8000/v1/chat/completions \
AI_CPP_QWEN_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
ctest --test-dir build/brain-qwen -R '^brain_qwen_integration_test$' \
  --output-on-failure
```

実モデルテストはfallbackを成功扱いにせず、期待する `AcquireObject` が得られなければ失敗する。

## 制限

- goal/intent/constraint enumは現在のBrainが処理できる限定集合だけである。
- FNV-1a target IDには理論上collisionがあるため、永続entity registryが必要になれば置換する。
- adapterはHTTP/1.1の小規模なlocal server向けで、TLS、proxy、retry、認証更新は対象外である。
- 実モデルの精度・latency・GPUメモリはserver未起動環境では測れない。Fakeテストの合格と区別する。
- Qwen modelは外部取得物であり、本repositoryの配布物・学習済み重みではない。
