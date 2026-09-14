#pragma once
#include "ai/agent/components.h"

namespace ai::agent {
class Agent {
    public:
        Agent(std::shared_ptr<IInputParser>, std::shared_ptr<IRouter>,
              std::shared_ptr<IPlanner>, std::shared_ptr<IExecutor>,
              std::shared_ptr<IEvaluator>, std::shared_ptr<IAggregator>,
              std::shared_ptr<IMemoryManager>, std::shared_ptr<IReasoner>);
        AgentResponse process(std::string_view input);

    private:
        void finish_turn(const ParsedInput &, std::string_view response);
        std::shared_ptr<IInputParser> parser_;
        std::shared_ptr<IRouter> router_;
        std::shared_ptr<IPlanner> planner_;
        std::shared_ptr<IExecutor> executor_;
        std::shared_ptr<IEvaluator> evaluator_;
        std::shared_ptr<IAggregator> aggregator_;
        std::shared_ptr<IMemoryManager> memory_;
        std::shared_ptr<IReasoner> reasoner_;
        std::vector<ConversationTurn> recent_;
        std::string summary_;
};
} // namespace ai::agent
