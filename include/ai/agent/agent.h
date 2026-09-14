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
        std::shared_ptr<IInputParser> _parser;
        std::shared_ptr<IRouter> _router;
        std::shared_ptr<IPlanner> _planner;
        std::shared_ptr<IExecutor> _executor;
        std::shared_ptr<IEvaluator> _evaluator;
        std::shared_ptr<IAggregator> _aggregator;
        std::shared_ptr<IMemoryManager> _memory;
        std::shared_ptr<IReasoner> _reasoner;
        std::vector<ConversationTurn> _recent;
        std::string _summary;
};
} // namespace ai::agent
