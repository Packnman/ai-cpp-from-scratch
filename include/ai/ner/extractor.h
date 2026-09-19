#pragma once

#include "ai/ner/entity.h"

#include <memory>

namespace ai::ner {

class IEntityExtractor {
    public:
        virtual ~IEntityExtractor() = default;
        virtual std::vector<EntityMention>
        extract(std::string_view text,
                std::string_view utterance_id = {}) const = 0;
};

class RuleEntityExtractor final : public IEntityExtractor {
    public:
        std::vector<EntityMention>
        extract(std::string_view text,
                std::string_view utterance_id = {}) const override;
};

class ModelEntityExtractor final : public IEntityExtractor {
    public:
        explicit ModelEntityExtractor(const std::string &bundle_path);
        ~ModelEntityExtractor() override;
        ModelEntityExtractor(ModelEntityExtractor &&) noexcept;
        ModelEntityExtractor &operator=(ModelEntityExtractor &&) noexcept;
        ModelEntityExtractor(const ModelEntityExtractor &) = delete;
        ModelEntityExtractor &operator=(const ModelEntityExtractor &) = delete;
        std::vector<EntityMention>
        extract(std::string_view text,
                std::string_view utterance_id = {}) const override;
        std::size_t parameter_bytes() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
};

class HybridEntityExtractor final : public IEntityExtractor {
    public:
        HybridEntityExtractor(std::shared_ptr<IEntityExtractor> rules,
                              std::shared_ptr<IEntityExtractor> model);
        std::vector<EntityMention>
        extract(std::string_view text,
                std::string_view utterance_id = {}) const override;

    private:
        std::shared_ptr<IEntityExtractor> _rules;
        std::shared_ptr<IEntityExtractor> _model;
};

} // namespace ai::ner
