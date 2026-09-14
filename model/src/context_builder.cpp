#include "context_builder.h"
#include "tokenizer_character.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace {
void Append(TokenIds &nDestination, const TokenIds &c_nSource) {
    nDestination.insert(nDestination.end(), c_nSource.begin(), c_nSource.end());
}

TokenIds Utterance(const TokenConversation &c_tokTokenizer, int nSpeaker,
                   const std::string &c_strText) {
    // 発話は話者IDと終端IDを含む不可分な単位として組み立てる。
    TokenIds nResult = {nSpeaker};
    const auto nText = c_tokTokenizer.encode(c_strText);
    Append(nResult, nText);
    nResult.push_back(TokenConversation::UTTERANCE_END);
    return nResult;
}

TokenIds Turn(const TokenConversation &c_tokTokenizer,
              const ConversationContextTurn &c_trnTurn) {
    // QAの片側だけが残らないよう、完全な1往復を一つの選択単位にする。
    auto nResult = Utterance(c_tokTokenizer, TokenConversation::SPEAKER_A,
                             c_trnTurn.strQuestion);
    Append(nResult, Utterance(c_tokTokenizer, TokenConversation::SPEAKER_B,
                              c_trnTurn.strResponse));
    return nResult;
}

using Frequencies = std::unordered_map<int, int>;

Frequencies TokenFrequencies(const TokenIds &c_nIds) {
    Frequencies nResult;
    // 話者や区切りの一致を関連度と誤認しないよう、特殊IDはBM25から除外する。
    for (int nId : c_nIds)
        if (nId >= TokenConversation::SPECIAL_COUNT)
            ++nResult[nId];
    return nResult;
}

std::vector<double> Bm25(const TokenIds &c_nQuery,
                         const std::vector<TokenIds> &c_nDocuments) {
    std::vector<double> dblResult(c_nDocuments.size(), 0.0);
    if (c_nDocuments.empty())
        return dblResult;
    const auto nQuery = TokenFrequencies(c_nQuery);
    if (nQuery.empty())
        return dblResult;
    std::vector<Frequencies> nFrequencies;
    std::unordered_map<int, int> nDocumentFrequency;
    double dblAverageLength = 0.0;
    for (const auto &c_nDocument : c_nDocuments) {
        nFrequencies.push_back(TokenFrequencies(c_nDocument));
        dblAverageLength += c_nDocument.size();
        for (const auto &[nId, nCount] : nFrequencies.back()) {
            (void)nCount;
            ++nDocumentFrequency[nId];
        }
    }
    dblAverageLength /= c_nDocuments.size();
    if (dblAverageLength == 0.0)
        return dblResult;
    // 一般的なBM25の定数を固定し、同じ入力に対する選択を再現可能にする。
    constexpr double dblK1 = 1.2;
    constexpr double dblB = 0.75;
    for (std::size_t nDocument = 0; nDocument < c_nDocuments.size();
         ++nDocument) {
        for (const auto &[nId, nQueryCount] : nQuery) {
            const auto itrFrequency = nFrequencies[nDocument].find(nId);
            if (itrFrequency == nFrequencies[nDocument].end())
                continue;
            const double dblDf = nDocumentFrequency[nId];
            const double dblIdf = std::log(
                1.0 + (c_nDocuments.size() - dblDf + 0.5) / (dblDf + 0.5));
            const double dblTf = itrFrequency->second;
            const double dblDenominator =
                dblTf + dblK1 * (1.0 - dblB +
                                 dblB * c_nDocuments[nDocument].size() /
                                     dblAverageLength);
            dblResult[nDocument] +=
                nQueryCount * dblIdf * dblTf * (dblK1 + 1.0) / dblDenominator;
        }
    }
    return dblResult;
}

struct Item {
        std::size_t nOriginal;
        TokenIds nTokens;
        double dblScore = 0.0;
};

std::vector<Item>
UniqueTurns(const TokenConversation &c_tokTokenizer,
            const std::vector<ConversationContextTurn> &c_trnHistory) {
    std::set<std::string> strSeen;
    std::vector<Item> itmReverse;
    // 逆順に走査することで、完全一致するターンのうち最新のものだけを残す。
    for (std::size_t n = c_trnHistory.size(); n > 0; --n) {
        const auto &c_trnTurn = c_trnHistory[n - 1];
        const std::string strKey =
            c_trnTurn.strQuestion + '\0' + c_trnTurn.strResponse;
        if (strSeen.insert(strKey).second)
            itmReverse.push_back({n - 1, Turn(c_tokTokenizer, c_trnTurn)});
    }
    std::reverse(itmReverse.begin(), itmReverse.end());
    return itmReverse;
}

bool Take(std::vector<bool> &isSelected, const std::vector<Item> &c_itmItems,
          std::size_t nIndex, std::size_t &nUsed, std::size_t nLimit) {
    if (isSelected[nIndex] ||
        nUsed + c_itmItems[nIndex].nTokens.size() > nLimit)
        return false;
    isSelected[nIndex] = true;
    nUsed += c_itmItems[nIndex].nTokens.size();
    return true;
}

std::vector<std::size_t> SelectTurns(std::vector<Item> &itmItems,
                                     const TokenIds &c_nQuery,
                                     std::size_t nCapacity,
                                     std::size_t nRecentLimit) {
    std::vector<TokenIds> nDocuments;
    for (const auto &c_itmItem : itmItems)
        nDocuments.push_back(c_itmItem.nTokens);
    const auto dblScores = Bm25(c_nQuery, nDocuments);
    for (std::size_t n = 0; n < itmItems.size(); ++n)
        itmItems[n].dblScore = dblScores[n];

    std::vector<bool> isSelected(itmItems.size(), false);
    std::size_t nUsed = 0;
    // 最新の完全ターンは、直近60%枠を超えても総容量に収まれば先に確保する。
    if (!itmItems.empty())
        Take(isSelected, itmItems, itmItems.size() - 1, nUsed, nCapacity);
    // 続いて新しい順に直近60%枠を埋める。
    for (std::size_t n = itmItems.size(); n > 0; --n)
        Take(isSelected, itmItems, n - 1, nUsed, std::max(nRecentLimit, nUsed));

    std::vector<std::size_t> nRank(itmItems.size());
    for (std::size_t n = 0; n < nRank.size(); ++n)
        nRank[n] = n;
    std::stable_sort(
        nRank.begin(), nRank.end(), [&](std::size_t nA, std::size_t nB) {
            if (itmItems[nA].dblScore != itmItems[nB].dblScore)
                return itmItems[nA].dblScore > itmItems[nB].dblScore;
            return itmItems[nA].nOriginal > itmItems[nB].nOriginal;
        });
    // 残り40%を質問との関連度順に使い、同点なら新しいターンを優先する。
    for (std::size_t n : nRank)
        Take(isSelected, itmItems, n, nUsed, nCapacity);
    // 関連候補で使い切れなかった余剰を直近履歴へ移譲する。
    for (std::size_t n = itmItems.size(); n > 0; --n)
        Take(isSelected, itmItems, n - 1, nUsed, nCapacity);

    std::vector<std::size_t> nResult;
    for (std::size_t n = 0; n < itmItems.size(); ++n)
        if (isSelected[n])
            nResult.push_back(n);
    return nResult;
}
} // namespace

BuiltContext g_buildConversationContext(
    const TokenConversation &c_tokTokenizer,
    const std::vector<ConversationContextTurn> &c_trnHistory,
    const std::string &c_strQuestion, std::size_t nMaxTokens) {
    TokenCharacter tokValidate(c_strQuestion);
    const auto nQuestion =
        Utterance(c_tokTokenizer, TokenConversation::SPEAKER_A, c_strQuestion);
    // BEGIN・現在質問・回答話者IDは必須領域として履歴より先に予算を確保する。
    const std::size_t nBaseSize = 1 + nQuestion.size() + 1;
    if (nBaseSize > nMaxTokens)
        throw std::invalid_argument(
            "Current question exceeds the input token budget");

    auto itmTurns = UniqueTurns(c_tokTokenizer, c_trnHistory);
    const std::size_t nCapacity = nMaxTokens - nBaseSize;
    const auto nSelected =
        SelectTurns(itmTurns, c_tokTokenizer.encode(c_strQuestion), nCapacity,
                    nCapacity * 60 / 100);
    BuiltContext ctxResult;
    ctxResult.nTokens = {TokenConversation::BEGIN};
    for (std::size_t n : nSelected) {
        Append(ctxResult.nTokens, itmTurns[n].nTokens);
        ctxResult.nTurnIndices.push_back(itmTurns[n].nOriginal);
    }
    Append(ctxResult.nTokens, nQuestion);
    ctxResult.nTokens.push_back(TokenConversation::SPEAKER_B);
    return ctxResult;
}

std::vector<std::string>
g_splitDocumentSentences(const std::string &c_strDocument) {
    if (c_strDocument.empty())
        throw std::invalid_argument("Document must not be empty");
    TokenCharacter tokValidate(c_strDocument);
    std::vector<std::string> strResult;
    std::size_t nStart = 0;
    // byte位置をUTF-8文字単位で進め、マルチバイト文字の途中を境界にしない。
    for (std::size_t n = 0; n < c_strDocument.size();) {
        const unsigned char nByte = c_strDocument[n];
        const std::size_t nLength =
            nByte < 0x80 ? 1 : (nByte < 0xe0 ? 2 : (nByte < 0xf0 ? 3 : 4));
        const bool isAsciiBoundary =
            nLength == 1 &&
            (c_strDocument[n] == '.' || c_strDocument[n] == '!' ||
             c_strDocument[n] == '?' || c_strDocument[n] == '\n');
        const bool isJapaneseBoundary =
            nLength == 3 && (c_strDocument.compare(n, 3, "。") == 0 ||
                             c_strDocument.compare(n, 3, "！") == 0 ||
                             c_strDocument.compare(n, 3, "？") == 0);
        n += nLength;
        if (isAsciiBoundary || isJapaneseBoundary) {
            // 文末記号と改行は情報を失わないよう直前の文に含める。
            strResult.push_back(c_strDocument.substr(nStart, n - nStart));
            nStart = n;
        }
    }
    if (nStart < c_strDocument.size())
        strResult.push_back(c_strDocument.substr(nStart));
    return strResult;
}

BuiltContext g_buildDocumentContext(
    const TokenConversation &c_tokTokenizer, const std::string &c_strDocument,
    const std::vector<ConversationContextTurn> &c_trnHistory,
    const std::string &c_strQuestion, std::size_t nMaxTokens) {
    const auto ctxBase = g_buildConversationContext(c_tokTokenizer, {},
                                                    c_strQuestion, nMaxTokens);
    const std::size_t nCapacity = nMaxTokens - ctxBase.nTokens.size();
    const std::size_t nHistoryLimit = nCapacity * 60 / 100;
    auto itmTurns = UniqueTurns(c_tokTokenizer, c_trnHistory);

    std::size_t nUsed = 0;
    std::vector<std::size_t> nTurns;
    // 文書モードでも、まず残容量の60%を新しい完全QAへ割り当てる。
    for (std::size_t n = itmTurns.size(); n > 0; --n)
        if (nUsed + itmTurns[n - 1].nTokens.size() <= nHistoryLimit) {
            nUsed += itmTurns[n - 1].nTokens.size();
            nTurns.push_back(n - 1);
        }
    std::reverse(nTurns.begin(), nTurns.end());

    const auto strSentences = g_splitDocumentSentences(c_strDocument);
    std::set<std::string> strSeen;
    std::vector<Item> itmSentences;
    for (std::size_t n = 0; n < strSentences.size(); ++n)
        if (!strSentences[n].empty() && strSeen.insert(strSentences[n]).second)
            itmSentences.push_back({n, c_tokTokenizer.encode(strSentences[n])});
    std::vector<TokenIds> nDocuments;
    for (const auto &c_itmSentence : itmSentences)
        nDocuments.push_back(c_itmSentence.nTokens);
    const auto dblScores =
        Bm25(c_tokTokenizer.encode(c_strQuestion), nDocuments);
    for (std::size_t n = 0; n < itmSentences.size(); ++n)
        itmSentences[n].dblScore = dblScores[n];
    std::stable_sort(itmSentences.begin(), itmSentences.end(),
                     [](const Item &c_a, const Item &c_b) {
                         if (c_a.dblScore != c_b.dblScore)
                             return c_a.dblScore > c_b.dblScore;
                         return c_a.nOriginal < c_b.nOriginal;
                     });

    std::vector<Item> itmSelectedSentences;
    // 残り容量は質問とのBM25関連度が高い文書文から使用する。
    for (const auto &c_itmSentence : itmSentences)
        if (nUsed + c_itmSentence.nTokens.size() + 1 <= nCapacity) {
            nUsed += c_itmSentence.nTokens.size();
            itmSelectedSentences.push_back(c_itmSentence);
        }

    std::set<std::size_t> nTurnSet(nTurns.begin(), nTurns.end());
    // 文書文で余った容量を、未選択の直近履歴へ移譲する。
    for (std::size_t n = itmTurns.size(); n > 0; --n)
        if (!nTurnSet.count(n - 1) &&
            nUsed + itmTurns[n - 1].nTokens.size() <= nCapacity) {
            nUsed += itmTurns[n - 1].nTokens.size();
            nTurns.push_back(n - 1);
            nTurnSet.insert(n - 1);
        }
    // 選択は関連度・新しさで行うが、promptへ格納する前に元の時系列へ戻す。
    std::sort(nTurns.begin(), nTurns.end());
    std::sort(itmSelectedSentences.begin(), itmSelectedSentences.end(),
              [](const Item &c_a, const Item &c_b) {
                  return c_a.nOriginal < c_b.nOriginal;
              });

    BuiltContext ctxResult;
    ctxResult.nTokens = {TokenConversation::BEGIN};
    for (const auto &c_itmSentence : itmSelectedSentences) {
        Append(ctxResult.nTokens, c_itmSentence.nTokens);
        ctxResult.nDocumentSentenceIndices.push_back(c_itmSentence.nOriginal);
    }
    if (!itmSelectedSentences.empty())
        ctxResult.nTokens.push_back(TokenConversation::UTTERANCE_END);
    for (std::size_t n : nTurns) {
        Append(ctxResult.nTokens, itmTurns[n].nTokens);
        ctxResult.nTurnIndices.push_back(itmTurns[n].nOriginal);
    }
    Append(
        ctxResult.nTokens,
        Utterance(c_tokTokenizer, TokenConversation::SPEAKER_A, c_strQuestion));
    ctxResult.nTokens.push_back(TokenConversation::SPEAKER_B);
    return ctxResult;
}
