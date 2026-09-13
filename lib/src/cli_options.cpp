#include "cli_options.h"
#include <stdexcept>

CliOptions::CliOptions(int nArgc, char **lpArgv, int nStart) {
    for (int nArg = nStart; nArg < nArgc;) {
        const std::string strName = lpArgv[nArg];
        if (strName.rfind("--", 0) != 0)
            throw std::invalid_argument("Expected option name: " + strName);
        const bool hasValue = nArg + 1 < nArgc &&
                              std::string(lpArgv[nArg + 1]).rfind("--", 0) != 0;
        if (!_strOptions
                 .emplace(strName, hasValue ? lpArgv[nArg + 1] : std::string())
                 .second) {
            throw std::invalid_argument("Missing or duplicate option");
        }
        nArg += hasValue ? 2 : 1;
    }
}

bool CliOptions::flag(const std::string &strName) {
    const auto itr = _strOptions.find(strName);
    if (itr == _strOptions.end())
        return false;
    if (!itr->second.empty())
        throw std::invalid_argument("Flag does not take a value: " + strName);
    _strOptions.erase(itr);
    return true;
}

bool CliOptions::contains(const std::string &strName) const {
    return _strOptions.count(strName) != 0;
}

std::string CliOptions::string(const std::string &strName,
                               const std::string &strDefault) {
    const auto itrValue = _strOptions.find(strName);
    if (itrValue == _strOptions.end()) {
        return strDefault;
    }
    const auto strValue = itrValue->second;
    if (strValue.empty())
        throw std::invalid_argument("Missing option value: " + strName);
    _strOptions.erase(itrValue);
    return strValue;
}

int CliOptions::integer(const std::string &strName, int nDefault) {
    if (!contains(strName)) {
        return nDefault;
    }
    const auto strValue = string(strName);
    std::size_t nEnd = 0;
    const int nValue = std::stoi(strValue, &nEnd);
    if (nEnd != strValue.size()) {
        throw std::invalid_argument("Invalid integer option: " + strName);
    }
    return nValue;
}

float CliOptions::real(const std::string &strName, float fDefault) {
    if (!contains(strName)) {
        return fDefault;
    }
    const auto strValue = string(strName);
    std::size_t nEnd = 0;
    const float fValue = std::stof(strValue, &nEnd);
    if (nEnd != strValue.size()) {
        throw std::invalid_argument("Invalid float option: " + strName);
    }
    return fValue;
}

void CliOptions::finish() const {
    if (!_strOptions.empty()) {
        throw std::invalid_argument("Unknown option: " +
                                    _strOptions.begin()->first);
    }
}
