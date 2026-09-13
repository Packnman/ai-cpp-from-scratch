#pragma once

#include <map>
#include <stdexcept>
#include <string>

class CliOptions {
    public:
        CliOptions(int nArgc, char **lpArgv, int nStart);

    private:
        std::map<std::string, std::string> _strOptions;

    public:
        bool contains(const std::string &strName) const;
        bool flag(const std::string &strName);
        std::string string(const std::string &strName,
                           const std::string &strDefault = {});
        int integer(const std::string &strName, int nDefault);
        float real(const std::string &strName, float fDefault);
        void finish() const;
};
