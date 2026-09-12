#pragma once

#include <map>
#include <stdexcept>
#include <string>

class CliOptions
{
public:
    CliOptions( int nArgc, char** lpArgv, int nStart )
    {
        for( int nArg = nStart; nArg < nArgc; nArg += 2 )
        {
            if( nArg + 1 >= nArgc || !_strOptions.emplace( lpArgv[nArg], lpArgv[nArg + 1] ).second )
            {
                throw std::invalid_argument( "Missing or duplicate option" );
            }
        }
    }
    bool contains( const std::string& strName ) const { return _strOptions.count( strName ) != 0; }
    std::string string( const std::string& strName, const std::string& strDefault = {} )
    {
        const auto itrValue = _strOptions.find( strName );
        if( itrValue == _strOptions.end() ) return strDefault;
        const auto strValue = itrValue->second;
        _strOptions.erase( itrValue );
        return strValue;
    }
    int integer( const std::string& strName, int nDefault )
    {
        if( !contains( strName ) ) return nDefault;
        const auto strValue = string( strName );
        std::size_t nEnd = 0;
        const int nValue = std::stoi( strValue, &nEnd );
        if( nEnd != strValue.size() ) throw std::invalid_argument( "Invalid integer option: " + strName );
        return nValue;
    }
    float real( const std::string& strName, float fDefault )
    {
        if( !contains( strName ) ) return fDefault;
        const auto strValue = string( strName );
        std::size_t nEnd = 0;
        const float fValue = std::stof( strValue, &nEnd );
        if( nEnd != strValue.size() ) throw std::invalid_argument( "Invalid float option: " + strName );
        return fValue;
    }
    void finish() const
    {
        if( !_strOptions.empty() ) throw std::invalid_argument( "Unknown option: " + _strOptions.begin()->first );
    }
private:
    std::map<std::string, std::string> _strOptions;
};
