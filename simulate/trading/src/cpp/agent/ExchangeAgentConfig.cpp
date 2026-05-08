/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include "ExchangeAgentConfig.hpp"

#include <fmt/format.h>

//-------------------------------------------------------------------------

namespace taosim::config
{

//-------------------------------------------------------------------------

void ExchangeAgentConfig::configure(pugi::xml_node node)
{
    try {
        setPriceIncrement(node);
        setVolumeIncrement(node);
        setBaseDecimals(node);
        setQuoteDecimals(node);
    }
    catch (...) {
        handleException();
    }
}

//-------------------------------------------------------------------------

const ExchangeAgentConfig::Parameters& ExchangeAgentConfig::parameters() const noexcept
{
    return m_parameters;
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::setPriceIncrement(pugi::xml_node node)
{
    static constexpr const char* attrName = "priceDecimals";

    pugi::xml_attribute attr = node.attribute(attrName);
    if (attr.empty()) return;

    if (uint32_t value = attr.as_uint(); value < Parameters::kMinimumPriceIncrementDecimals) {
        throw ExchangeAgentConfigException{fmt::format(
            "Value of attribute '{}' should be at least {}, was {}",
            attrName,
            Parameters::kMinimumPriceIncrementDecimals,
            value)};
    } else {
        m_parameters.priceIncrementDecimals = value;
    }
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::setVolumeIncrement(pugi::xml_node node)
{
    static constexpr const char* attrName = "volumeDecimals";

    pugi::xml_attribute attr = node.attribute(attrName);
    if (attr.empty()) return;

    if (uint32_t value = attr.as_uint(); value < Parameters::kMinimumVolumeIncrementDecimals) {
        throw ExchangeAgentConfigException{fmt::format(
            "Value of attribute '{}' should be at least {}, was {}",
            attrName,
            Parameters::kMinimumVolumeIncrementDecimals,
            value)};
    } else {
        m_parameters.volumeIncrementDecimals = value;
    }
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::setBaseDecimals(pugi::xml_node node)
{
    static constexpr const char* attrName = "baseDecimals";

    pugi::xml_attribute attr = node.attribute(attrName);
    if (attr.empty()) return;

    if (uint32_t value = attr.as_uint(); value < Parameters::kMinimumVolumeIncrementDecimals) {
        throw ExchangeAgentConfigException{fmt::format(
            "Value of attribute '{}' should be at least {}, was {}",
            attrName,
            Parameters::kMinimumVolumeIncrementDecimals,
            value)};
    } else {
        m_parameters.baseIncrementDecimals = value;
    }
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::setQuoteDecimals(pugi::xml_node node)
{
    static constexpr const char* attrName = "quoteDecimals";

    pugi::xml_attribute attr = node.attribute(attrName);
    if (attr.empty()) return;

    if (uint32_t value = attr.as_uint(); value < Parameters::kMinimumPriceIncrementDecimals) {
        throw ExchangeAgentConfigException{fmt::format(
            "Value of attribute '{}' should be at least {}, was {}",
            attrName,
            Parameters::kMinimumPriceIncrementDecimals,
            value)};
    } else {
        m_parameters.quoteIncrementDecimals = value;
    }
}

//-------------------------------------------------------------------------

void ExchangeAgentConfig::handleException()
{
    try {
        throw;
    }
    catch (const ExchangeAgentConfigException& exc) {
        fmt::println("{}", exc.what());
        throw;
    }
}

//-------------------------------------------------------------------------

ExchangeAgentConfigException::ExchangeAgentConfigException(std::string msg) noexcept
    : m_msg{std::move(msg)}
{}

//-------------------------------------------------------------------------

const char* ExchangeAgentConfigException::what() const noexcept
{
    return m_msg.c_str();
}

//-------------------------------------------------------------------------

}  // namespace taosim::config

//-------------------------------------------------------------------------
