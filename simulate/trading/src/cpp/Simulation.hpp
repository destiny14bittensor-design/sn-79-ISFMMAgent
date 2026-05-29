/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "Agent.hpp"
#include "IConfigurable.hpp"
#include "IMessageable.hpp"
#include "LocalAgentManager.hpp"
#include <taosim/message/Message.hpp>
#include <taosim/message/MessageQueue.hpp>
#include <taosim/simulation/SharedResources.hpp>
#include <taosim/simulation/SimulationConfig.hpp>
#include <taosim/simulation/SimulationSignals.hpp>
#include <taosim/simulation/SimulationState.hpp>
#include <taosim/simulation/ISimulation.hpp>
#include <taosim/replay/ReplayDesc.hpp>
#include <common.hpp>

#include <fmt/core.h>

#include <barrier>
#include <map>
#include <tuple>
#include <valarray>

//-------------------------------------------------------------------------

class MultiBookExchangeAgent;

//-------------------------------------------------------------------------

class Simulation
    : public taosim::simulation::ISimulation,
      public IMessageable,
      public IConfigurable
{
public:
    Simulation() noexcept;
    Simulation(
        uint32_t blockIdx,
        uint32_t blockDim,
        const fs::path& logDir,
        const taosim::simulation::SharedResources* sharedResources,
        bool replayMode = false,
        taosim::replay::ReplayDesc = {});

    void dispatchMessage(
        Timestamp occurrence,
        Timestamp delay,
        const std::string& source,
        const std::string& target,
        const std::string& type,
        MessagePayload::Ptr payload = MessagePayload::create<EmptyPayload>()) const;

    template<typename... PrioArgs>
    requires std::constructible_from<taosim::message::PrioritizedMessage, Message::Ptr, PrioArgs...>
    void dispatchMessageWithPriority(
        Timestamp occurrence,
        Timestamp delay,
        const std::string& source,
        const std::string& target,
        const std::string& type,
        MessagePayload::Ptr payload,
        PrioArgs&&... prioArgs) const
    {
        queueMessageWithPriority(
            Message::create(occurrence, occurrence + delay, source, target, type, payload),
            std::forward<PrioArgs>(prioArgs)...);
    }

    void dispatchGenericMessage(
        Timestamp occurrence,
        Timestamp delay,
        const std::string& source,
        const std::string& target,
        const std::string& type,
        std::map<std::string, std::string> payload) const;

    void queueMessage(Message::Ptr msg) const;

    template<typename... Args>
    requires std::constructible_from<taosim::message::PrioritizedMessage, Args...>
    void queueMessageWithPriority(Args&&... args) const
    {
        m_messageQueue.push(taosim::message::PrioritizedMessage(std::forward<Args>(args)...));
    }

    template<typename Fn>
    void simulate(std::barrier<Fn>& barrier)
    {
        if (m_state == taosim::simulation::SimulationState::STOPPED) return;
        else if (m_state == taosim::simulation::SimulationState::INACTIVE) start();

        while (m_time.current < m_time.start + m_time.duration) {
            step();
            barrier.arrive_and_wait();
        }

        stop();
        fmt::println("end");
    }

    void simulate();

    [[nodiscard]] taosim::accounting::Account& account(const LocalAgentId& id) const noexcept;
    [[nodiscard]] auto&& agents(this auto&& self) noexcept { return self.m_localAgentManager->agents(); }
    [[nodiscard]] Timestamp currentTimestamp() const noexcept;
    [[nodiscard]] Timestamp duration() const noexcept;
    [[nodiscard]] MultiBookExchangeAgent* exchange() const noexcept { return m_exchange; }
    [[nodiscard]] DistributedProxyAgent* proxy() const noexcept { return m_proxy; }
    [[nodiscard]] taosim::simulation::SimulationSignals& signals() const noexcept;
    [[nodiscard]] std::mt19937& rng() const noexcept;
    [[nodiscard]] const taosim::simulation::SimulationConfig& config() const noexcept { return m_config2; }
    [[nodiscard]] const std::unique_ptr<LocalAgentManager>& localAgentManager() const noexcept { return m_localAgentManager; }
    [[nodiscard]] auto&& time(this auto&& self) noexcept { return self.m_time; }
    [[nodiscard]] uint32_t blockIdx() const noexcept { return m_blockIdx; }
    [[nodiscard]] auto&& logWindow(this auto&& self) noexcept { return self.m_logWindow; }
    [[nodiscard]] auto&& messageQueue(this auto&& self) noexcept { return self.m_messageQueue; }
    [[nodiscard]] bool replayMode() const noexcept { return m_replayMode; }
    [[nodiscard]] const auto& replayDesc() const noexcept { return m_replayDesc; }
    [[nodiscard]] auto&& timestampToMidPrice(this auto&& self) noexcept { return self.m_timestampToMidPrice; }
    [[nodiscard]] std::string_view id() const noexcept { return m_id; }
    [[nodiscard]] std::string_view configSv() const noexcept { return m_config; }
    [[nodiscard]] auto&& state(this auto&& self) noexcept { return self.m_state; }
    [[nodiscard]] const taosim::simulation::SharedResources* sharedResources() const noexcept { return m_sharedResources; }

    [[nodiscard]] bool shouldAdjustLimitPrice(Message::Ptr msg) const noexcept
    {
        return m_replayMode
            && m_replayDesc.adjustLimitPrices
            && !isReplacedAgent(msg->source)
            && !isInitAgent(msg->source);
    }

    [[nodiscard]] bool isReplacedAgent(const std::string& name) const noexcept;
    [[nodiscard]] bool isInitAgent(const std::string& name) const noexcept;

    [[nodiscard]] BookId bookIdCanon(BookId bookId) const noexcept
    {
        return m_blockIdx * m_blockDim + bookId;
    }
    
    virtual const fs::path& logDir() const noexcept override { return m_logDir; }
    virtual void receiveMessage(Message::Ptr msg) override;
    virtual void configure(const pugi::xml_node& node) override;

    template<typename... Args>
    void logDebug(fmt::format_string<Args...> fmt, Args&&... args) const noexcept
    {
        if (m_debug) {
            fmt::println(fmt, std::forward<Args>(args)...);
        }
    }
    
    void setDebug(bool flag) noexcept { m_debug = flag; }
    [[nodiscard]] bool debug() const noexcept { return m_debug; }

    template<typename... Args>
    void logError(fmt::format_string<Args...> fmt, Args&&... args) const noexcept
    {
        if (m_error) {
            fmt::println(fmt, std::forward<Args>(args)...);
        }
    }

    void setError(bool flag) noexcept { m_error = flag; }
    [[nodiscard]] bool error() const noexcept { return m_error; }

    // GBM trajectory cache shared across agents that sample the same path at
    // configure-time (e.g. all StylizedTraderAgent instances within a block use
    // identical S0/mu/sigma/N and a seed derived only from bookId).  Single
    // threaded by design — configure() runs serially per block.
    const std::valarray<double>& getOrComputeGbmPath(
        uint64_t seed, double S0, double mu, double sigma, uint32_t N);

    void step();

    [[nodiscard]] static std::unique_ptr<Simulation> fromXML(pugi::xml_node node);

private:
    void configureAgents(pugi::xml_node node);
    void configureLogging(pugi::xml_node node);
    void deliverMessage(Message::Ptr msg);
    void start();
    void stop();

    void updateTime(Timestamp newTime)
    {
        if (newTime == m_time.current) [[unlikely]] return;
        Timestamp oldTime = std::exchange(m_time.current, newTime);
        m_signals.time({.begin = oldTime + 1, .end = newTime});
    }

    mutable taosim::message::MessageQueue m_messageQueue;
    taosim::simulation::SimulationState m_state{taosim::simulation::SimulationState::INACTIVE};
    struct { Timestamp start, duration, step, current; } m_time;
    mutable taosim::simulation::SimulationSignals m_signals;
    std::unique_ptr<LocalAgentManager> m_localAgentManager;
    MultiBookExchangeAgent* m_exchange{};
    DistributedProxyAgent* m_proxy;
    mutable std::mt19937 m_rng;
    std::string m_id;
    std::string m_config;
    bool m_debug = false;
    bool m_error = false;
    fs::path m_logDir;
    taosim::simulation::SimulationConfig m_config2;
    uint32_t m_blockIdx{};
    uint32_t m_blockDim{};
    fs::path m_baseLogDir;
    Timestamp m_logWindow{};
    bool m_replayMode{};
    taosim::replay::ReplayDesc m_replayDesc;
    const taosim::simulation::SharedResources* m_sharedResources{};
    std::unordered_map<Timestamp, taosim::decimal_t> m_timestampToMidPrice;
    // (seed, S0, mu, sigma, N) -> generated GBM trajectory.  std::map so
    // references into stored valarrays remain valid as entries are inserted.
    std::map<std::tuple<uint64_t, double, double, double, uint32_t>,
             std::valarray<double>> m_gbmPathCache;

    friend class LocalAgentManager;
    friend class MultiBookExchangeAgent;
};

//-------------------------------------------------------------------------
