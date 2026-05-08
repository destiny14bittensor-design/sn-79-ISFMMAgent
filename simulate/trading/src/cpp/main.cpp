/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#include <taosim/checkpoint/helpers.hpp>
#include <taosim/replay/ReplayDesc.hpp>
#include <taosim/replay/helpers.hpp>
#include <taosim/simulation/SimulationManager.hpp>

#include <fmt/format.h>

#include <CLI/CLI.hpp>
#include <cstdio>
#ifdef OVERRIDE_NEW_DELETE
#include <mimalloc-new-delete.h>
#endif

//-------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::setvbuf(stderr, nullptr, _IOLBF, 0);

    CLI::App app{"ExchangeSimulator v2.0"};

    auto initGroup = app.add_option_group("Init");

    fs::path configPath;
    auto optConfigPath = initGroup->add_option(
        "-f,--config-file", configPath, "Simulation config file")
        ->check(CLI::ExistingFile)
        ->transform([](auto&& p) { return fs::absolute(p); });

    fs::path baseDir;
    app.add_option("-d,--dir", baseDir, "Base directory in which to preserve the run artifacts")
        ->needs(optConfigPath)
        ->default_val(fs::current_path() / "logs")
        ->transform([](auto&& p) { return fs::absolute(p); });

    taosim::checkpoint::CheckpointToken ckptToken;
    initGroup->add_option(
        "-c,--load-checkpoint",
        ckptToken,
        fmt::format(
            "Checkpoint directory"
            " OR run directory containing checkpoint directories under '{}'"
            " OR special token (one of [{}], assumes run directories in 'logs/')",
            taosim::checkpoint::CheckpointManager::s_storeDirName,
            fmt::join(taosim::checkpoint::s_specialTokens, ", ")))
        ->transform(&taosim::checkpoint::postProcessToken);

    fs::path exchangeConfigPath;
    initGroup->add_option("-e,--exchange-mode", exchangeConfigPath, "Run in exchange mode")
        ->transform([](auto&& p) { return fs::absolute(p); });

    taosim::replay::ReplayDesc replayDesc;

    auto optReplayPath = initGroup->add_option(
        "-r,--replay-dir", replayDesc.dir, "Log directory to use in a replay context")
        ->check(CLI::ExistingDirectory)
        ->transform(&taosim::replay::helpers::cleanReplayPath);

    app.add_option("--book-id", replayDesc.bookId, "Book to replay")
        ->needs(optReplayPath);

    app.add_option(
        "--replaced-agents",
        replayDesc.replacedAgents,
        "Comma-separated list of agent base names which to replace during replay")
        ->delimiter(',')
        ->needs(optReplayPath);

    app.add_flag(
        "--adjust-limit-prices",
        replayDesc.adjustLimitPrices,
        "Adjust limit prices of passive agents using historical mid price data")
        ->needs(optReplayPath);

    initGroup->require_option(1);

    CLI11_PARSE(app, argc, argv);

    fmt::println("{}", app.get_description());

    if (!configPath.empty()) {
        auto mngr = taosim::simulation::SimulationManager::fromConfig(configPath, baseDir);
        mngr->runSimulations();
    }
    else if (!ckptToken.empty()) {
        auto mngr = taosim::simulation::SimulationManager::fromCheckpoint(ckptToken);
        mngr->runSimulations();
    }
    else if (!replayDesc.dir.empty()) {
        auto mngr = taosim::simulation::SimulationManager::fromReplay(replayDesc);
        if (replayDesc.bookId) {
            mngr->runReplay();
        } else {
            mngr->runReplayAdvanced();
        }
    }

    fmt::println(" - all simulations finished, exiting");

    return 0;
}

//-------------------------------------------------------------------------
