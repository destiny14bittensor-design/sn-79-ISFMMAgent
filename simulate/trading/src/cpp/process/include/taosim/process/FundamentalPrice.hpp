/*
 * SPDX-FileCopyrightText: 2025 Rayleigh Research <to@rayleigh.re>
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <taosim/simulation/ISimulation.hpp>
#include <taosim/process/Process.hpp>
#include "common.hpp"

#include <Eigen/Dense>
#include <pugixml.hpp>

//-------------------------------------------------------------------------

namespace taosim::process
{

//-------------------------------------------------------------------------

struct FundamentalPriceDesc
{
    simulation::ISimulation* simulation;
    uint64_t bookId;
    uint64_t seedInterval;
    double X0;
    double mu;
    double sigma;
    double dt;
    double lambda;
    double muJump;
    double sigmaJump;
    double hurst{0.5};
    double epsilon{0.0};
    Timestamp gracePeriod{};   // suppress seed-update warnings before this t
    ProcessDesc proc;
    const Eigen::MatrixXd* L{};
};

struct FundamentalPriceState
{
    double dJ{};
    double t{};
    double W{};
    Eigen::VectorXd X;
    Eigen::VectorXd V;
    double BH{};
    int lastCount{};
    uint64_t lastSeed{};
    Timestamp lastSeedTime{};
    double value{};
};

//-------------------------------------------------------------------------

class FundamentalPrice : public Process
{
public:
    FundamentalPrice() noexcept = default;
    FundamentalPrice(const FundamentalPriceDesc& desc) noexcept;

    [[nodiscard]] auto&& state(this auto&& self) noexcept { return self.m_state; }
    [[nodiscard]] auto&& rng(this auto&& self) noexcept { return self.m_rng; }

    virtual void update(Timestamp timestamp) override;
    virtual double value() const override { return m_state.value; }

    [[nodiscard]] static std::unique_ptr<FundamentalPrice> fromXML(
        simulation::ISimulation* simulation,
        pugi::xml_node node,
        uint64_t bookId,
        double X0,
        const Eigen::MatrixXd* L);

private:
    void cholesky_step(int64_t i);

    simulation::ISimulation* m_simulation;
    std::mt19937* m_rng;
    uint64_t m_bookId;
    uint64_t m_seedInterval;
    Timestamp m_gracePeriod{};
    std::string m_seedfile;
    double m_X0, m_mu, m_sigma, m_dt;
    FundamentalPriceState m_state;
    const Eigen::MatrixXd* m_L;
    std::normal_distribution<double> m_gaussian;
    double m_epsilon;
    double m_hurst;
    std::normal_distribution<double> m_fractionalGaussian;
    std::normal_distribution<double> m_jump;
    std::poisson_distribution<int> m_poisson;
};

//-------------------------------------------------------------------------

}  // namespace taosim::process

//-------------------------------------------------------------------------