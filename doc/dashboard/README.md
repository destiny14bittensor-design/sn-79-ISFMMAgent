# **τaos** ☯ **‪ي‬n 79**<!-- omit in toc -->
# Dashboard Guide<!-- omit in toc -->

This document serves to provide details on the data displayed at the [τaos dashboard](https://taos.simulate.trading).

- [Validator Page](#validator-page)
  - [Validator Info](#validator-info)
  - [Simulation Config](#simulation-config)
  - [Fee Policy](#fee-policy)
    - [Fee Parameters](#fee-parameters)
    - [Maker-Taker Ratio Chart](#maker-taker-ratio-chart)
  - [Trade Data](#trade-data)
    - [Trade Price Plot](#trade-price-plot)
    - [Trade Quantity Plot](#trade-quantity-plot)
    - [Trades Table](#trades-table)
  - [Books Table](#books-table)
  - [Agents Table](#agents-table)
  - [Incentives Plot](#incentives-plot)
- [Book Page](#book-page)
  - [Book Info](#book-info)
  - [Trade Data](#trade-data-1)
    - [Trade Price Plot](#trade-price-plot-1)
    - [Trade Quantity Plot](#trade-quantity-plot-1)
    - [Trades Table](#trades-table-1)
  - [Orderbook Data](#orderbook-data)
    - [Best Levels Plot](#best-levels-plot)
    - [Depth Plots](#depth-plots)
  - [Agents Table](#agents-table-1)
  - [Dynamic Fee Rates Plot](#dynamic-fee-rates-plot)
- [Agent Page](#agent-page)
  - [Agent Info](#agent-info)
  - [Score Plot](#score-plot)
  - [Performance Plot](#performance-plot)
  - [Trades Table](#trades-table-2)
  - [Requests Plot](#requests-plot)
  - [Daily Volume Plot](#daily-volume-plot)
  - [Round-Trip Volume Plot](#round-trip-volume-plot)
  - [Realized PnL Plot](#realized-pnl-plot)
  - [Kappa-3 Plots](#kappa-3-plots)
  - [Unrealized Profit \& Loss Plots](#unrealized-profit--loss-plots)
  - [Fee Rates Plot](#fee-rates-plot)
  - [Balances Plots](#balances-plots)


## Validator Page
The main page at which visitors to the dashboard land is the Validators page.  This page displays overview data for the simulation hosted by each sn79 validator.

### Validator Info
![alt text](validator_info.png)

The top part of the dashboard page displays basic details of the selected validator.

The first row just indicates the UID, hotkey address and current simulation time.

The second row contains metagraph data for the validator - stake, vTrust, last update, emission and dividends.  See the [bittensor documentation](https://docs.learnbittensor.org/subnets/metagraph) for details on the meaning of these variables.

The third row displays the current resource usage of the validator hosting instance.

### Simulation Config
![alt text](validator_config.png)

The next section displays the most central simulation config parameters.

- **Books** - Number of order books in the simulation.

- **Duration** - Total simulation runtime in simulation time.

- **Time Unit** - Smallest time increment in the simulation.

- **Init Period** - Initial warm-up/stabilization period before miner agents are able to participate.

- **Publish Interval** - Frequency at which state updates are published to miners.

- **Init Price** - Starting price for assets at beginning of simulation.

- **Base Precision** - Decimal places for BASE quantities.

- **Quote Precision** - Decimal places for QUOTE quantities.

- **Price Precision** - Decimal places for price values.

- **Vol Precision** - Decimal places for volumes.

- **Capital Type** - Distribution method for initial capital allocation.

- **Miner Wealth** - Initial total value of assets allocated to each miner agent.

- **Max Orders** - Maximum number of concurrently open orders per agent on each book.

- **Init Agents** - Number of initialization agents present in the simulation.

- **Init Wealth** - Initial wealth allocated to each initialization agent.

- **HFT Agents** - Number of high-frequency trading agents.

- **HFT Wealth** - Total initial capital allocated to each HFT agent.

- **ST Agents** - Number of stylized trading agents.

- **ST Wealth** - Total initial capital allocated to each stylized trading agent.

- **FT Agents** - Number of fundamental trading agents.

- **FT Wealth** - Total initial capital allocated to each fundamental trader.

### Fee Policy
![alt text](validator_fee_policy.png)

The next section displays configuration parameters for the fees applied in the simulation, as well as a visualization of the central parameter determining the fees in the [Dynamic Incentive Structure (DIS)](https://simulate.trading/taos-im-dis-paper) fee mechanism.

#### Fee Parameters

- **Policy** - Fee structure model being applied (e.g., DIS - Dynamic Incentive Structure).

- **Target MTR** - Target maker-to-taker ratio, the desired proportion of maker trades to taker trades executed by miner agents.

- **Trade Window** - Time window (in seconds or time units) used for calculating MTR.

- **Base Maker Rate** - Baseline fee rate charged on maker trades.

- **Base Taker Rate** - Baseline fee rate charged on taker trades.

- **Max Rebate** - Maximum rebate percentage.

- **Max Fee** - Maximum fee percentage.

- **Fee Shape** - Parameter controlling the evolution of fee values.

- **Rebate Shape** - Parameter controlling the evolution of rebate values.

#### Maker-Taker Ratio Chart 

This provides a visual representation of the maker-to-taker ratio over time across all the simulated books.  The fees applicable to trades are dependent on the MTR: divergences from the target towards higher maker ratio incur fees on makers during trades while takers receive a rebate, and if there is a higher proportion of trades where miner agents are the taker then makers will receive rebate while takers pay a fee.

### Trade Data

![alt text](validator_trade_data.png)

The next row contains information and visualizations related to trading activity throughout the simulation.  

#### Trade Price Plot
The Trade Price plot displays the history of last traded prices for all books, along with the fundamental price value for each.  The fundamental value is an internal variable associated with each book which contributes to the evolution of the price series.

#### Trade Quantity Plot
The Trade Quantity plot illustrates the total quantity traded in the last interval at the time the trade price was published.


#### Trades Table
The Trades table shows details of the latest 25 trades on each book:

- **Time** - Simulation timestamp of when the trade was executed.

- **Book** - Order book identifier where the trade occurred.  Note that clicking on the book ID here will redirect to the Book details page for that orderbook.

- **Taker** - Agent ID of the taker (the agent whose order removed liquidity).  Note that clicking on the agent ID here will redirect to the Agent details page for that UID.

- **Maker** - Agent ID of the maker (the agent whose order provided liquidity).  Note that clicking on the agent ID here will redirect to the Agent details page for that UID.

- **Side** - Direction of the taker's order (BUY or SELL).

- **Price** - Execution price at which the trade occurred.

- **Quantity** - Amount of base asset traded in the transaction.

- **Maker Fee** - Fee charged to or rebate earned by the maker agent (negative value indicates rebate).

- **Taker Fee** - Fee charged to or rebate earned by the taker agent (negative value indicates rebate).

### Books Table

![alt text](validator_books.png)

The Books table displays the current state of the top 5 levels of each orderbook:

- **id** - Unique identifier for the order book.  Note that clicking on the book ID here will redirect to the Book details page for that orderbook.

- **bid_5, bid_4, bid_3, bid_2, bid_1** - Price levels for the top 5 bids (buy orders), with bid_1 being the best (highest) bid.

- **qty** - Quantity available at each corresponding bid price level.

- **ask_1, ask_2, ask_3, ask_4, ask_5** - Price levels for the top 5 asks (sell orders), with ask_1 being the best (lowest) ask.

- **qty** - Quantity available at each corresponding ask price level.

### Agents Table

![alt text](validator_agents.png)

The Agents table provides summary performance information for all miners in the subnet.  Values are all aggregated over all books.

- **Pos** - Ranking position of the agent based on score.

- **Agent** - Unique identifier for the agent (equal to miner UID).  Note that clicking on the agent ID here will redirect to the Agent details page for that UID.

- **24H Vol [QUOTE]** - The agent's total trading volume in QUOTE asset over the last 24 simulation hours for whichever book they traded in the least.

- **24H RT [QUOTE]** - The agent's total round-tripped volume in QUOTE asset over the last 24 simulation hours for whichever book they traded in the least.

- **Activity** - Activity score based on round-tripped trading volume executed in the latest assessment window.

- **Realized PnL** - Realized Profit and Loss from closed positions over the latest assessment window in QUOTE asset.

- **Median Kappa** - Median of Kappa-3 ratio values over all books.

- **Penalty** - Penalty factor applied to the agent's score due to inconsistent realized Sharpe performance in one or more books.

- **Kappa Score** - Final Kappa-based score calculated using activity-weighted median normalized realized Kappa-3 ratio with outlier penalty for latest assessment period.

- **Score** - Final composite score determining agent ranking; obtained as an exponential moving average of the Sharpe score over a period of observations.

- **ΔInv [QUOTE]** - Total change in miner inventory value since the start of simulation.

- **BASE Balance** - Current balance of BASE held by the agent.

- **BASE Loan** - Quantity of BASE borrowed by the agent via leveraged orders.

- **BASE Collat.** - Collateral posted in BASE for borrowing.

- **QUOTE Balance** - Current balance of QUOTE held by the agent.

- **QUOTE Loan** - Quantity of QUOTE borrowed by the agent via leveraged orders.

- **QUOTE Collat.** - Collateral posted in QUOTE for borrowing.

### Incentives Plot

![alt text](validator_incentives.png)

The incentives plot displays a history of the incentive value (as read from the metagraph) per UID.


## Book Page
This page displays overview data for a particular simulated book.  It is most easily accessed by clicking the links in either the Trades or Books table at the Validators page.

### Book Info

![alt text](book_info.png)

The first two rows display basic information about the selected book.

The unique identifier for the book and the current simulation time are presented in the first row.

The second row shows the latest traded price, the volume traded in latest interval, and the current best bid and ask price levels along with the quantity of open orders present and the best levels.

### Trade Data

![alt text](book_trade_data.png)

The next row contains information and visualizations related to trading activity in the selected book.

#### Trade Price Plot
The Trade Price plot displays the history of last traded prices for the selected book, along with the fundamental price value.

#### Trade Quantity Plot
The Trade Quantity plot illustrates the total quantity traded in the last interval at the time the trade price was published.

#### Trades Table
The Trades table shows details of the latest 25 trades on each book:

- **Time** - Simulation timestamp of when the trade was executed.

- **Taker** - Agent ID of the taker (the agent whose order removed liquidity).  Note that clicking on the agent ID here will redirect to the Agent details page for that UID.

- **Maker** - Agent ID of the maker (the agent whose order provided liquidity).  Note that clicking on the agent ID here will redirect to the Agent details page for that UID.

- **Side** - Direction of the taker's order (BUY or SELL).

- **Price** - Execution price at which the trade occurred.

- **Quantity** - Amount of base asset traded in the transaction.

- **Maker Fee** - Fee charged to or rebate earned by the maker agent (negative value indicates rebate).

- **Taker Fee** - Fee charged to or rebate earned by the taker agent (negative value indicates rebate).

### Orderbook Data

![alt text](book_orderbook_data.png)

The next row contains visualizations of the orderbook state.

#### Best Levels Plot

This plot displays the prices of the first 5 levels of the book over time.  This allows to inspect the spread size and available liquidity over time.

#### Depth Plots

These plots illustrate the cumulative quantity of orders open among the top 21 levels on each side of the book.  Each dot represents a book level at price indicated on the x-axis, with the y-axis showing how much volume exists up to that level.

### Agents Table

![alt text](book_agents.png)

The Agents table at the Book page displays statistics for agents calculated specifically on the selected book:

- **Agent** - Unique identifier for the agent.

- **24H Vol** - Agent's total trading volume in QUOTE asset over the last 24 simulation hours on the selected book.

- **24H Vol (Maker)** - Agent's maker trading volume (liquidity-providing trades) in QUOTE asset over the last 24 simulation hours on the selected book.

- **24H Vol (Taker)** - Agent's taker trading volume (liquidity-taking trades) in QUOTE asset over the last 24 simulation hours on the selected book.

- **24H RT [QUOTE]** - The agent's total round-tripped volume in QUOTE asset over the last 24 simulation hours for the selected book.

- **Activity** - Activity factor indicating agent's trading engagement level as a function of round-tripped volume.  This is multiplied onto the Kappa-3 score for each book to reward miners who achieve high risk-adjusted performance while also trading significant volume.

- **Realized PnL** - Realized Profit and Loss from closed positions over the latest assessment window in QUOTE asset for the selected book.

- **Kappa** - Kappa-3 ratio for the selected book.

- **Kappa Score** - Kappa3-based score calculated as activity-weighted and normalized Kappa-3 ratio for latest assessment period on the selected book.

- **ΔInv [QUOTE]** - Total change in miner inventory value since the start of simulation or registration of the UID (whichever is more recent).

- **Maker Fee** - Current maker fee rate at the time of observation for the agent on the selected book.

- **Taker Fee** - Current taker fee rate at the time of observation for the agent on the selected book.

- **Initial BASE** - Starting balance of BASE asset for this book at simulation start or registration.

- **BASE** - Current balance of BASE asset held by the agent on this book.

- **BASE Loan** - Amount of BASE asset borrowed by the agent on this book.

- **BASE Collat.** - Collateral posted in BASE asset for borrowing on this book.

- **Initial QUOTE** - Starting balance of QUOTE asset for this book at simulation start or registration.

- **QUOTE** - Current balance of QUOTE asset held by the agent on this book.

- **QUOTE Loan** - Amount of QUOTE asset borrowed by the agent on this book.

- **QUOTE Collat.** - Collateral posted in QUOTE asset for borrowing on this book.


### Dynamic Fee Rates Plot

![alt text](book_fee_rates.png)

This plot displays a history of the maker and taker fee rates applicable to miner agents in this book on the right y-axis, and the MTR for this book on the left y-axis.


## Agent Page

This page displays detailed statistics for a particular agent over all books for either a specific validator or all validators in the subnet. It is most easily accessed by clicking the links in either the Trades or Agents table at the Validators page.

### Agent Info

![alt text](agent_info.png)

The first two rows display basic information about the selected agent.

The unique identifier for the agent and the current simulation time are presented in the first row.

The second row shows the key metagraph statistics for the agent - consensus, emission, incentive and trust.

### Score Plot

![alt text](agent_score.png)

The topmost left plot displays the total score for the agent as assigned by the selected validator(s) on the right y-axis, and the incentive of the agent on the left y-axis.

### Performance Plot

![alt text](agent_performance.png)

The Performance plot illustrates the ranking in terms of score of the agent over time as assigned by the selected validator(s), as well as an average ranking taken over all selected validators.  The ranking indicates where among the subnet miners this agent places; higher ranking indicates outperformance of others.

### Trades Table

![alt text](agent_trades.png)

The Trades table shows details of the latest 5 trades on each book for the select agent:

- **Time** - Simulation timestamp of when the trade was executed.

- **Book** - Order book identifier where the trade occurred.  Note that clicking on the book ID here will redirect to the Book details page for that orderbook.

- **Side** - Direction of the taker's order (BUY or SELL).

- **Price** - Execution price at which the trade occurred.

- **Quantity** - Amount of base asset traded in the transaction.

- **Role** - The role of the selected agent in the trade, either maker (providing liquidity with passive order) or taker (taking liquidity with aggressive order).

- **Fee [QUOTE]** - Fee charged to or rebate earned in the trade by the selected agent.

- **Validator** - Hotkey of the validator in whose simulation the trade took place.

### Requests Plot

![alt text](agent_requests.png)

This plot illustrates statistics related to communication with validators; the left hand y-axis shows the average response time of the miner via the dotted blue line, while the right y-axis indicates counts of requests completed with different status:

- **Requests** - Total count of requests received in the observation window.
- **Success** - Responses successfully received by validators from the selected miner.
- **Failures** - Responses which failed to be received by the validator due to reasons other than timeout (e.g. network configuration issue).
- **Timeouts** - Responses not received by the validator due to exceeding the response timeout.
- **Rejections** - Responses not sent to validator due to blacklisting rules.

### Daily Volume Plot

![alt text](agent_volume.png)

This plot illustrates the average volumes over all selected validators which were traded by the selected agent over the last 24 simulation hours.  Volumes which the agent had traded in maker and taker role are illustrated as well as the total volume.  The agent's activity factor on average and for each book is also plotted; this is a function of the total traded volume and was applied to the unrealized Sharpe in obtaining the final unrealized score.  Although after release of version 0.2.0 the unrealized Sharpe is no longer used in obtaining miner scores, the total trading volume is important to monitor due to the trading volume cap indicated by the dashed red line - agents will be restricted from placing any new orders on books where the total trading volume exceeds this threshold.

### Round-Trip Volume Plot

![alt text](agent_roundtrip_volume.png)

This plot illustrates the average volumes over all selected validators which were round-tripped (either bought then sold or sold and then bought to open and close a position and thus realized a profit or loss) by the selected agent over the last 24 simulation hours.  The realized activity factor which is multiplied onto the realized Sharpe in obtaining the final realized Sharpe score for the agent is also plotted on average and for each book.

### Realized PnL Plot

![alt text](agent_realized_pnl.png)

This plot illustrates the realized PnL achieved by the agent in the most recent Sharpe assessment window over time.  Realized PnL is calculated from round tripped trades, using the price difference and fees/rebates to calculate the profit or loss realized through trading activity.

### Kappa-3 Plots

![alt text](agent_kappa.png)

The Kappa-3 plot displays the raw (unnormalized and unweighted) Kappa-3 ratio achieved by the agent on all books, as well as the median value.
The Kappa-3 Score plot displays the normalized and weighted Kappa-3 Score calculated for the agent for each book, as well as the median value.  The outlier penalty applied to the score is also plotted.

### Unrealized Profit & Loss Plots

![alt text](agent_pnl.png)

The Total Inventory Value Change plot illustrates the unrealized PnL (change in total inventory value) achieved by the agent since start of simulation or registration, for each book as well as in total.
The Unrealized PnL plot ilustrates the profit and loss (change in inventory value) achieved by the agent over the preceding Sharpe assessment window, for each book individually and in total.

### Fee Rates Plot

![alt text](agent_fees.png)

This plot displays a history of the fee rates paid by the agent in trades, for each book and on average.  The average and per-book MTR is also plotted.

### Balances Plots

![alt text](agent_balances.png)

The BASE and QUOTE balances and loans for the agent are plotted for each book as well as in total over all books.