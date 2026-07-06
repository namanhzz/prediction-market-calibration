# Domain Classification Rules

This document describes the domain classification systems used for Kalshi and Polymarket markets.

## 1. Kalshi: Prefix-Based Taxonomy

Kalshi markets are classified by extracting the alphanumeric prefix from the `event_ticker` field using:

```sql
regexp_extract(event_ticker, '^([A-Z0-9]+)', 1)
```

For example, `KXNFLGAME-25-AFC-CHI-DET` yields the prefix `KXNFLGAME`.

The prefix is then looked up in `SUBCATEGORY_PATTERNS` (571 tuples), each mapping a prefix pattern to a `(group, category, subcategory)` triple. Lookup is case-insensitive substring matching; more specific patterns are listed first.

### `get_group()` and `get_hierarchy()`

```python
def get_hierarchy(category: str) -> tuple:
    cat_upper = category.upper()
    for pattern, group, cat, subcat in SUBCATEGORY_PATTERNS:
        if pattern in cat_upper:
            return (group, cat, subcat)
    return ("Other", "Other", category)

def get_group(category: str) -> str:
    return get_hierarchy(category)[0]
```

### Domain Composition

Only markets in the following 6 domains are included in the analysis; all others are classified as "Other" and excluded.

| Domain | Principal Ticker Prefixes | Description |
|--------|--------------------------|-------------|
| Sports | `NFLGAME`, `NBAGAME`, `MLBGAME`, `NCAAFGAME`, `NCAAMBGAME`, `NHLGAME`, `WNBAGAME`, `UFCFIGHT`, `ATPMATCH`, `PGATOUR`, `EPLGAME`, `SOCCERGAME` | Game outcomes, spreads, totals, player props |
| Crypto | `BTCD`, `BTCMAXY`, `ETHD`, `ETHMAXY`, `DOGE`, `SOL`, `XRP`, `SHIBA`, `COIN` | Daily/weekly price moves, token prices |
| Politics | `PRES`, `SENATEAZ`, `HOUSEMOV`, `GOVPARTYVA`, `TRUMP`, `BIDEN`, `CABINET`, `MAYORNYCPARTY`, `ELECTION` | Elections, administration actions, policy |
| Finance | `FEDDECISION`, `INXU`, `NASDAQ100U`, `TNOTE`, `USDJPY`, `GAS`, `CPI`, `IPO`, `TARIFF` | Index moves, Fed decisions, macro indicators |
| Weather | `HIGHNY`, `RAINNYC`, `SNOW`, `TORNADO`, `HURCAT`, `ARCTICICE`, `WEATHER` | Temperature, precipitation, severe weather |
| Entertainment | `OSCAR`, `EMMY`, `GRAMMY`, `NETFLIX`, `BOX`, `SPOTIFY`, `TIKTOK` | Awards, streaming, box office |

The full list of 571 prefix-to-domain mappings is in `src/classify.py`.

---

## 2. Polymarket: Regex Title Matching

Polymarket lacks structured ticker prefixes, so markets are classified by applying compiled regex patterns to the market `title` (question) field. First match wins; patterns are ordered by specificity.

### Sports

```
(?i)\b(NFL|NBA|MLB|NHL|UFC|MMA|boxing|tennis|golf|F1|Formula 1|NASCAR|
Super Bowl|World Series|Stanley Cup|NCAA|March Madness|
Premier League|Champions League|La Liga|Serie A|Bundesliga|Ligue 1|
Europa League|World Cup|Copa America|MLS|WNBA|PGA|ATP|WTA|
Grand Slam|Wimbledon|US Open|Australian Open|French Open|
Ryder Cup|Olympics|Olympic|medal|
playoff|postseason|All[- ]Star|MVP|Cy Young|Heisman|
touchdown|home run|strikeout|rushing yards|rebound|assist|
point spread|moneyline|sack|interception|three-pointer|
batting average|ERA|free throw|penalty kick|
49ers|Packers|Chiefs|Eagles|Cowboys|Patriots|Dolphins|Bills|Ravens|Bengals|
Lakers|Celtics|Warriors|Bucks|Nuggets|Knicks|76ers|Heat|Nets|Suns|
Yankees|Dodgers|Braves|Astros|Mets|Red Sox|Cubs|Phillies|Padres|
Maple Leafs|Bruins|Rangers|Lightning|Avalanche|Panthers|Oilers|
Arsenal|Liverpool|Manchester|Chelsea|Barcelona|Real Madrid|Bayern|Juventus|
game \d|win.*series|win.*championship|win.*title)\b
```

### Politics

```
(?i)\b(president|presidential|election|senate|congress|governor|
democrat|republican|GOP|Trump|Biden|Obama|
vote|ballot|primary|caucus|nominee|nomination|
Supreme Court|executive order|impeach|filibuster|debt ceiling|
shutdown|electoral college|swing state|poll|approval rating|
cabinet|attorney general|secretary of state|Speaker|majority leader|
veto|legislation|midterm|runoff|recall|referendum|
RFK|DeSantis|Haley|Ramaswamy|Newsom|Pence|Kamala|Harris|
McConnell|Pelosi|AOC|Schumer|Vance|Vivek|
indictment|classified documents|
NATO|Ukraine|Russia|China.*Taiwan|Israel|Iran|sanction|
TikTok.*ban|government.*ban|Congress.*pass|
federal|POTUS|White House|inaugurat)\b
```

### Crypto

```
(?i)\b(Bitcoin|BTC|Ethereum|ETH|crypto|Solana|SOL|Dogecoin|DOGE|XRP|
token|blockchain|DeFi|altcoin|stablecoin|USDC|USDT|Tether|
Binance|Coinbase|FTX|SBF|halving|mining|NFT|airdrop|
memecoin|Pepe|Shiba|TVL|DEX|CEX|
Bitcoin ETF|Ethereum ETF|spot ETF|
Layer 2|rollup|zkSync|Arbitrum|Polygon|Avalanche|Cardano|ADA|
Ripple|Litecoin|Polkadot|Chainlink|Uniswap)\b
```

### Finance

```
(?i)\b(S&P|S&P 500|NASDAQ|Dow Jones|Russell 2000|stock|
Fed |Federal Reserve|interest rate|rate cut|rate hike|FOMC|
CPI|inflation|GDP|recession|tariff|
IPO|treasury|yield curve|bond|
unemployment|jobs report|nonfarm payroll|non-farm payroll|PCE|PPI|
housing starts|retail sales|consumer confidence|
earnings|revenue|market cap|PE ratio|
merger|acquisition|bankruptcy|credit rating|
oil price|gold price|commodity|forex|
quantitative|balance sheet|FDIC|
trade deficit|budget|fiscal|monetary policy)\b
```

### Classifier Function

```python
def classify_polymarket_domain(title: str) -> str:
    if not title:
        return "Other"
    for pattern, domain in POLYMARKET_DOMAIN_PATTERNS:
        if pattern.search(title):
            return domain
    return "Other"
```

Markets not matching any pattern are classified as "Other" (42.5% of Polymarket markets). Only Sports, Crypto, Politics, and Finance are used in the cross-platform comparison; Weather and Entertainment have negligible Polymarket coverage.
