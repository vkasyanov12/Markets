# Markets

The Market code was designed to pull stock market data and visualize it by  using pandas library. The Market file can also pull cryptocurrency data if one so desires. The Markets file contains four classes know as MarketListings, Asset, Technical_Analysis and Graph. 

- MarketListings: Pulls all the possible listings from the nasdaq exchange and coinmarketcap.com
- Asset: Data structure that holds the asset data.
- Technical_Analysis: Used by the Asset class to create stock indicators.
- Graph: Used by the Asset class in order to graph the data.

# Portfolio

The Portfolio code was designed to generate a portfolio using the Asset class from the Markets code. The portfolio generator only has two momentum strategies so far. The return based momentum where there the stocks are chosen based upon the increase or decrease of the stock. The other is the 52 week strategy. It uses multiprocessing in order to fix the latency problem that comes with pulling lots of stock data.
