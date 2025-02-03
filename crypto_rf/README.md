# Crypto Algorithmic Trading 

## 1. Daily Data Downloads
We download hourly data from CoinCompare. To do so, we need to follow a few steps. 

1. Package and upload crypto_rf subdirectory to S3
```python crypto_rf/automate/package_and_upload.py```
2. Lambda Function
- We have a lambda function that executes save_hour_data.py and save_unvierse.py at specified timeframes, scheduled by EventBridge. 
- The save_universe function saves the top 250 cryptos by market cap at 12:15 AM each day. 
- The save_hour_data function saves the hourly price data for the top 250 cryptos at 11:45 pm each day. 


## 2. Trading Approaches
### 2.1 Technical Kitchen-Sink
Idea: throw a bunch of technical features into a random forest model to predict the outperforms over a X hour horizon. Use the probability classifications to build a decile based portfolio and long the top and short the bottom. Can improve with ensembling multiple ML models. 

### 2.2 Crypto-Weekend Performance as a Risk On/Off Signal 
Does the crypto performance over the weekend signal the equity performance for the following week? Need to make sure we model at approriate entry times. 