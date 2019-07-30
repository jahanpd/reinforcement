library("crypto")
library("rio")

start_date <- Sys.Date()
end_date <- Sys.Date() - 365*3

start_date
end_date

btc <-crypto_history(coin = c("BTC"), limit = 100000, start_date = "2019-06-19",
end_date = "2016-05-01", coin_list = NULL, sleep = 5)

export(btc, "btc.csv")