library(anytime)
library(tseries)
library(TSA)
library(zoo)
library(rugarch)
library(car)
library(ggplot2)
library(forecast)
library(Metrics)

#load and manipulate Ford data
ford_data = read.csv("C:/Users/Owner/Documents/MSDS Fall22/Time Series/FORD.csv")
ford_dates = anydate(ford_data$Date)
ford_adj_close = ford_data$Adj.Close

ford_ts = ts(ford_dates, ford_adj_close)
plot(ford_dates, ford_adj_close, type = 'l')


#acf and pacf of ford adj close price
acf(ford_adj_close)
pacf(ford_adj_close)

#stationarity of original dataset
adf.test(ford_adj_close)

#take log differece to make the data stationary for ARIMA
ford_adj_close_logdiff = diff(log(ford_adj_close),1)
plot(ford_dates[2:length(ford_adj_close)],ford_adj_close_logdiff, type = 'l')

#dicky fuller test on both log differenced data sets to determine stationarity
adf.test(ford_adj_close_logdiff)

#use sqaured data for GARCH 
ford_adj_close_logdiff_sq = (ford_adj_close_logdiff)^2
plot(ford_dates[2:length(ford_adj_close)],ford_adj_close_logdiff_sq, type = 'l')


#ARI(6,1), ARI(7,1), IMA(1,6), IMA(1,7), ARIMA(6,1,6) 
acf(ford_adj_close_logdiff)
pacf(ford_adj_close_logdiff)
eacf(ford_adj_close_logdiff, ar.max = 30, ma.max = 30)

#GARCH(6,0) 
acf(ford_adj_close_logdiff_sq)
pacf(ford_adj_close_logdiff_sq)
eacf(ford_adj_close_logdiff_sq)

#fit garch model and see how well it fits with qqplot of residuals

#parameter estimation for Ford ARIMA
#ARIMA(7,1,7) seems to be the best based on the lowest AIC (most negative)
#ARIMA(1,1,0) does close to the same job as 7,1,7 but has way less parameters
arima(log(ford_adj_close), order=c(6,1,0),method='ML') # maximum likelihood
arima(log(ford_adj_close), order=c(7,1,0),method='ML') # maximum likelihood
arima(log(ford_adj_close), order=c(0,1,6),method='ML') # maximum likelihood
arima(log(ford_adj_close), order= c(0,1,7),method = 'ML') # maximum likelihood
arima(log(ford_adj_close), order = c(6,1,6),method = 'ML') # maximum likelihood
arima(log(ford_adj_close), order = c(7,1,7),method = 'ML') # maximum likelihood
arima(log(ford_adj_close), order = c(8,1,8),method = 'ML') # maximum likelihood
arima(log(ford_adj_close), order=c(1,1,0),method='ML') # maximum likelihood
arima(log(ford_adj_close), order=c(0,1,1),method='ML') # maximum likelihood
arima(log(ford_adj_close), order=c(1,1,1),method='ML') # maximum likelihood
arima(log(ford_adj_close), order=c(2,1,0),method='ML') # maximum likelihood
arima(log(ford_adj_close), order=c(0,1,2),method='ML') # maximum likelihood
arima(log(ford_adj_close), order=c(2,1,2),method='ML') # maximum likelihood


auto.arima(ford_adj_close_logdiff,stationary = TRUE, seasonal = FALSE)
auto.arima(ford_adj_close,stationary = TRUE, seasonal = FALSE)


#Model Diagnostics on Ford ARIMA
#using parameter redundancy we lowered the order of the model to reduce overfitting

#Residuals do not seem to be following normal dist according to Shapiro Wilkes Test
arima_model = arima(log(ford_adj_close), order=c(1,1,0),method='ML')
acf(arima_model$residuals)
qqPlot(arima_model$residuals)
shapiro.test(arima_model$residuals)
Box.test(arima_model$residuals, type = "Ljung-Box")
jarque.bera.test(arima_model$residuals)
checkresiduals(arima_model)


arima_model = arima(log(ford_adj_close), order=c(0,1,1),method='ML')
acf(arima_model$residuals)
qqPlot(arima_model$residuals)
shapiro.test(arima_model$residuals)
jarque.bera.test(arima_model$residuals)
Box.test(arima_model$residuals, type = "Ljung-Box")
checkresiduals(arima_model)


arima_model = arima(log(ford_adj_close), order=c(2,1,2),method='ML')
acf(arima_model$residuals)
qqPlot(arima_model$residuals)
shapiro.test(arima_model$residuals)
Box.test(arima_model$residuals, type = "Ljung-Box")
checkresiduals(arima_model)


#parameter estimation for Ford GARCH
plot(ford_dates[2:length(ford_adj_close)],ford_adj_close_logdiff^2, type = 'l')
acf(ford_adj_close_logdiff^2)
pacf(ford_adj_close_logdiff^2)
eacf(ford_adj_close_logdiff^2)

garch_model = garch(ford_adj_close_logdiff, order = c(6,0))
qqPlot(garch_model$residuals)
shapiro.test(garch_model$residuals)

garch_model = garch(ford_adj_close_logdiff, order = c(1,1))
qqPlot(garch_model$residuals)
shapiro.test(garch_model$residuals)

#ugarch - how do we use?
spec1= ugarchspec(
  variance.model = list(garchOrder = c(1,0)),
  mean.model = list(armaOrder = c(1,0), include.mean = FALSE)
)
spec1
GARCH = ugarchfit(spec = spec1, data = ford_adj_close_logdiff)
GARCH
names(GARCH@fit)
plot(GARCH@fit$sigma, type = 'l')

garch_2 = garch(diff())

residuals = GARCH@fit$residuals
acf(residuals, main = "Autocorrelation of residuals")
acf(residuals^2, main = "Autocorrelation of residuals squared")
jarque.bera.test(residuals)
Box.test(residuals, type = "Ljung-Box")
Box.test(residuals^2, type = "Ljung-Box")
shapiro.test(residuals)
shapiro.test(residuals^2)
qqPlot(residuals)
qqPlot(residuals^2)
hist(residuals)
hist(residuals^2)



#Forecasting of final Ford ARIMA model
final_arima_model = arima(log(ford_adj_close), order=c(1,1,0),method='ML')
autoplot(forecast(final_arima_model))


###############################################################################
#load and manipulate Temp data
avg_temp = read.csv("C:/Users/Owner/Documents/MSDS Fall22/Time Series/Monthly Avg Temp.csv")
avg_temp_dates = anydate(avg_temp$Date)
avg_temp_ts = ts(avg_temp_dates, avg_temp$Value)
plot(avg_temp_dates, avg_temp$Value, type='l')

total_observ = length(avg_temp$Value)
train_num = ceiling(total_observ*0.8)
test_num = total_observ - train_num
avg_temp_test = avg_temp$Value[(train_num+1):total_observ]
length(avg_temp_test)

#dicky fuller test on both data sets to determine stationarity
adf.test(avg_temp$Value)

#parameter estimation for Avg Temp
#s = 12
#take seasonal differencing of 12 for monthly data
avg_temp_seasDiff = diff(avg_temp$Value, lag = 12)
plot(avg_temp_seasDiff, type = 'l')

#take 1 non seasonal difference on top of the seasonal difference
plot(diff(avg_temp_seasDiff), type = 'l')

#Determine Non Seasonal and Seasonal Orders of SARIMA model 
acf(diff(avg_temp_seasDiff, lag = 1), lag.max = 50)
pacf(diff(avg_temp_seasDiff, lag = 1), lag.max = 50)

#ACF - 2 non seas, 2/3 seas
#PACF - 1/2/3 non seasonal, 3/4 seas
#SARIMA(1/2/3,1,2)x(3/4,1,2/3) ~12

#Find Best Model according to AIC and w/ Parameter Redundancy
arima(avg_temp$Value, order=c(1,1,2),seasonal=list(order=c(3,1,2),period=12), method = "ML")
#aic = 1264.96

arima(avg_temp$Value, order=c(1,1,2),seasonal=list(order=c(3,1,3),period=12), method = "ML")
#aic = 1266.67

arima(avg_temp$Value, order=c(2,1,2),seasonal=list(order=c(3,1,2),period=12), method = "ML")
#aic = 1271.82

arima(avg_temp$Value, order=c(2,1,2),seasonal=list(order=c(3,1,3),period=12), method = "ML")
#error

arima(avg_temp$Value, order=c(3,1,2),seasonal=list(order=c(3,1,2),period=12), method = "ML")
#error

arima(avg_temp$Value, order=c(3,1,2),seasonal=list(order=c(3,1,3),period=12), method = "ML")
#error

arima(avg_temp$Value, order=c(1,1,2),seasonal=list(order=c(4,1,2),period=12), method = "ML")
#aic = 1267.35

arima(avg_temp$Value, order=c(1,1,2),seasonal=list(order=c(4,1,3),period=12), method = "ML")
#aic = 1269.53

arima(avg_temp$Value, order=c(2,1,2),seasonal=list(order=c(4,1,2),period=12), method = "ML")
#aic = 1277.2

arima(avg_temp$Value, order=c(2,1,2),seasonal=list(order=c(4,1,3),period=12), method = "ML")
#error

arima(avg_temp$Value, order=c(3,1,2),seasonal=list(order=c(4,1,2),period=12), method = "ML")
#error

arima(avg_temp$Value, order=c(3,1,2),seasonal=list(order=c(4,1,3),period=12), method = "ML")
#error


#Model Diagnostics
sarima_model = arima(avg_temp$Value, order=c(1,1,2),seasonal=list(order=c(3,1,2),period=12), method = "ML")
acf(sarima_model$residuals)
qqPlot(sarima_model$residuals)
shapiro.test(sarima_model$residuals)
jarque.bera.test(sarima_model$residuals)
Box.test(sarima_model$residuals, type = "Ljung-Box")
checkresiduals(sarima_model)


#Forecast
#forecast 50 values
sarima_model_pred = predict(sarima_model,50)$pred

#full plot with forecasted values compared to actual values
plot(avg_temp_dates, avg_temp$Value, type='l', lwd= 2)
lines(avg_temp_dates[(train_num + 1):total_observ], sarima_model_pred, col='red', lwd = 3)

#zoomed plot with forecasted values compared to actual values
plot(avg_temp_dates[(train_num+1):total_observ], avg_temp_test, ylim = c(20,90), type='l')
lines(avg_temp_dates[(train_num+1):total_observ], sarima_model_pred, col='red', lwd = 3)

#forecast seems too lag a few months??

#forecasting evaluation
mse(avg_temp_test, sarima_model_pred)
ae(avg_temp_test, sarima_model_pred)
mae(avg_temp_test, sarima_model_pred)


#sq diff
#abs val of diff


