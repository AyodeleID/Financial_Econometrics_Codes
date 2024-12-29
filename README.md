# Financial_Econometrics_Codes_and_Investment Banking Analysis

This repository contains Python and R codes for financial econometrics analysis, including:
- Modeling returns with random walk models
- ARMA and ARIMA models
- ARCH, GARCH, and IGARCH models
- VAR models
- Cointegration models
- Correlation models (CCC, DCC, Copula-GARCH)






#Modeling Returns with Random Walk Models in Python

#Explanations of each line of Code are Provided each steps
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Explanation: We import numpy for numerical computations, pandas for data manipulation, and matplotlib for visualizing the random walk.

# Step 1: Define parameters for the random walk
n_steps = 100  # Number of time steps
initial_price = 100  # Starting price of the asset
mean_return = 0.001  # Mean return (drift)
volatility = 0.02  # Standard deviation of returns (volatility)

# Explanation: We define key parameters for the random walk:
# - `n_steps`: Number of time periods to simulate
# - `initial_price`: The starting price of the stock
# - `mean_return`: The expected daily return (drift component)
# - `volatility`: Random shock or noise in the returns

# Step 2: Simulate returns
np.random.seed(42)  # For reproducibility
random_shocks = np.random.normal(loc=mean_return, scale=volatility, size=n_steps)

# Explanation: 
# - `np.random.normal`: Generates random shocks from a normal distribution with mean `mean_return` and standard deviation `volatility`.
# - `size=n_steps`: We generate one random shock per time step.
# - `np.random.seed(42)`: Ensures reproducibility of results.

# Step 3: Generate prices from returns
prices = [initial_price]
for shock in random_shocks:
    new_price = prices[-1] * (1 + shock)
    prices.append(new_price)

# Explanation:
# - Start with `initial_price`.
# - For each `shock`, calculate the new price using the formula: `new_price = old_price * (1 + shock)`.
# - Append the new price to the list of prices.

# Step 4: Visualize the random walk
time = range(len(prices))
plt.figure(figsize=(10, 6))
plt.plot(time, prices, label="Simulated Random Walk", color="blue")
plt.title("Random Walk Model for Asset Prices")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

# Explanation: 
# - Use `plt.plot` to visualize the random walk.
# - Add titles and labels to make the chart informative.



#Applying Random Walk Model to Real Data - Note that you can also use an Excel file as well.
# Step 1: Load historical stock price data
# Replace 'your_file.csv' with the actual file path
data = pd.read_csv('your_file.csv')

# Explanation:
# - Use `pd.read_csv` to load data from a CSV file.
# - Replace `'your_file.csv'` with the actual file path containing historical stock prices.

# Step 2: Calculate daily returns
data['Return'] = data['Close'].pct_change()

# Explanation:
# - `pct_change`: Calculates percentage changes between consecutive closing prices.
# - This gives daily returns, a common input for financial modeling.

# Step 3: Visualize the returns
plt.figure(figsize=(10, 6))
plt.plot(data['Return'], label="Daily Returns", color="orange")
plt.title("Daily Returns of Asset")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend()
plt.grid()
plt.show()

# Explanation:
# - Plot the calculated daily returns to observe their behavior.




##################################################################################################################





#ARMA (Autoregressive Moving Average) ###See Python Code and R also

#Pyhton
# Import necessary libraries
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Explanation:
# - `ARIMA` (from statsmodels) includes ARMA functionality.
# - Other libraries are for data manipulation (`pandas`) and visualization (`matplotlib`).

# Step 1: Generate synthetic time series data
np.random.seed(42)
n_steps = 200
noise = np.random.normal(0, 1, n_steps)
synthetic_series = np.cumsum(noise)

# Explanation:
# - `np.random.normal`: Generates random noise for the time series.
# - `np.cumsum`: Creates a random walk by cumulatively summing the noise.

# Step 2: Fit an ARMA model
model = ARIMA(synthetic_series, order=(2, 0, 2))  # ARMA(p=2, d=0, q=2)
arma_fit = model.fit()

# Explanation:
# - `order=(p, d, q)`: Specifies the AR (p), differencing (d), and MA (q) orders.
#   For ARMA, `d=0` since we don't apply differencing.
# - `fit()`: Estimates the model parameters.

# Step 3: Summarize and forecast
print(arma_fit.summary())

# Explanation:
# - `summary()`: Provides detailed statistics about the ARMA model's fit.

# Step 4: Forecast future values
forecast = arma_fit.forecast(steps=10)
plt.plot(synthetic_series, label="Original Series")
plt.plot(range(len(synthetic_series), len(synthetic_series) + 10), forecast, label="Forecast", color="red")
plt.legend()
plt.title("ARMA Model with Forecast")
plt.show()

# Explanation:
# - `forecast()`: Predicts future values based on the fitted model.
# - The forecast is plotted alongside the original series.





# Install all necessary R packages Using the code below
install.packages(c("forecast", "readxl", "openxlsx", "readr", "ggplot2", "plotly", "dplyr", "tidyr"))

# Load all libraries that you need to always import before running the code , but still depends on what you are doing working on 
library(forecast)   # For ARIMA modeling and forecasting
library(stats)      # For residuals and plotting (pre-installed)
library(readxl)     # For reading Excel files
library(openxlsx)   # For writing Excel files
library(readr)      # For reading CSV files
library(ggplot2)    # For visualizations
library(plotly)     # For interactive plots
library(dplyr)      # For data manipulation
library(tidyr)      # For data reshaping

#R
# Load necessary library
library(forecast)

# Explanation:
# - The `forecast` package is commonly used for time series modeling in R.

# Step 1: Generate synthetic time series data
set.seed(42)  # For reproducibility
n_steps <- 200  # Number of steps
noise <- rnorm(n_steps, mean = 0, sd = 1)  # Random noise
synthetic_series <- cumsum(noise)  # Create a random walk

# Explanation:
# - `set.seed(42)` ensures reproducible results.
# - `rnorm`: Generates random normal noise.
# - `cumsum`: Simulates a random walk by cumulatively summing the noise.

# Step 2: Fit an ARMA model
arma_model <- Arima(synthetic_series, order = c(2, 0, 2))

# Explanation:
# - `Arima`: Fits an ARIMA model.
# - `order = c(2, 0, 2)`: Specifies ARMA(2,2) (p=2, d=0, q=2).

# Step 3: Summarize the model
summary(arma_model)

# Explanation:
# - `summary`: Provides model details, including coefficients and diagnostics.

# Step 4: Forecast future values
forecast_arma <- forecast(arma_model, h = 10)

# Explanation:
# - `forecast`: Predicts the next `h` time steps (here, 10 steps).

# Step 5: Visualize the forecast
plot(forecast_arma, main = "ARMA Model with Forecast")

# Explanation:
# - `plot`: Displays the original series and the forecasted values.



#Using a true dataset for ARIMA But on R (I used R because of simplicity) ---Remember to import all necessary libraries please, see what I did below

# Step 1: Load necessary libraries
install.packages("forecast")  # Install the forecast package
library(forecast)             # Load the forecast library for ARIMA modeling
install.packages("readr")     # Install readr for importing CSV files/ you can also use Excel
library(readr)

# Explanation:
# - `forecast`: Contains the `Arima` function for fitting ARIMA models.
# - `readr`: Used to load external datasets like CSV files/ or excel files.

# Step 2: Import your real-life data
# Download or use an existing dataset with time series data (e.g., stock prices or economic indicators)
# Ensure the CSV file is saved in the working directory
data <- read_csv("example_data.csv")  # Replace with your file name
#See mine for example
# Import the data from the specified path
data <- read_excel("C:/Users/ayode/Desktop/RESEARCH PUBLICATIONS/Economic Complexity on Renewable Energy/Extended Data for Impact of Eco Complexity on Ren Energy.xlsx")

# Explanation:
# - Replace `"example_data.csv"` with the path to your dataset on your laptop, don't forget (PATH of your file)
# - The file should contain at least two columns: date and value (e.g., stock prices).

# Step 3: Extract the time series data
time_series <- ts(data$value, start = c(2010, 1), frequency = 12)

# Explanation:
# - `ts`: Converts the imported data into a time series object.
# - `start`: Defines the start year and month (e.g., January 2010).
# - `frequency`: Indicates the number of observations per year (e.g., 12 for monthly data).

# Step 4: Plot the original time series
plot(time_series, main = "Original Time Series", ylab = "Value", xlab = "Time")

# Explanation:
# - Visualize the original data to check for trends or seasonality.

# Step 5: Fit an ARIMA model
arima_model <- Arima(time_series, order = c(2, 1, 2))

# Explanation:
# - `order = c(2, 1, 2)` specifies the ARIMA model:
#   - `p = 2`: Number of autoregressive terms.
#   - `d = 1`: Differencing to make the series stationary.
#   - `q = 2`: Number of moving average terms.

# Step 6: Summarize the model
summary(arima_model)

# Explanation:
# - Provides detailed statistics about the ARIMA model fit, including coefficients and diagnostics.

# Step 7: Plot residuals
residuals <- residuals(arima_model)
plot(residuals, main = "Residuals from ARIMA Model", type = "l")

# Explanation:
# - `residuals`: Extracts residuals from the ARIMA model.
# - Plotting the residuals helps check if they resemble white noise.

# Step 8: Forecast future values
forecasted_values <- forecast(arima_model, h = 12)  # Forecast next 12 periods
plot(forecasted_values, main = "ARIMA Model Forecast")

# Explanation:
# - `forecast`: Predicts the next `h` steps (e.g., 12 months in this case).
# - Visualize the forecasted values alongside the historical data.


#EWMA (Exponentially Weighted Moving Average) in R ONLY also
# Step 1: Calculate EWMA
alpha <- 0.2  # Smoothing factor
ewma <- stats::filter(synthetic_series, filter = alpha, method = "recursive")

# Explanation:
# - `filter`: Applies the EWMA formula recursively.
# - `alpha`: Determines the weight given to recent observations (higher = more weight).

# Step 2: Plot original series and EWMA
plot(synthetic_series, type = "l", col = "blue", lty = 1, main = "EWMA Smoothing")
lines(ewma, col = "orange", lty = 2)

# Explanation:
# - `lines`: Adds the EWMA curve to the plot.
# - Visualizes how EWMA smooths the original series.




#########################################################################################################################

#ARCH, GARCH and IGARCH models

# Load the necessary libraries
install.packages("rugarch")  # For ARCH/GARCH/IGARCH modeling
library(rugarch)  # Provides functions for modeling and forecasting volatility

# Explanation:
# The `rugarch` package is used to specify and fit ARCH/GARCH/IGARCH models efficiently.

# Load real-life financial data (stock returns or indices)
install.packages("quantmod")  # For retrieving financial data
library(quantmod)

# Explanation:
# The `quantmod` package is used to download financial data directly from sources like Yahoo Finance. (you can always use your own sources too)

# Get stock price data (e.g., Apple Inc.)
getSymbols("AAPL", src = "yahoo", from = "2015-01-01", to = "2023-01-01")
stock_data <- na.omit(AAPL$AAPL.Adjusted)  # Use adjusted closing prices

# Explanation:
# `getSymbols` fetches historical stock price data from Yahoo Finance.
# `na.omit` removes any missing values in the dataset.

# Calculate log returns
log_returns <- diff(log(stock_data)) * 100
log_returns <- na.omit(log_returns)

# Explanation:
# Log returns are calculated as the percentage change in log prices.
# This transformation ensures stationarity, which is required for volatility modeling.



#Then using the ARCH and GARCH models in R

#ARCH Model
# Step 1: Specify the ARCH(1) model
arch_spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 0)),
                        mean.model = list(armaOrder = c(0, 0)))

# Explanation:
# `ugarchspec`: Defines the model specification.
# - `model = "sGARCH"`: Standard GARCH model; ARCH is a special case where MA(q)=0.
# - `garchOrder = c(1, 0)`: Specifies an ARCH(1) model (1 lag of variance).

# Step 2: Fit the ARCH model
arch_fit <- ugarchfit(spec = arch_spec, data = log_returns)

# Explanation:
# `ugarchfit`: Fits the specified ARCH model to the log returns data.

# Step 3: Summarize the results
summary(arch_fit)

# Explanation:
# Prints details about the fitted ARCH model, including parameter estimates.

# Step 4: Plot the fitted volatility
plot(arch_fit, which = "all")

# Explanation:
# Visualizes the estimated conditional variance (volatility) over time.





#GARCH Model
# Step 1: Specify the GARCH(1,1) model
garch_spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                         mean.model = list(armaOrder = c(0, 0)))

# Explanation:
# `garchOrder = c(1, 1)`: Specifies a GARCH(1,1) model with 1 lag of ARCH and 1 lag of GARCH terms.

# Step 2: Fit the GARCH model
garch_fit <- ugarchfit(spec = garch_spec, data = log_returns)

# Explanation:
# Fits the GARCH model to the log returns data.

# Step 3: Summarize the results
summary(garch_fit)

# Explanation:
# Displays GARCH parameter estimates and model diagnostics.

# Step 4: Plot the fitted volatility
plot(garch_fit, which = "all")

# Explanation:
# Visualizes the estimated volatility from the GARCH model.






IGARCH Model
# Step 1: Specify the IGARCH(1,1) model
igarch_spec <- ugarchspec(variance.model = list(model = "iGARCH", garchOrder = c(1, 1)),
                          mean.model = list(armaOrder = c(0, 0)))

# Explanation:
# `model = "iGARCH"`: Specifies an Integrated GARCH model.
# IGARCH imposes a unit root in the volatility process, meaning shocks have a long-lasting impact.

# Step 2: Fit the IGARCH model
igarch_fit <- ugarchfit(spec = igarch_spec, data = log_returns)

# Explanation:
# Fits the IGARCH model to the log returns data.

# Step 3: Summarize the results
summary(igarch_fit)

# Explanation:
# Provides parameter estimates for the IGARCH model, which are constrained to ensure stationarity.

# Step 4: Plot the fitted volatility
plot(igarch_fit, which = "all")

# Explanation:
# Visualizes the long-term impact of shocks on volatility using the IGARCH model.





######################################################################################################################


#VAR Model Implementation in R
#You need to do these first before running the code
install.packages("vars")    # For VAR modeling
install.packages("readxl")  # For importing Excel files
install.packages("ggplot2") # For visualization

#Reassuring that you have all the necessary libraries
library(vars)   # Contains functions for VAR model fitting and diagnostics
library(readxl) # For importing Excel files
library(ggplot2) # For visualization



# Import data from Excel
data <- read_excel(Paste your file path here)

# Explanation:
# - Replace the path with the correct location of your Excel file.
# - Ensure the file contains at least two time series columns for VAR analysis.

# Select relevant columns (e.g., Time Series 1 and Time Series 2)
time_series_data <- data.frame(
  Series1 = data$Variable1,  # Replace `Variable1` with the actual column name (your own variable)
  Series2 = data$Variable2   # Replace `Variable2` with the actual column name (your own variable)
)

# Convert data to a time series object
time_series <- ts(time_series_data, start = c(2010, 1), frequency = 12)

# Explanation:
# - `data.frame`: Combines selected columns into a data frame.
# - `ts`: Converts the data into a time series object, specifying the start date and frequency (12 for monthly data).
# Fit the VAR model with lag selection
lag_selection <- VARselect(time_series, lag.max = 10, type = "const")

# Explanation:
# - `VARselect`: Identifies the optimal lag length using criteria like AIC, BIC, and HQ.
# - `lag.max = 10`: Tests lags up to 10 periods.

print(lag_selection$selection)  # Display the selected lag lengths

# Fit the VAR model using the optimal lag
var_model <- VAR(time_series, p = lag_selection$selection["AIC(n)"], type = "const")

# Explanation:
# - `VAR`: Fits the VAR model with the selected lag (`p`).
# - `type = "const"`: Includes a constant in the model.

# Display the VAR model summary
summary(var_model)

# Explanation:
# - `summary`: Provides parameter estimates, standard errors, and model diagnostics.






######################################################################################################################

#Cointegration and VECM in R

# Step 1: Install and load necessary libraries
install.packages("urca")    # For cointegration tests
install.packages("vars")    # For VAR modeling (used in cointegration analysis)
install.packages("readxl")  # For importing Excel files
install.packages("ggplot2") # For visualization

library(urca)    # Contains functions for Johansen cointegration test
library(vars)    # Used for VAR modeling and lag selection
library(readxl)  # To import data from Excel
library(ggplot2) # For data visualization

# Step 2: Import real-life data from Excel
# Replace with the path to your Excel file
data <- read_excel("C:/Users/ayode/Desktop/RESEARCH PUBLICATIONS/Economic Complexity on Renewable Energy/Extended Data for Impact of Eco Complexity on Ren Energy.xlsx")

# Explanation:
# - Replace the file path with the path to your file.
# - Ensure the file contains at least two time series columns.

# Step 3: Prepare the data for analysis
# Select the relevant columns (replace `Variable1` and `Variable2` with actual column names)
time_series_data <- data.frame(
  Series1 = data$Variable1,  # First time series
  Series2 = data$Variable2   # Second time series
)

# Convert to a time series object
time_series <- ts(time_series_data, start = c(2010, 1), frequency = 12)

# Explanation:
# - Combines selected columns into a `data.frame` and converts them into a `ts` object.
# - Adjust `start` and `frequency` based on your data (e.g., monthly data -> frequency = 12).

# Step 4: Test for stationarity (ADF test)
# Test each series for stationarity
adf_series1 <- ur.df(time_series[, 1], type = "trend", lags = 12)
adf_series2 <- ur.df(time_series[, 2], type = "trend", lags = 12)

# Display the ADF test results
summary(adf_series1)
summary(adf_series2)

# Explanation:
# - `ur.df`: Conducts Augmented Dickey-Fuller tests to check for stationarity.
# - If the p-value > 0.05, the series is non-stationary, which is required for cointegration analysis.

# Step 5: Perform Johansen Cointegration Test
johansen_test <- ca.jo(time_series, type = "trace", K = 2, ecdet = "const")

# Explanation:
# - `ca.jo`: Conducts Johansen's cointegration test.
# - `type = "trace"`: Tests for cointegration using the trace statistic.
# - `K = 2`: Specifies the lag order (adjust based on your data's characteristics).
# - `ecdet = "const"`: Includes a constant in the cointegration equation.

# Display Johansen test results
summary(johansen_test)

# Step 6: Extract Cointegration Vector
cointegration_vector <- johansen_test@V[, 1]  # First cointegrating vector
print(cointegration_vector)

# Explanation:
# - The cointegration vector shows the long-run relationship between the variables.

# Step 7: Fit a VECM (Vector Error Correction Model)
vecm_model <- cajorls(johansen_test, r = 1)  # Use the first cointegrating relationship
summary(vecm_model)

# Explanation:
# - `cajorls`: Estimates the VECM model based on the cointegrating relationship.
# - Provides short-run dynamics and error correction term.

# Step 8: Visualize the time series
ggplot(data.frame(Time = time(time_series), Series1 = time_series[, 1], Series2 = time_series[, 2]), aes(x = Time)) +
  geom_line(aes(y = Series1, color = "Series 1")) +
  geom_line(aes(y = Series2, color = "Series 2")) +
  labs(title = "Time Series Plot", y = "Values", x = "Time") +
  theme_minimal()

# Explanation:
# - Plots both time series to observe if they move together over time.



######################################################################################################################

6.	Correlation models: constant conditional correlation (CCC) model; 
dynamic conditional correlation (DCC) model; 
correlation targeting; copula-GARCH models.

install.packages("rmgarch")    # For CCC and DCC models
install.packages("copula")     # For copula functions
install.packages("readxl")     # For importing Excel files
install.packages("ggplot2")    # For data visualization

library(rmgarch)  # Multivariate GARCH models (CCC, DCC)
library(copula)   # Copula functions for dependence modeling
library(readxl)   # Importing Excel files
library(ggplot2)  # Data visualization


# Import real-life financial data from Excel ---I just used my own file path, replace with yours below
data <- read_excel("C:/Users/ayode/Desktop/RESEARCH PUBLICATIONS/Economic Complexity on Renewable Energy/Extended Data for Impact of Eco Complexity on Ren Energy.xlsx")

# Select relevant columns for analysis (replace `Variable1` and `Variable2` with actual column names)
time_series_data <- data.frame(
  Series1 = data$Variable1,  # Replace `Variable1` with your column name
  Series2 = data$Variable2   # Replace `Variable2` with your column name
)

# Calculate log returns for both series
time_series_data$LogReturn1 <- diff(log(time_series_data$Series1)) * 100
time_series_data$LogReturn2 <- diff(log(time_series_data$Series2)) * 100

# Remove NAs caused by differencing
time_series_data <- na.omit(time_series_data)

# Explanation:
# - Calculate log returns to make the data stationary.
# - `diff(log(...))` computes percentage changes in log prices.

###########

# CCC Model
# Step 1: Specify the CCC-GARCH model
spec_ccc <- ugarchspec(mean.model = list(armaOrder = c(1, 0)),
                       variance.model = list(model = "sGARCH"),
                       distribution.model = "norm")
cccspec <- dccspec(uspec = multispec(replicate(2, spec_ccc)), dccOrder = c(1, 1), model = "CCC")

# Explanation:
# - `ugarchspec`: Defines univariate GARCH models for each series.
# - `dccspec`: Specifies the multivariate GARCH model with constant correlation.

# Step 2: Fit the CCC model
cccfitted <- dccfit(cccspec, data = time_series_data[, c("LogReturn1", "LogReturn2")])

# Explanation:
# - `dccfit`: Fits the CCC-GARCH model to the data.

# Step 3: Summarize the CCC model
summary(cccfitted)

# Explanation:
# - Summarizes the constant conditional correlations between the series.


###########

#DCC Model
# Step 1: Specify the DCC-GARCH model
spec_dcc <- dccspec(uspec = multispec(replicate(2, spec_ccc)), dccOrder = c(1, 1), model = "DCC")

# Explanation:
# - `model = "DCC"`: Specifies the Dynamic Conditional Correlation model.

# Step 2: Fit the DCC model
dccfitted <- dccfit(spec_dcc, data = time_series_data[, c("LogReturn1", "LogReturn2")])

# Explanation:
# - `dccfit`: Fits the DCC-GARCH model to the data.

# Step 3: Extract and plot dynamic correlations
dcc_correlations <- rcor(dccfitted)
plot(dcc_correlations[1, 2, ], type = "l", main = "Dynamic Conditional Correlations", ylab = "Correlation", xlab = "Time")

# Explanation:
# - `rcor`: Extracts the dynamic conditional correlations over time.
# - Plot shows how the correlation evolves dynamically.


############

#Correlation Targeting
# Set up GARCH specification with correlation targeting
target_spec <- dccspec(uspec = multispec(replicate(2, spec_ccc)), dccOrder = c(1, 1), model = "DCC", 
                       target = cor(time_series_data[, c("LogReturn1", "LogReturn2")]))

# Explanation:
# - `target`: Specifies the fixed correlation matrix for correlation targeting.
# - Uses the sample correlation of the log returns.

# Fit the model
target_fitted <- dccfit(target_spec, data = time_series_data[, c("LogReturn1", "LogReturn2")])

# Explanation:
# - Fits a DCC model with fixed correlation targeting.



#############

#Copula-GARCH Model
# Step 1: Specify univariate GARCH models
garch_uspec <- ugarchspec(mean.model = list(armaOrder = c(1, 0)),
                          variance.model = list(model = "sGARCH"),
                          distribution.model = "norm")

# Step 2: Fit the univariate GARCH models
garch_fit1 <- ugarchfit(garch_uspec, data = time_series_data$LogReturn1)
garch_fit2 <- ugarchfit(garch_uspec, data = time_series_data$LogReturn2)

# Extract residuals and standardize them
residuals1 <- residuals(garch_fit1, standardize = TRUE)
residuals2 <- residuals(garch_fit2, standardize = TRUE)

# Combine residuals into a matrix
residuals_matrix <- cbind(residuals1, residuals2)

# Step 3: Fit a Gaussian copula
copula_fit <- normalCopula(dim = 2)
fit <- fitCopula(copula_fit, pobs(residuals_matrix), method = "ml")

# Explanation:
# - `normalCopula`: Specifies a Gaussian copula.
# - `pobs`: Converts residuals into pseudo-observations.
# - `fitCopula`: Fits the copula model using maximum likelihood estimation.

