import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Read the data
order = pd.read_csv('orders.csv', parse_dates=['date'], dayfirst=True, usecols=['order_id', 'date'])
order_details = pd.read_csv('order_details.csv')
pizza_data = pd.read_csv('pizzas.csv')

# Merge the data
merged_order = pd.merge(order, order_details, on='order_id', how='inner')
sales_data = pd.merge(merged_order, pizza_data, on='pizza_id', how='left')
sales_data['date'] = pd.to_datetime(sales_data['date'], dayfirst=True)

# Resample the data to get monthly sales
monthly_sales = sales_data.resample('M', on='date').sum()
monthly_sales['total_sales_amount'] = monthly_sales['quantity'] * monthly_sales['price']

# Train the ARIMA model
model = ARIMA(monthly_sales['total_sales_amount'], order=(1, 1, 1))
result = model.fit()

# Forecast the sales for the next month
def plot_sales_forecast():
    # Forecast the sales for the next month
    next_month_forecast = result.forecast(steps=5)

    print("Forecasted Sales for Next Month:")
    print(next_month_forecast)

    # Plot the historical sales and forecasted sales
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales.index, monthly_sales['total_sales_amount'], marker='o', label='Historical Sales')
    plt.plot(next_month_forecast.index, next_month_forecast, marker='o', label='Forecasted Sales')
    plt.title('Monthly Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Total Sales Amount')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_sales_forecast()
