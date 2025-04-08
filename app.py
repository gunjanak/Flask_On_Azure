from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import numpy as np
from helper import stock_dataFrame, create_sequences, inference

app = Flask(__name__)

def generate_plotly_plot(result_df, symbol):
    """Generate an interactive Plotly plot and return its HTML div."""
    fig = px.line(result_df, x=result_df.index, y=['Actual', 'Predicted'], 
                  markers=True, title=f'Actual vs Predicted Stock Prices for {symbol}')

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price",
        template="plotly_white",
        legend_title="Legend"
    )

    return fig.to_html(full_html=False)  # Return only the div content

def calculate_mape(y_actual, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    y_actual, y_pred = np.array(y_actual), np.array(y_pred)
    return np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

@app.route('/', methods=['GET', 'POST'])
def home():
    df_html = None
    plot_html = None
    mape_value = None
    forecasted_price = None  # Store next-day forecast

    if request.method == 'POST':
        symbol = request.form.get('symbol')
        if symbol:
            df = stock_dataFrame(symbol)
            df.reset_index(inplace=True)

            # Create sequences
            X, y = create_sequences(df)
            print(X.shape)
            print(type(X))

            # Convert NumPy arrays to lists
            y_list = y.tolist()
            prediction = inference(X)
            print(X[-5:])
            prediction_list = prediction
            print(prediction_list[-5:])

            # Take only the last 'len(y)' dates from df
            df_dates = df.iloc[-len(y):]['Date'].values

            # Create DataFrame for predictions
            result_df = pd.DataFrame({
                'Date': df_dates,
                'Actual': y_list,
                'Predicted': prediction_list
            })
            result_df.set_index('Date', inplace=True)

            # Calculate MAPE
            mape_value = calculate_mape(y_list, prediction_list)

            # Select last 10 records for display
            result_df_tail = result_df.tail(10)
            df_html = result_df_tail.to_html(classes='table table-striped table-bordered')

            # Generate interactive plot
            plot_html = generate_plotly_plot(result_df, symbol)

            # # --- 🔹 Next-Day Forecast ---
            print(df.tail())
            df =df[['Close']]
            last_5_days = df.iloc[-5:]  
            print(f"Last 5 days: {last_5_days}")
            last_5_days_array = last_5_days.values  
            last_5_days_array = last_5_days_array.reshape(1, 5)
            print(last_5_days_array.shape)
            print(type(last_5_days_array))
           
            
            forecasted_price = inference(last_5_days_array)  # Pass list to inference()
            print(forecasted_price)
            forecasted_price = round(float(forecasted_price[0]), 2)  # Convert to readable format

    return render_template('index.html', df_html=df_html, plot_html=plot_html, 
                           mape_value=mape_value, forecasted_price=forecasted_price)

    

if __name__ == '__main__':
    app.run(debug=True)
