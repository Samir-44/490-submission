import streamlit as st
from logic.predict_spy import predict_with_spy_model, predict_next_n_days
from logic.retrain_model import retrain_and_predict, retrain_and_predict_multi
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Stock Price Prediction (Pretrained vs Retrainable LSTM)")

# --- Initialize prediction history ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- User inputs ---
ticker = st.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL").upper()
n_days = st.slider("How many future days to predict?", min_value=1, max_value=5, value=1)

# --- Model visibility options ---
show_spy = st.checkbox("Show SPY-Based Model", value=True)
show_retrain = st.checkbox("Show Retrainable Model", value=True)

if st.button("Predict"):
    col1, col2 = st.columns(2)
    spy_preds = []
    retrain_preds = []

    try:
        # --- SPY-based prediction ---
        if show_spy:
            with col1:
                with st.spinner("SPY-based model is predicting..."):
                    if n_days == 1:
                        spy_preds = [predict_with_spy_model(ticker)]
                        st.metric("SPY-Based Model", f"${spy_preds[0]:.2f}")
                    else:
                        spy_preds = predict_next_n_days(ticker, n_days=n_days)
                        st.markdown("### 📈 SPY-Based Multi-Day Prediction")
                        for i, p in enumerate(spy_preds, 1):
                            st.write(f"Day {i}: **${p:.2f}**")

        # --- Retrainable model prediction ---
        if show_retrain:
            with col2:
                with st.spinner("Retraining & predicting..."):
                    if n_days == 1:
                        retrain_preds = [retrain_and_predict(ticker)]
                        st.metric("Retrainable Model", f"${retrain_preds[0]:.2f}")
                    else:
                        retrain_preds = retrain_and_predict_multi(ticker, n_days=n_days)
                        st.markdown("### 📈 Retrainable Multi-Day Prediction")
                        for i, p in enumerate(retrain_preds, 1):
                            st.write(f"Day {i}: **${p:.2f}**")

        # --- Store in history ---
        st.session_state.history.append({
            "Ticker": ticker,
            "SPY-Based": [f"${p:.2f}" for p in spy_preds] if show_spy else ["Hidden"],
            "Retrainable": [f"${p:.2f}" for p in retrain_preds] if show_retrain else ["Hidden"]
        })

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Display prediction history ---
if st.session_state.history:
    st.subheader(" Prediction History")

    # Convert lists to strings for display
    history_df = pd.DataFrame(st.session_state.history[::-1])
    history_df["SPY-Based"] = history_df["SPY-Based"].apply(lambda x: ", ".join(x))
    history_df["Retrainable"] = history_df["Retrainable"].apply(lambda x: ", ".join(x))

    st.dataframe(history_df, use_container_width=True)

    # --- Download button ---
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    if st.button(" Clear History"):
        st.session_state.history.clear()
        st.info("History cleared — click 'Predict' to refresh the table.")


# --- Historical Price Chart ---
if ticker:
    st.subheader(" Historical Stock Price (Last 90 Days)")

    try:
        hist_data = yf.download(ticker, period="90d")['Close']
        if not hist_data.empty:
            fig, ax = plt.subplots(figsize=(18, 8))
            ax.plot(hist_data.index, hist_data.values, label="Close Price", color="blue", linewidth=2)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            fig.autofmt_xdate()
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.set_title(f"{ticker} - Last 90 Days Closing Price")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("No historical data available for this ticker.")
    except Exception as e:
        st.warning(f"Couldn't load historical data: {e}")