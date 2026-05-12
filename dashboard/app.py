import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Regime Trader", layout="wide")
st.title("Regime Trader — smoke test")
st.metric("NAV", "$100,000")
st.write(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
fig = px.line(pd.DataFrame({"x": range(10), "y": np.random.randn(10)}), x="x", y="y")
st.plotly_chart(fig)
st.success("All imports OK")
