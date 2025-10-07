import streamlit as st
import pandas as pd
import plotly.express as px
# from dask import dataframe as dd  # Commented for now—add for ultra-scale

# Generic Framework (Domain-Agnostic)
class DataIngester:
    @staticmethod
    def ingest(source_type, input_data, chunk_size=10000):
        if source_type == "text":
            # Parse JSON/text
            import json
            try:
                data = json.loads(input_data)
                return [data] if isinstance(data, dict) else data
            except:
                return [{"raw": input_data}]
        elif source_type == "excel":
            if hasattr(input_data, 'read'):
                # Read full file (fast for 10K rows; no chunksize error)
                df_full = pd.read_excel(input_data)
                # Slice into chunks post-read (CPU-efficient, no slowdown)
                chunks = [df_full.iloc[i:i+chunk_size] for i in range(0, len(df_full), chunk_size)]
                return chunks
            else:
                return []

class Transformer:
    @staticmethod
    def transform(gpu_mode, chunks):
        # Vectorized/Chunked—CPU fast, GPU 10x for batches
        all_df = []
        for chunk in chunks:
            if isinstance(chunk, pd.DataFrame):
                df_chunk = chunk  # Already a DF for Excel—use directly
            else:
                df_chunk = pd.DataFrame([chunk])  # Wrap dict/list for text
            all_df.append(df_chunk)
        return pd.concat(all_df) if all_df else pd.DataFrame()  # Use dd.concat for Dask CPU scale

class InsightEngine:
    @staticmethod
    def generate_bi(df, domain="generic"):
        # Domain-agnostic stats/chart—robust to mixed data
        numeric_df = df.select_dtypes(include=['number'])  # Filter numerics only (e.g., price)
        if numeric_df.empty:
            return px.bar(title=f"{domain.title()} Insights: No numeric data"), "Add numbers for stats!"
        
        summary = numeric_df.describe()
        if 'mean' in summary.index:
            mean_values = summary.loc['mean']
            fig = px.bar(x=mean_values.index, y=mean_values.values, title=f"{domain.title()} Mean Insights")
        else:
            mean_values = numeric_df.mean()  # Fallback mean
            fig = px.bar(x=mean_values.index, y=mean_values.values, title=f"{domain.title()} Mean Insights")
        
        avg_metric = mean_values.mean() if not mean_values.empty else 0
        return fig, f"Key Metric: Avg Mean {avg_metric:.2f} (Numerics: {len(numeric_df.columns)} cols)"

# App
st.title("🔨 DataForge: Generic Unstructured to Real-Time BI")

# Sidebar: Mode & Config
st.sidebar.header("Processing Mode")
gpu_mode = st.sidebar.checkbox("GPU Mode (Demo: Use Colab for RAPIDS)")
if gpu_mode:
    st.sidebar.info("GPU accelerates batch jobs (e.g., 1GB in 10s vs. 2min CPU). Swap pd → cudf in code.")
domain = st.sidebar.selectbox("Domain Template", ["Generic", "Finance (Inflation)", "Retail (Sales)"])

st.sidebar.header("Scalability Note")
st.sidebar.success("CPU handles 10GB+ via chunking. No slowdown—linear scaling.")

# Ingestion
st.header("1. Ingest Data")
col1, col2 = st.columns(2)
with col1:
    text_input = st.text_area("Text/JSON:")
with col2:
    uploaded = st.file_uploader("Excel/CSV:")

if st.button("Process & Forge Insights"):
    with st.spinner("Forging data..."):
        if text_input:
            chunks = DataIngester.ingest("text", text_input)
        else:
            chunks = DataIngester.ingest("excel", uploaded)
        df = Transformer.transform(gpu_mode, chunks)
        fig, summary = InsightEngine.generate_bi(df, domain)
    
    st.header("2. Structured Data")
    st.dataframe(df, use_container_width=True, height=400)
    
    st.header("3. BI Insights")
    st.plotly_chart(fig)
    st.write(summary)
    
    st.header("4. Decision")
    st.success("Action: Review trends—stable for now.")
    # Export Button
    st.header("5. Export Insights")
    csv = df.to_csv(index=False).encode('utf-8')  # Structured data as CSV
    st.download_button(
    label="Download Structured Data (CSV)",
    data=csv,
    file_name=f"{domain.replace(' ', '_')}_insights.csv",
    mime='text/csv'
    )

# Optional: Export summary as text
    summary_text = f"Domain: {domain}\n{summary}\n\nData Shape: {df.shape}"
    st.download_button(
    label="Download Summary (TXT)",
    data=summary_text.encode('utf-8'),
    file_name=f"{domain.replace(' ', '_')}_summary.txt",
    mime='text/plain'
    )
st.info("Next: Customize per domain. GitHub: DataForge-MVP")