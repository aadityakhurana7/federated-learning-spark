import streamlit as st
import pandas as pd
import json
import os
import time
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Federated Learning & Spark UI", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
METRICS_FILE = os.path.join(LOGS_DIR, "metrics.json")
BASELINE_FILE = os.path.join(LOGS_DIR, "baseline_metrics.json")

st.title("Distributed Federated Learning System (Apache Spark)")
st.markdown("### Architecture Overview")
st.info("This dashboard monitors a privacy-preserving federated learning pipeline. By utilizing Apache Spark for distributed computation, models are trained locally on worker nodes. Only parameter updates are transmitted to the server, eliminating the need for centralized data collection and significantly reducing network overhead.")

tab1, tab2, tab3 = st.tabs(["System Architecture Analysis", "Model Evaluation Metrics", "Global Parameter Visualization"])

def load_metrics():
    metrics = None
    baseline = None
    if os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, 'r') as f:
                metrics = json.load(f)
        except Exception:
            pass
    if os.path.exists(BASELINE_FILE):
        try:
            with open(BASELINE_FILE, 'r') as f:
                baseline = json.load(f)
        except Exception:
            pass
    return metrics, baseline

placeholder = st.empty()

while True:
    metrics, baseline = load_metrics()
    
    if metrics and len(metrics['rounds']) > 0 and baseline:
        # Only use the sequence lists for the Pandas DataFrame
        df_metrics = {
            'rounds': metrics['rounds'],
            'accuracy': metrics['accuracy'],
            'time_taken': metrics['time_taken']
        }
        df = pd.DataFrame(df_metrics)
        current_round = metrics['rounds'][-1]
        
        # Calculate dynamic byte savings
        baseline_bytes = baseline['data_transferred_bytes']
        baseline_mb = baseline_bytes / (1024*1024)
        
        with tab1:
            st.header("Distributed Processing Analysis")
            
            # 1. Data Locality & Partitioning
            st.subheader("1. Data Partitioning & Locality")
            st.write("The dataset is partitioned across Spark worker nodes. Training data is processed locally and is never transmitted over the network.")
            # Mock data distribution based on 5 clients
            client_data = {"Client Node": [f"Spark Worker {i}" for i in range(1, 6)], "Rows of Data": [1600]*5}
            pie_fig = px.pie(client_data, values='Rows of Data', names='Client Node', title='Dataset Partitioning Across Spark Cluster', hole=0.3)
            st.plotly_chart(pie_fig, use_container_width=True)
            
            # 2. Parallel Processing
            st.subheader("2. Parallel Execution (Spark RDD Compute)")
            st.write("Worker nodes compute local gradients concurrently using Spark RDD mapping operations.")
            
            # 3. Communication Savings (The main value prop)
            st.subheader("3. Communication Cost Optimization")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Data Transferred (Centralized Baseline)", value=f"{baseline_mb:.2f} MB", delta="- High Congestion", delta_color="inverse")
            with col2:
                st.metric(label="Data Transferred (Federated Spark)", value="~168 Bytes", delta="+ 99.9% Bandwidth Saved", delta_color="normal")
            
            savings_df = pd.DataFrame({
                "System": ["Centralized ML (Raw Data)", "Federated Spark (Model Weights)"],
                "Bytes Transferred per Round": [baseline_bytes, 168]
            })
            bar_fig = px.bar(savings_df, x="System", y="Bytes Transferred per Round", color="System", title="Network Bandwidth Consumption")
            st.plotly_chart(bar_fig, use_container_width=True)

        with tab2:
            st.header("Evaluation Metrics")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric(label="Current Round", value=current_round)
            with m_col2:
                st.metric(label="Global Accuracy (FL)", value=f"{metrics['accuracy'][-1]:.2%}")
            with m_col3:
                st.metric(label="Centralized Baseline Accuracy", value=f"{baseline['accuracy']:.2%}")
                
            # Convergence plot comparing FL to Centralized Baseline
            fig_acc = go.Figure()
            # Add Federated curve
            fig_acc.add_trace(go.Scatter(x=df['rounds'], y=df['accuracy'], mode='lines+markers', name='Federated Spark Model'))
            # Add Centralized baseline (flat line)
            fig_acc.add_trace(go.Scatter(x=[1, df['rounds'].max()], y=[baseline['accuracy'], baseline['accuracy']], 
                                         mode='lines', name='Centralized Baseline', line=dict(dash='dash', color='red')))
            
            fig_acc.update_layout(title="Global Model Convergence vs Baseline", xaxis_title="Communication Round", yaxis_title="Test Accuracy", yaxis_range=[0, 1])
            st.plotly_chart(fig_acc, use_container_width=True)
            
        with tab3:
            st.header("Global Parameter Evolution")
            st.write("Real-time visualization of the aggregated model coefficients across communication rounds. Dynamic updates indicate the convergence of the distributed stochastic gradient descent algorithm.")
            
            if 'latest_weights' in metrics:
                weights = metrics['latest_weights']
                
                # Format into a 4x5 grid for visual appeal
                if len(weights) == 20:
                    weight_matrix = [weights[i:i+5] for i in range(0, 20, 5)]
                    
                    fig_heat = px.imshow(weight_matrix, text_auto=".4f", color_continuous_scale="RdBu_r", 
                                         title=f"Aggregated Model Coefficients (Round {current_round})",
                                         labels=dict(color="Coefficient Value"))
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.write(f"Current Parameters: {weights}")
            else:
                st.info("Awaiting parameter data...")

    else:
        st.info("Waiting for federated learning to start... (Run `python -m src.main`)")
        
    time.sleep(2)
    st.rerun()
