# ==========================================
# Streamlit Web UI (Deployment Optimized - No GUI)
# Filename: app.py
# ==========================================
import os
import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt 
# 注意：已移除 stpyvista 匯入以避免 Linux 環境崩潰

from LLM_composite_web_function_PPO_2D import initialize_system, run_LLM

# 1. 頁面基本設定
st.set_page_config(
    page_title="AI Composite Design Assistant",
    page_icon="🧪",
    layout="wide"
)

# 2. 初始化系統
@st.cache_resource
def load_backend():
    try:
        success = initialize_system()
        return success
    except Exception as e:
        st.error(f"System initialization error: {e}")
        return False

system_ready = load_backend()

# 3. 側邊欄
with st.sidebar:
    st.title("⚙️ System Console")
    st.markdown("---")
    if system_ready:
        st.success("✅ System Core Connected")
        st.info(f"🧠 LLM Model: Llama-3 (Ready)")
        st.info(f"🤖 PPO Model: Loaded") 
        st.info(f"🏗️ Physics Engine: Surrogate Model")
    else:
        st.error("❌ System Initialization Failed")
    
    st.markdown("---")
    st.markdown("Created by **NCKU Engineering Science LAiMM Lab**")

# 4. 主畫面
st.title("🧪 AI Smart Composite Design Platform")
st.markdown("Welcome to the Smart Design Assistant. Please enter a command or click an example below:")

col1, col2, col3 = st.columns(3)
button_prompt = None

if col1.button("🔍 Query System Functions", use_container_width=True):
    button_prompt = "Hello, what can you do?"
if col2.button("🚀 Design Optimal Material", use_container_width=True):
    button_prompt = "Design a Glass/Epoxy material with the maximum strain energy density."
if col3.button("📈 Predict Composite with 75° Weave angle", use_container_width=True):
    button_prompt = "Predict the performance of a 75-degree, plain weave Carbon Fiber and Epoxy."

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

final_prompt = button_prompt if button_prompt else st.chat_input("Please enter your design requirements or questions...")

if final_prompt:
    st.chat_message("user").markdown(final_prompt)
    st.session_state.messages.append({"role": "user", "content": final_prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🤖 AI is thinking...")
        
        try:
            start_time = time.time()
            result = run_LLM(final_prompt) 
            end_time = time.time()
            
            if result is None:
                message_placeholder.error("⚠️ Sorry, a system error occurred.")
                response_content = "System error occurred."
            else:
                task_type = result.get('task_type', 'general_chat')
                params = result.get('params', {})
                data = result.get('data')
                fig = result.get('figure')
                plotter_path = result.get('plotter') # 現在這回傳的是圖片路徑
                reply_text = result.get('reply') 
                
                # 工具函數定義
                def format_unit_label(label):
                    if label.startswith('CLTE'): return f"{label} (1/K)"
                    if label in ['E1', 'E2', 'E3', 'G12', 'G13', 'G23']: return f"{label} (Pa)"
                    if '_' in label:
                        parts = label.rsplit('_', 1)
                        name = parts[0].replace('_', ' ')
                        if name.lower() == 'energy density': name = 'strain energy density'
                        unit = 'J/m³' if parts[1] == 'Jm3' else parts[1]
                        return f"{name} ({unit})"
                    return label

                def format_scientific(val):
                    return f"{val:.4e}" if isinstance(val, (int, float)) else val

                if task_type == 'general_chat':
                    final_reply = reply_text if reply_text else "I'm not quite sure."
                    message_placeholder.markdown(final_reply)
                    response_content = final_reply

                elif task_type == 'design':
                    display_tgt = params.get('target', 'Unknown').replace('max_', '').replace('_', ' ')
                    header_text = f"**✅ Task Completed! ({end_time - start_time:.2f}s)**\n\n"
                    message_placeholder.markdown(header_text)
                    st.divider()
                    
                    d_col1, d_col2 = st.columns([1, 1])
                    with d_col1:
                        st.subheader("🚀 Optimization Report")
                        if data and "optimized" in data:
                            init_state, opt_state = data['initial'], data['optimized']
                            df_compare = pd.DataFrame({
                                "Parameter": ["Resin", "Fiber", "Weave", "Angle (°)", "Width (mm)", "Height (mm)", "Score"],
                                "Initial": [init_state['resin'], init_state['fiber'], init_state['weave'], f"{init_state['geo'][0]:.2f}", f"{init_state['geo'][1]:.2f}", f"{init_state['geo'][2]:.2f}", f"{init_state['score']:.4e}"],
                                "AI Optimized": [opt_state['resin'], opt_state['fiber'], opt_state['weave'], f"**{opt_state['geo'][0]:.2f}**", f"{opt_state['geo'][1]:.2f}", f"{opt_state['geo'][2]:.2f}", f"**{opt_state['score']:.4e}**"]
                            })
                            st.table(df_compare)
                    with d_col2:
                        if plotter_path and os.path.exists(plotter_path):
                            st.subheader("🧊 Optimized 3D Structure")
                            st.image(plotter_path, use_container_width=True)
                    
                    response_content = f"Design task completed for {display_tgt}."

                elif task_type == 'prediction':
                    message_placeholder.markdown(f"**✅ Task Completed! ({end_time - start_time:.2f}s)**")
                    st.divider()
                    p_col1, p_col2 = st.columns([1, 1])
                    with p_col1:
                        if data:
                            df_e = pd.DataFrame.from_dict(data['elastic_modulus'], orient='index', columns=['Value'])
                            df_e.index = df_e.index.map(format_unit_label)
                            df_e['Value'] = df_e['Value'].map(format_scientific)
                            st.table(df_e)
                    with p_col2:
                        if fig: st.pyplot(fig)
                        if plotter_path and os.path.exists(plotter_path):
                            st.markdown("#### 3D Woven Preview")
                            st.image(plotter_path, use_container_width=True)
                    
                    response_content = "Performance prediction task completed."

        except Exception as e:
            message_placeholder.error(f"Unexpected error: {e}")
            response_content = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response_content})