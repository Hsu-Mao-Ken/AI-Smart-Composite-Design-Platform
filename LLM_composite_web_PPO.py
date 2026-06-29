# ==========================================
# Streamlit Web UI (Layout Optimized Final)
# Filename: app.py
# ==========================================
import os
# --- 關鍵修正：強制關閉視窗化功能，改用離屏渲染 ---
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_USE_IPYVTK"] = "false"
# 確保 vtk 也不會嘗試連接 X Server
os.environ["VTK_USE_X"] = "0"


import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt 
from stpyvista import stpyvista 

from LLM_composite_web_function_PPO import initialize_system, run_LLM

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

# --- 快速指令按鈕區 ---
col1, col2, col3 = st.columns(3)
button_prompt = None

if col1.button("🔍 Query System Functions", use_container_width=True):
    button_prompt = "Hello, what can you do?"
if col2.button("🚀 Design Optimal Material", use_container_width=True):
    button_prompt = "Design a Glass/Epoxy material with the maximum strain energy density."
if col3.button("📈 Predict Composite with 75° Weave angle", use_container_width=True):
    button_prompt = "Predict the performance of a 75-degree, plain weave Carbon Fiber and Epoxy."

# 初始化聊天紀錄
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 聊天輸入處理
chat_input_prompt = st.chat_input("Please enter your design requirements or questions...")

# 優先使用按鈕的輸入
final_prompt = button_prompt if button_prompt else chat_input_prompt

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
                plotter = result.get('plotter') 
                reply_text = result.get('reply') 
                
                # 工具函數：格式化標籤與數值
                def format_unit_label(label):
                    if label.startswith('CLTE'): return f"{label} (1/K)"
                    if label in ['E1', 'E2', 'E3', 'G12', 'G13', 'G23']: return f"{label} (Pa)"
                    if '_' in label:
                        parts = label.rsplit('_', 1)
                        name = parts[0].replace('_', ' ')
                        if name.lower() == 'energy density': name = 'strain energy density'
                        unit = 'J/m³' if parts[1] == 'Jm3' else parts[1]
                        return f"{name} ({unit})"
                    if label.lower() == 'energy density': return 'strain energy density'
                    return label

                def format_scientific(val):
                    return f"{val:.4e}" if isinstance(val, (int, float)) else val

                # [Case 1] 一般對話
                if task_type == 'general_chat':
                    final_reply = reply_text if reply_text else "I'm not quite sure."
                    message_placeholder.markdown(final_reply)
                    response_content = final_reply

                # [Case 2] 設計任務
                elif task_type == 'design':
                    display_tgt = params.get('target', 'Unknown').replace('max_', '').replace('_', ' ')
                    header_text = f"**✅ Task Completed! ({end_time - start_time:.2f}s)**\n\n"
                    header_text += f"**Detected Intent:** `DESIGN` | **Target:** `{display_tgt}`"
                    message_placeholder.markdown(header_text)
                    
                    st.divider()
                    d_col1, d_col2 = st.columns([1, 1])
                    
                    with d_col1:
                        st.subheader("🚀 Optimization Report")
                        if data and "optimized" in data:
                            init_state = data['initial']
                            opt_state = data['optimized']
                            df_compare = pd.DataFrame({
                                "Parameter": ["Resin", "Fiber", "Weave", "Angle (°)", "Width (mm)", "Height (mm)", "Score"],
                                "Initial": [init_state['resin'], init_state['fiber'], init_state['weave'], f"{init_state['geo'][0]:.2f}", f"{init_state['geo'][1]:.2f}", f"{init_state['geo'][2]:.2f}", f"{init_state['score']:.4e}"],
                                "AI Optimized": [opt_state['resin'], opt_state['fiber'], opt_state['weave'], f"**{opt_state['geo'][0]:.2f}**", f"{opt_state['geo'][1]:.2f}", f"{opt_state['geo'][2]:.2f}", f"**{opt_state['score']:.4e}**"]
                            })
                            st.table(df_compare)
                            if opt_state['improvement_pct'] > 0:
                                st.success(f"📈 Improvement: +{opt_state['improvement_pct']:.2f}%")
                    
                    with d_col2:
                        if plotter:
                            st.subheader("🧊 Optimized 3D Structure")
                            plotter.window_size = [600, 350]
                            stpyvista(plotter, key=f"pv_design_{int(time.time())}")
                    
                    response_content = f"Design task completed for {display_tgt}."

                # [Case 3] 預測任務
                elif task_type == 'prediction':
                    header_text = f"**✅ Task Completed! ({end_time - start_time:.2f}s)**\n\n"
                    header_text += f"**Detected Intent:** `PREDICTION`"
                    message_placeholder.markdown(header_text)
                    
                    st.divider()
                    p_col1, p_col2 = st.columns([1, 1])
                    
                    with p_col1:
                        st.subheader("📊 Material Properties")
                        
                        if data and "elastic_modulus" in data and "plastic_props" in data:
                            # 處理彈性資料
                            df_e = pd.DataFrame.from_dict(data['elastic_modulus'], orient='index', columns=['Value'])
                            df_e.index = df_e.index.map(format_unit_label)
                            df_e['Value'] = df_e['Value'].map(format_scientific)
                            
                            # 處理塑性資料
                            df_p = pd.DataFrame.from_dict(data['plastic_props'], orient='index', columns=['Value'])
                            df_p.index = df_p.index.map(format_unit_label)
                            df_p['Value'] = df_p['Value'].map(format_scientific)
                            
                            # 建立視覺分隔行 (標題列)
                            df_head_e = pd.DataFrame({"Value": [""]}, index=["[ Elastic Properties ]"])
                            df_head_p = pd.DataFrame({"Value": [""]}, index=["[ Plastic Characteristics ]"])
                            df_space = pd.DataFrame({"Value": [""]}, index=[" "]) # 空白行
                            
                            # 串接成單一 DataFrame 後一次性輸出，保證欄位完美對齊
                            df_combined = pd.concat([df_head_e, df_e, df_space, df_head_p, df_p])
                            st.table(df_combined)

                    with p_col2:
                        st.subheader("📈 Visual Analysis")
                        
                        if fig:
                            st.markdown("#### Stress-Strain Curve")
                            # **[修改核心 1] 調整曲線圖佔用比例，讓它變得更小 (由原本 4:2 改為 5.5:4.5)**
                            # 左邊數字越小，圖表就被擠得越小
                            fig_col, _ = st.columns([6.0, 4.0]) 
                            with fig_col:
                                st.pyplot(fig)
                        
                        if plotter:
                            st.markdown("#### 3D Woven Preview")
                            # **[修改核心 2] 用嵌套欄位限制 3D 圖的寬度，解決拉伸問題**
                            # 用 6:4 的比例限制 3D 圖，右側留白，強制壓縮 3D 圖寬度
                            pv_col, _ = st.columns([6.5, 3.5])
                            with pv_col:
                                plotter.window_size = [400, 300]
                                stpyvista(plotter, key=f"pv_pred_{int(time.time())}")
                    
                    response_content = "Performance prediction task completed."

        except Exception as e:
            message_placeholder.error(f"Unexpected error: {e}")
            response_content = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": response_content})