import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def get_material_params(name, material_type="fiber", verbose=True):
    """
    Find material parameters using keyword containment logic.
    """
    # Pre-processing: 轉小寫、去頭尾空白
    query = str(name).lower().strip()
    
    # --- A. Alias Mapping (關鍵字優先對照表) ---
    # 重要: Python 3.7+ 字典有順序性。
    # 請將「特徵較強」的關鍵字放在前面，「通用」的關鍵字放在後面。
    # 這樣如果輸入 "Optical Epoxy"，會先對到 "Optical" 而對應到 CEL。
    
    alias_map = {
        # === 特殊樹脂 (Specific Resins) ===
        # 對應 CEL-400 (透明/光學)
        "cel": "cel",
        "optical": "cel",
        "transparent": "cel",
        "clear": "cel",
        
        # 對應 Plaskon (模塑/封裝/黑色)
        "plaskon": "plaskon",
        "molding": "plaskon",
        "compound": "plaskon",
        "smt": "plaskon",
        "emc": "plaskon",
        
        # === 通用樹脂 (Generic Resins) ===
        # 對應 ELER (標準環氧) - 放在最後當作預設
        "eler": "eler",
        "epoxy": "eler",
        "resin": "eler",
        
        # === 纖維 (Fibers) ===
        "glass": "e-glass",     # Glass -> E-glass
        "fiberglass": "e-glass",
        "t300": "carbon",       # T300 -> Carbon
        "graphite": "carbon"
    }
    
    # 邏輯修正：改為「檢查是否包含」
    mapped_target = None
    for keyword, target in alias_map.items():
        if keyword in query:
            # print(f"   [Alias] Input '{name}' contains keyword '{keyword}' -> Mapped to '{target}'")
            mapped_target = target
            break # 找到第一個(權重最高的)關鍵字就停止，避免被後面的 generic 覆蓋
            
    # 如果有對應到別名，就用別名去搜尋；否則用原始輸入
    final_query = mapped_target if mapped_target else query

    # Select DataFrame
    if material_type == "fiber":
        df = df_fiber
        col_name = df.columns[0] 
    else:
        df = df_resin
        col_name = df.columns[0]
    
    if df.empty:
        return None

    # --- B. Database Search ---
    # 使用最終決定的查詢詞 (final_query) 去資料庫找
    
    match_row = None
    for idx, row in df.iterrows():
        db_name_str = str(row[col_name]).lower().strip()
        
        # 1. 資料庫名稱 包含 查詢詞 (e.g., query="cel" in db="cel-400")
        if final_query in db_name_str:
            match_row = row
            break
            
        # 2. 查詢詞 包含 資料庫名稱 (e.g., query="carbon t300" contains db="carbon")
        # 避免匹配到太短的字串 (長度>2)
        if len(db_name_str) > 2 and db_name_str in final_query:
            match_row = row
            break

    # --- C. Return Result ---
    if match_row is not None:
        match_name = match_row[col_name]
        if verbose:
            print(f"Search Result [{material_type}]: Found '{match_name}'")
        return match_row.values[1:].astype(float)
    else:
        print(f"Warning: No match found for {material_type} name '{name}' (searched as '{final_query}').")
        return None

def get_weave_pattern(style_name="plain", verbose=True):
    """
    功能: 根據名稱回傳編織矩陣 (0/1)
    輸入 (Input): style_name (str) - plain, twill, satin
    輸出 (Output): numpy array (25,)
    """
    style = style_name.lower().strip()
    
    # 定義 5x5 的樣式 (1 = Weft over Warp, 0 = Warp over Weft)
    patterns = {
        # 平紋 (Plain): 1/1 交錯
        "plain": [
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1
        ],
        
        # 斜紋 (Twill): 2/2 Twill (這也是標準的，對角線位移)
        # 每一行向右移動一格
        "twill": [ 
            1, 1, 0, 0, 0,  # Row 0
            0, 1, 1, 0, 0,  # Row 1 (Shift 1)
            0, 0, 1, 1, 0,  # Row 2 (Shift 1)
            0, 0, 0, 1, 1,  # Row 3 (Shift 1)
            1, 0, 0, 0, 1   # Row 4 (Wrap around) - 註: 5x5 做 2/2 Twill 邊界會剛好接不上，這是幾何限制
        ],
        
        # 緞紋 (Satin): 5-Harness Satin (Counter = 2)
        # 這是紡織學上標準的 5枚緞紋
        "satin": [ 
            0, 0, 1, 0, 0,  # (0,0)
            1, 0, 0, 0, 0,  # (1,2)
            0, 0, 0, 1, 0,  # (2,4)
            0, 1, 0, 0, 0,  # (3,1)
            0, 0, 0, 0, 1   # (4,3)
        ]
    }
    
    if style in patterns:
        if verbose:
            print(f"Weave Style Selected: {style}")
        return np.array(patterns[style], dtype=float)
    else:
        print(f"Warning: Unknown style '{style}'. Defaulting to 'plain'.")
        return np.array(patterns["plain"], dtype=float)

def get_geometry_params(user_geo_dict=None, verbose=True):
    """
    功能: 生成幾何參數向量 (含角度轉換與間距自動計算)
    修正: 
      1. 增加對 None 值的檢查
      2. 修正高度限制為固定的 0.1 ~ 0.4 mm，移除與寬度的連動限制
    """
    # 1. 定義預設值 (Defaults)
    defaults = {
        "angle": 90.0,
        "yarn_width": 0.6,
        "yarn_height": 0.2
    }
    
    # 2. 定義別名對照表
    alias_map = {
        "width": "yarn_width",
        "w": "yarn_width",
        "height": "yarn_height",
        "h": "yarn_height",
        "deg": "angle",
        "degree": "angle"
    }

    # 複製預設值作為起點
    user_input = defaults.copy()
    
    if user_geo_dict:
        # A. 預處理: 轉小寫、處理別名
        clean_dict = {}
        for k, v in user_geo_dict.items():
            k_lower = str(k).lower().strip()
            real_key = alias_map.get(k_lower, k_lower)
            clean_dict[real_key] = v
            
        # B. 更新數值 (檢查 val 是否為 None)
        for key, val in clean_dict.items():
            if key in user_input:
                if val is not None:
                    try:
                        user_input[key] = float(val)
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid value for {key}: {val}. Using default.")
                else:
                    pass

    # --- 3. 數值範圍檢查 (修正重點) ---
    u_angle = np.clip(user_input["angle"], 30.0, 90.0)
    
    # 寬度限制: 0.2 ~ 1.0
    width = np.clip(user_input["yarn_width"], 0.2, 1.0)
    
    # 高度限制: 0.1 ~ 0.4 (獨立限制，不依賴寬度)
    height = np.clip(user_input["yarn_height"], 0.1, 0.4)

    # --- 4. 計算模型參數 ---
    # 模型訓練時使用的是與 90 度的夾角差 (0~60)
    model_angle = abs(u_angle - 90.0)
    
    # 間距計算公式 (維持不變)
    # multiplier = 1.5 + (u_angle - 90) * (-0.025)
    # space = width * multiplier
    multiplier = 1.5 + (u_angle - 90.0) * (-0.025)
    space = width * multiplier
    
    # --- 5. 輸出確認 ---
    if verbose:
        print(f"Geometry Processing:")
        print(f"  - Effective Config: Angle={u_angle}, Width={width}, Height={height}")
        print(f"  - User Angle: {u_angle:.1f}° -> Model Angle: {model_angle:.1f}°")
        print(f"  - Space: {space:.3f} mm (Multiplier: {multiplier:.2f}x)")

    ordered_values = [model_angle, width, height, space]
    return np.array(ordered_values, dtype=float)

class EffectiveModel(nn.Module):
    def __init__(self):
        super().__init__()

        # --- 1. Image Branch (圖像路徑) ---
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        
        self.flatten = nn.Flatten()
        
        # [修正點 1] 這裡原本是 2*2*2=8，但根據錯誤訊息，PyTorch 實際算出來是 18
        # 所以我們直接改成 18 讓它通過
        self.img_dense_block = nn.Sequential(
            nn.Linear(18, 960), 
            nn.ReLU(),
            nn.Linear(960, 480),
            nn.ReLU(),
            nn.Linear(480, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU()
        )

        # --- 2. Info Branch (性質路徑) ---
        self.info_dense_block = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 48),
            nn.ReLU(),
            nn.Linear(48, 36),
            nn.ReLU(),
            nn.Linear(36, 24),
            nn.ReLU()
        )

        # --- 3. Combined Path (結合後路徑) ---
        self.c_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1, padding=0)
        self.c_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=1, padding=1)
        self.c_conv3 = nn.Conv2d(in_channels=16, out_channels=12, kernel_size=2, stride=1, padding=1)
        
        # [修正點 2] 預防下一個錯誤
        # 經過計算，PyTorch 的 padding=1 會讓尺寸稍微變大：
        # Input 12x12 -> c_conv1(valid) -> 11x11
        # 11x11 -> c_conv2(pad=1) -> 12x12
        # 12x12 -> c_conv3(pad=1) -> 13x13
        # 所以最終 Flatten 大小是：12 (channels) * 13 * 13 = 2028
        self.final_dense_block = nn.Sequential(
            nn.Linear(12 * 13 * 13, 1920), 
            nn.ReLU(),
            nn.Linear(1920, 960),
            nn.ReLU(),
            nn.Linear(960, 480),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(480, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, 12)
        )

    def forward(self, img_input, info_input):
        # 轉置圖片維度 (Batch, 5, 5, 1) -> (Batch, 1, 5, 5)
        if img_input.shape[-1] == 1: 
            x1 = img_input.permute(0, 3, 1, 2)
        else:
            x1 = img_input
            
        x1 = self.relu(self.conv1(x1))
        x1 = self.relu(self.conv2(x1))
        x1 = self.dropout1(x1)
        x1 = self.relu(self.conv3(x1))
        x1 = self.relu(self.conv4(x1))
        x1 = self.pool1(x1)
        x1 = self.flatten(x1)
        
        # Debug: 如果未來改參數又報錯，請取消下面這行註解看實際形狀
        # print(f"x1 shape: {x1.shape}")
        
        x1 = self.img_dense_block(x1)

        x2 = self.info_dense_block(info_input)

        combined = torch.cat((x1, x2), dim=1)
        
        # Reshape: 144 -> 12x12 image
        combined = combined.view(-1, 1, 12, 12)

        c = self.relu(self.c_conv1(combined))
        c = self.relu(self.c_conv2(c))
        c = self.relu(self.c_conv3(c))
        c = self.flatten(c)
        
        # Debug: 如果未來改參數又報錯，請取消下面這行註解看實際形狀
        # print(f"c shape: {c.shape}")
        
        output = self.final_dense_block(c)
        
        return output
    

class PlasticModel(nn.Module):
    def __init__(self):
        super(PlasticModel, self).__init__()

        # --- 1. Image Branch ---
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        
        self.flatten = nn.Flatten()
        
        self.img_dense_block = nn.Sequential(
            nn.Linear(18, 960), nn.ReLU(),
            nn.Linear(960, 480), nn.ReLU(),
            nn.Linear(480, 240), nn.ReLU(),
            nn.Linear(240, 120), nn.ReLU()
        )

        # --- 2. Info Branch ---
        self.info_dense_block = nn.Sequential(
            nn.Linear(25, 64), nn.ReLU(),
            nn.Linear(64, 48), nn.ReLU(),
            nn.Linear(48, 36), nn.ReLU(),
            nn.Linear(36, 24), nn.ReLU()
        )

        # --- 3. Combined Path (MLP Fusion) ---
        self.final_dense_block = nn.Sequential(
            # Layer 1: 144 -> 512
            nn.Linear(144, 512),
            nn.ReLU(),
            nn.Dropout(0.1), 
            
            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(),
            
            # Layer 3: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
           
            # Output Layer: 128 -> 40
            nn.Linear(128, 40)
        )

    def forward(self, img_input, info_input):
        # --- Image Branch ---
        if img_input.shape[-1] == 1: 
            x1 = img_input.permute(0, 3, 1, 2)
        else:
            x1 = img_input
            
        x1 = self.relu(self.conv1(x1))
        x1 = self.relu(self.conv2(x1))
        x1 = self.dropout1(x1)
        x1 = self.relu(self.conv3(x1))
        x1 = self.relu(self.conv4(x1))
        x1 = self.pool1(x1)
        x1 = self.flatten(x1)
        
        x1 = self.img_dense_block(x1) 
        
        # --- Info Branch ---
        x2 = self.info_dense_block(info_input)

        # --- Combined Path ---
        combined = torch.cat((x1, x2), dim=1) 
        output = self.final_dense_block(combined)
        
        return output
    
# ==========================================
# 2. 雙模型管理器 (Dual Predictor)
# ==========================================
class DualPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Predictor on {self.device}...")
        
        # 定義檔案路徑 (請確認檔名與您截圖中的一致)
        self.paths = {
            "eff_model": "effective_model_pytorch_half.pth",
            "eff_in": "effective_input_scaler.pkl",
            "eff_out": "effective_output_scaler.pkl",
            "pla_model": "plastic_model_pytorch.pth",
            "pla_in": "plastic_input_scaler.pkl",
            "pla_out": "plastic_output_scaler.pkl"
        }
        
        # --- 載入 Effective System (彈性) ---
        try:
            self.model_eff = EffectiveModel().to(self.device) 
            # 這裡加了 strict=False 是為了容錯，建議最好還是讓架構完全一致
            self.model_eff.load_state_dict(torch.load(self.paths["eff_model"], map_location=self.device))
            self.model_eff.eval()
            self.scaler_eff_in = joblib.load(self.paths["eff_in"])
            self.scaler_eff_out = joblib.load(self.paths["eff_out"])
            print("Effective Model loaded.")
        except Exception as e:
            print(f"Error loading Effective Model: {e}")
            self.model_eff = None

        # --- 載入 Plastic System (塑性) ---
        try:
            self.model_pla = PlasticModel().to(self.device)
            # 載入您剛剛訓練好的權重
            self.model_pla.load_state_dict(torch.load(self.paths["pla_model"], map_location=self.device))
            self.model_pla.eval() # 確保 Dropout 關閉
            self.scaler_pla_in = joblib.load(self.paths["pla_in"])
            self.scaler_pla_out = joblib.load(self.paths["pla_out"])
            print("Plastic Model loaded.")
        except Exception as e:
            print(f"Error loading Plastic Model: {e}")
            self.model_pla = None

    def predict(self, weave_style, geo_dict, resin_name, fiber_name):
        # 1. 獲取參數
        weave_vec = get_weave_pattern(weave_style) 
        geo_vec = get_geometry_params(geo_dict)    
        fiber_vec = get_material_params(fiber_name, "fiber") 
        resin_vec_full = get_material_params(resin_name, "resin")

        if resin_vec_full is None or fiber_vec is None:
            return {"error": "Material not found"}

        results = {}

        # 2. 預測 - 彈性模型
        if self.model_eff:
            # 切片: 取前 3 個 (E, v, CTE)
            resin_vec_eff = resin_vec_full[:3] 
            info_raw = np.concatenate([geo_vec, resin_vec_eff, fiber_vec])
            
            # 預處理
            info_scaled = self.scaler_eff_in.transform(info_raw.reshape(1, -1))
            img_tensor = torch.tensor(weave_vec.reshape(1, 1, 5, 5), dtype=torch.float32).to(self.device)
            info_tensor = torch.tensor(info_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                pred_scaled = self.model_eff(img_tensor, info_tensor)
                pred_final = self.scaler_eff_out.inverse_transform(pred_scaled.cpu().numpy())
            results["elastic"] = pred_final.flatten()

        # 3. 預測 - 塑性模型
        if self.model_pla:
            # 切片: 取全部 (包含塑性參數)
            resin_vec_pla = resin_vec_full[:] 
            info_raw = np.concatenate([geo_vec, resin_vec_pla, fiber_vec])
            
            # 預處理
            info_scaled = self.scaler_pla_in.transform(info_raw.reshape(1, -1))
            img_tensor = torch.tensor(weave_vec.reshape(1, 1, 5, 5), dtype=torch.float32).to(self.device)
            info_tensor = torch.tensor(info_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                pred_scaled = self.model_pla(img_tensor, info_tensor)
                pred_final = self.scaler_pla_out.inverse_transform(pred_scaled.cpu().numpy())
            results["plastic"] = pred_final.flatten()
            
        return results

# ==========================================
# Advanced Helper: Smart Yield Finding
# (專門適配微應變 0.05% 與雙線性數據)
# ==========================================
def find_smart_yield_point(stress, strain):
    """
    自動偵測雙線性轉折點 (Knee Point)。
    優先使用雙線交點法 (Bilinear Intersection)，失敗則退回微量偏差法。
    
    Returns:
        (yield_stress, yield_strain, method_name, debug_info)
    """
    # 資料太少直接跳過
    if len(strain) < 5: 
        return None, None, "Data too short", {}
    
    # --- 方法 A: 雙線性交點法 (Bilinear Intersection) ---
    # 1. 擬合第一段 (彈性段): 取前 20% 點
    n_start = max(3, int(len(strain) * 0.2))
    slope1, intercept1 = np.polyfit(strain[:n_start], stress[:n_start], 1)
    
    # 2. 擬合第二段 (塑性段): 取後 30% 點
    n_end = max(3, int(len(strain) * 0.3))
    slope2, intercept2 = np.polyfit(strain[-n_end:], stress[-n_end:], 1)
    
    debug_info = {
        "slope_elastic": slope1,
        "slope_plastic": slope2,
        "intercept_plastic": intercept2
    }

    # 3. 檢查斜率是否有明顯變化 (避免抓到純直線)
    # 如果前後斜率差異 > 1% 才算有轉折
    if abs(slope1 - slope2) / (abs(slope1) + 1e-9) > 0.01:
        # 解聯立方程式: m1*x + c1 = m2*x + c2
        # x = (c2 - c1) / (m1 - m2)
        yield_strain = (intercept2 - intercept1) / (slope1 - slope2)
        yield_stress = slope1 * yield_strain + intercept1
        
        # 檢查交點是否在合理範圍 (允許稍微超出模擬邊界)
        if 0 <= yield_strain <= np.max(strain) * 1.5:
            return yield_stress, yield_strain, "Bilinear Intersection", debug_info

    # --- 方法 B: 微量偏差法 (Micro-Offset 0.002%) ---
    # 備案：如果交點法失敗，嘗試用極小的 offset
    micro_offset = 0.00002 # 0.002%
    
    offset_line = slope1 * (strain - micro_offset) + intercept1
    diff = stress - offset_line
    idx = np.where(diff < 0)[0]
    
    if len(idx) > 0:
        cross_idx = idx[0]
        return stress[cross_idx], strain[cross_idx], "0.002% Offset", debug_info

    return None, None, "No Yield Found (Linear)", debug_info


# ==========================================
# Main Function: Composite Evaluation
# (Modified to return Figure object for Web UI)
# ==========================================
def evaluate_composite(weave_style, geo_dict, resin_name, fiber_name, show_plot=True, verbose=True):
    """
    接收設計參數，執行預測，計算物理性質。
    回傳: (metrics, fig)
      - metrics: 物理性質字典
      - fig: Matplotlib Figure 物件 (若 show_plot=False 則為 None)
    """
    
    # 檢查預測器
    if 'predictor' not in globals():
        print("[ERROR] 'predictor' is not initialized.")
        return None, None

    # 1. 執行預測
    if verbose:
        print(f"Running Analysis: {weave_style} / {geo_dict} / {resin_name} / {fiber_name}")
    
    results = predictor.predict(weave_style, geo_dict, resin_name, fiber_name)
    
    if "error" in results:
        if verbose: print(f"[ERROR] Prediction failed: {results['error']}")
        return None, None

    # 準備回傳結構
    metrics = {
        "elastic_modulus": {},   
        "plastic_props": {}      
    }

    # 2. 處理彈性性質 (Elastic)
    if "elastic" in results:
        vals = results["elastic"]
        labels = ["E1", "E2", "E3", "G12", "G23", "G13", "v12", "v23", "v13", "CLTE1", "CLTE2", "CLTE3"]
        
        for l, v in zip(labels, vals):
            metrics["elastic_modulus"][l] = v

        if verbose:
            print("\n[Result 1] Linear Elastic Properties:")
            print("-" * 55)
            print(f"{'Property':<10} | {'Value':<15} | {'Unit':<5}")
            print("-" * 55)
            for l, v in zip(labels, vals):
                if l.startswith("E") or l.startswith("G"):
                    print(f"{l:<10} | {v/1e9:.4f}{'':<9} | {'GPa':<5}")
                elif l.startswith("CLTE"):
                    print(f"{l:<10} | {v:.2e}{'':<9} | {'1/K':<5}")
                else:
                    print(f"{l:<10} | {v:.4f}{'':<9} | {'-':<5}")
            print("-" * 55)

    # 3. 處理塑性行為 (Plastic)
    fig = None # 預設回傳 None
    
    if "plastic" in results:
        p_vals = results["plastic"]
        stress = p_vals[0::2]
        strain = p_vals[1::2]
        
        # --- 使用智慧搜尋找降伏點 ---
        y_str, y_eps, method_name, debug = find_smart_yield_point(stress, strain)
        
        # 計算其他指標
        slope_elastic = (stress[1] - stress[0]) / (strain[1] - strain[0]) if len(strain)>1 else 0
        slope_plastic = (stress[-1] - stress[-2]) / (strain[-1] - strain[-2]) if len(strain)>1 else 0
        
        if hasattr(np, 'trapezoid'):
            energy_density = np.trapezoid(stress, strain)
        else:
            energy_density = np.trapz(stress, strain)

        # 存入 metrics
        metrics["plastic_props"] = {
            "slope_elastic_Pa": slope_elastic,
            "slope_plastic_Pa": slope_plastic,
            "energy_density_Jm3": energy_density,
            "yield_strength_Pa": y_str if y_str else 0.0,
            "yield_strain": y_eps if y_eps else 0.0
        }

        # 顯示結果
        if verbose:
            print("\n[Result 2] Non-linear Plastic Behavior (Micro-Strain):")
            print("-" * 55)
            print(f"  > Initial Modulus     : {slope_elastic / 1e9:.4f} GPa")
            print(f"  > Plastic Modulus     : {slope_plastic / 1e9:.4f} GPa")
            print(f"  > Strain Energy       : {energy_density:.4f} J/m^3")
            
            if y_str:
                print(f"  > Yield Strength      : {y_str / 1e6:.4f} MPa")
            else:
                print(f"  > Yield Strength      : Not Found (Linear within 0.05%)")
            print("-" * 55)

        # 4. 繪圖邏輯 (修改為適合網頁排版的原生大小)
        if show_plot:
            # [修改 1] 直接在生成時指定較小的比例 (例如 5.5 x 3.5)，讓原生的文字與圖表比例最完美
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            
            # 1. 主曲線
            ax.plot(strain, stress, 'r-o', markersize=4, linewidth=2, label=f"{resin_name}/{fiber_name}")
            
            # 2. 標記降伏點與特徵線
            if y_str and y_eps:
                ax.plot(y_eps, y_str, 'b*', markersize=12, label='Yield Point', zorder=5)
                
                # 如果是交點法，畫出兩條延伸線
                if "Intersection" in method_name and debug:
                    x_range = np.linspace(0, np.max(strain)*1.1, 100)
                    
                    # 綠色虛線: 彈性延伸
                    y_elas = debug["slope_elastic"] * x_range 
                    ax.plot(x_range, y_elas, 'g--', alpha=0.4, linewidth=1, label="Elastic Ext.")
                    
                    # 藍色虛線: 塑性延伸
                    y_plas = debug["slope_plastic"] * x_range + debug["intercept_plastic"]
                    ax.plot(x_range, y_plas, 'b--', alpha=0.4, linewidth=1, label="Plastic Ext.")

            # [修改 2] 縮小標題與標籤的字體大小，配合縮小後的畫布
            ax.set_title(f"Predicted Stress-Strain ({weave_style})", fontsize=11)
            ax.set_xlabel("Strain (-)", fontsize=9)
            ax.set_ylabel("Stress (Pa)", fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # [修改 3] 將圖例縮小，避免擋住曲線
            ax.legend(fontsize=8, loc='best')
            
            # [修改 4] 加入 tight_layout 自動切除多餘白邊
            fig.tight_layout()
            
            # [修改 5] 移除 plt.show() 與 plt.close(fig) 的邏輯。
            # 因為要將 fig 傳遞給 Streamlit 渲染，絕對不能在這裡 close 掉，否則會破壞物件。

    return metrics, fig

# ==========================================
# 2. 模型管理器 (Predictor)
# ==========================================
class PlasticPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing Predictor on {self.device}...")
        
        # 定義檔案路徑 (請確認檔名與您截圖中的一致)
        self.paths = {
            "pla_model": "plastic_model_pytorch.pth",
            "pla_in": "plastic_input_scaler.pkl",
            "pla_out": "plastic_output_scaler.pkl"
        }

        # --- 載入 Plastic System (塑性) ---
        try:
            self.model_pla = PlasticModel().to(self.device)
            # 載入您剛剛訓練好的權重
            self.model_pla.load_state_dict(torch.load(self.paths["pla_model"], map_location=self.device))
            self.model_pla.eval() # 確保 Dropout 關閉
            self.scaler_pla_in = joblib.load(self.paths["pla_in"])
            self.scaler_pla_out = joblib.load(self.paths["pla_out"])
            print("Plastic Model loaded.")
        except Exception as e:
            print(f"Error loading Plastic Model: {e}")
            self.model_pla = None

    def predict(self, weave_style, geo_dict, resin_name, fiber_name):
        # 1. 獲取參數
        weave_vec = get_weave_pattern(weave_style, verbose=False) 
        geo_vec = get_geometry_params(geo_dict, verbose=False)    
        fiber_vec = get_material_params(fiber_name, "fiber", verbose=False) 
        resin_vec_full = get_material_params(resin_name, "resin", verbose=False)

        if resin_vec_full is None or fiber_vec is None:
            return {"error": "Material not found"}

        results = {}

        
        # 2. 預測 - 塑性模型
        if self.model_pla:
            # 切片: 取全部 (包含塑性參數)
            resin_vec_pla = resin_vec_full[:] 
            info_raw = np.concatenate([geo_vec, resin_vec_pla, fiber_vec])
            
            # 預處理
            info_scaled = self.scaler_pla_in.transform(info_raw.reshape(1, -1))
            img_tensor = torch.tensor(weave_vec.reshape(1, 1, 5, 5), dtype=torch.float32).to(self.device)
            info_tensor = torch.tensor(info_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                pred_scaled = self.model_pla(img_tensor, info_tensor)
                pred_final = self.scaler_pla_out.inverse_transform(pred_scaled.cpu().numpy())
            results["plastic"] = pred_final.flatten()
            
        return results

class CompositeEnvPPO:
    def __init__(self, predictor, df_resin, df_fiber):
        self.predictor = predictor
        
        # --- 1. 定義選項清單 ---
        self.weave_options = ["plain", "twill", "satin"]
        self.resin_options = df_resin.iloc[:, 0].astype(str).tolist()
        self.fiber_options = df_fiber.iloc[:, 0].astype(str).tolist()
        
        self.resin_props_matrix = df_resin.iloc[:, 1:].astype(float).values
        self.fiber_props_matrix = df_fiber.iloc[:, 1:].astype(float).values
        
        print(f"PPO Env Initialized: {len(self.resin_options)} resins, {len(self.fiber_options)} fibers.")

        # --- 2. 動作空間定義 (PPO 混合動作概念) ---
        # 注意：我們不需要 Gym 的 Action Space，這裡純粹記錄維度供 Agent 參考
        self.action_dims = {
            "weave": len(self.weave_options), # 離散 (Discrete)
            "resin": len(self.resin_options), # 離散 (Discrete)
            "fiber": len(self.fiber_options), # 離散 (Discrete)
            "geo": 3                          # 連續 (Continuous: Angle, Width, Height)
        }

        # --- 3. 狀態空間維度 ---
        self.n_resin_props = self.resin_props_matrix.shape[1]
        self.n_fiber_props = self.fiber_props_matrix.shape[1]
        
        # 狀態順序：Target(3) + Weave_OneHot(3) + Geo(4) + Resin(Props+OneHot) + Fiber(Props+OneHot)
        self.state_dim = (3 + len(self.weave_options) + 4 + 
                          (self.n_resin_props + len(self.resin_options)) + 
                          (self.n_fiber_props + len(self.fiber_options)))
        
        print(f"State Dimension: {self.state_dim}")

        # --- 4. 幾何參數真實邊界 ---
        self.geo_min = np.array([30.0, 0.2, 0.1])
        self.geo_max = np.array([90.0, 1.0, 0.4])
        
        # --- 5. 目標縮放尺度 (Reward Scaling) ---
        self.target_scales = {
            0: 1e-3,   # Energy
            1: 1e-10,  # Stiffness 
            2: 1e-6    # Yield
        }

        # --- 6. 內部變數 ---
        self.cur_weave_idx = 0
        self.cur_resin_idx = 0
        self.cur_fiber_idx = 0
        
        # PPO 專用：內部儲存 [-1, 1] 的正規化幾何數值
        self.cur_geo_norm = np.zeros(3) 
        self.cur_geo_real = np.zeros(3) 
        
        self.current_target = 0 
        self.steps = 0
        self.max_steps = 30 # PPO 單局不用太長
        self.best_score_episode = 0 

    def reset(self, target_type="max_energy"):
        # 1. 隨機初始離散選項
        self.cur_weave_idx = random.randint(0, len(self.weave_options)-1)
        self.cur_resin_idx = random.randint(0, len(self.resin_options)-1)
        self.cur_fiber_idx = random.randint(0, len(self.fiber_options)-1)
        
        # 2. 隨機初始連續幾何 (在 [-1, 1] 之間)
        self.cur_geo_norm = np.random.uniform(-1.0, 1.0, size=(3,))
        self._update_real_geo() # 計算真實物理幾何數值
        
        # 3. 設定目標
        target_map = {"max_energy": 0, "max_stiffness": 1, "max_yield": 2}
        self.current_target = target_map.get(target_type, 0)
        
        self.steps = 0
        
        # 4. 初始化分數基準
        raw_score = self._calculate_physics_score()
        scale = self.target_scales.get(self.current_target, 1.0)
        
        self.prev_score_norm = raw_score * scale
        self.best_score_episode = self.prev_score_norm
        
        return self._get_observation()

    def step(self, action_dict):
        """
        接收 PPO Agent 傳來的混合動作字典
        action_dict = {
            'weave': int,
            'resin': int,
            'fiber': int,
            'geo': np.array([angle_norm, width_norm, height_norm]) # range: [-1, 1]
        }
        """
        self.steps += 1
        
        # --- 1. 執行動作 (直接覆蓋為絕對值) ---
        self.cur_weave_idx = int(action_dict['weave'])
        self.cur_resin_idx = int(action_dict['resin'])
        self.cur_fiber_idx = int(action_dict['fiber'])
        
        # 裁剪幾何數值確保在 [-1, 1] 內
        self.cur_geo_norm = np.clip(action_dict['geo'], -1.0, 1.0)
        self._update_real_geo() # 將 [-1,1] 轉回真實的物理維度
        
        # --- 2. 呼叫預測模型 ---
        raw_score = self._calculate_physics_score()
        
        # --- 3. PPO 獎勵計算 (純絕對分數版) ---
        scale = self.target_scales.get(self.current_target, 1.0)
        norm_score = raw_score * scale
        
        # 1. 核心邏輯：獎勵就是當前表現的絕對分數！
        # 表現越好，每一步拿到的分就越高。模型為了總分最大化，會盡快找到最高分並保持住。
        reward = norm_score
        
        # 2. 失敗懲罰
        # 如果物理計算失敗或分數極低，給予明確懲罰
        if raw_score <= 1e-9:
            reward = -5.0 
            
        # 3. 數值裁剪 (Clipping)
        # 確保極端情況下不會把神經網路的梯度算爆
        reward = np.clip(reward, -5.0, 10.0)

        self.prev_score_norm = norm_score
        done = self.steps >= self.max_steps
        
        # [關鍵] 把 Raw Score 放在 Info 裡傳出去，供 Log 與繪圖使用
        info = {
            "raw_score": raw_score,
            "real_geo": self.cur_geo_real.copy()
        }
        
        return self._get_observation(), reward, done, info

    def _update_real_geo(self):
        """將 [-1, 1] 的神經網路輸出，線性映射回真實物理範圍"""
        self.cur_geo_real = self.geo_min + 0.5 * (self.cur_geo_norm + 1.0) * (self.geo_max - self.geo_min)

    def _get_observation(self):
        # 1. Target (One-Hot)
        target_vec = np.zeros(3)
        target_vec[self.current_target] = 1.0
        
        # 2. Weave (One-Hot)
        s_weave_oh = np.zeros(len(self.weave_options))
        s_weave_oh[self.cur_weave_idx] = 1.0
        
        # 3. Geometry (包含 Space 計算，維持神經網路友善範圍)
        angle_real, width_real, height_real = self.cur_geo_real
        multiplier = 1.5 + (angle_real - 90.0) * (-0.025)
        space_real = width_real * multiplier
        
        # 將 space 也稍微縮小一點放進狀態裡，或直接用 norm 值
        s_geo = np.array([
            self.cur_geo_norm[0], 
            self.cur_geo_norm[1], 
            self.cur_geo_norm[2], 
            space_real / 2.0 # 簡單除以 2 讓數值大概落在 0~1 之間
        ])
        
        # 4. Resin (Properties + One-Hot)
        s_resin_props = np.log10(self.resin_props_matrix[self.cur_resin_idx] + 1e-9)
        s_resin_oh = np.zeros(len(self.resin_options))
        s_resin_oh[self.cur_resin_idx] = 1.0
        
        # 5. Fiber (Properties + One-Hot)
        s_fiber_props = np.log10(self.fiber_props_matrix[self.cur_fiber_idx] + 1e-9)
        s_fiber_oh = np.zeros(len(self.fiber_options))
        s_fiber_oh[self.cur_fiber_idx] = 1.0
        
        # 依序串接：Target -> Weave -> Geo -> Resin -> Fiber
        state = np.concatenate([
            target_vec,
            s_weave_oh,
            s_geo,
            s_resin_props, s_resin_oh,
            s_fiber_props, s_fiber_oh
        ])
        return state.astype(np.float32)

    def _calculate_physics_score(self):
        weave_name = self.weave_options[self.cur_weave_idx]
        resin_name = self.resin_options[self.cur_resin_idx]
        fiber_name = self.fiber_options[self.cur_fiber_idx]
        
        # PPO 必須使用還原後的真實數值去跑模擬
        angle, width, height = self.cur_geo_real
        
        geo_dict = {
            "angle": angle,
            "width": width,
            "height": height
        }
        
        result = self.predictor.predict(weave_name, geo_dict, resin_name, fiber_name)
        
        if "error" in result or "plastic" not in result:
            return 0.0
            
        plastic_data = result["plastic"]
        stress = plastic_data[0::2]
        strain = plastic_data[1::2]
        
        val = 0.0
        
        if self.current_target == 0: # Max Energy
            val = np.trapezoid(stress, strain)
        elif self.current_target == 1: # Max Stiffness
            if len(strain) > 1:
                val = (stress[1] - stress[0]) / (strain[1] - strain[0])
        elif self.current_target == 2: # Max Yield
            try:
                y_str, _, _, _ = find_smart_yield_point(stress, strain) # 確保外部有此函數
                if y_str: val = y_str
            except:
                val = 0.0
                
        return val
    
from torch.distributions import Categorical, Normal

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dims):
        super().__init__()
        
        # 提取動作維度資訊
        self.n_weave = action_dims["weave"]
        self.n_resin = action_dims["resin"]
        self.n_fiber = action_dims["fiber"]
        self.n_geo = action_dims["geo"]
        
        # ==========================================
        # 1. Critic Network (評論家：評估狀態價值)
        # ==========================================
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1) # 輸出單一 Value
        )
        
        # ==========================================
        # 2. Actor Network (演員：特徵提取層)
        # ==========================================
        # 為了避免連續與離散動作互相干擾，Actor 使用獨立的特徵提取
        self.actor_feature = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        
        # ==========================================
        # 3. Actor Output Heads (演員：多頭輸出)
        # ==========================================
        # 離散動作頭 (輸出 Logits)
        self.head_weave = nn.Linear(256, self.n_weave)
        self.head_resin = nn.Linear(256, self.n_resin)
        self.head_fiber = nn.Linear(256, self.n_fiber)
        
        # 連續動作頭 (幾何參數)
        # Mean: 使用 Tanh 確保輸出範圍在 [-1, 1]
        self.head_geo_mean = nn.Sequential(
            nn.Linear(256, self.n_geo),
            nn.Tanh() 
        )
        # Log_Std: 標準差設定為獨立的可訓練參數 (不依賴 State，這在 PPO 中更穩定)
        self.geo_log_std = nn.Parameter(torch.zeros(self.n_geo))

    def forward(self):
        # PPO 通常不直接呼叫 forward，而是拆分成 act 和 evaluate
        raise NotImplementedError
        
    def act(self, state):
        """
        環境互動階段使用：根據狀態抽出動作，並計算該動作的對數機率 (Log Prob)。
        """
        # 1. 提取特徵
        actor_features = self.actor_feature(state)
        
        # 2. 離散動作分佈 (Categorical)
        dist_weave = Categorical(logits=self.head_weave(actor_features))
        dist_resin = Categorical(logits=self.head_resin(actor_features))
        dist_fiber = Categorical(logits=self.head_fiber(actor_features))
        
        # 3. 連續動作分佈 (Normal 高斯分佈)
        geo_mean = self.head_geo_mean(actor_features)
        geo_std = self.geo_log_std.exp().expand_as(geo_mean) # 將 log_std 轉回標準差
        dist_geo = Normal(geo_mean, geo_std)
        
        # 4. 抽樣動作 (Sampling)
        action_weave = dist_weave.sample()
        action_resin = dist_resin.sample()
        action_fiber = dist_fiber.sample()
        action_geo = dist_geo.sample()
        
        # 5. 計算 Log Probabilities (用於 PPO Loss)
        # 連續動作的 log_prob 會有 3 個值，我們需要把它們 sum 起來代表這個幾何組合的總機率
        action_logprob = (dist_weave.log_prob(action_weave) + 
                          dist_resin.log_prob(action_resin) + 
                          dist_fiber.log_prob(action_fiber) + 
                          dist_geo.log_prob(action_geo).sum(dim=-1))
                          
        # 6. 計算 Value
        state_value = self.critic(state)
        
        # 將動作打包成字典回傳給環境
        action_dict = {
            'weave': action_weave.item() if state.dim() == 1 else action_weave.cpu().numpy(),
            'resin': action_resin.item() if state.dim() == 1 else action_resin.cpu().numpy(),
            'fiber': action_fiber.item() if state.dim() == 1 else action_fiber.cpu().numpy(),
            'geo': action_geo.detach().cpu().numpy() if state.dim() == 1 else action_geo.detach().cpu().numpy()
        }
        
        # 回傳：動作字典, 展平的動作張量(存記憶體用), Log機率, 價值
        action_tensor = torch.cat([
            action_weave.unsqueeze(-1).float(), 
            action_resin.unsqueeze(-1).float(), 
            action_fiber.unsqueeze(-1).float(), 
            action_geo
        ], dim=-1)
        
        return action_dict, action_tensor, action_logprob.detach(), state_value.detach()

    def evaluate(self, state, action_tensor):
        """
        神經網路更新階段使用：計算給定動作的 Log Prob、狀態價值、以及資訊熵 (Entropy)。
        """
        actor_features = self.actor_feature(state)
        
        # 重建分佈
        dist_weave = Categorical(logits=self.head_weave(actor_features))
        dist_resin = Categorical(logits=self.head_resin(actor_features))
        dist_fiber = Categorical(logits=self.head_fiber(actor_features))
        
        geo_mean = self.head_geo_mean(actor_features)
        geo_std = self.geo_log_std.exp().expand_as(geo_mean)
        dist_geo = Normal(geo_mean, geo_std)
        
        # 解析傳入的 action_tensor [Weave, Resin, Fiber, Geo(3)]
        action_weave = action_tensor[:, 0]
        action_resin = action_tensor[:, 1]
        action_fiber = action_tensor[:, 2]
        action_geo = action_tensor[:, 3:6]
        
        # 計算 Log Prob
        action_logprobs = (dist_weave.log_prob(action_weave) + 
                           dist_resin.log_prob(action_resin) + 
                           dist_fiber.log_prob(action_fiber) + 
                           dist_geo.log_prob(action_geo).sum(dim=-1))
                           
        # 計算 Entropy (鼓勵探索)
        dist_entropy = (dist_weave.entropy() + 
                        dist_resin.entropy() + 
                        dist_fiber.entropy() + 
                        dist_geo.entropy().sum(dim=-1))
                        
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    
    
# ==========================================
# 2. Updated PPO Optimization Helper (為 LLM 串接設計)
# ==========================================
def optimize_composite(
    target_type,       # e.g. "max_stiffness"
    weave_style,       # e.g. "plain"
    geo_dict,          # e.g. {"angle": 45, "width": 0.5, "height": 0.2}
    resin_name,        # e.g. "Epoxy"
    fiber_name,        # e.g. "Carbon"
    model_path="ppo_best_model.pth", 
    max_steps=3,       # PPO 給絕對值，1步即達最佳解，預設跑3步確認穩定性
    verbose=True       
):
    """
    接收手動定義的初始參數，使用 PPO 直接給出該目標下的最佳設計。
    """
    
    # 0. 檢查環境依賴
    if 'env' not in globals():
        print("[ERROR] 'env' (CompositeEnvPPO) is not initialized in globals().")
        return None
    if 'PPOActorCritic' not in globals():
        print("[ERROR] 'PPOActorCritic' class is not defined.")
        return None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 內部 Helper: 搜尋邏輯 (與 get_material_params 保持一致) ---
    def find_index_smartly(user_input, option_list, material_type="fiber"):
        if user_input in option_list:
            return option_list.index(user_input)
            
        query = str(user_input).lower().strip()
        
        alias_map = {
            "cel": "cel", "optical": "cel", "transparent": "cel", "clear": "cel",
            "plaskon": "plaskon", "molding": "plaskon", "compound": "plaskon", "smt": "plaskon", "emc": "plaskon",
            "eler": "eler", "epoxy": "eler", "resin": "eler",
            "glass": "e-glass", "fiberglass": "e-glass",
            "t300": "carbon", "graphite": "carbon"
        }
        
        mapped_target = None
        for keyword, target in alias_map.items():
            if keyword in query:
                mapped_target = target
                break
        final_query = mapped_target if mapped_target else query
        
        for idx, opt_name in enumerate(option_list):
            db_name_str = str(opt_name).lower().strip()
            if final_query in db_name_str:
                return idx
            if len(db_name_str) > 2 and db_name_str in final_query:
                return idx
        return None
    # -------------------------------------------------------------

    if verbose:
        print(f"\n{'='*70}")
        print(f"[AI Optimization (PPO)] Target: {target_type}")
        print(f"User Input          : {resin_name} + {fiber_name} ({weave_style})")
        print(f"{'='*70}\n")

    # 1. 載入 PPO 模型 (Actor-Critic)
    policy = PPOActorCritic(env.state_dim, env.action_dims).to(device)
    
    if not os.path.exists(model_path):
        print(f"[Warning] {model_path} not found. Trying final model...")
        model_path = "ppo_final_model.pth"
        
    if os.path.exists(model_path):
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval() # 評估模式，鎖定 Dropout/BatchNorm
    else:
        print("[ERROR] No model file found. Cannot optimize.")
        return None

    # 2. 強制設定環境至「使用者指定的初始狀態」
    state = env.reset(target_type=target_type)
    
    # (A) 設定編織
    if weave_style in env.weave_options:
        env.cur_weave_idx = env.weave_options.index(weave_style)
    else:
        print(f"[Warning] Weave '{weave_style}' not found. Using default.")

    # (B) 設定樹脂
    r_idx = find_index_smartly(resin_name, env.resin_options, "resin")
    if r_idx is not None:
        env.cur_resin_idx = r_idx
    real_resin_name = env.resin_options[env.cur_resin_idx]

    # (C) 設定纖維
    f_idx = find_index_smartly(fiber_name, env.fiber_options, "fiber")
    if f_idx is not None:
        env.cur_fiber_idx = f_idx
    real_fiber_name = env.fiber_options[env.cur_fiber_idx]

    if verbose:
        print(f"Mapped Material: {real_resin_name} + {real_fiber_name}")

    # (D) 設定幾何 (PPO 環境需要同時處理 Real 數值與 Normal 數值)
    g_angle = geo_dict.get("angle")
    g_width = geo_dict.get("width")
    g_height = geo_dict.get("height")

    # 取值或使用預設
    angle_val = float(g_angle) if g_angle is not None else 60.0
    width_val = float(g_width) if g_width is not None else 0.5
    height_val = float(g_height) if g_height is not None else 0.2
    
    # 寫入真實幾何並 Clip
    env.cur_geo_real = np.array([angle_val, width_val, height_val])
    env.cur_geo_real = np.clip(env.cur_geo_real, env.geo_min, env.geo_max)
    
    # [關鍵轉換] 將真實物理數值反向映射回 [-1, 1]，供神經網路讀取
    env.cur_geo_norm = 2.0 * (env.cur_geo_real - env.geo_min) / (env.geo_max - env.geo_min) - 1.0

    # (E) 更新 State Vector 與 Initial Score
    state = env._get_observation()
    initial_score = env._calculate_physics_score()
    
    metrics = {
        "target": target_type,
        "initial": {
            "weave": env.weave_options[env.cur_weave_idx],
            "resin": env.resin_options[env.cur_resin_idx],
            "fiber": env.fiber_options[env.cur_fiber_idx],
            "geo": env.cur_geo_real.copy(),
            "score": initial_score
        }
    }

    if verbose:
        print(f"{'Step':<4} | {'AI Decision (Absolute Values)':<35} | {'Score':<10} | {'Status'}")
        print("-" * 70)

    # 3. 開始 PPO 優化迴圈
    final_score = initial_score
    
    for t in range(max_steps):
        state_tensor = torch.unsqueeze(torch.FloatTensor(state), 0).to(device)
        
        # 使用確定性策略 (取 Argmax 與 Mean)
        with torch.no_grad():
            actor_features = policy.actor_feature(state_tensor)
            
            w_idx = torch.argmax(policy.head_weave(actor_features), dim=1).item()
            r_idx = torch.argmax(policy.head_resin(actor_features), dim=1).item()
            f_idx = torch.argmax(policy.head_fiber(actor_features), dim=1).item()
            geo_norm = policy.head_geo_mean(actor_features).squeeze(0).cpu().numpy()
            
        action_dict = {
            'weave': w_idx,
            'resin': r_idx,
            'fiber': f_idx,
            'geo': geo_norm
        }
        
        prev_score = env._calculate_physics_score()
        next_state, reward, done, info = env.step(action_dict)
        curr_score = info['raw_score']
        
        if verbose:
            real_geo = info['real_geo']
            act_str = f"W:{env.weave_options[w_idx][:3].upper()} | Geo:[{real_geo[0]:.1f}, {real_geo[1]:.2f}, {real_geo[2]:.2f}]"
            change = "(+)" if curr_score > prev_score else ("(-)" if curr_score < prev_score else "(=)")
            print(f"{t+1:<4} | {act_str:<35} | {curr_score:.4e} | {change}")

        state = next_state
        final_score = curr_score
        if done: break

    # 4. 整理結果與詳細報告
    improvement_pct = ((final_score - initial_score) / (initial_score + 1e-9)) * 100
    
    metrics["optimized"] = {
        "weave": env.weave_options[env.cur_weave_idx],
        "resin": env.resin_options[env.cur_resin_idx],
        "fiber": env.fiber_options[env.cur_fiber_idx],
        "geo": env.cur_geo_real.copy(), # 回傳真實數值
        "score": final_score,
        "improvement_pct": improvement_pct
    }
    
    print("-" * 70)
    print(f"[PPO Optimization Report] Target: {target_type}")
    print(f"{'Parameter':<15} | {'Initial State':<22} | {'Optimized State':<22}")
    print("-" * 70)
    
    print(f"{'Weave':<15} | {metrics['initial']['weave']:<22} | {metrics['optimized']['weave']:<22}")
    print(f"{'Resin':<15} | {metrics['initial']['resin']:<22} | {metrics['optimized']['resin']:<22}")
    print(f"{'Fiber':<15} | {metrics['initial']['fiber']:<22} | {metrics['optimized']['fiber']:<22}")
    print(f"{'Angle (deg)':<15} | {metrics['initial']['geo'][0]:<22.2f} | {metrics['optimized']['geo'][0]:<22.2f}")
    print(f"{'Width (mm)':<15} | {metrics['initial']['geo'][1]:<22.2f} | {metrics['optimized']['geo'][1]:<22.2f}")
    print(f"{'Height (mm)':<15} | {metrics['initial']['geo'][2]:<22.2f} | {metrics['optimized']['geo'][2]:<22.2f}")
    
    print("-" * 70)
    print(f"{'Score':<15} | {initial_score:<22.4e} | {final_score:<22.4e}")
    print(f"{'Improvement':<15} | {'--':<22} | {improvement_pct:+.2f}%")
    print("=" * 70)

    return metrics

# Cell 5: Load Llama-3.1 Agent (Standard Float16 Mode)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==========================================
# PLEASE PASTE YOUR HUGGING FACE TOKEN HERE
# ==========================================
# 透過 Streamlit 的 secrets 功能讀取金鑰
import streamlit as st
YOUR_HF_TOKEN = st.secrets["HF_TOKEN"]

class LLMAgent:
    def __init__(self, hf_token):
        # Using Llama 3.1
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing LLM Agent ({self.model_id})...")
        print(f"Device detected: {self.device}")

        try:
            # 1. Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                token=hf_token
            )
            
            # 2. Load Model (Standard Float16)
            print("Loading model in Float16 mode (Requires ~16GB VRAM/RAM)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float16, 
                device_map="auto",
                token=hf_token
            )
            
            # 3. Create Pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512, # [修改] 增加長度以容納對話回應
                temperature=0.1,
                do_sample=True
            )
            print("Success: Llama-3.1 Agent loaded.")
            
        except Exception as e:
            print(f"Error loading LLM: {e}")
            print("Please check your Hugging Face token.")
            self.pipe = None

    # [修改] 增加 resin_options 和 fiber_options 參數
    def parse_instruction(self, user_text, resin_options=[], fiber_options=[]):
        """
        Convert natural language to JSON string.
        Enhanced to support General Chat and Material Awareness.
        """
        if not self.pipe:
            return None
            
        # 1. 準備上下文資訊 (將材料轉為字串)
        # 取前 20 個避免 Prompt 太長
        resin_str = ", ".join(str(r) for r in resin_options[:20]) if resin_options else "Standard Resins"
        fiber_str = ", ".join(str(f) for f in fiber_options[:20]) if fiber_options else "Standard Fibers"

        # 2. 構建 System Prompt (使用 f-string 注入資料)
        # 注意：JSON 的大括號在 f-string 中需要用雙大括號 {{ }} 跳脫
        system_prompt = f"""
        You are an intelligent AI assistant for Composite Material Design (Engineering Science).
        Your goal is to assist users in designing or predicting material properties using a deep learning surrogate model.

        [Context - Available Materials]
        - Resins: {resin_str}, ...
        - Fibers: {fiber_str}, ...

        Your task is to analyze the user's input and output a strictly valid JSON object.

        --- Rules for "task_type" ---
        1. "prediction": User provides parameters (weave, resin, fiber, angle) and asks for properties.
        2. "design": User asks to optimize/maximize/find best config for a target (stiffness, energy, yield).
        3. "general_chat": User says hello, asks "how to use", asks for material list, or asks non-technical questions.

        --- JSON Structure ---
        {{
            "task_type": "prediction" | "design" | "general_chat",
            "target": "max_stiffness" | "max_energy" | "max_yield" | null,
            "weave": "plain" | "twill" | "satin" (Default: "plain"),
            "geo": {{ "angle": float, "width": float, "height": float }} (Extract explicitly mentioned values only),
            "resin": string (Extract material name) | null,
            "fiber": string (Extract material name) | null,
            "reply": string (REQUIRED for "general_chat". A helpful, polite response guiding the user. Null for others.)
        }}

        --- Few-Shot Examples ---
        User: "Hi, what can you do?"
        Output: {{ "task_type": "general_chat", "reply": "Hello! I am your Composite Design Assistant. I can help you 'Predict properties' or 'Design optimal materials'. For example, try asking: 'Design a high stiffness Glass fiber/Epoxy composite'.", "target": null, "weave": "plain", "geo": {{}}, "resin": null, "fiber": null }}

        User: "Optimize for max stiffness using Epoxy."
        Output: {{ "task_type": "design", "target": "max_stiffness", "resin": "Epoxy", "reply": null, "weave": "plain", "geo": {{}}, "fiber": null }}

        Output ONLY the JSON string. No markdown, no explanations.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        print("Agent is parsing instruction...")
        outputs = self.pipe(
            messages, 
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = outputs[0]["generated_text"][-1]["content"]
        
        # Cleanup
        clean_json = generated_text.replace("```json", "").replace("```", "").strip()
        
        return clean_json


import pyvista as pv
import numpy as np

# ==========================================
# 1. 3D 模型生成核心 (最終完美整合版：修復緞紋穿模)
# ==========================================
def generate_3d_woven_plotter(width=1.0, height=0.2, angle=90, weave_style="plain"):
    """
    產生 3D 編織模型的 PyVista Plotter 物件，完美支援平紋、斜紋與緞紋，並修復穿模
    """
    num_yarns = 8 
    theta = np.radians(angle) 
    
    pitch = (width * 1.1) / max(np.sin(theta), 0.1)
    
    # [修改 1] 保留最適合斜紋與緞紋展示的視覺誇張係數 2.5
    visual_z_scale = 2.5 
    amp = (height / 2) * visual_z_scale

    style = weave_style.lower()
    
    # [整合核心] 根據不同的編織法給予專屬的相位偏移
    if style in ["twill", "斜紋"]:
        d1, d2 = np.pi / 2, np.pi / 2
        # **斜紋不需要額外偏移，保留完美的長浮動流暢交錯**
        offset_u, offset_v = 0.0, 0.0
    elif style in ["satin", "段紋", "緞紋"]:
        d1, d2 = 2 * np.pi / 5, 4 * np.pi / 5 
        # **[修改 2] 核心修復：緞紋也需要特定的中心對齊相位校正，解決穿模問題**
        offset_u = ((num_yarns - 1) / 2) * d2
        offset_v = ((num_yarns - 1) / 2) * d1
    else:
        # 預設為 plain 平紋
        d1, d2 = np.pi, np.pi            
        # **平紋加入相位校正**
        offset_u = ((num_yarns - 1) / 2) * d2
        offset_v = ((num_yarns - 1) / 2) * d1

    vec_warp = np.array([1.0, 0.0, 0.0]) 
    vec_weft = np.array([np.cos(theta), np.sin(theta), 0.0])

    fabric_span = (num_yarns - 1) * pitch
    yarn_len_logical = fabric_span + width * 2.5

    plotter = pv.Plotter(notebook=True)
    tube_z_scale = (height / width) * (visual_z_scale * 0.8)

    # --- 建立經紗 (Warp) ---
    for i in range(num_yarns):
        v_logical = (i - (num_yarns - 1) / 2) * pitch
        u_points = np.linspace(-yarn_len_logical/2, yarn_len_logical/2, 200)
        
        points = []
        for u in u_points:
            pos = u * vec_warp + v_logical * vec_weft
            # **將 offset_u 整合進cosine方程式中**
            z = amp * np.cos(i * d1 + u * (d2 / pitch) + offset_u)
            points.append([pos[0], pos[1], z])
            
        tube = pv.Spline(np.array(points), 200).tube(radius=width/2.5)
        tube = tube.scale([1.0, 1.0, tube_z_scale], inplace=False)
        plotter.add_mesh(tube, color="crimson", smooth_shading=True, specular=0.5)

    # --- 建立緯紗 (Weft) ---
    for j in range(num_yarns):
        u_logical = (j - (num_yarns - 1) / 2) * pitch
        v_points = np.linspace(-yarn_len_logical/2, yarn_len_logical/2, 200)
        
        points = []
        for v in v_points:
            pos = u_logical * vec_warp + v * vec_weft
            z = amp * np.cos(v * (d1 / pitch) + j * d2 + offset_v + np.pi)
            points.append([pos[0], pos[1], z])

        tube = pv.Spline(np.array(points), 200).tube(radius=width/2.5)
        tube = tube.scale([1.0, 1.0, tube_z_scale], inplace=False)
        plotter.add_mesh(tube, color="dodgerblue", smooth_shading=True, specular=0.5)

    plotter.set_background("white")
    plotter.add_axes()
    plotter.view_isometric()
    
    # [修改 3] 保留最適合斜紋與緞紋展示的 15 度攝影機仰角
    plotter.camera.elevation = 15 
    plotter.camera.azimuth = 45
    plotter.camera.zoom(1.2)
    
    return plotter


import json

# ==========================================
# 2. 修改後的主程式介面 (包含穩健的參數讀取)
# ==========================================
def run_LLM(user_query):
    print("="*60)
    print(f"User Input: {user_query}")
    print("="*60)
    
    # --- Check Agent ---
    if 'my_agent' not in globals() or my_agent is None:
        print("Error: Agent not loaded. Please run Cell 5 first.")
        return None

    try:
        # --- Step 1: 準備材料上下文 ---
        if 'env' in globals() and env is not None:
            r_opts = env.resin_options
            f_opts = env.fiber_options
        else:
            r_opts = []
            f_opts = []

        # --- Step 2: LLM Parsing ---
        json_str = my_agent.parse_instruction(user_query, resin_options=r_opts, fiber_options=f_opts)
        
        try:
            params = json.loads(json_str)
        except json.JSONDecodeError:
            print(f"JSON Parsing Failed. Raw: {json_str}")
            return {
                "task_type": "general_chat",
                "reply": "抱歉，我無法理解您的指令格式，請再試一次。",
                "params": {}, "data": None, "figure": None
            }

        task_type = params.get("task_type", "general_chat")
        print(f"\n[Step 1] Intent Detected: {task_type.upper()}")

        # 準備回傳結構
        result_package = {
            "task_type": task_type,
            "params": params,
            "data": None,
            "figure": None,    # 用於 Matplotlib (2D 數據圖)
            "plotter": None,   # 用於 PyVista (3D 模型)
            "reply": params.get("reply", None)
        }

        # --- 定義一個穩健的參數讀取小工具 ---
        # 這能防止 float(None) 的錯誤
        def safe_get_float(dictionary, key, default_value):
            val = dictionary.get(key)
            if val is None:
                return float(default_value)
            try:
                return float(val)
            except (ValueError, TypeError):
                return float(default_value)

        # --- Step 3: 分流處理 ---

        # === 分支 A: 一般對話 ===
        if task_type == "general_chat":
            return result_package

        # === 分支 B: 最佳化設計 (Optimization) ===
        elif task_type == "design":
            target = params.get("target", "max_stiffness")
            weave_style = params.get("weave", "plain")
            geo_dict = params.get("geo", {})
            if geo_dict is None: geo_dict = {} # 防呆：如果 geo 是 None，設為空字典

            resin_name = params.get("resin", "Epoxy")
            fiber_name = params.get("fiber", "Carbon")

            print(f"    Target: {target}")
            print(f"    Initial Point: {resin_name}/{fiber_name}, Geo={geo_dict}")
            
            # 執行最佳化
            opt_metrics = optimize_composite(
                target_type=target,
                weave_style=weave_style,
                geo_dict=geo_dict,
                resin_name=resin_name,
                fiber_name=fiber_name,
                verbose=True
            )
            result_package["data"] = opt_metrics
            
            # [修正] 生成對應的 3D 模型 (使用最佳化後的參數)
            try:
                # **從最佳化結果中提取最終的幾何陣列 (預設給定一個安全陣列以防萬一)**
                opt_geo = opt_metrics.get("optimized", {}).get("geo", [90.0, 1.0, 0.2])
                opt_weave = opt_metrics.get("optimized", {}).get("weave", "plain")
                
                # **根據 opt_geo 的陣列順序 [angle, width, height] 取值**
                a = float(opt_geo[0])
                w = float(opt_geo[1])
                h = float(opt_geo[2])
                
                # **更新 Print 訊息以方便在終端機除錯**
                print(f"    Generating Optimized 3D Model (Angle:{a}°, Width:{w:.2f}, Height:{h:.2f}, Style:{opt_weave})...")
                result_package["plotter"] = generate_3d_woven_plotter(width=w, height=h, angle=a, weave_style=opt_weave)
            except Exception as e:
                print(f"    Warning: Failed to generate optimized 3D model: {e}")

        # === 分支 C: 物理預測 (Prediction) ===
        elif task_type == "prediction":
            weave_style = params.get("weave", "plain")
            geo_dict = params.get("geo", {})
            if geo_dict is None: geo_dict = {} # 防呆

            resin_name = params.get("resin", "Epoxy")
            fiber_name = params.get("fiber", "Carbon")

            print(f"    Configuration: {resin_name}/{fiber_name}, Geo={geo_dict}")
            
            # 執行預測
            eval_metrics, fig = evaluate_composite(
                weave_style=weave_style, 
                geo_dict=geo_dict, 
                resin_name=resin_name, 
                fiber_name=fiber_name, 
                show_plot=True, 
                verbose=True
            )
            result_package["data"] = eval_metrics
            result_package["figure"] = fig 
            
            # [修正] 生成對應的 3D 模型 (加入防呆)
            try:
                w = safe_get_float(geo_dict, "width", 1.0)
                h = safe_get_float(geo_dict, "height", 0.2)
                
                # 角度防呆邏輯
                raw_angle = geo_dict.get("angle")
                if raw_angle is None:
                    raw_angle = params.get("angle")
                
                a = float(raw_angle) if raw_angle is not None else 90.0
                
                print(f"    Generating Prediction 3D Model ({a}°, Style:{weave_style})...")
                result_package["plotter"] = generate_3d_woven_plotter(width=w, height=h, angle=a, weave_style=weave_style)
            except Exception as e:
                print(f"    Warning: Failed to generate 3D model: {e}")
            
        return result_package

    except Exception as e:
        print(f"Agent Error: {e}")
        return {
            "task_type": "general_chat",
            "reply": f"系統發生錯誤: {str(e)}",
            "params": {}, "data": None, "figure": None, "plotter": None
        }



# ==========================================
# 5. Initialization Function (Entry Point)
# ==========================================
def initialize_system():
    global predictor, plastic_predictor, env, my_agent, df_resin, df_fiber, Net
    print(">>> [Backend] Starting Initialization...")
    
    # 1. Load Data
    if 'df_resin' not in globals() or df_resin.empty:
        if os.path.exists("resin_material_property.csv"):
            df_resin = pd.read_csv("resin_material_property.csv")
            df_fiber = pd.read_csv("fiber_material_property.csv")
        else:
            print("CSV files not found.")
            return False
            
    # 2. Load Predictors
    if 'predictor' not in globals() or predictor is None: 
        predictor = DualPredictor()
    if 'plastic_predictor' not in globals() or plastic_predictor is None: 
        plastic_predictor = PlasticPredictor()
        
    # 3. Load Env
    if 'env' not in globals() or env is None: 
        env = CompositeEnvPPO(plastic_predictor, df_resin, df_fiber)
    
    # 4. Load LLM
    if 'my_agent' not in globals() or my_agent is None:
        print("Loading LLM...")
        if "hf_" in YOUR_HF_TOKEN:
            my_agent = LLMAgent(YOUR_HF_TOKEN)
        else:
            print("Invalid Token")
            return False
        
    print(">>> [Backend] Ready.")
    return True