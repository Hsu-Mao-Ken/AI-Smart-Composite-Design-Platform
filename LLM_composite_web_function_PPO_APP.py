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


# --- 2. Enhanced Search Function (包含高強度防呆與透明宣告) ---
def get_material_params(name, material_type="fiber", verbose=True):
    """
    Find material parameters using keyword containment logic.
    Enhanced to prevent silent overrides and handle NaN/Null safely.
    """
    # [新增防呆 1] 阻絕無效輸入，避免 str(None) 變成 "none" 導致後續誤判
    if name is None:
        return None
        
    # Pre-processing: 轉小寫、去頭尾空白
    query = str(name).lower().strip()
    
    # [新增防呆 2] 擋下 LLM 可能產生的無效字串
    if query in ["nan", "null", "none", ""]:
        return None
    
    # --- A. Alias Mapping (關鍵字優先對照表) ---
    alias_map = {
        # === 特殊樹脂 (Specific Resins) ===
        "cel": "cel",
        "optical": "cel",
        "transparent": "cel",
        "clear": "cel",
        
        "plaskon": "plaskon",
        "molding": "plaskon",
        "compound": "plaskon",
        "smt": "plaskon",
        "emc": "plaskon",
        
        # === 通用樹脂 (Generic Resins) ===
        "eler": "eler",
        "epoxy": "eler",
        "resin": "eler",
        
        # === 纖維 (Fibers) ===
        "glass": "e-glass",
        "fiberglass": "e-glass",
        "t300": "carbon",
        "graphite": "carbon"
    }
    
    # [防禦機制] 定義哪些字眼是「模糊/通用的」，需要向使用者透明宣告
    generic_keywords = ["epoxy", "resin", "fiber", "compound"]
    
    mapped_target = None
    for keyword, target in alias_map.items():
        if keyword in query:
            mapped_target = target
            
            # [核心修復] 如果觸發的是通用名詞，給出明確的透明宣告 (避免靜默取代)
            if keyword in generic_keywords and verbose:
                print(f"    [System Notice] User specified generic term '{keyword}'. Transparently defaulting to baseline '{target.upper()}'.")
            
            break # 找到第一個(權重最高的)關鍵字就停止
            
    # 如果有對應到別名，就用別名去搜尋；否則用原始輸入
    final_query = mapped_target if mapped_target else query

    # Select DataFrame
    if material_type == "fiber":
        df = df_fiber
    else:
        df = df_resin
        
    if df.empty:
        return None
        
    col_name = df.columns[0] 

    # --- B. Database Search ---
    match_row = None
    for idx, row in df.iterrows():
        db_name_str = str(row[col_name]).lower().strip()
        
        # 1. 資料庫名稱 包含 查詢詞
        if final_query in db_name_str:
            match_row = row
            break
            
        # 2. 查詢詞 包含 資料庫名稱 (長度需 > 2 避免誤判)
        if len(db_name_str) > 2 and db_name_str in final_query:
            match_row = row
            break

    # --- C. Return Result ---
    if match_row is not None:
        match_name = match_row[col_name]
        if verbose:
            # 【修正】拔除 mapped_target is None 的限制，只要成功找到就印出確認訊息！
            print(f"    Search Result [{material_type}]: Successfully matched and loaded '{match_name}'")
        return match_row.values[1:].astype(float)
    else:
        if verbose:
            print(f"    Warning: No exact or alias match found for {material_type} '{name}'.")
        return None

def get_weave_pattern(style_name="plain", verbose=True):
    """
    功能: 根據名稱回傳編織矩陣 (0/1)
    增強: 具備高強度防呆，拒絕靜默取代 (Silent Override)，遇錯回傳 None
    輸入 (Input): style_name (str) - plain, twill, satin
    輸出 (Output): numpy array (25,) 或 None
    """
    # 1. [防呆機制] 擋下 None 或浮點數 NaN，避免 .lower() 引發系統崩潰
    if style_name is None:
        return None
        
    style = str(style_name).lower().strip()
    
    # 2. [防呆機制] 擋下 LLM 的常見幻覺字串
    if style in ["nan", "null", "none", ""]:
        return None
        
    # 定義 5x5 的樣式 (1 = Weft over Warp, 0 = Warp over Weft)
    patterns = {
        "plain": [
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 1, 0, 1
        ],
        "twill": [ 
            1, 1, 0, 0, 0,  
            0, 1, 1, 0, 0,  
            0, 0, 1, 1, 0,  
            0, 0, 0, 1, 1,  
            1, 0, 0, 0, 1   
        ],
        "satin": [ 
            0, 0, 1, 0, 0,  
            1, 0, 0, 0, 0,  
            0, 0, 0, 1, 0,  
            0, 1, 0, 0, 0,  
            0, 0, 0, 0, 1   
        ]
    }
    
    # 3. [嚴格驗證] 只有在字典裡的才放行，否則回傳 None 觸發澄清機制
    if style in patterns:
        if verbose:
            print(f"    Search Result [weave]: Successfully loaded '{style}' pattern.")
        return np.array(patterns[style], dtype=float)
    else:
        if verbose:
            # 拔除靜默取代 (Defaulting to plain)，改為警告並回傳 None
            print(f"    Warning: Unknown weave style '{style_name}'. Rejecting silent override.")
        return None

import math
import numpy as np

def get_geometry_params(user_geo_dict=None, verbose=True):
    """
    功能: 生成幾何參數向量 (含角度轉換與間距自動計算)
    增強: 拔除 np.clip 靜默取代，加入嚴格邊界驗證與缺失值透明宣告
    """
    # 1. 定義預設值 (Defaults) 與 合法範圍 (Bounds)
    defaults = {
        "angle": 90.0,
        "width": 0.6,
        "height": 0.2
    }
    
    bounds = {
        "angle": (30.0, 90.0),
        "width": (0.2, 1.0),
        "height": (0.1, 0.4)
    }
    
    # 2. 定義別名對照表
    alias_map = {
        "yarn_width": "width",
        "w": "width",
        "yarn_height": "height",
        "h": "height",
        "deg": "angle",
        "degree": "angle"
    }

    # --- 3. 提取與清洗輸入值 ---
    clean_dict = {}
    if user_geo_dict and isinstance(user_geo_dict, dict):
        for k, v in user_geo_dict.items():
            k_lower = str(k).lower().strip()
            real_key = alias_map.get(k_lower, k_lower)
            clean_dict[real_key] = v

    final_geo = {}
    
    # --- 4. 數值驗證與透明宣告 (取代靜默代入) ---
    for key, default_val in defaults.items():
        val = clean_dict.get(key)
        
        # 狀況 A: 缺失值或 LLM 幻覺字串 (None, NaN) -> 進行透明宣告並代入基準
        if val is None or str(val).strip().lower() in ["nan", "null", "none", ""]:
            if verbose:
                print(f"    [System Notice] Geometry '{key}' not provided. Transparently defaulting to {default_val}.")
            final_geo[key] = default_val
            continue
            
        # 狀況 B: 數值解析
        try:
            f_val = float(val)
            if math.isnan(f_val):
                raise ValueError("Value is NaN")
            final_geo[key] = f_val
        except (ValueError, TypeError):
            if verbose:
                print(f"    Warning: Invalid format for '{key}' ('{val}'). Rejecting silent override.")
            return None # 拒絕靜默代入，直接回傳 None 觸發外部澄清

    # --- 5. 嚴格範圍檢查 (拔除 np.clip) ---
    for key, (min_val, max_val) in bounds.items():
        if not (min_val <= final_geo[key] <= max_val):
            if verbose:
                print(f"    Warning: Geometry '{key}' value {final_geo[key]} is OUT OF BOUNDS [{min_val}, {max_val}]. Rejecting evaluation.")
            return None # 發現超出物理極限，直接阻斷，交由 Pydantic 或外部去報錯

    # --- 6. 計算模型參數 ---
    u_angle = final_geo["angle"]
    width = final_geo["width"]
    height = final_geo["height"]

    # 模型訓練時使用的是與 90 度的夾角差 (0~60)
    model_angle = abs(u_angle - 90.0)
    
    # 間距計算公式
    multiplier = 1.5 + (u_angle - 90.0) * (-0.025)
    space = width * multiplier
    
    # --- 7. 輸出確認 ---
    if verbose:
        print(f"    Geometry Processing:")
        print(f"      - Effective Config: Angle={u_angle:.1f}°, Width={width:.2f}, Height={height:.2f}")
        print(f"      - Model Input Angle: {model_angle:.1f}°")
        print(f"      - Space: {space:.3f} mm (Multiplier: {multiplier:.2f}x)")

    ordered_values = [model_angle, width, height, space]
    return np.array(ordered_values, dtype=float)

class EffectiveModel(nn.Module):
    def __init__(self):
        super(EffectiveModel, self).__init__()

        # ==========================================
        # 1. Image Branch (維持不變)
        # ==========================================
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=2, stride=2, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        
        self.flatten = nn.Flatten()
        
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

        # ==========================================
        # 2. Info Branch (維持不變)
        # ==========================================
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

        # ==========================================
        # 3. Combined Path (重大修改)
        # ==========================================
        # 這裡不再轉回圖片，而是使用 MLP 直接融合
        # 輸入維度: 120 (Image特徵) + 24 (Info特徵) = 144
        
        self.final_dense_block = nn.Sequential(
            # Layer 1: 144 -> 512
            nn.Linear(144, 512),
            nn.ReLU(),
            nn.Dropout(0.1),       # [優化] 防止過擬合
            
            # Layer 2: 512 -> 256
            nn.Linear(512, 256),
            nn.ReLU(),
            
            # Layer 3: 256 -> 128
            nn.Linear(256, 128),
            nn.ReLU(),
           
            # Output Layer: 128 -> 40
            # 輸出 20 個點的 (Stress, Strain) 座標
            nn.Linear(128, 12)
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
        
        x1 = self.img_dense_block(x1) # Output: (Batch, 120)
        
        # --- Info Branch ---
        x2 = self.info_dense_block(info_input) # Output: (Batch, 24)

        # --- Combined Path ---
        # 1. 拼接
        combined = torch.cat((x1, x2), dim=1) # Shape: (Batch, 144)
        
        # 2. [修改點] 直接進入 MLP，不再 reshape 也不再做卷積
        output = self.final_dense_block(combined)
        
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
        
        self.paths = {
            "eff_model": "effective_model_pytorch.pth",
            "eff_in": "effective_input_scaler.pkl",
            "eff_out": "effective_output_scaler.pkl",
            "pla_model": "plastic_model_pytorch.pth",
            "pla_in": "plastic_input_scaler.pkl",
            "pla_out": "plastic_output_scaler.pkl"
        }
        
        # --- 載入 Effective System (彈性) ---
        try:
            self.model_eff = EffectiveModel().to(self.device) 
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
            self.model_pla.load_state_dict(torch.load(self.paths["pla_model"], map_location=self.device))
            self.model_pla.eval()
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

        # ---------------------------------------------------------
        # 🛡️ [防護機制] 全面攔截底層回傳的 None，避免張量運算崩潰
        # ---------------------------------------------------------
        if weave_vec is None:
            return {"error": f"Invalid weave style: '{weave_style}'"}
        if geo_vec is None:
            return {"error": "Invalid or out-of-bounds geometry parameters."}
        if resin_vec_full is None:
            return {"error": f"Resin material not found: '{resin_name}'"}
        if fiber_vec is None:
            return {"error": f"Fiber material not found: '{fiber_name}'"}

        results = {}

        # 2. 預測 - 彈性模型
        if self.model_eff:
            try:
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
            except Exception as e:
                return {"error": f"Effective model inference failed: {str(e)}"}

        # 3. 預測 - 塑性模型
        if self.model_pla:
            try:
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
            except Exception as e:
                return {"error": f"Plastic model inference failed: {str(e)}"}
            
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


import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Main Function: Composite Evaluation
# (Modified to return Figure object for Web UI)
# ==========================================
def evaluate_composite(weave_style, geo_dict, resin_name, fiber_name, show_plot=True, verbose=True):
    """
    接收設計參數，執行預測，計算物理性質。
    增強: 拒絕靜默失敗，遇到錯誤主動拋出例外 (Raise Exception)，並加入數學防呆
    回傳: (metrics, fig)
    """
    
    # 檢查預測器
    if 'predictor' not in globals():
        raise RuntimeError("Predictor is not initialized. Please load the DualPredictor first.")

    # 1. 執行預測
    if verbose:
        print(f"Running Analysis: {weave_style} / {geo_dict} / {resin_name} / {fiber_name}")
    
    results = predictor.predict(weave_style, geo_dict, resin_name, fiber_name)
    
    # 🛡️ [核心防禦 1] 拒絕靜默失敗：如果底層預測出錯，直接拋出 ValueError 讓外層 LLM 轉成對話警告！
    if "error" in results:
        raise ValueError(f"Backend Evaluation Failed: {results['error']}")

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
        
        # 🛡️ [核心防禦 2] 預防除以零 (ZeroDivisionError) 導致網頁崩潰
        dx_elastic = (strain[1] - strain[0]) if len(strain) > 1 else 0
        slope_elastic = (stress[1] - stress[0]) / dx_elastic if dx_elastic != 0 else 0.0
        
        dx_plastic = (strain[-1] - strain[-2]) if len(strain) > 1 else 0
        slope_plastic = (stress[-1] - stress[-2]) / dx_plastic if dx_plastic != 0 else 0.0
        
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

        # 4. 繪圖邏輯
        if show_plot:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            
            # 1. 主曲線
            ax.plot(strain, stress, 'r-o', markersize=4, linewidth=2, label=f"{resin_name}/{fiber_name}")
            
            # 2. 標記降伏點與特徵線
            if y_str and y_eps:
                ax.plot(y_eps, y_str, 'b*', markersize=12, label='Yield Point', zorder=5)
                
                # 如果是交點法，畫出兩條延伸線
                if "Intersection" in method_name and debug:
                    x_range = np.linspace(0, np.max(strain)*1.1, 100)
                    
                    y_elas = debug["slope_elastic"] * x_range 
                    ax.plot(x_range, y_elas, 'g--', alpha=0.4, linewidth=1, label="Elastic Ext.")
                    
                    y_plas = debug["slope_plastic"] * x_range + debug["intercept_plastic"]
                    ax.plot(x_range, y_plas, 'b--', alpha=0.4, linewidth=1, label="Plastic Ext.")

            ax.set_title(f"Predicted Stress-Strain ({weave_style})", fontsize=11)
            ax.set_xlabel("Strain (-)", fontsize=9)
            ax.set_ylabel("Stress (Pa)", fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=8, loc='best')
            fig.tight_layout()
            
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

        # ---------------------------------------------------------
        # 🛡️ [防護機制] 全面攔截底層回傳的 None，避免張量運算崩潰
        # ---------------------------------------------------------
        if weave_vec is None:
            return {"error": f"Invalid weave style: '{weave_style}'"}
        if geo_vec is None:
            return {"error": "Invalid or out-of-bounds geometry parameters."}
        if resin_vec_full is None:
            return {"error": f"Resin material not found: '{resin_name}'"}
        if fiber_vec is None:
            return {"error": f"Fiber material not found: '{fiber_name}'"}

        results = {}

        # 2. 預測 - 塑性模型
        if self.model_pla:
            try:
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
            except Exception as e:
                # 捕捉神經網路推理時的意外錯誤
                return {"error": f"Plastic model inference failed: {str(e)}"}
            
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
        
        try:
            result = self.predictor.predict(weave_name, geo_dict, resin_name, fiber_name)
        except Exception:
            # 萬一底層預測器拋出例外，RL 環境不能崩潰，而是給 0 分當作嚴厲懲罰
            return 0.0
        
        if "error" in result or "plastic" not in result:
            return 0.0 # 代理人提出無效參數，回傳 0 分予以懲罰
            
        plastic_data = result["plastic"]
        stress = plastic_data[0::2]
        strain = plastic_data[1::2]
        
        val = 0.0
        
        if self.current_target == 0: # Max Energy
            # [修復] 相容不同 Numpy 版本的積分函數
            if hasattr(np, 'trapezoid'):
                val = np.trapezoid(stress, strain)
            else:
                val = np.trapz(stress, strain)
                
        elif self.current_target == 1: # Max Stiffness
            # [修復] 預防除以零 (ZeroDivisionError)
            if len(strain) > 1:
                dx = strain[1] - strain[0]
                if dx != 0:
                    val = (stress[1] - stress[0]) / dx
                else:
                    val = 0.0
                    
        elif self.current_target == 2: # Max Yield
            try:
                # 確保外部有此函數，並捕捉可能的例外
                if 'find_smart_yield_point' in globals():
                    y_str, _, _, _ = find_smart_yield_point(stress, strain)
                    if y_str: val = y_str
            except Exception:
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
        
        # ---------------------------------------------------------
        # 🛡️ [防護機制] 強制轉型為 LongTensor，避免 Categorical 崩潰
        # ---------------------------------------------------------
        action_weave = action_tensor[:, 0].long()
        action_resin = action_tensor[:, 1].long()
        action_fiber = action_tensor[:, 2].long()
        # 幾何參數保持 FloatTensor 即可
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
    
    
import torch
import numpy as np
import os

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
    max_steps=5,       # PPO 給絕對值，1步即達最佳解，預設跑5步確認穩定性
    verbose=True       
):
    """
    接收手動定義的初始參數，使用 PPO 直接給出該目標下的最佳設計。
    增強: 拒絕靜默失敗與靜默取代，嚴格驗證邊界並拋出例外。
    """
    
    # 0. 檢查環境依賴 (拒絕靜默失敗，主動拋出錯誤)
    if 'env' not in globals():
        raise RuntimeError("PPO Environment ('env') is not initialized.")
    if 'PPOActorCritic' not in globals():
        raise RuntimeError("'PPOActorCritic' class is not defined.")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 內部 Helper: 搜尋邏輯 ---
    def find_index_smartly(user_input, option_list, material_type="fiber"):
        if user_input is None or str(user_input).strip().lower() in ["nan", "null", "none", ""]:
            return None
            
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
        # 拒絕靜默失敗
        raise FileNotFoundError("No PPO model file found. Cannot perform optimization.")

    # 2. 強制設定環境至「使用者指定的初始狀態」
    state = env.reset(target_type=target_type)
    
    # 🛡️ (A) 設定編織 (拒絕靜默取代)
    if weave_style in env.weave_options:
        env.cur_weave_idx = env.weave_options.index(weave_style)
    else:
        raise ValueError(f"Invalid weave style '{weave_style}'. Cannot initialize PPO state.")

    # 🛡️ (B) 設定樹脂 (拒絕靜默取代)
    r_idx = find_index_smartly(resin_name, env.resin_options, "resin")
    if r_idx is not None:
        env.cur_resin_idx = r_idx
    else:
        raise ValueError(f"Resin material '{resin_name}' not found. Cannot initialize PPO state.")
    real_resin_name = env.resin_options[env.cur_resin_idx]

    # 🛡️ (C) 設定纖維 (拒絕靜默取代)
    f_idx = find_index_smartly(fiber_name, env.fiber_options, "fiber")
    if f_idx is not None:
        env.cur_fiber_idx = f_idx
    else:
        raise ValueError(f"Fiber material '{fiber_name}' not found. Cannot initialize PPO state.")
    real_fiber_name = env.fiber_options[env.cur_fiber_idx]

    if verbose:
        print(f"Mapped Material: {real_resin_name} + {real_fiber_name}")

    # 🛡️ (D) 設定幾何 (拔除 np.clip，加入嚴格邊界驗證)
    g_angle = geo_dict.get("angle")
    g_width = geo_dict.get("width")
    g_height = geo_dict.get("height")

    try:
        angle_val = float(g_angle) if g_angle is not None else 60.0
        width_val = float(g_width) if g_width is not None else 0.5
        height_val = float(g_height) if g_height is not None else 0.2
    except (ValueError, TypeError):
        raise ValueError("Geometry parameters must be valid numerical values.")
        
    geo_arr = np.array([angle_val, width_val, height_val])
    
    # 嚴格邊界驗證
    if not (np.all(geo_arr >= env.geo_min) and np.all(geo_arr <= env.geo_max)):
        raise ValueError(f"Initial geometry {geo_arr} is out of physical bounds. Rejecting optimization.")
        
    env.cur_geo_real = geo_arr
    
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
    print(f"{'Angle (deg)':<15} | {metrics['initial']['geo'][0]:<22.1f} | {metrics['optimized']['geo'][0]:<22.1f}")
    print(f"{'Width (mm)':<15} | {metrics['initial']['geo'][1]:<22.2f} | {metrics['optimized']['geo'][1]:<22.2f}")
    print(f"{'Height (mm)':<15} | {metrics['initial']['geo'][2]:<22.2f} | {metrics['optimized']['geo'][2]:<22.2f}")
    
    print("-" * 70)
    print(f"{'Score':<15} | {initial_score:<22.4e} | {final_score:<22.4e}")
    print(f"{'Improvement':<15} | {'--':<22} | {improvement_pct:+.2f}%")
    print("=" * 70)

    return metrics

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

# Cell 5: Load Llama-3.1 Agent (Standard Float16 Mode)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==========================================
# PLEASE PASTE YOUR HUGGING FACE TOKEN HERE
# ==========================================
# 透過 Streamlit 的 secrets 功能讀取金鑰
import streamlit as st
from huggingface_hub import InferenceClient
import re

# 透過 Streamlit 的 secrets 功能讀取金鑰
YOUR_HF_TOKEN = st.secrets["HF_TOKEN"]

class LLMAgent:
    def __init__(self, hf_token):
        # 使用 Llama 3.1
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"
        print(f"Connecting to Hugging Face Inference API ({self.model_id})...")

        try:
            # [修改] 放棄本地載入，改用線上 InferenceClient
            self.client = InferenceClient(model=self.model_id, token=hf_token)
            print("Success: Llama-3.1 API Client connected.")
            
        except Exception as e:
            print(f"Error connecting to LLM API: {e}")
            print("Please check your Hugging Face token.")
            self.client = None

    # 原本的 parse_instruction 也要配合 API 的呼叫方式進行修改
    def parse_instruction(self, user_text, resin_options=[], fiber_options=[]):
        """
        Convert natural language to JSON string via Hugging Face API.
        """
        if not self.client:
            return None
            
        # 1. 準備上下文資訊 (將材料轉為字串)
        resin_str = ", ".join(str(r) for r in resin_options[:20]) if resin_options else "Standard Resins"
        fiber_str = ", ".join(str(f) for f in fiber_options[:20]) if fiber_options else "Standard Fibers"

        # 2. 構建 System Prompt (保持不變)
        system_prompt = f"""
        You are an intelligent AI assistant for Composite Material Design (Engineering Science).
        Your goal is to assist users in designing or predicting material properties using a deep learning surrogate model.

        [Context - Available Materials]
        - Resins: {resin_str}, ...
        - Fibers: {fiber_str}, ...

        Your task is to analyze the user's input and output a strictly valid JSON object.

        --- Rules for "task_type" ---
        1. "prediction": User asks to calculate, evaluate, or predict properties. (Classify as this EVEN IF parameters or materials are missing).
        2. "design": User asks to optimize, maximize, or design a composite for a target. (Classify as this EVEN IF parameters or materials are missing).
        3. "general_chat": ONLY for greetings, asking for material lists, or non-technical questions. Do NOT use this if the user wants to design or predict.

        --- JSON Structure ---
        {{
            "task_type": "prediction" | "design" | "general_chat",
            "target": "max_stiffness" | "max_energy" (use this for strain energy density) | "max_yield" | null,
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

        # [修改] 呼叫線上 API 進行對話生成
        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.1
            )
            
            # 從 API 回傳的物件中提取文字
            generated_text = response.choices[0].message.content
            
            # 使用 Regex 嚴格提取 JSON 區塊
            match = re.search(r'\{.*\}', generated_text, flags=re.DOTALL)
            if match:
                clean_json = match.group(0)
            else:
                clean_json = generated_text.replace("```json", "").replace("```", "").strip()
                
            return clean_json
            
        except Exception as e:
            print(f"Hugging Face API Error: {e}")
            return None


import json
import math
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Literal, Optional, Dict, Any

# =========================================================
# 🛡️ 1. 定義 Pydantic Schema (符合審查要求的高強度防呆與範圍驗證)
# =========================================================
class GeoParams(BaseModel):
    # 使用 Field 限制數值範圍 (ge: 大於等於, le: 小於等於)
    # 若 LLM 回傳 999 或是負數，將直接引發 ValidationError 被系統攔截
    angle: Optional[float] = Field(default=None, ge=30.0, le=90.0)
    width: Optional[float] = Field(default=None, ge=0.2, le=1.0)
    height: Optional[float] = Field(default=None, ge=0.1, le=0.4)

class LLMOutputSchema(BaseModel):
    task_type: Literal["prediction", "design", "general_chat"]
    target: Optional[Literal["max_stiffness", "max_energy", "max_yield"]] = None
    weave: Optional[Literal["plain", "twill", "satin"]] = None
    resin: Optional[str] = None
    fiber: Optional[str] = None
    geo: Optional[GeoParams] = Field(default_factory=GeoParams)
    reply: Optional[str] = None

    @model_validator(mode='after')
    def validate_materials(self):
        # Required-field 驗證：任務若為 prediction 或 design，必須明確指定材料
        if self.task_type in ["prediction", "design"]:
            def is_invalid(m):
                return m is None or str(m).strip().lower() in ["none", "null", "nan", ""]
            
            if is_invalid(self.resin):
                raise ValueError("Missing 'resin'. 請明確指定樹脂 (Resin) 材料。")
            if is_invalid(self.fiber):
                raise ValueError("Missing 'fiber'. 請明確指定纖維 (Fiber) 材料。")
        return self


# =========================================================
# 2. 修改後的主程式介面 (包含 Schema 驗證與避免靜默取代機制)
# =========================================================
def run_LLM(user_query):
    print("="*60)
    print(f"User Input: {user_query}")
    print("="*60)
    
    if 'my_agent' not in globals() or my_agent is None:
        print("Error: Agent not loaded. Please run Cell 5 first.")
        return None

    try:
        # --- Step 1: 準備材料上下文 ---
        r_opts = env.resin_options if 'env' in globals() and env is not None else []
        f_opts = env.fiber_options if 'env' in globals() and env is not None else []

        # --- Step 2: LLM Parsing ---
        json_str = my_agent.parse_instruction(user_query, resin_options=r_opts, fiber_options=f_opts)
        
        # ---------------------------------------------------------
        # 🛡️ [防護機制啟動] 導入 Pydantic 進行 Schema 驗證
        # ---------------------------------------------------------
        try:
            raw_dict = json.loads(json_str)
            if not isinstance(raw_dict, dict) or not raw_dict:
                raise ValueError("解析結果必須為有效的 JSON 物件。")
            
            # 透過 Pydantic 進行驗證
            validated_data = LLMOutputSchema(**raw_dict)
            
            # 相容 Pydantic v1 與 v2 的寫法，轉回字典供後續使用
            params = validated_data.model_dump() if hasattr(validated_data, "model_dump") else validated_data.dict()
            
        except json.JSONDecodeError:
            print(f"JSON Parsing Failed. Raw: {json_str}")
            return {
                "task_type": "error",
                "reply": "【系統錯誤】無法解析您的指令格式，請用更清晰的語言再試一次。",
                "params": {}, "data": None, "figure": None, "plotter": None
            }
        except ValidationError as e:
            error_msgs = [f"- {err['loc'][-1]}: {err['msg']}" for err in e.errors()]
            print("⚠️ Safe Clarification Triggered (Schema Validation Failed)")
            return {
                "task_type": raw_dict.get("task_type", "error"), # 👈 保留原始判斷，不篡改
                "reply": "【安全澄清機制】您的輸入包含無效參數或超出物理範圍：\n" + "\n".join(error_msgs) + "\n請修正後再試。",
                "params": raw_dict, "data": None, "figure": None, "plotter": None
            }
        except ValueError as e:
            print(f"⚠️ Safe Clarification Triggered: {e}")
            return {
                "task_type": raw_dict.get("task_type", "error"), # 👈 保留原始判斷
                "reply": f"【安全澄清機制】{str(e)}",
                "params": raw_dict, "data": None, "figure": None, "plotter": None
            }
        # ---------------------------------------------------------
        # 🛡️ 防護機制結束
        # ---------------------------------------------------------

        task_type = params["task_type"]
        print(f"\n[Step 1] Intent Detected: {task_type.upper()}")

        # === 處理缺失值並透明宣告 (避免靜默取代 Silent Override) ===
        transparent_notice = ""
        if task_type in ["prediction", "design"]:
            assumed = []
            if params["weave"] is None:
                params["weave"] = "plain"
                assumed.append("Weave=plain")
            if params["geo"]["angle"] is None:
                params["geo"]["angle"] = 60.0
                assumed.append("Angle=60.0°")
            if params["geo"]["width"] is None:
                params["geo"]["width"] = 0.6
                assumed.append("Width=0.6mm")
            if params["geo"]["height"] is None:
                params["geo"]["height"] = 0.2
                assumed.append("Height=0.2mm")
                
            if assumed:
                transparent_notice = "\n\n【系統提示】部分參數未指定，已透明化啟用基準設定：" + ", ".join(assumed) + "。"

        # 準備回傳結構
        result_package = {
            "task_type": task_type,
            "params": params,
            "data": None,
            "figure": None,
            "plotter": None,
            "reply": params.get("reply") or ""
        }

        # --- Step 3: 分流處理 ---

        # === 分支 A: 一般對話 ===
        if task_type == "general_chat":
            return result_package

        # === 分支 B: 最佳化設計 (Optimization) ===
        elif task_type == "design":
            target = params.get("target", "max_stiffness")
            weave_style = params["weave"]
            geo_dict = params["geo"]
            resin_name = params["resin"]
            fiber_name = params["fiber"]

            print(f"    Target: {target}")
            print(f"    Initial Point: {resin_name}/{fiber_name}, Geo={geo_dict}")
            
            if 'optimize_composite' in globals():
                opt_metrics = optimize_composite(
                    target_type=target,
                    weave_style=weave_style,
                    geo_dict=geo_dict,
                    resin_name=resin_name,
                    fiber_name=fiber_name,
                    verbose=True
                )
                result_package["data"] = opt_metrics
                
                try:
                    opt_geo = opt_metrics.get("optimized", {}).get("geo", [90.0, 1.0, 0.2])
                    opt_weave = opt_metrics.get("optimized", {}).get("weave", "plain")
                    a, w, h = float(opt_geo[0]), float(opt_geo[1]), float(opt_geo[2])
                    print(f"    Generating Optimized 3D Model...")
                    if 'generate_3d_woven_plotter' in globals():
                        result_package["plotter"] = generate_3d_woven_plotter(width=w, height=h, angle=a, weave_style=opt_weave)
                except Exception as e:
                    print(f"    Warning: Failed to generate optimized 3D model: {e}")

        # === 分支 C: 物理預測 (Prediction) ===
        elif task_type == "prediction":
            weave_style = params["weave"]
            geo_dict = params["geo"]
            resin_name = params["resin"]
            fiber_name = params["fiber"]

            print(f"    Configuration: {resin_name}/{fiber_name}, Geo={geo_dict}")
            
            if 'evaluate_composite' in globals():
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
                
                try:
                    w, h, a = float(geo_dict["width"]), float(geo_dict["height"]), float(geo_dict["angle"])
                    print(f"    Generating Prediction 3D Model ({a}°)...")
                    if 'generate_3d_woven_plotter' in globals():
                        result_package["plotter"] = generate_3d_woven_plotter(width=w, height=h, angle=a, weave_style=weave_style)
                except Exception as e:
                    print(f"    Warning: Failed to generate 3D model: {e}")
        
        # 將透明宣告附加到回覆訊息中
        result_package["reply"] += transparent_notice
        return result_package

    except Exception as e:
        print(f"Agent Error: {e}")
        t_type = params.get("task_type", "error") if 'params' in locals() else "error"
        return {
            "task_type": t_type,
            "reply": f"【系統錯誤】發生未預期錯誤: {str(e)}",
            "params": params if 'params' in locals() else {}, "data": None, "figure": None, "plotter": None
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