import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
import pandas as pd

# ==========================================
# 1. 定義虛擬學生 (The Environment)
# ==========================================
class SimulatedStudent:
    def __init__(self, initial_skill=0.0, learning_rate=0.05):
        # 學生初始能力 (範圍 -3 到 3，對應標準常態分佈概念)
        self.true_skill = initial_skill
        self.learning_rate = learning_rate
        self.history = []

    def answer_question(self, difficulty):
        """
        根據 IRT (Item Response Theory) 決定是否答對
        P(correct) = 1 / (1 + exp(-(skill - difficulty)))
        """
        # 計算答對機率
        logit = self.true_skill - difficulty
        prob_correct = 1 / (1 + np.exp(-logit))
        
        # 模擬作答結果 (0: 錯, 1: 對)
        is_correct = 1 if np.random.rand() < prob_correct else 0
        
        # 模擬學習過程：
        # 如果答對了，能力提升；如果答錯但題目沒有太難，也可能學到一點
        # 這裡簡化：答對能力 +LR, 答錯且題目難度接近能力 +0.1*LR
        if is_correct:
            self.true_skill += self.learning_rate
        else:
            # 從錯誤中學習 (Error-driven learning)，但效果較小
            if abs(difficulty - self.true_skill) < 1.0: 
                self.true_skill += self.learning_rate * 0.1
        
        return is_correct, prob_correct

# ==========================================
# 2. 定義 AI 家教 (The Agent)
# ==========================================
class AITutor:
    def __init__(self):
        # 使用 SGDClassifier 模擬線上學習的 Logistic Regression
        # loss='log_loss' 啟用邏輯回歸
        self.model = SGDClassifier(loss='log_loss', learning_rate='constant', eta0=0.1, random_state=42)
        self.is_trained = False
        
        # 定義題目難度池 (Easy, Medium, Hard)
        # 對應數值 (例如: -2, 0, 2)
        self.difficulty_levels = [-2.0, -1.0, 0.0, 1.0, 2.0]
        self.difficulty_names = ['Easy', 'Medium-Easy', 'Medium', 'Medium-Hard', 'Hard']

    def predict_success_prob(self, difficulty):
        """預測學生在特定難度下的答對機率"""
        if not self.is_trained:
            return 0.5 # 尚未訓練時，假設機率為 0.5
        
        # scikit-learn 需要 2D array
        X = np.array([[difficulty]])
        # predict_proba 回傳 [[prob_0, prob_1]]
        return self.model.predict_proba(X)[0][1]

    def select_action(self):
        """
        決策邏輯 (Policy):
        目標是讓學生處於「心流通道」(Flow Channel)，
        假設最佳學習的答對機率約為 70% (0.7)。
        太簡單(P>0.9)會無聊，太難(P<0.4)會挫折。
        """
        if not self.is_trained:
            # 冷啟動：隨機選題
            return np.random.choice(self.difficulty_levels)
        
        best_diff = self.difficulty_levels[1] # Default Medium
        min_gap = float('inf')

        # 評估三個難度的預期答對率，選擇最接近 0.7 的那個
        for diff in self.difficulty_levels:
            prob = self.predict_success_prob(diff)
            gap = abs(prob - 0.7)
            if gap < min_gap:
                min_gap = gap
                best_diff = diff
        
        return best_diff

    def update_model(self, difficulty, result):
        """根據剛發生的 (題目, 結果) 更新模型"""
        X = np.array([[difficulty]])
        y = np.array([result])
        # partial_fit 允許增量訓練
        self.model.partial_fit(X, y, classes=[0, 1])
        self.is_trained = True

# ==========================================
# 3. 執行模擬 (Simulation Loop)
# ==========================================
def run_simulation(steps=30):
    # 初始化
    student = SimulatedStudent(initial_skill=-1.0) # 假設學生一開始基礎較差
    tutor = AITutor()
    
    records = [] # 紀錄過程
    
    print(f"開始模擬教學 ({steps} 題)...")
    
    for t in range(steps):
        # 1. AI 決定出什麼難度的題目
        difficulty = tutor.select_action()
        
        # 2. 學生作答
        is_correct, true_prob = student.answer_question(difficulty)
        
        # 3. AI 根據結果更新模型 (Learn)
        tutor.update_model(difficulty, is_correct)
        
        # 記錄數據
        est_prob = tutor.predict_success_prob(difficulty)
        records.append({
            'step': t + 1,
            'difficulty': difficulty,
            'is_correct': is_correct,
            'true_skill': student.true_skill,
            'est_prob': est_prob, # AI 預測的勝率
            'true_prob': true_prob # 真實的勝率
        })

    return pd.DataFrame(records)

# 執行並繪圖
df_results = run_simulation(steps=150)

# ==========================================
# 4. 結果視覺化 (Visualization)
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# 繪製學生真實能力的變化
color = 'tab:blue'
ax1.set_xlabel('Question Number (Time)')
ax1.set_ylabel('Student True Skill (Z-score)', color=color)
ax1.plot(df_results['step'], df_results['true_skill'], color=color, linewidth=2, label='Student Skill')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# 繪製題目難度變化
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Question Difficulty', color=color)
ax2.step(df_results['step'], df_results['difficulty'], color=color, where='post', linestyle='--', label='Difficulty Assigned')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Simulation: AI Tutor Adapting Difficulty to Student Skill Growth')
fig.tight_layout()
plt.show()

# 顯示前幾筆數據
print(df_results.head())