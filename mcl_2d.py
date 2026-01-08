import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats

# === 設定 ===
DT = 0.1             # 時間刻み [s]
SIM_STEPS = 200      # シミュレーションのステップ数
PARTICLE_NUM = 100   # パーティクルの数
NOISE_VEL = 0.8    # 速度に対するノイズ分散 (大きいと前後にブレる)
NOISE_OMG = 0.2    # 角速度に対するノイズ分散 (大きいと進行方向がブレる)
NOISE_SENSOR = 0.2 # 距離計測の誤差標準偏差 (大きいとパーティクルが収束しにくい)

# ランドマークの位置 (x, y)
LANDMARKS = [np.array([5.0, 0.0]), 
             np.array([-8.0, 10.0]), 
             np.array([-5.0, -5.0])]

class Particle:
    """ロボットの状態（x, y, theta）と重みを持つパーティクル"""
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

class Robot:
    """真のロボット（シミュレータ）"""
    def __init__(self):
        self.x = 3.0
        self.y = 0.0
        self.theta = 0.0
    
    def move(self, v, omega):
        # 実際の移動（ノイズなしとするが、環境によってはノイズを入れる）
        self.theta += omega * DT
        self.x += v * np.cos(self.theta) * DT
        self.y += v * np.sin(self.theta) * DT
        
    def observe(self):
        # ランドマークまでの距離を観測（+観測ノイズ）
        z_list = []
        for lm in LANDMARKS:
            d = np.sqrt((self.x - lm[0])**2 + (self.y - lm[1])**2)
            # 観測ノイズを加える
            # d += np.random.normal(0.0, 0.5)
            d += np.random.normal(0.0, NOISE_SENSOR)
            z_list.append(d)
        return z_list

class MCL:
    """MCLアルゴリズム"""
    def __init__(self):
        # 初期位置 (0,0,0) 付近にばら撒く
        self.particles = []
        for i in range(PARTICLE_NUM):
            p = Particle(np.random.normal(0.0, 0.5),
                         np.random.normal(0.0, 0.5),
                         np.random.normal(0.0, 0.1),
                         1.0/PARTICLE_NUM)
            self.particles.append(p)
            
    def motion_update(self, v, omega):
        # 動作モデル：全パーティクルを移動（ノイズ混入）
        for p in self.particles:
            # 速度・角速度にノイズが入る
            # v_noise = v + np.random.normal(0.0, 0.2)
            v_noise = v + np.random.normal(0.0, NOISE_VEL)
            omega_noise = omega + np.random.normal(0.0, NOISE_OMG) # 角度はブレやすい
            # omega_noise = omega + np.random.normal(0.0, 0.1)

            p.theta += omega_noise * DT
            p.x += v_noise * np.cos(p.theta) * DT
            p.y += v_noise * np.sin(p.theta) * DT

    def observation_update(self, observations):
        # 観測モデル：尤度計算
        for p in self.particles:
            likelihood = 1.0
            for i, lm in enumerate(LANDMARKS):
                # パーティクルから見たランドマーク距離
                d_pred = np.sqrt((p.x - lm[0])**2 + (p.y - lm[1])**2)
                # 観測値(observations[i])との差で重み付け
                # 距離が近いほど確率密度が高い
                pdf = scipy.stats.norm.pdf(observations[i], loc=d_pred, scale=1.0)
                likelihood *= pdf
            p.weight *= likelihood
            
        # 重みの正規化
        total_weight = sum([p.weight for p in self.particles])
        if total_weight > 0:
            for p in self.particles:
                p.weight /= total_weight
                
    def resampling(self):
        # シンプルな系統サンプリング
        weights = [p.weight for p in self.particles]
        # 重みに応じてインデックスを抽選
        indices = np.random.choice(range(PARTICLE_NUM), size=PARTICLE_NUM, p=weights, replace=True)
        
        new_particles = []
        for i in indices:
            old_p = self.particles[i]
            # コピーを作成（重みはリセット）
            new_p = Particle(old_p.x, old_p.y, old_p.theta, 1.0/PARTICLE_NUM)
            new_particles.append(new_p)
        self.particles = new_particles

# === アニメーション実行部分 ===
fig, ax = plt.subplots(figsize=(8, 8))
robot = Robot()
mcl = MCL()

# 描画用オブジェクトの初期化
particle_scat = ax.scatter([], [], s=10, color='blue', alpha=0.5, label='Particles')
robot_scat, = ax.plot([], [], 'r-', linewidth=2, label='Robot Trajectory')
true_pos_scat, = ax.plot([], [], 'ro', label='Current Robot')
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# 軌跡保存用
robot_history_x = []
robot_history_y = []

def init():
    # 背景のランドマーク描画
    lx = [lm[0] for lm in LANDMARKS]
    ly = [lm[1] for lm in LANDMARKS]
    ax.scatter(lx, ly, marker='*', s=200, color='orange', label='Landmarks')
    
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.grid(True)
    ax.legend()
    return particle_scat, robot_scat, true_pos_scat, time_text

def update(frame):
    # 1. ロボットが動く (直進1.0, 回転一定 で円を描く)
    v = 2.0
    omega = 0.4
    robot.move(v, omega)
    
    # 履歴保存
    robot_history_x.append(robot.x)
    robot_history_y.append(robot.y)
    
    # 2. ロボットが観測
    obs = robot.observe()
    
    # 3. MCL更新サイクル
    mcl.motion_update(v, omega)  # 予測
    mcl.observation_update(obs)  # 更新
    mcl.resampling()             # リサンプリング
    
    # 4. 描画更新
    # パーティクル
    px = [p.x for p in mcl.particles]
    py = [p.y for p in mcl.particles]
    particle_scat.set_offsets(np.c_[px, py])
    
    # ロボット軌跡
    robot_scat.set_data(robot_history_x, robot_history_y)
    true_pos_scat.set_data([robot.x], [robot.y])
    
    time_text.set_text(f'Step: {frame}')
    
    return particle_scat, robot_scat, true_pos_scat, time_text

# アニメーション作成
ani = animation.FuncAnimation(fig, update, frames=SIM_STEPS,
                              init_func=init, interval=100, blit=True)

plt.title("2D MCL Animation (Inspired by LNPR)")
plt.show()