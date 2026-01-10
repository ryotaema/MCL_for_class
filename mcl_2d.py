import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import multivariate_normal
import math
import random
import copy

def state_transition(nu, omega, time, pose):
    t0 = pose[2]
    if math.fabs(omega) < 1e-10: 
        return pose + np.array( [nu*math.cos(t0), 
                                 nu*math.sin(t0), 
                                 omega ] ) * time
    else:
        return pose + np.array( [nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)), 
                                 nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)), 
                                 omega*time ] )

def observation_function(pose, landmark_pos):
    diff = landmark_pos - pose[0:2]
    phi = math.atan2(diff[1], diff[0]) - pose[2]
    while phi >= np.pi: phi -= 2*np.pi
    while phi < -np.pi: phi += 2*np.pi
    return np.array( [math.hypot(*diff), phi ] )

class Landmark:
    def __init__(self, x, y, lid):
        self.pos = np.array([x, y])
        self.id = lid

class Map:
    def __init__(self):
        self.landmarks = []
    
    def append_landmark(self, x, y):
        self.landmarks.append(Landmark(x, y, len(self.landmarks)))

class Particle: 
    def __init__(self, init_pose, weight):
        self.pose = init_pose
        self.weight = weight
        
    def motion_update(self, nu, omega, time, noise_rate_pdf): 
        ns = noise_rate_pdf.rvs()
        pnu = nu + ns[0]*math.sqrt(abs(nu)/time) + ns[1]*math.sqrt(abs(omega)/time)
        pomega = omega + ns[2]*math.sqrt(abs(nu)/time) + ns[3]*math.sqrt(abs(omega)/time)
        self.pose = state_transition(pnu, pomega, time, self.pose)
        
    def observation_update(self, observation, envmap, distance_dev_rate, direction_dev):
        for d in observation:
            obs_dist = d[0]
            obs_phi = d[1]
            obs_id = d[2]
            
            pos_on_map = envmap.landmarks[obs_id].pos
            particle_suggest_pos = observation_function(self.pose, pos_on_map)
            
            distance_dev = distance_dev_rate * particle_suggest_pos[0]
            cov = np.diag(np.array([distance_dev**2, direction_dev**2]))
            
            try:
                p = multivariate_normal(mean=particle_suggest_pos, cov=cov).pdf([obs_dist, obs_phi])
                self.weight *= p
            except:
                pass

class Mcl:
    def __init__(self, envmap, init_pose, num, motion_noise_stds, distance_dev_rate, direction_dev):
        self.particles = [Particle(init_pose, 1.0/num) for i in range(num)]
        self.map = envmap
        self.distance_dev_rate = distance_dev_rate
        self.direction_dev = direction_dev

        v = motion_noise_stds
        c = np.diag([v["nn"]**2, v["no"]**2, v["on"]**2, v["oo"]**2])
        self.motion_noise_rate_pdf = multivariate_normal(cov=c)
        
        self.pose = self.particles[0].pose 
        
    def set_ml(self):
        xs = np.array([p.pose[0] for p in self.particles])
        ys = np.array([p.pose[1] for p in self.particles])
        ts = np.array([p.pose[2] for p in self.particles])
        ws = np.array([p.weight for p in self.particles])
        
        if np.sum(ws) != 0:
            ws = ws / np.sum(ws)
        
        x_mean = np.sum(xs * ws)
        y_mean = np.sum(ys * ws)
        
        v_cos = np.sum(np.cos(ts) * ws)
        v_sin = np.sum(np.sin(ts) * ws)
        t_mean = math.atan2(v_sin, v_cos)
        
        self.pose = np.array([x_mean, y_mean, t_mean])
        
    def motion_update(self, nu, omega, time): 
        for p in self.particles: 
            p.motion_update(nu, omega, time, self.motion_noise_rate_pdf)
            
    def observation_update(self, observation): 
        for p in self.particles:
            p.observation_update(observation, self.map, self.distance_dev_rate, self.direction_dev) 
        self.set_ml() 
        self.resampling() 
            
    def resampling(self): 
        ws = np.cumsum([e.weight for e in self.particles])
        if ws[-1] < 1e-100: ws = [e + 1e-100 for e in ws]
            
        step = ws[-1]/len(self.particles)
        r = np.random.uniform(0.0, step)
        cur_pos = 0
        ps = []
        
        while(len(ps) < len(self.particles)):
            if r < ws[cur_pos]:
                ps.append(self.particles[cur_pos])
                r += step
            else:
                cur_pos += 1

        self.particles = [copy.deepcopy(e) for e in ps]
        for p in self.particles: p.weight = 1.0/len(self.particles)

class EstimationAgent: 
    def __init__(self, time_interval, nu, omega, estimator):
        self.estimator = estimator
        self.time_interval = time_interval
        self.nu = nu
        self.omega = omega
        self.prev_nu = 0.0
        self.prev_omega = 0.0
        self.poses = []
        
    def decision(self, observation=None): 
        self.estimator.motion_update(self.prev_nu, self.prev_omega, self.time_interval)
        self.prev_nu, self.prev_omega = self.nu, self.omega
        self.estimator.observation_update(observation)
        self.poses.append(self.estimator.pose)
        return self.nu, self.omega

class RealRobot:
    def __init__(self, init_pose, agent, envmap):
        self.pose = init_pose
        self.agent = agent
        self.map = envmap
        
    def one_step(self, time_interval):
        obs = []
        for lm in self.map.landmarks:
            z = observation_function(self.pose, lm.pos)
            z[0] += np.random.normal(0.0, 0.1)
            z[1] += np.random.normal(0.0, 0.05)
            if z[0] < 10.0:
                obs.append([z[0], z[1], lm.id])
        
        nu, omega = self.agent.decision(obs)
        self.pose = state_transition(nu, omega, time_interval, self.pose)

def main():
    TIME_INTERVAL = 0.1
    SIM_STEPS = 300
    
    m = Map()
    # for ln in [(-4,2), (2,-3), (3,3), (0, 5), (-2, -4)]: 
    for ln in [(-4,2),  (3,3), (-2,-2)]: 
        m.append_landmark(*ln)

    initial_pose = np.array([0.0, 0.0, 0.0])

    motion_noise = {"nn":0.5, "no":0.5, "on":0.5, "oo":0.5}
    # motion_noise = {"nn":0.19, "no":0.001, "on":0.13, "oo":0.2}
    # パーティクルの設定
    estimator = Mcl(m, np.array([0.0, 0.0, 0.0]), 100, motion_noise, distance_dev_rate=0.2, direction_dev=0.05)
    
    agent = EstimationAgent(TIME_INTERVAL, 0.4, 20.0/180*math.pi, estimator)
    robot = RealRobot(initial_pose, agent, m)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    initial_xs = [p.pose[0] for p in estimator.particles]
    initial_ys = [p.pose[1] for p in estimator.particles]
    initial_us = [math.cos(p.pose[2]) for p in estimator.particles]
    initial_vs = [math.sin(p.pose[2]) for p in estimator.particles]

    p_arrows = ax.quiver(initial_xs, initial_ys, initial_us, initial_vs, 
                         color='blue', alpha=0.5, scale=5.0, 
                         scale_units='xy', angles='xy', label='Particles')
    
    r_body, = ax.plot([], [], 'ro', markersize=10, label='Robot')
    r_dir, = ax.plot([], [], 'r-', linewidth=2)
    traj_line, = ax.plot([], [], 'r-', linewidth=1, alpha=0.5, label='Est Trajectory')
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        ax.set_aspect('equal')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.grid(True)
        
        lx = [lm.pos[0] for lm in m.landmarks]
        ly = [lm.pos[1] for lm in m.landmarks]
        ax.scatter(lx, ly, s=200, marker='*', color='orange', zorder=10, label='Landmarks')
        ax.legend()
        return r_body, r_dir, p_arrows, traj_line, time_text

    def update(frame):
        robot.one_step(TIME_INTERVAL)
        
        rx, ry, rt = robot.pose
        r_body.set_data([rx], [ry])
        r_dir.set_data([rx, rx + 0.5*math.cos(rt)], [ry, ry + 0.5*math.sin(rt)])
        
        px = [p.pose[0] for p in estimator.particles]
        py = [p.pose[1] for p in estimator.particles]
        p_len = [p.weight * len(estimator.particles) for p in estimator.particles]
        pu = [l * math.cos(p.pose[2]) for p, l in zip(estimator.particles, p_len)]
        pv = [l * math.sin(p.pose[2]) for p, l in zip(estimator.particles, p_len)]
        
        p_arrows.set_offsets(np.c_[px, py])
        p_arrows.set_UVC(pu, pv)
        
        hist = agent.poses
        hx = [h[0] for h in hist]
        hy = [h[1] for h in hist]
        traj_line.set_data(hx, hy)
        
        time_text.set_text(f"Step: {frame}")
        
        return r_body, r_dir, p_arrows, traj_line, time_text

    ani = animation.FuncAnimation(fig, update, frames=SIM_STEPS, init_func=init, interval=100, blit=False)
    plt.show()

if __name__ == "__main__":
    main()