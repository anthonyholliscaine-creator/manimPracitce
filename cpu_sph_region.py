# cpu_sph_manim.py
# CPU port of SebLague Episode-01 FluidSim compute solver → Manim visualization
# Uses explicit neighbor-grid (spatial hash) and same kernel formulas as the GPU version.

from manim import *
import numpy as np
import math
import scipy.integrate as integrate
from collections import defaultdict

# --------------------------
# Simulation core (CPU SPH)
# --------------------------


class SPHSimulation2D:
    def __init__(
        self,
        num_particles=None,
        bounds=(12.0, 6.0),
        smoothing_radius=0.5,
        dt=1 / 120,
        gravity=-9.8,
        pressure_multiplier=.01,
        near_pressure_multiplier=90.0,
        viscosity_strength=0.7,
        collision_damping=0.7,
        interaction_point=None,
        interaction_strength=0.0,
        dot_radius=0.012,
        height_func=None,
        max_particles=150,
    ):
        # Store params
        self.dot_radius = float(dot_radius)
        self.bounds = np.array(bounds, dtype=float)
        self.h = float(smoothing_radius)
        self.dt = float(dt)
        self.gravity = float(gravity)
        self.pressure_multiplier = float(pressure_multiplier)
        self.near_pressure_multiplier = float(near_pressure_multiplier)
        self.viscosity_strength = float(viscosity_strength)
        self.collision_damping = float(collision_damping)
        self.interaction_point = (
            np.array(interaction_point, dtype=float)
            if interaction_point is not None
            else None
        )
        self.interaction_strength = float(interaction_strength)
        self.predictionFactor = 1.0/100  # visible dynamics; tune if desired
        self.max_particles = int(max_particles)

        W = float(self.bounds[0])
        H = float(self.bounds[1])

        # height func default
        if height_func is None:
            self.height_func = lambda x: H / 2.0
        else:
            self.height_func = height_func

        # compute area under curve
        area, _ = integrate.quad(self.height_func, 0.0, W)
        area = float(area)
        area = max(area, 1e-8)

        # estimate number of particles from dot_radius (use circle area packing)
        if num_particles is None:
            # packing_factor = 0.9  # ~0.9 for loose packing; tune if you want denser/looser
            # approx_dot_area = math.pi * (self.dot_radius ** 2) * packing_factor
            raw_n = max(1, int(math.floor(area / (4*dot_radius**2))))
            raw_n = min(raw_n, self.max_particles)
            self.n = raw_n
        else:
            self.n = int(max(1, min(int(num_particles), self.max_particles)))

        # allocate arrays based on computed self.n
        self.pos = np.zeros((self.n, 2), dtype=float)
        self.vel = np.zeros((self.n, 2), dtype=float)
        self.pred = np.zeros_like(self.pos)
        self.density = np.zeros((self.n,), dtype=float)
        self.near_density = np.zeros((self.n,), dtype=float)

        # initialize particles under the curve
        self._init_particles()

        # neighbor grid params
        self.cell_size = self.h
        self.grid = defaultdict(list)

        # kernel scaling factors (matching original formulas)
        h = self.h
        self.Poly6ScalingFactor = 4.0 / (math.pi * (h ** 8))
        self.SpikyPow3ScalingFactor = 10.0 / (math.pi * (h ** 5))
        # self.SpikyPow2ScalingFactor = 3.0 / (math.pi * (h ** 3))
        self.SpikyPow2ScalingFactor = 6.0 / (math.pi * (h ** 4))
        self.SpikyPow3DerivativeScalingFactor = 30.0 / (math.pi * (h ** 5))
        self.SpikyPow2DerivativeScalingFactor = 12.0 / (math.pi * (h ** 4))

        # initial predict/densities
        self.pred = self.pos.copy()
        self.build_grid()
        self.compute_densities()

        # to avoid being stuck on wall
        self.wall_perturb_strength = 1.0
        self.wall_perturb_distance = dot_radius

        # rest density: geometric area per particle (consistent with user's request)
        # self.rest_density = area / float(self.n)
        self.rest_density = (self.h/self.dot_radius)**2
        self.step_count = 0

    def _init_particles(self):
        W, H = self.bounds
        dx = 2.1 * self.dot_radius
        dy = 2.1 * self.dot_radius

        xs = np.arange(dx * 0.5, W, dx)
        points = []

        for x in xs:
            h_x = float(self.height_func(x))
            if h_x <= 0.0:
                continue
            if h_x > H:
                h_x = H

            n_rows = max(1, int(math.floor(h_x / dy)))

            if n_rows == 1:
                ys = [h_x * 0.5]
            else:
                ys = np.arange(dy * 0.5, h_x - dy * 0.5, dy)

            for y in ys:
                points.append((x, y))

        # Convert to numpy
        pts = np.array(points, dtype=float)

        # The lattice is now the source of truth
        self.n = pts.shape[0]

        # Allocate arrays AFTER we know n
        self.pos = np.zeros((self.n, 2), dtype=float)
        self.vel = np.zeros((self.n, 2), dtype=float)
        self.pred = np.zeros((self.n, 2), dtype=float)
        self.density = np.zeros((self.n,), dtype=float)
        self.near_density = np.zeros((self.n,), dtype=float)

        # Map to centered coordinates
        centered = np.zeros_like(pts)
        centered[:, 0] = pts[:, 0] - 0.5 * W
        centered[:, 1] = pts[:, 1] - 0.5 * H

        self.pos[:] = centered

        # Small jitter
        self.pos += 0.02 * (np.random.rand(self.n, 2) - 0.5)
        self.vel[:] = 0.0

    # kernels

    def smoothing_poly6(self, r):
        h = self.h
        if r < h:
            v = (h * h - r * r)
            return v * v * v * self.Poly6ScalingFactor
        return 0.0

    def spiky_pow3(self, r):
        h = self.h
        if r < h:
            v = (h - r)
            return v * v * v * self.SpikyPow3ScalingFactor
        return 0.0

    def spiky_pow2(self, r):
        h = self.h
        if r < h:
            v = (h - r)
            return v * v * self.SpikyPow2ScalingFactor
        return 0.0

    def deriv_spiky_pow3(self, r):
        h = self.h
        if r <= h:
            v = (h - r)
            return -v * v * self.SpikyPow3DerivativeScalingFactor
        return 0.0

    def deriv_spiky_pow2(self, r):
        h = self.h
        if r <= h:
            v = (h - r)
            return -v * self.SpikyPow2DerivativeScalingFactor
        return 0.0

    def viscosity_kernel(self, r):
        return self.smoothing_poly6(r)

    # neighbor grid
    def build_grid(self):
        self.grid.clear()
        coords = np.floor(self.pred / self.cell_size).astype(int)
        for i, c in enumerate(coords):
            key = (int(c[0]), int(c[1]))
            self.grid[key].append(i)
        self._coords = coords

    def neighbors_of(self, idx):
        cx, cy = self._coords[idx]
        res = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (cx + dx, cy + dy)
                if key in self.grid:
                    res.extend(self.grid[key])
        return res

    # external forces & predict
    def apply_external_forces_and_predict(self):
        g = np.array((0.0, self.gravity))
        for i in range(self.n):
            accel = g.copy()
            # lateral random force (mean 0, sigma 1)
            accel[0] += np.random.normal(0.0, 1.0)
            if self.interaction_point is not None and self.interaction_strength != 0.0:
                dir_to = self.interaction_point - self.pos[i]
                d = np.linalg.norm(dir_to)
                if d < self.h * 2.0 and d > 1e-6:
                    accel += (dir_to / d) * (
                        self.interaction_strength * (1 - d / (self.h * 2.0))
                    )
            # wall perturbation (left/right walls)
            # x = self.pos[i, 0]  # centered coordinates
            # dist_right = self.bounds[0]/2 - x
            # dist_left = x + self.bounds[0]/2

            # if dist_right < self.wall_perturb_distance:
            #     # strength = 1.0 - dist_right / self.wall_perturb_distance
            #     # accel[0] -= self.wall_perturb_strength * strength
            #     accel[0] -= self.wall_perturb_strength

            # elif dist_left < self.wall_perturb_distance:
            #     # strength = 1.0 - dist_left / self.wall_perturb_distance
            #     # accel[0] += self.wall_perturb_strength * strength
            #     accel[0] += self.wall_perturb_strength

            self.vel[i] += accel * self.dt
            self.pred[i] = self.pos[i] + self.vel[i] * self.predictionFactor
            # self.vel[i] += accel * self.dt
            # self.pred[i] = self.pos[i] + self.vel[i] * self.predictionFactor

    # densities
    def compute_densities(self):
        n = self.n
        self.density.fill(0.0)
        self.near_density.fill(0.0)
        for i in range(n):
            pi = self.pred[i]
            neighs = self.neighbors_of(i)
            den = 0.0
            nden = 0.0
            for j in neighs:
                if j == i:
                    continue
                offset = self.pred[j] - pi
                r2 = offset[0] * offset[0] + offset[1] * offset[1]
                if r2 > self.h * self.h:
                    continue
                r = math.sqrt(r2)
                den += self.spiky_pow2(r)
                nden += self.spiky_pow3(r)
            den += self.spiky_pow2(0.0)
            nden += self.spiky_pow3(0.0)
            self.density[i] = den
            self.near_density[i] = nden

    # pressure forces
    def apply_pressure_forces(self):
        n = self.n
        new_vel = self.vel
        rest = getattr(self, "rest_density", 0.0)
        for i in range(n):
            pi = self.pred[i]
            neighs = self.neighbors_of(i)
            density = self.density[i]
            near_density = self.near_density[i]

            pressure = (density - rest) * self.pressure_multiplier
            near_pressure = self.near_pressure_multiplier * near_density

            pressure_force = np.zeros(2, dtype=float)
            for j in neighs:
                if j == i:
                    continue
                pj = self.pred[j]
                offset = pj - pi
                r2 = offset[0] * offset[0] + offset[1] * offset[1]
                if r2 > self.h * self.h:
                    continue
                r = math.sqrt(r2)
                dir_vec = offset / r if r > 1e-9 else np.array((0.0, 1.0))
                neigh_density = self.density[j] if self.density[j] != 0 else 1e-6
                neigh_near_density = (
                    self.near_density[j] if self.near_density[j] != 0 else 1e-6
                )
                neigh_pressure = (neigh_density - rest) * \
                    self.pressure_multiplier
                neigh_near_pressure = self.near_pressure_multiplier * neigh_near_density

                shared_pressure = 0.5 * (pressure + neigh_pressure)
                shared_near_pressure = 0.5 * \
                    (near_pressure + neigh_near_pressure)

                pressure_force += dir_vec * (
                    self.deriv_spiky_pow2(r) * shared_pressure / neigh_density
                )
                pressure_force += dir_vec * (
                    self.deriv_spiky_pow3(
                        r) * shared_near_pressure / neigh_near_density
                )
            accel = pressure_force / (density if density != 0 else 1e-6)
            new_vel[i] += accel * self.dt

    # viscosity
    def apply_viscosity(self):
        n = self.n
        for i in range(n):
            pi = self.pred[i]
            vel_i = self.vel[i]
            neighs = self.neighbors_of(i)
            viscosity_force = np.zeros(2, dtype=float)
            for j in neighs:
                if j == i:
                    continue
                pj = self.pred[j]
                offset = pj - pi
                r2 = offset[0] * offset[0] + offset[1] * offset[1]
                if r2 > self.h * self.h:
                    continue
                r = math.sqrt(r2)
                vel_j = self.vel[j]
                viscosity_force += (vel_j - vel_i) * self.viscosity_kernel(r)
            self.vel[i] += viscosity_force * self.viscosity_strength * self.dt

    # integrate and collisions
    def integrate_and_collide(self):
        half = 0.5 * self.bounds
        for i in range(self.n):
            self.pos[i] += self.vel[i] * self.dt
            for k in range(2):
                if abs(self.pos[i, k]) > half[k]:
                    self.pos[i, k] = math.copysign(half[k], self.pos[i, k])
                    self.vel[i, k] *= -self.collision_damping

    # single simulation step
    def step(self):
        self.apply_external_forces_and_predict()
        self.build_grid()
        self.compute_densities()
        self.apply_pressure_forces()
        self.apply_viscosity()
        self.integrate_and_collide()


# --------------------------
# Manim Scene wrapper
# --------------------------


class FluidDots(VGroup):
    def __init__(self, sim: SPHSimulation2D, axes: Axes, dot_radius=0.04, **kwargs):
        super().__init__(**kwargs)
        self.sim = sim
        self.axes = axes
        self.W, self.H = float(sim.bounds[0]), float(sim.bounds[1])
        self.dots = VGroup(*[Dot(radius=dot_radius, color=BLUE)
                           for _ in range(sim.n)])
        self.add(self.dots)

    def update_from_sim(self):
        halfW = 0.5 * self.W
        halfH = 0.5 * self.H
        for i, d in enumerate(self.dots):
            x_sim, y_sim = self.sim.pos[i]
            x_ui = x_sim + halfW
            y_ui = y_sim + halfH
            d.move_to(self.axes.c2p(x_ui, y_ui))


class FluidSPHScene(Scene):
    def construct(self):
        W = 6.0

        def f(x):
            # return 1 + 0.5 * x
            # return 1 + 3 * np.sin(np.pi*x/W)
            return (x-3)**2

        # label = MathTex(rf"f(x) = 1 + 3 \cdot \sin(\frac{{\pi}}{{{W}}} x)")
        # label = MathTex(rf"f(x) = 1+\frac{{1}}{{2}} \cdot x")
        label = MathTex(rf"f(x) = (x-3)^2")
        label.color = RED_D
        label.scale(0.8)

        xs = np.linspace(0.0, W, 500)
        max_f = float(np.ceil(max(f(x) for x in xs)))
        H = max(max_f, 0)+1

        axes = Axes(
            x_range=[0, W, 1],
            y_range=[0, H+1, 1],
            x_length=W,
            y_length=W,
            tips=False,
            axis_config={"include_numbers": True},
        )
        axes.to_corner(DL)

        x_label = MathTex("x")
        y_label = MathTex("f(x)")

        # Default x label placement is fine
        x_label.next_to(axes.x_axis, RIGHT)

        # Move y label to vertical center of y-axis
        # y_label.rotate(PI/2)
        y_label.move_to(axes.y_axis.get_center())
        y_label.shift(LEFT * 0.5)   # adjust horizontal spacing if needed

        axis_labels = VGroup(x_label, y_label)

        graph_f = axes.plot(lambda x: f(x), x_range=[
                            0, W], color=PURE_RED, stroke_width=6)
        # graph_f.set_stroke(width=3)
        graph_f.set_z_index(-10)

        # place in upper-right of the axes
        # label.next_to(axes.c2p(W, H), UL, buff=0.2)
        label.to_corner(UR, buff=1)

        area, _ = integrate.quad(f, 0, W)
        avg_f = area / W
        graph_avgf = axes.plot(lambda x: avg_f, x_range=[
                               0, W], color=PURE_YELLOW)
        graph_avgf.set_stroke(width=6)
        graph_avgf.set_z_index(-9)

        labelavg = MathTex(rf"avg(f) = {avg_f}")
        # labelavg = MathTex(rf"avg(f) = \frac{{6+\pi}}{{\pi}}")
        labelavg.color = PURE_YELLOW
        labelavg.next_to(label, DOWN*2)

        border_top = Line(axes.c2p(0, H), axes.c2p(W, H), stroke_width=4)
        border_top.set_color(WHITE)
        border_right = Line(axes.c2p(W, 0), axes.c2p(W, H), stroke_width=4)
        border_right.set_color(WHITE)

        dot_radius = 0.05
        # because the area of circumscribed square is 4r^2
        raw_num = area / (4 * dot_radius ** 2)
        numDots = int(max(1, min(int(math.floor(raw_num)), 2000)))

        sim = SPHSimulation2D(
            num_particles=numDots,
            bounds=(W, H),
            smoothing_radius=2.5*dot_radius,
            dt=.003,
            gravity=-9.8,
            pressure_multiplier=15.0,  # Use 90-100 or so
            near_pressure_multiplier=154.0,  # Use about 170-175
            viscosity_strength=0.34,
            collision_damping=0.9,
            dot_radius=dot_radius,
            height_func=f,
        )

        dots = FluidDots(sim, axes, dot_radius)

        # Note: May need to adjust predictionFactor

        self.add(axes, axis_labels)
        self.add(label)
        self.add(graph_f)
        self.add(border_top, border_right)
        self.add(dots)

        steps_per_frame = 25
        total_frames = 75
        for frame in range(total_frames):
            if frame == 1:
                sim.step()
                dots.update_from_sim()
                self.wait(0.09)
                continue
            elif frame > 1:
                for _ in range(steps_per_frame):
                    sim.step()
            dots.update_from_sim()
            self.wait(0.07)

        graph_avgf.set_z_index(20)
        labelavg.set_z_index(21)

        self.play(Create(graph_avgf), Write(labelavg))
        self.wait(0.5)

        # steps_per_frame = 4
        # total_frames = 400
        # for frame in range(total_frames):
        #     for _ in range(steps_per_frame):
        #         sim.step()
        #     if frame % 5 == 0:
        #         print(f"[frame] {frame}/{total_frames}, n={sim.n}")
        #     dots.update_from_sim()
        #     self.wait(0.07)
