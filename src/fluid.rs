use std::f64::consts::{PI, TAU};

use crate::math::{Aabb, Onb, Ray, SpectralCurve, Spectrum4, Vec3, Wavelengths4};
use crate::rng::Rng;

#[derive(Clone, Copy, Debug, Default)]
pub struct FluidTelemetryFrame {
    pub time_sec: f64,
    pub kinetic_energy: f64,
    pub enstrophy: f64,
    pub peak_vorticity: f64,
    pub total_mass: f64,
    /// Plume center of mass normalized to [-1, 1]^3 inside the simulation volume.
    pub center_of_mass: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoissonSolverKind {
    ConjugateGradient,
    RedBlackGaussSeidel,
}

pub struct FluidSolver3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub h: f64,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub w: Vec<f64>,
    pub density: Vec<f64>,
    pub temperature: Vec<f64>,
    pub pressure: Vec<f64>,
    pub divergence: Vec<f64>,
    curl_x: Vec<f64>,
    curl_y: Vec<f64>,
    curl_z: Vec<f64>,
    curl_mag: Vec<f64>,
    tmp_fwd: Vec<f64>,
    tmp_bwd: Vec<f64>,
    tmp_out: Vec<f64>,
    cg_r: Vec<f64>,
    cg_d: Vec<f64>,
    cg_q: Vec<f64>,
    pub vorticity_eps: f64,
    pub buoyancy_beta: f64,
    pub gravity_alpha: f64,
    pub solver_kind: PoissonSolverKind,
    pub solver_iters: usize,
}

impl FluidSolver3D {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        assert!(nx >= 6 && ny >= 6 && nz >= 6);
        let total = nx * ny * nz;
        let h = 1.0 / (nx.max(ny).max(nz) as f64);
        Self {
            nx,
            ny,
            nz,
            h,
            u: vec![0.0; total],
            v: vec![0.0; total],
            w: vec![0.0; total],
            density: vec![0.0; total],
            temperature: vec![0.0; total],
            pressure: vec![0.0; total],
            divergence: vec![0.0; total],
            curl_x: vec![0.0; total],
            curl_y: vec![0.0; total],
            curl_z: vec![0.0; total],
            curl_mag: vec![0.0; total],
            tmp_fwd: vec![0.0; total],
            tmp_bwd: vec![0.0; total],
            tmp_out: vec![0.0; total],
            cg_r: vec![0.0; total],
            cg_d: vec![0.0; total],
            cg_q: vec![0.0; total],
            vorticity_eps: 5.5,
            buoyancy_beta: 6.8,
            gravity_alpha: 0.45,
            solver_kind: PoissonSolverKind::ConjugateGradient,
            solver_iters: 32,
        }
    }

    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    /// Trilinearly sample any grid-centered scalar field at continuous index coordinates `(x, y, z)`.
    #[inline]
    pub fn sample_trilinear(&self, field: &[f64], x: f64, y: f64, z: f64) -> f64 {
        let xc = x.clamp(0.5, (self.nx as f64) - 1.500_001);
        let yc = y.clamp(0.5, (self.ny as f64) - 1.500_001);
        let zc = z.clamp(0.5, (self.nz as f64) - 1.500_001);

        let i0 = xc.floor() as usize;
        let j0 = yc.floor() as usize;
        let k0 = zc.floor() as usize;
        let i1 = i0 + 1;
        let j1 = j0 + 1;
        let k1 = k0 + 1;

        let tx = xc - (i0 as f64);
        let ty = yc - (j0 as f64);
        let tz = zc - (k0 as f64);

        let c000 = field[self.idx(i0, j0, k0)];
        let c100 = field[self.idx(i1, j0, k0)];
        let c010 = field[self.idx(i0, j1, k0)];
        let c110 = field[self.idx(i1, j1, k0)];
        let c001 = field[self.idx(i0, j0, k1)];
        let c101 = field[self.idx(i1, j0, k1)];
        let c011 = field[self.idx(i0, j1, k1)];
        let c111 = field[self.idx(i1, j1, k1)];

        let c00 = c000 * (1.0 - tx) + c100 * tx;
        let c10 = c010 * (1.0 - tx) + c110 * tx;
        let c01 = c001 * (1.0 - tx) + c101 * tx;
        let c11 = c011 * (1.0 - tx) + c111 * tx;

        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;

        c0 * (1.0 - tz) + c1 * tz
    }

    #[inline]
    fn stencil_minmax(&self, field: &[f64], x: f64, y: f64, z: f64) -> (f64, f64) {
        let xc = x.clamp(0.5, (self.nx as f64) - 1.500_001);
        let yc = y.clamp(0.5, (self.ny as f64) - 1.500_001);
        let zc = z.clamp(0.5, (self.nz as f64) - 1.500_001);
        let i0 = xc.floor() as usize;
        let j0 = yc.floor() as usize;
        let k0 = zc.floor() as usize;
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for dk in 0..=1 {
            for dj in 0..=1 {
                for di in 0..=1 {
                    let v = field[self.idx(i0 + di, j0 + dj, k0 + dk)];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
        }
        (lo, hi)
    }

    /// Semi-Lagrangian single-step advection (`dt_sign` = +1.0 for forward backtrace, -1.0 for reverse).
    fn advect_semi_lagrangian(&self, src: &[f64], dst: &mut [f64], dt: f64) {
        let scale = dt / self.h;
        dst.fill(0.0);
        for k in 1..(self.nz - 1) {
            for j in 1..(self.ny - 1) {
                for i in 1..(self.nx - 1) {
                    let id = self.idx(i, j, k);
                    let bx = (i as f64) - scale * self.u[id];
                    let by = (j as f64) - scale * self.v[id];
                    let bz = (k as f64) - scale * self.w[id];
                    dst[id] = self.sample_trilinear(src, bx, by, bz);
                }
            }
        }
    }

    /// Second-order MacCormack advection with Selle et al. local 8-cell extrema clamping.
    pub fn advect_maccormack_in_place(&mut self, field_kind: u8, dt: f64, dissipation: f64) {
        let src_snapshot: Vec<f64> = match field_kind {
            0 => self.u.clone(),
            1 => self.v.clone(),
            2 => self.w.clone(),
            3 => self.density.clone(),
            _ => self.temperature.clone(),
        };

        let mut fwd = std::mem::take(&mut self.tmp_fwd);
        let mut bwd = std::mem::take(&mut self.tmp_bwd);
        let mut out = std::mem::take(&mut self.tmp_out);

        self.advect_semi_lagrangian(&src_snapshot, &mut fwd, dt);
        self.advect_semi_lagrangian(&fwd, &mut bwd, -dt);

        out.fill(0.0);
        let scale = dt / self.h;
        for k in 1..(self.nz - 1) {
            for j in 1..(self.ny - 1) {
                for i in 1..(self.nx - 1) {
                    let id = self.idx(i, j, k);
                    let bx = (i as f64) - scale * self.u[id];
                    let by = (j as f64) - scale * self.v[id];
                    let bz = (k as f64) - scale * self.w[id];

                    let near_wall = bx < 1.5
                        || bx > (self.nx as f64) - 2.5
                        || by < 1.5
                        || by > (self.ny as f64) - 2.5
                        || bz < 1.5
                        || bz > (self.nz as f64) - 2.5;

                    let val = if near_wall {
                        fwd[id]
                    } else {
                        let mc = fwd[id] + 0.5 * (src_snapshot[id] - bwd[id]);
                        let (lo, hi) = self.stencil_minmax(&src_snapshot, bx, by, bz);
                        mc.clamp(lo, hi)
                    };
                    out[id] = val * dissipation;
                }
            }
        }

        match field_kind {
            0 => self.u.copy_from_slice(&out),
            1 => self.v.copy_from_slice(&out),
            2 => self.w.copy_from_slice(&out),
            3 => {
                for (d, &o) in self.density.iter_mut().zip(out.iter()) {
                    *d = o.max(0.0);
                }
            }
            _ => {
                for (t, &o) in self.temperature.iter_mut().zip(out.iter()) {
                    *t = o.max(0.0);
                }
            }
        }

        self.tmp_fwd = fwd;
        self.tmp_bwd = bwd;
        self.tmp_out = out;
    }

    pub fn enforce_wall_velocity_bc(&mut self) {
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let id = self.idx(i, j, k);
                    if i <= 1 || i + 1 >= self.nx || j == 0 || j + 1 >= self.ny || k == 0 || k + 1 >= self.nz {
                        self.u[id] = 0.0;
                    }
                    if j <= 1 || j + 1 >= self.ny || i == 0 || i + 1 >= self.nx || k == 0 || k + 1 >= self.nz {
                        self.v[id] = 0.0;
                    }
                    if k <= 1 || k + 1 >= self.nz || i == 0 || i + 1 >= self.nx || j == 0 || j + 1 >= self.ny {
                        self.w[id] = 0.0;
                    }
                }
            }
        }
    }

    /// Inject Boussinesq thermal buoyancy + 3D Vorticity Confinement (`f = eps * h * (N x omega)`).
    pub fn apply_buoyancy_and_vorticity_confinement(&mut self, dt: f64) {
        let inv_2h = 0.5 / self.h;

        // 1. Compute 3D curl omega = grad x u and |omega|
        self.curl_mag.fill(0.0);
        for k in 1..(self.nz - 1) {
            for j in 1..(self.ny - 1) {
                for i in 1..(self.nx - 1) {
                    let id = self.idx(i, j, k);
                    let dw_dy = (self.w[self.idx(i, j + 1, k)] - self.w[self.idx(i, j - 1, k)]) * inv_2h;
                    let dv_dz = (self.v[self.idx(i, j, k + 1)] - self.v[self.idx(i, j, k - 1)]) * inv_2h;
                    let du_dz = (self.u[self.idx(i, j, k + 1)] - self.u[self.idx(i, j, k - 1)]) * inv_2h;
                    let dw_dx = (self.w[self.idx(i + 1, j, k)] - self.w[self.idx(i - 1, j, k)]) * inv_2h;
                    let dv_dx = (self.v[self.idx(i + 1, j, k)] - self.v[self.idx(i - 1, j, k)]) * inv_2h;
                    let du_dy = (self.u[self.idx(i, j + 1, k)] - self.u[self.idx(i, j - 1, k)]) * inv_2h;

                    let ox = dw_dy - dv_dz;
                    let oy = du_dz - dw_dx;
                    let oz = dv_dx - du_dy;
                    self.curl_x[id] = ox;
                    self.curl_y[id] = oy;
                    self.curl_z[id] = oz;
                    self.curl_mag[id] = (ox * ox + oy * oy + oz * oz).sqrt();
                }
            }
        }

        // 2. Apply f_conf = eps * h * (N x omega) and Boussinesq upward lift
        let eps_h = self.vorticity_eps * self.h;
        for k in 2..(self.nz - 2) {
            for j in 2..(self.ny - 2) {
                for i in 2..(self.nx - 2) {
                    let id = self.idx(i, j, k);
                    let gx = (self.curl_mag[self.idx(i + 1, j, k)]
                        - self.curl_mag[self.idx(i - 1, j, k)])
                        * inv_2h;
                    let gy = (self.curl_mag[self.idx(i, j + 1, k)]
                        - self.curl_mag[self.idx(i, j - 1, k)])
                        * inv_2h;
                    let gz = (self.curl_mag[self.idx(i, j, k + 1)]
                        - self.curl_mag[self.idx(i, j, k - 1)])
                        * inv_2h;
                    let glen = (gx * gx + gy * gy + gz * gz).sqrt() + 1e-10;
                    let nx = gx / glen;
                    let ny = gy / glen;
                    let nz = gz / glen;

                    let ox = self.curl_x[id];
                    let oy = self.curl_y[id];
                    let oz = self.curl_z[id];

                    let fx = eps_h * (ny * oz - nz * oy);
                    let fy = eps_h * (nz * ox - nx * oz);
                    let fz = eps_h * (nx * oy - ny * ox);

                    self.u[id] += dt * fx;
                    self.v[id] += dt * fy;
                    self.w[id] += dt * fz;
                }
            }
        }

        for k in 1..(self.nz - 1) {
            for j in 1..(self.ny - 1) {
                for i in 1..(self.nx - 1) {
                    let id = self.idx(i, j, k);
                    let lift = self.buoyancy_beta * self.temperature[id]
                        - self.gravity_alpha * self.density[id];
                    self.v[id] += dt * lift;
                }
            }
        }

        self.enforce_wall_velocity_bc();
    }

    /// Compute staggered MAC face-flux divergence `rhs = -h^2 * div(u)` for exact SPD system `(-Lap_h) p = rhs`.
    pub fn compute_divergence_norm(&mut self) -> f64 {
        let inv_h = 1.0 / self.h;
        let mut sum_sq = 0.0;
        self.divergence.fill(0.0);
        for k in 1..(self.nz - 1) {
            for j in 1..(self.ny - 1) {
                for i in 1..(self.nx - 1) {
                    let id = self.idx(i, j, k);
                    let du = self.u[self.idx(i + 1, j, k)] - self.u[id];
                    let dv = self.v[self.idx(i, j + 1, k)] - self.v[id];
                    let dw = self.w[self.idx(i, j, k + 1)] - self.w[id];
                    let div_val = (du + dv + dw) * inv_h;
                    // Store `-h^2 * div(u) = -h * (du + dv + dw)` as exact SPD right-hand side
                    self.divergence[id] = -self.h * (du + dv + dw);
                    sum_sq += div_val * div_val;
                }
            }
        }
        let interior_n = ((self.nx - 2) * (self.ny - 2) * (self.nz - 2)) as f64;
        (sum_sq / interior_n).sqrt()
    }

    /// Evaluate exact MAC discrete negative Laplacian `out = A * x` with pure Neumann solid walls.
    fn apply_spd_laplacian(&self, x: &[f64], out: &mut [f64]) {
        out.fill(0.0);
        for k in 1..(self.nz - 1) {
            for j in 1..(self.ny - 1) {
                for i in 1..(self.nx - 1) {
                    let id = self.idx(i, j, k);
                    let mut diag = 0.0;
                    let mut nb_sum = 0.0;

                    if i > 1 {
                        diag += 1.0;
                        nb_sum += x[self.idx(i - 1, j, k)];
                    }
                    if i + 2 < self.nx {
                        diag += 1.0;
                        nb_sum += x[self.idx(i + 1, j, k)];
                    }
                    if j > 1 {
                        diag += 1.0;
                        nb_sum += x[self.idx(i, j - 1, k)];
                    }
                    if j + 2 < self.ny {
                        diag += 1.0;
                        nb_sum += x[self.idx(i, j + 1, k)];
                    }
                    if k > 1 {
                        diag += 1.0;
                        nb_sum += x[self.idx(i, j, k - 1)];
                    }
                    if k + 2 < self.nz {
                        diag += 1.0;
                        nb_sum += x[self.idx(i, j, k + 1)];
                    }

                    out[id] = diag * x[id] - nb_sum;
                }
            }
        }
    }

    /// Red-Black SOR Gauss-Seidel solver for the MAC Poisson pressure equation.
    pub fn solve_poisson_gauss_seidel(&mut self, iterations: usize) {
        const OMEGA: f64 = 1.65;
        for _ in 0..iterations {
            for color in 0..2 {
                for k in 1..(self.nz - 1) {
                    for j in 1..(self.ny - 1) {
                        for i in 1..(self.nx - 1) {
                            if (i + j + k) % 2 != color {
                                continue;
                            }
                            let id = self.idx(i, j, k);
                            let mut diag = 0.0;
                            let mut nb_sum = 0.0;
                            if i > 1 {
                                diag += 1.0;
                                nb_sum += self.pressure[self.idx(i - 1, j, k)];
                            }
                            if i + 2 < self.nx {
                                diag += 1.0;
                                nb_sum += self.pressure[self.idx(i + 1, j, k)];
                            }
                            if j > 1 {
                                diag += 1.0;
                                nb_sum += self.pressure[self.idx(i, j - 1, k)];
                            }
                            if j + 2 < self.ny {
                                diag += 1.0;
                                nb_sum += self.pressure[self.idx(i, j + 1, k)];
                            }
                            if k > 1 {
                                diag += 1.0;
                                nb_sum += self.pressure[self.idx(i, j, k - 1)];
                            }
                            if k + 2 < self.nz {
                                diag += 1.0;
                                nb_sum += self.pressure[self.idx(i, j, k + 1)];
                            }
                            let gs_target = (self.divergence[id] + nb_sum) / diag;
                            self.pressure[id] =
                                (1.0 - OMEGA) * self.pressure[id] + OMEGA * gs_target;
                        }
                    }
                }
            }
        }
    }

    /// Matrix-free Conjugate Gradient (CG) solver for the SPD Poisson pressure equation.
    pub fn solve_poisson_conjugate_gradient(&mut self, max_iters: usize) {
        let mut r = std::mem::take(&mut self.cg_r);
        let mut d = std::mem::take(&mut self.cg_d);
        let mut q = std::mem::take(&mut self.cg_q);

        // Compute initial residual r = b - A * p (warm-started from previous frame's pressure!)
        let p_snap = std::mem::take(&mut self.pressure);
        self.apply_spd_laplacian(&p_snap, &mut q);
        self.pressure = p_snap;
        let mut delta_new = 0.0;
        for i in 0..r.len() {
            let ri = self.divergence[i] - q[i];
            r[i] = ri;
            d[i] = ri;
            delta_new += ri * ri;
        }

        if delta_new > 1e-24 {
            for _ in 0..max_iters {
                self.apply_spd_laplacian(&d, &mut q);
                let mut dq = 0.0;
                for i in 0..d.len() {
                    dq += d[i] * q[i];
                }
                if dq.abs() < 1e-28 {
                    break;
                }
                let alpha = delta_new / dq;
                let mut delta_next = 0.0;
                for i in 0..self.pressure.len() {
                    self.pressure[i] += alpha * d[i];
                    r[i] -= alpha * q[i];
                    delta_next += r[i] * r[i];
                }
                if delta_next < 1e-20 {
                    break;
                }
                let beta = delta_next / delta_new;
                for i in 0..d.len() {
                    d[i] = r[i] + beta * d[i];
                }
                delta_new = delta_next;
            }
        }

        self.cg_r = r;
        self.cg_d = d;
        self.cg_q = q;
    }

    /// Helmholtz-Hodge Pressure Projection: `u^{n+1} = u* - grad(p)` onto divergence-free subspace.
    pub fn project(&mut self) {
        self.enforce_wall_velocity_bc();
        self.compute_divergence_norm();

        match self.solver_kind {
            PoissonSolverKind::ConjugateGradient => {
                self.solve_poisson_conjugate_gradient(self.solver_iters);
            }
            PoissonSolverKind::RedBlackGaussSeidel => {
                self.solve_poisson_gauss_seidel(self.solver_iters);
            }
        }

        // Subtract exact staggered MAC face pressure gradients: u[i] -= (p[i] - p[i-1]) / h
        let inv_h = 1.0 / self.h;
        for k in 1..(self.nz - 1) {
            for j in 1..(self.ny - 1) {
                for i in 1..(self.nx - 1) {
                    let id = self.idx(i, j, k);
                    let pc = self.pressure[id];
                    if i > 1 {
                        self.u[id] -= (pc - self.pressure[self.idx(i - 1, j, k)]) * inv_h;
                    }
                    if j > 1 {
                        self.v[id] -= (pc - self.pressure[self.idx(i, j - 1, k)]) * inv_h;
                    }
                    if k > 1 {
                        self.w[id] -= (pc - self.pressure[self.idx(i, j, k - 1)]) * inv_h;
                    }
                }
            }
        }
        self.enforce_wall_velocity_bc();
    }

    /// Inject swirling dual cyber-vent plasma/smoke plumes modulated across the 15-second timeline.
    pub fn inject_cyberpunk_sources(&mut self, time_sec: f64, dt: f64) {
        let cx = 0.5 * (self.nx as f64);
        let cz = 0.5 * (self.nz as f64);
        let base_j = 3usize.min(self.ny - 3);

        // Harmonic pulsing + orbiting dual jets creating colliding vortex rings
        let pulse = 0.65 + 0.35 * (time_sec * TAU * 0.55).sin()
            + 0.25 * (time_sec * TAU * 1.35).cos().max(0.0);
        let angle = time_sec * 1.85;
        let orbit_r = (self.nx as f64) * 0.14;

        let jets = [
            (
                cx + orbit_r * angle.cos(),
                (base_j as f64) + 1.5,
                cz + orbit_r * angle.sin(),
                -2.6 * angle.sin(),
                7.2 * pulse,
                2.6 * angle.cos(),
            ),
            (
                cx - orbit_r * angle.cos(),
                (base_j as f64) + 2.0,
                cz - orbit_r * angle.sin(),
                2.6 * angle.sin(),
                6.4 * pulse,
                -2.6 * angle.cos(),
            ),
        ];

        let r_cells = ((self.nx as f64) * 0.105).max(2.2);
        let r2 = r_cells * r_cells;

        for &(jx, jy, jz, vx, vy, vz) in &jets {
            let k_min = ((jz - r_cells - 1.0).floor().max(1.0)) as usize;
            let k_max = ((jz + r_cells + 1.0).ceil().min((self.nz - 2) as f64)) as usize;
            let j_min = ((jy - r_cells - 1.0).floor().max(1.0)) as usize;
            let j_max = ((jy + r_cells + 1.0).ceil().min((self.ny - 2) as f64)) as usize;
            let i_min = ((jx - r_cells - 1.0).floor().max(1.0)) as usize;
            let i_max = ((jx + r_cells + 1.0).ceil().min((self.nx - 2) as f64)) as usize;

            for k in k_min..=k_max {
                for j in j_min..=j_max {
                    for i in i_min..=i_max {
                        let dx = (i as f64) - jx;
                        let dy = (j as f64) - jy;
                        let dz = (k as f64) - jz;
                        let d2 = dx * dx + dy * dy + dz * dz;
                        if d2 < r2 {
                            let w_falloff = (-2.2 * d2 / r2).exp();
                            let id = self.idx(i, j, k);
                            self.density[id] =
                                (self.density[id] + dt * 9.5 * pulse * w_falloff).min(3.8);
                            self.temperature[id] =
                                (self.temperature[id] + dt * 8.2 * pulse * w_falloff).min(4.5);
                            self.u[id] += dt * vx * w_falloff * 4.2;
                            self.v[id] += dt * vy * w_falloff * 4.2;
                            self.w[id] += dt * vz * w_falloff * 4.2;
                        }
                    }
                }
            }
        }
    }

    /// Execute one complete incompressible Navier-Stokes timestep and return telemetry.
    pub fn step(&mut self, sim_dt: f64, timeline_sec: f64) -> FluidTelemetryFrame {
        self.advect_maccormack_in_place(0, sim_dt, 0.996);
        self.advect_maccormack_in_place(1, sim_dt, 0.996);
        self.advect_maccormack_in_place(2, sim_dt, 0.996);

        self.inject_cyberpunk_sources(timeline_sec, sim_dt);
        self.apply_buoyancy_and_vorticity_confinement(sim_dt);
        self.project();

        self.advect_maccormack_in_place(3, sim_dt, 0.996);
        self.advect_maccormack_in_place(4, sim_dt, 0.985);

        self.compute_telemetry(timeline_sec)
    }

    pub fn compute_telemetry(&self, timeline_sec: f64) -> FluidTelemetryFrame {
        let dv = self.h * self.h * self.h;
        let inv_2h = 0.5 / self.h;
        let mut ke = 0.0;
        let mut enstrophy = 0.0;
        let mut peak_vort = 0.0_f64;
        let mut mass = 0.0;
        let mut com = Vec3::ZERO;

        for k in 1..(self.nz - 1) {
            for j in 1..(self.ny - 1) {
                for i in 1..(self.nx - 1) {
                    let id = self.idx(i, j, k);
                    let u2 = self.u[id] * self.u[id]
                        + self.v[id] * self.v[id]
                        + self.w[id] * self.w[id];
                    ke += 0.5 * u2 * dv;

                    let dw_dy =
                        (self.w[self.idx(i, j + 1, k)] - self.w[self.idx(i, j - 1, k)]) * inv_2h;
                    let dv_dz =
                        (self.v[self.idx(i, j, k + 1)] - self.v[self.idx(i, j, k - 1)]) * inv_2h;
                    let du_dz =
                        (self.u[self.idx(i, j, k + 1)] - self.u[self.idx(i, j, k - 1)]) * inv_2h;
                    let dw_dx =
                        (self.w[self.idx(i + 1, j, k)] - self.w[self.idx(i - 1, j, k)]) * inv_2h;
                    let dv_dx =
                        (self.v[self.idx(i + 1, j, k)] - self.v[self.idx(i - 1, j, k)]) * inv_2h;
                    let du_dy =
                        (self.u[self.idx(i, j + 1, k)] - self.u[self.idx(i, j - 1, k)]) * inv_2h;
                    let ox = dw_dy - dv_dz;
                    let oy = du_dz - dw_dx;
                    let oz = dv_dx - du_dy;
                    let w2 = ox * ox + oy * oy + oz * oz;
                    enstrophy += 0.5 * w2 * dv;
                    peak_vort = peak_vort.max(w2.sqrt());

                    let rho = self.density[id];
                    if rho > 1e-8 {
                        mass += rho * dv;
                        let nx_norm = 2.0 * ((i as f64) / ((self.nx - 1) as f64)) - 1.0;
                        let ny_norm = 2.0 * ((j as f64) / ((self.ny - 1) as f64)) - 1.0;
                        let nz_norm = 2.0 * ((k as f64) / ((self.nz - 1) as f64)) - 1.0;
                        com += Vec3::new(nx_norm, ny_norm, nz_norm) * (rho * dv);
                    }
                }
            }
        }
        if mass > 1e-12 {
            com /= mass;
        }
        FluidTelemetryFrame {
            time_sec: timeline_sec,
            kinetic_energy: ke,
            enstrophy,
            peak_vorticity: peak_vort,
            total_mass: mass,
            center_of_mass: com.clamp(-1.0, 1.0),
        }
    }
}

// ============================================================================
// Heterogeneous Participating Medium Coupled to the Spectral Path Tracer
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct MediumInteraction {
    pub t: f64,
    pub point: Vec3,
    pub albedo: Spectrum4,
    pub emission: Spectrum4,
}

pub struct FluidMedium {
    pub bounds: Aabb,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub density: Vec<f64>,
    pub temperature: Vec<f64>,
    pub max_density: f64,
    pub sigma_t_scale: f64,
    pub majorant_sigma_t: f64,
    pub phase_g: f64,
    pub scatter_albedo: SpectralCurve,
    pub hot_core_emission: SpectralCurve,
}

impl FluidMedium {
    pub fn from_solver(solver: &FluidSolver3D, bounds: Aabb, sigma_t_scale: f64, phase_g: f64) -> Self {
        let mut max_density = 0.0_f64;
        for &d in &solver.density {
            max_density = max_density.max(d);
        }
        let majorant_sigma_t = (max_density * sigma_t_scale).max(1e-4);
        Self {
            bounds,
            nx: solver.nx,
            ny: solver.ny,
            nz: solver.nz,
            density: solver.density.clone(),
            temperature: solver.temperature.clone(),
            max_density,
            sigma_t_scale,
            majorant_sigma_t,
            phase_g: phase_g.clamp(-0.9, 0.9),
            // Cool cyan-violet atmospheric mist albedo + hot magenta-cyan plasma core glow
            scatter_albedo: SpectralCurve::rgb(0.84, 0.90, 0.96),
            hot_core_emission: SpectralCurve::NeonDischarge {
                continuum: Vec3::new(0.18, 0.72, 1.15),
                peak1_nm: 486.0, // Hydrogen-beta / cyan plasma line
                sigma1_nm: 14.0,
                amp1: 1.6,
                peak2_nm: 618.0, // Neon magenta-red core filament
                sigma2_nm: 16.0,
                amp2: 1.1,
            },
        }
    }

    #[inline]
    fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.ny + j) * self.nx + i
    }

    #[inline]
    pub fn sample_density_and_temp(&self, p: Vec3) -> (f64, f64) {
        if p.x < self.bounds.min.x
            || p.x > self.bounds.max.x
            || p.y < self.bounds.min.y
            || p.y > self.bounds.max.y
            || p.z < self.bounds.min.z
            || p.z > self.bounds.max.z
        {
            return (0.0, 0.0);
        }
        let ext = self.bounds.extent();
        let ux = (p.x - self.bounds.min.x) / ext.x.max(1e-9);
        let uy = (p.y - self.bounds.min.y) / ext.y.max(1e-9);
        let uz = (p.z - self.bounds.min.z) / ext.z.max(1e-9);

        // Smooth edge feathering so bounding-box walls never create hard volumetric edges
        let fx = (4.0 * ux * (1.0 - ux)).clamp(0.0, 1.0);
        let fz = (4.0 * uz * (1.0 - uz)).clamp(0.0, 1.0);
        let fy = (1.0 - ((uy - 0.95).max(0.0) / 0.05)).clamp(0.0, 1.0);
        let feather = (fx * fz * fy).sqrt();

        let gx = (ux * ((self.nx - 1) as f64)).clamp(0.5, (self.nx as f64) - 1.500_001);
        let gy = (uy * ((self.ny - 1) as f64)).clamp(0.5, (self.ny as f64) - 1.500_001);
        let gz = (uz * ((self.nz - 1) as f64)).clamp(0.5, (self.nz as f64) - 1.500_001);

        let i0 = gx.floor() as usize;
        let j0 = gy.floor() as usize;
        let k0 = gz.floor() as usize;
        let tx = gx - (i0 as f64);
        let ty = gy - (j0 as f64);
        let tz = gz - (k0 as f64);

        let trilin = |buf: &[f64]| -> f64 {
            let c000 = buf[self.idx(i0, j0, k0)];
            let c100 = buf[self.idx(i0 + 1, j0, k0)];
            let c010 = buf[self.idx(i0, j0 + 1, k0)];
            let c110 = buf[self.idx(i0 + 1, j0 + 1, k0)];
            let c001 = buf[self.idx(i0, j0, k0 + 1)];
            let c101 = buf[self.idx(i0 + 1, j0, k0 + 1)];
            let c011 = buf[self.idx(i0, j0 + 1, k0 + 1)];
            let c111 = buf[self.idx(i0 + 1, j0 + 1, k0 + 1)];

            let c00 = c000 * (1.0 - tx) + c100 * tx;
            let c10 = c010 * (1.0 - tx) + c110 * tx;
            let c01 = c001 * (1.0 - tx) + c101 * tx;
            let c11 = c011 * (1.0 - tx) + c111 * tx;
            (c00 * (1.0 - ty) + c10 * ty) * (1.0 - tz) + (c01 * (1.0 - ty) + c11 * ty) * tz
        };

        (trilin(&self.density) * feather, trilin(&self.temperature) * feather)
    }

    /// Woodcock Delta Tracking free-flight distance sampler in `[ray.t_min, surface_t]`.
    pub fn sample_interaction(
        &self,
        ray: &Ray,
        surface_t: f64,
        wl: &Wavelengths4,
        rng: &mut Rng,
    ) -> Option<MediumInteraction> {
        if self.max_density <= 1e-6 {
            return None;
        }
        let (t_enter, t_exit) = self.bounds.intersect(ray, ray.t_min, surface_t)?;
        let mut t = t_enter.max(ray.t_min);
        let limit = t_exit.min(surface_t);
        if t >= limit {
            return None;
        }

        let inv_maj = 1.0 / self.majorant_sigma_t;
        for _ in 0..256 {
            let u = (1.0 - rng.next_f64()).max(1e-15);
            t -= u.ln() * inv_maj;
            if t >= limit {
                return None;
            }
            let p = ray.at(t);
            let (rho, temp) = self.sample_density_and_temp(p);
            let sigma_t = rho * self.sigma_t_scale;
            if rng.next_f64() * self.majorant_sigma_t < sigma_t {
                let albedo = self.scatter_albedo.eval4(wl);
                // Incandescent plasma core volumetric glow where thermal field `temp` is intense
                let glow_factor = (temp - 0.35).max(0.0).powi(2) * 0.95;
                let emission = if glow_factor > 1e-5 {
                    self.hot_core_emission.eval4(wl) * glow_factor
                } else {
                    Spectrum4::ZERO
                };
                return Some(MediumInteraction {
                    t,
                    point: p,
                    albedo,
                    emission,
                });
            }
        }
        None
    }

    /// Unbiased Novak et al. (2014) Ratio Tracking volumetric shadow transmittance `T in [0, 1]`.
    pub fn transmittance(&self, ray: &Ray, max_dist: f64, rng: &mut Rng) -> f64 {
        if self.max_density <= 1e-6 {
            return 1.0;
        }
        let Some((t_enter, t_exit)) = self.bounds.intersect(ray, ray.t_min, max_dist) else {
            return 1.0;
        };
        let mut t = t_enter.max(ray.t_min);
        let limit = t_exit.min(max_dist);
        if t >= limit {
            return 1.0;
        }

        let mut tr = 1.0;
        let inv_maj = 1.0 / self.majorant_sigma_t;
        for _ in 0..256 {
            let u = (1.0 - rng.next_f64()).max(1e-15);
            t -= u.ln() * inv_maj;
            if t >= limit {
                break;
            }
            let (rho, _) = self.sample_density_and_temp(ray.at(t));
            let sigma_t = (rho * self.sigma_t_scale).min(self.majorant_sigma_t);
            tr *= 1.0 - sigma_t * inv_maj;
            if tr < 0.05 {
                let q = (1.0 - tr).min(0.95);
                if rng.next_f64() < q {
                    return 0.0;
                }
                tr /= 1.0 - q;
            }
        }
        tr.clamp(0.0, 1.0)
    }

    /// Henyey-Greenstein phase function PDF & value (`p(cos_theta)` [sr^-1]).
    #[inline]
    pub fn phase_hg_pdf(&self, wo: Vec3, wi: Vec3) -> f64 {
        let g = self.phase_g;
        if g.abs() < 1e-4 {
            return 1.0 / (4.0 * PI);
        }
        // Note: `wo` points back along incoming ray (-ray.dir), `wi` points toward scattered ray;
        // scattering deflection cosine is `(-wo).dot(wi)`.
        let cos_theta = (-wo).dot(wi).clamp(-1.0, 1.0);
        let denom = (1.0 + g * g - 2.0 * g * cos_theta).max(1e-12);
        (1.0 - g * g) / (4.0 * PI * denom * denom.sqrt())
    }

    /// Exact inverse-CDF Henyey-Greenstein direction sampler.
    #[inline]
    pub fn sample_phase_hg(&self, wo: Vec3, rng: &mut Rng) -> (Vec3, f64) {
        let g = self.phase_g;
        if g.abs() < 1e-4 {
            let (x, y, z) = rng.uniform_sphere();
            return (Vec3::new(x, y, z), 1.0 / (4.0 * PI));
        }
        let s = 2.0 * rng.next_f64() - 1.0;
        let sqr = (1.0 - g * g) / (1.0 + g * s);
        let cos_theta = ((1.0 + g * g - sqr * sqr) / (2.0 * g)).clamp(-1.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = TAU * rng.next_f64();

        let forward = (-wo).normalize();
        let onb = Onb::from_normal(forward);
        let local = Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
        let wi = onb.to_world(local).normalize();
        let pdf = self.phase_hg_pdf(wo, wi);
        (wi, pdf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_pressure_projection_cg_and_gauss_seidel() {
        for kind in [
            PoissonSolverKind::ConjugateGradient,
            PoissonSolverKind::RedBlackGaussSeidel,
        ] {
            let mut solver = FluidSolver3D::new(18, 22, 18);
            solver.solver_kind = kind;
            solver.solver_iters = if kind == PoissonSolverKind::ConjugateGradient {
                45
            } else {
                90
            };

            // Inject strongly divergent localized velocity burst
            for k in 6..12 {
                for j in 5..11 {
                    for i in 6..12 {
                        let id = solver.idx(i, j, k);
                        solver.u[id] = ((i as f64) - 8.5) * 3.0;
                        solver.v[id] = ((j as f64) - 7.5) * 4.5;
                        solver.w[id] = ((k as f64) - 8.5) * 2.5;
                    }
                }
            }
            solver.enforce_wall_velocity_bc();
            let div_before = solver.compute_divergence_norm();
            assert!(div_before > 1.0);

            solver.project();
            let div_after = solver.compute_divergence_norm();
            assert!(
                div_after < div_before * 0.02,
                "{:?}: divergence before={} after={}",
                kind,
                div_before,
                div_after
            );
        }
    }

    #[test]
    fn test_maccormack_positivity_and_vorticity_confinement() {
        let mut solver = FluidSolver3D::new(16, 20, 16);
        let f1 = solver.step(0.08, 0.2);
        let f2 = solver.step(0.08, 0.4);
        assert!(f1.kinetic_energy > 0.0);
        assert!(f2.enstrophy > 0.0);
        for &d in &solver.density {
            assert!(d >= 0.0 && d.is_finite());
        }
    }
}
