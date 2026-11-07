use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

// ============================================================================
// 3D & 4D Vector Algebra
// ============================================================================

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x: 0.0, y: 0.0, z: 0.0 };
    pub const ONE: Self = Self { x: 1.0, y: 1.0, z: 1.0 };
    pub const X: Self = Self { x: 1.0, y: 0.0, z: 0.0 };
    pub const Y: Self = Self { x: 0.0, y: 1.0, z: 0.0 };
    pub const Z: Self = Self { x: 0.0, y: 0.0, z: 1.0 };

    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub const fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x.mul_add(rhs.x, self.y.mul_add(rhs.y, self.z * rhs.z))
    }

    #[inline]
    pub fn cross(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 1e-18 {
            self * (1.0 / len)
        } else {
            Self::Z
        }
    }

    #[inline]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }

    #[inline]
    pub fn exp(self) -> Self {
        Self::new(self.x.exp(), self.y.exp(), self.z.exp())
    }

    #[inline]
    pub fn sqrt(self) -> Self {
        Self::new(self.x.max(0.0).sqrt(), self.y.max(0.0).sqrt(), self.z.max(0.0).sqrt())
    }

    #[inline]
    pub fn min_elem(self, rhs: Self) -> Self {
        Self::new(self.x.min(rhs.x), self.y.min(rhs.y), self.z.min(rhs.z))
    }

    #[inline]
    pub fn max_elem(self, rhs: Self) -> Self {
        Self::new(self.x.max(rhs.x), self.y.max(rhs.y), self.z.max(rhs.z))
    }

    #[inline]
    pub fn max_comp(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }

    #[inline]
    pub fn min_comp(self) -> f64 {
        self.x.min(self.y).min(self.z)
    }

    #[inline]
    pub fn clamp(self, lo: f64, hi: f64) -> Self {
        Self::new(self.x.clamp(lo, hi), self.y.clamp(lo, hi), self.z.clamp(lo, hi))
    }

    #[inline]
    pub fn lerp(self, rhs: Self, t: f64) -> Self {
        self * (1.0 - t) + rhs * t
    }

    #[inline]
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * (2.0 * self.dot(normal))
    }

    /// Snell's law vector refraction. `incident` points toward surface (unit), `normal` points against incident (unit),
    /// `eta` = eta_i / eta_t. Returns `None` on Total Internal Reflection.
    #[inline]
    pub fn refract(incident: Self, normal: Self, eta: f64) -> Option<Self> {
        let cos_i = (-incident.dot(normal)).clamp(-1.0, 1.0);
        let sin2_t = eta * eta * (1.0 - cos_i * cos_i).max(0.0);
        if sin2_t > 1.0 {
            return None;
        }
        let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
        Some((incident * eta + normal * (eta * cos_i - cos_t)).normalize())
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl Index<usize> for Vec3 {
    type Output = f64;
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => &self.z,
        }
    }
}

impl IndexMut<usize> for Vec3 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => &mut self.z,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3 {
        rhs * self
    }
}

impl MulAssign<f64> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        let inv = 1.0 / rhs;
        self * inv
    }
}

impl DivAssign<f64> for Vec3 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn dot(self, rhs: Self) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }

    #[inline]
    pub fn xyz(self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
}

// ============================================================================
// 3x3 & 4x4 Matrices
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3 {
    pub m: [[f64; 3]; 3],
}

impl Mat3 {
    pub const IDENTITY: Self = Self {
        m: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    };

    #[inline]
    pub fn mul_vec3(&self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z,
        )
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        Self {
            m: [
                [self.m[0][0], self.m[1][0], self.m[2][0]],
                [self.m[0][1], self.m[1][1], self.m[2][1]],
                [self.m[0][2], self.m[1][2], self.m[2][2]],
            ],
        }
    }

    #[inline]
    pub fn determinant(&self) -> f64 {
        self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1])
            - self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0])
            + self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    pub m: [[f64; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    #[inline]
    pub fn translation(t: Vec3) -> Self {
        Self {
            m: [
                [1.0, 0.0, 0.0, t.x],
                [0.0, 1.0, 0.0, t.y],
                [0.0, 0.0, 1.0, t.z],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[inline]
    pub fn scale(s: Vec3) -> Self {
        Self {
            m: [
                [s.x, 0.0, 0.0, 0.0],
                [0.0, s.y, 0.0, 0.0],
                [0.0, 0.0, s.z, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[inline]
    pub fn rotation_x(rad: f64) -> Self {
        let (s, c) = rad.sin_cos();
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, -s, 0.0],
                [0.0, s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[inline]
    pub fn rotation_y(rad: f64) -> Self {
        let (s, c) = rad.sin_cos();
        Self {
            m: [
                [c, 0.0, s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [-s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[inline]
    pub fn rotation_z(rad: f64) -> Self {
        let (s, c) = rad.sin_cos();
        Self {
            m: [
                [c, -s, 0.0, 0.0],
                [s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[inline]
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Self {
        let f = (target - eye).normalize();
        let r = f.cross(up).normalize();
        let u = r.cross(f);
        Self {
            m: [
                [r.x, r.y, r.z, -r.dot(eye)],
                [u.x, u.y, u.z, -u.dot(eye)],
                [-f.x, -f.y, -f.z, f.dot(eye)],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    #[inline]
    pub fn perspective(fov_y_rad: f64, aspect: f64, near: f64, far: f64) -> Self {
        let f = 1.0 / (0.5 * fov_y_rad).tan();
        let nf = 1.0 / (near - far);
        Self {
            m: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, (far + near) * nf, 2.0 * far * near * nf],
                [0.0, 0.0, -1.0, 0.0],
            ],
        }
    }

    #[inline]
    pub fn transpose(&self) -> Self {
        let mut out = [[0.0; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                out[r][c] = self.m[c][r];
            }
        }
        Self { m: out }
    }

    #[inline]
    pub fn mul_vec4(&self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.m[0][0] * v.x + self.m[0][1] * v.y + self.m[0][2] * v.z + self.m[0][3] * v.w,
            self.m[1][0] * v.x + self.m[1][1] * v.y + self.m[1][2] * v.z + self.m[1][3] * v.w,
            self.m[2][0] * v.x + self.m[2][1] * v.y + self.m[2][2] * v.z + self.m[2][3] * v.w,
            self.m[3][0] * v.x + self.m[3][1] * v.y + self.m[3][2] * v.z + self.m[3][3] * v.w,
        )
    }

    #[inline]
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let v = self.mul_vec4(Vec4::new(p.x, p.y, p.z, 1.0));
        if (v.w - 1.0).abs() < 1e-15 || v.w.abs() < 1e-18 {
            v.xyz()
        } else {
            v.xyz() / v.w
        }
    }

    #[inline]
    pub fn transform_vector(&self, d: Vec3) -> Vec3 {
        Vec3::new(
            self.m[0][0] * d.x + self.m[0][1] * d.y + self.m[0][2] * d.z,
            self.m[1][0] * d.x + self.m[1][1] * d.y + self.m[1][2] * d.z,
            self.m[2][0] * d.x + self.m[2][1] * d.y + self.m[2][2] * d.z,
        )
    }

    /// Transform normal using an already-inverted matrix (`inv_matrix.transpose()`).
    #[inline]
    pub fn transform_normal_with_inverse(inv_matrix: &Self, n: Vec3) -> Vec3 {
        Vec3::new(
            inv_matrix.m[0][0] * n.x + inv_matrix.m[1][0] * n.y + inv_matrix.m[2][0] * n.z,
            inv_matrix.m[0][1] * n.x + inv_matrix.m[1][1] * n.y + inv_matrix.m[2][1] * n.z,
            inv_matrix.m[0][2] * n.x + inv_matrix.m[1][2] * n.y + inv_matrix.m[2][2] * n.z,
        )
        .normalize()
    }

    /// Full 4x4 cofactor/adjugate analytical matrix inversion.
    pub fn inverse(&self) -> Option<Self> {
        let a = &self.m;
        let s0 = a[0][0] * a[1][1] - a[1][0] * a[0][1];
        let s1 = a[0][0] * a[1][2] - a[1][0] * a[0][2];
        let s2 = a[0][0] * a[1][3] - a[1][0] * a[0][3];
        let s3 = a[0][1] * a[1][2] - a[1][1] * a[0][2];
        let s4 = a[0][1] * a[1][3] - a[1][1] * a[0][3];
        let s5 = a[0][2] * a[1][3] - a[1][2] * a[0][3];

        let c5 = a[2][2] * a[3][3] - a[3][2] * a[2][3];
        let c4 = a[2][1] * a[3][3] - a[3][1] * a[2][3];
        let c3 = a[2][1] * a[3][2] - a[3][1] * a[2][2];
        let c2 = a[2][0] * a[3][3] - a[3][0] * a[2][3];
        let c1 = a[2][0] * a[3][2] - a[3][0] * a[2][2];
        let c0 = a[2][0] * a[3][1] - a[3][0] * a[2][1];

        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
        if det.abs() < 1e-18 {
            return None;
        }
        let inv_det = 1.0 / det;

        Some(Self {
            m: [
                [
                    (a[1][1] * c5 - a[1][2] * c4 + a[1][3] * c3) * inv_det,
                    (-a[0][1] * c5 + a[0][2] * c4 - a[0][3] * c3) * inv_det,
                    (a[3][1] * s5 - a[3][2] * s4 + a[3][3] * s3) * inv_det,
                    (-a[2][1] * s5 + a[2][2] * s4 - a[2][3] * s3) * inv_det,
                ],
                [
                    (-a[1][0] * c5 + a[1][2] * c2 - a[1][3] * c1) * inv_det,
                    (a[0][0] * c5 - a[0][2] * c2 + a[0][3] * c1) * inv_det,
                    (-a[3][0] * s5 + a[3][2] * s2 - a[3][3] * s1) * inv_det,
                    (a[2][0] * s5 - a[2][2] * s2 + a[2][3] * s1) * inv_det,
                ],
                [
                    (a[1][0] * c4 - a[1][1] * c2 + a[1][3] * c0) * inv_det,
                    (-a[0][0] * c4 + a[0][1] * c2 - a[0][3] * c0) * inv_det,
                    (a[3][0] * s4 - a[3][1] * s2 + a[3][3] * s0) * inv_det,
                    (-a[2][0] * s4 + a[2][1] * s2 - a[2][3] * s0) * inv_det,
                ],
                [
                    (-a[1][0] * c3 + a[1][1] * c1 - a[1][2] * c0) * inv_det,
                    (a[0][0] * c3 - a[0][1] * c1 + a[0][2] * c0) * inv_det,
                    (-a[3][0] * s3 + a[3][1] * s1 - a[3][2] * s0) * inv_det,
                    (a[2][0] * s3 - a[2][1] * s1 + a[2][2] * s0) * inv_det,
                ],
            ],
        })
    }
}

impl Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut out = [[0.0; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                out[r][c] = self.m[r][0] * rhs.m[0][c]
                    + self.m[r][1] * rhs.m[1][c]
                    + self.m[r][2] * rhs.m[2][c]
                    + self.m[r][3] * rhs.m[3][c];
            }
        }
        Self { m: out }
    }
}

// ============================================================================
// Quaternion Algebra
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle_rad: f64) -> Self {
        let u = axis.normalize();
        let (s, c) = (0.5 * angle_rad).sin_cos();
        Self::new(c, u.x * s, u.y * s, u.z * s).normalize()
    }

    #[inline]
    pub fn from_euler_yxz(yaw: f64, pitch: f64, roll: f64) -> Self {
        let qy = Self::from_axis_angle(Vec3::Y, yaw);
        let qx = Self::from_axis_angle(Vec3::X, pitch);
        let qz = Self::from_axis_angle(Vec3::Z, roll);
        (qy * qx * qz).normalize()
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    #[inline]
    pub fn norm_squared(self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let n = self.norm();
        if n > 1e-18 {
            let inv = 1.0 / n;
            Self::new(self.w * inv, self.x * inv, self.y * inv, self.z * inv)
        } else {
            Self::IDENTITY
        }
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let n2 = self.norm_squared();
        if n2 > 1e-18 {
            let c = self.conjugate();
            Self::new(c.w / n2, c.x / n2, c.y / n2, c.z / n2)
        } else {
            Self::IDENTITY
        }
    }

    /// Fast Rodrigues-equivalent quaternion vector rotation `q * v * q^-1`.
    #[inline]
    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        let qv = Vec3::new(self.x, self.y, self.z);
        let t = 2.0 * qv.cross(v);
        v + self.w * t + qv.cross(t)
    }

    pub fn slerp(self, mut end: Self, t: f64) -> Self {
        let q1 = self.normalize();
        end = end.normalize();
        let mut dot = q1.w * end.w + q1.x * end.x + q1.y * end.y + q1.z * end.z;
        if dot < 0.0 {
            end = Self::new(-end.w, -end.x, -end.y, -end.z);
            dot = -dot;
        }
        if dot > 0.9995 {
            return Self::new(
                q1.w + t * (end.w - q1.w),
                q1.x + t * (end.x - q1.x),
                q1.y + t * (end.y - q1.y),
                q1.z + t * (end.z - q1.z),
            )
            .normalize();
        }
        let theta_0 = dot.clamp(-1.0, 1.0).acos();
        let theta = theta_0 * t;
        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();
        let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;
        Self::new(
            s0 * q1.w + s1 * end.w,
            s0 * q1.x + s1 * end.x,
            s0 * q1.y + s1 * end.y,
            s0 * q1.z + s1 * end.z,
        )
        .normalize()
    }

    #[inline]
    pub fn to_mat3(self) -> Mat3 {
        let q = self.normalize();
        let xx = q.x * q.x;
        let yy = q.y * q.y;
        let zz = q.z * q.z;
        let xy = q.x * q.y;
        let xz = q.x * q.z;
        let yz = q.y * q.z;
        let wx = q.w * q.x;
        let wy = q.w * q.y;
        let wz = q.w * q.z;
        Mat3 {
            m: [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
        }
    }

    #[inline]
    pub fn to_mat4(self) -> Mat4 {
        let r = self.to_mat3().m;
        Mat4 {
            m: [
                [r[0][0], r[0][1], r[0][2], 0.0],
                [r[1][0], r[1][1], r[1][2], 0.0],
                [r[2][0], r[2][1], r[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }
}

impl Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        )
    }
}

// ============================================================================
// Branchless Orthonormal Basis (Duff et al., JCGT 2017)
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct Onb {
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}

impl Onb {
    #[inline]
    pub fn from_normal(normal: Vec3) -> Self {
        let w = normal.normalize();
        let sign = (1.0_f64).copysign(w.z);
        let a = -1.0 / (sign + w.z);
        let b = w.x * w.y * a;
        let u = Vec3::new(1.0 + sign * w.x * w.x * a, sign * b, -sign * w.x);
        let v = Vec3::new(b, sign + w.y * w.y * a, -w.y);
        Self { u, v, w }
    }

    #[inline]
    pub fn to_world(&self, local: Vec3) -> Vec3 {
        self.u * local.x + self.v * local.y + self.w * local.z
    }

    #[inline]
    pub fn to_local(&self, world: Vec3) -> Vec3 {
        Vec3::new(world.dot(self.u), world.dot(self.v), world.dot(self.w))
    }
}

// ============================================================================
// Ray & Axis-Aligned Bounding Box (AABB)
// ============================================================================

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
    pub inv_dir: Vec3,
    pub t_min: f64,
    pub t_max: f64,
}

impl Ray {
    #[inline]
    pub fn new(origin: Vec3, dir: Vec3, t_min: f64, t_max: f64) -> Self {
        let d = dir.normalize();
        let inv_dir = Vec3::new(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        Self {
            origin,
            dir: d,
            inv_dir,
            t_min,
            t_max,
        }
    }

    #[inline]
    pub fn at(&self, t: f64) -> Vec3 {
        self.origin + self.dir * t
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub const EMPTY: Self = Self {
        min: Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        max: Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
    };

    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self {
            min: min.min_elem(max),
            max: min.max_elem(max),
        }
    }

    #[inline]
    pub fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min_elem(other.min),
            max: self.max.max_elem(other.max),
        }
    }

    #[inline]
    pub fn union_point(self, p: Vec3) -> Self {
        Self {
            min: self.min.min_elem(p),
            max: self.max.max_elem(p),
        }
    }

    #[inline]
    pub fn pad(self, delta: f64) -> Self {
        let d = Vec3::splat(delta);
        Self {
            min: self.min - d,
            max: self.max + d,
        }
    }

    #[inline]
    pub fn centroid(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn extent(&self) -> Vec3 {
        (self.max - self.min).max_elem(Vec3::ZERO)
    }

    #[inline]
    pub fn surface_area(&self) -> f64 {
        if self.min.x > self.max.x || self.min.y > self.max.y || self.min.z > self.max.z {
            return 0.0;
        }
        let d = self.extent();
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }

    #[inline]
    pub fn longest_axis(&self) -> usize {
        let d = self.extent();
        if d.x >= d.y && d.x >= d.z {
            0
        } else if d.y >= d.z {
            1
        } else {
            2
        }
    }

    /// IEEE-754 robust Kay-Kajiya slab intersection test. Returns `Some((t_enter, t_exit))`.
    #[inline]
    pub fn intersect(&self, ray: &Ray, mut t_min: f64, mut t_max: f64) -> Option<(f64, f64)> {
        let t0x = (self.min.x - ray.origin.x) * ray.inv_dir.x;
        let t1x = (self.max.x - ray.origin.x) * ray.inv_dir.x;
        let tmin_x = t0x.min(t1x);
        let tmax_x = t0x.max(t1x);
        t_min = t_min.max(tmin_x);
        t_max = t_max.min(tmax_x);
        if t_max < t_min {
            return None;
        }

        let t0y = (self.min.y - ray.origin.y) * ray.inv_dir.y;
        let t1y = (self.max.y - ray.origin.y) * ray.inv_dir.y;
        let tmin_y = t0y.min(t1y);
        let tmax_y = t0y.max(t1y);
        t_min = t_min.max(tmin_y);
        t_max = t_max.min(tmax_y);
        if t_max < t_min {
            return None;
        }

        let t0z = (self.min.z - ray.origin.z) * ray.inv_dir.z;
        let t1z = (self.max.z - ray.origin.z) * ray.inv_dir.z;
        let tmin_z = t0z.min(t1z);
        let tmax_z = t0z.max(t1z);
        t_min = t_min.max(tmin_z);
        t_max = t_max.min(tmax_z);

        if t_max >= t_min {
            Some((t_min, t_max))
        } else {
            None
        }
    }
}

// ============================================================================
// Hero-Wavelength Spectral Radiometry (380nm - 750nm) & ACES Filmic Tone Mapping
// ============================================================================

pub const LAMBDA_MIN: f64 = 380.0;
pub const LAMBDA_MAX: f64 = 750.0;
pub const LAMBDA_SPAN: f64 = LAMBDA_MAX - LAMBDA_MIN;

/// Pre-integrated CIE 1931 analytic observer integrals on [380, 750] nm (Wyman-Sloan-Shirley 2013):
/// Used to normalize Equal-Energy spectral power to D65 standard observer whitepoint (0.95047, 1.0, 1.08883).
const CIE_X_INTEGRAL: f64 = 106.597_642_6;
const CIE_Y_INTEGRAL: f64 = 106.850_917_4;
const CIE_Z_INTEGRAL: f64 = 106.334_644_8;

/// 4 stratified wavelengths [nm] for Hero Wavelength Spectral Sampling (Wilkie et al. 2014).
#[derive(Clone, Copy, Debug)]
pub struct Wavelengths4 {
    pub lambda: [f64; 4],
    pub hero_only: bool,
}

impl Wavelengths4 {
    #[inline]
    pub fn sample(u: f64) -> Self {
        let base = u.clamp(0.0, 0.999_999_999_999) * LAMBDA_SPAN;
        let step = LAMBDA_SPAN * 0.25;
        Self {
            lambda: [
                LAMBDA_MIN + base,
                LAMBDA_MIN + ((base + step) % LAMBDA_SPAN),
                LAMBDA_MIN + ((base + 2.0 * step) % LAMBDA_SPAN),
                LAMBDA_MIN + ((base + 3.0 * step) % LAMBDA_SPAN),
            ],
            hero_only: false,
        }
    }

    #[inline]
    pub fn hero(&self) -> f64 {
        self.lambda[0]
    }
}

/// 4-channel spectral radiance / throughput packet matching `Wavelengths4`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spectrum4 {
    pub c: [f64; 4],
}

impl Spectrum4 {
    pub const ZERO: Self = Self { c: [0.0; 4] };
    pub const ONE: Self = Self { c: [1.0; 4] };

    #[inline]
    pub const fn splat(v: f64) -> Self {
        Self { c: [v; 4] }
    }

    #[inline]
    pub fn max_comp(&self) -> f64 {
        self.c[0].max(self.c[1]).max(self.c[2]).max(self.c[3])
    }

    #[inline]
    pub fn average(&self) -> f64 {
        0.25 * (self.c[0] + self.c[1] + self.c[2] + self.c[3])
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.c[0] <= 0.0 && self.c[1] <= 0.0 && self.c[2] <= 0.0 && self.c[3] <= 0.0
    }

    #[inline]
    pub fn exp(&self) -> Self {
        Self {
            c: [
                self.c[0].exp(),
                self.c[1].exp(),
                self.c[2].exp(),
                self.c[3].exp(),
            ],
        }
    }

    #[inline]
    pub fn clamp(&self, lo: f64, hi: f64) -> Self {
        Self {
            c: [
                self.c[0].clamp(lo, hi),
                self.c[1].clamp(lo, hi),
                self.c[2].clamp(lo, hi),
                self.c[3].clamp(lo, hi),
            ],
        }
    }

    /// Collapse companion wavelengths when chromatic dispersion decouples ray geometry:
    /// Hero wavelength carries 4x weight so estimator remains strictly unbiased!
    #[inline]
    pub fn collapse_to_hero(&mut self, wl: &mut Wavelengths4) {
        if !wl.hero_only {
            wl.hero_only = true;
            self.c[0] *= 4.0;
            self.c[1] = 0.0;
            self.c[2] = 0.0;
            self.c[3] = 0.0;
        }
    }

    /// Convert sampled 4-wavelength spectral radiance to linear sRGB (IEC 61966-2-1, D65).
    #[inline]
    pub fn to_linear_srgb(&self, wl: &Wavelengths4) -> Vec3 {
        let mut x = 0.0;
        let mut y = 0.0;
        let mut z = 0.0;
        let count = if wl.hero_only { 1 } else { 4 };
        for i in 0..count {
            let l = self.c[i];
            if l != 0.0 {
                let (cx, cy, cz) = cie_xyz_1931(wl.lambda[i]);
                x += l * cx;
                y += l * cy;
                z += l * cz;
            }
        }
        // Monte Carlo integral factor: (LAMBDA_SPAN / 4.0). Note that when `hero_only` is true,
        // `self.c[0]` already absorbed the `* 4.0` MIS survival factor, so `LAMBDA_SPAN * 0.25` applies uniformly!
        let scale = LAMBDA_SPAN * 0.25;
        let cap_x = x * scale * (0.95047 / CIE_X_INTEGRAL);
        let cap_y = y * scale * (1.00000 / CIE_Y_INTEGRAL);
        let cap_z = z * scale * (1.08883 / CIE_Z_INTEGRAL);

        let r = 3.240_454_2 * cap_x - 1.537_138_5 * cap_y - 0.498_531_4 * cap_z;
        let g = -0.969_266_0 * cap_x + 1.876_010_8 * cap_y + 0.041_556_0 * cap_z;
        let b = 0.055_643_4 * cap_x - 0.204_025_9 * cap_y + 1.057_225_2 * cap_z;
        Vec3::new(r, g, b)
    }
}

impl Add for Spectrum4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            c: [
                self.c[0] + rhs.c[0],
                self.c[1] + rhs.c[1],
                self.c[2] + rhs.c[2],
                self.c[3] + rhs.c[3],
            ],
        }
    }
}

impl AddAssign for Spectrum4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.c[i] += rhs.c[i];
        }
    }
}

impl Mul for Spectrum4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            c: [
                self.c[0] * rhs.c[0],
                self.c[1] * rhs.c[1],
                self.c[2] * rhs.c[2],
                self.c[3] * rhs.c[3],
            ],
        }
    }
}

impl MulAssign for Spectrum4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..4 {
            self.c[i] *= rhs.c[i];
        }
    }
}

impl Mul<f64> for Spectrum4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            c: [
                self.c[0] * rhs,
                self.c[1] * rhs,
                self.c[2] * rhs,
                self.c[3] * rhs,
            ],
        }
    }
}

impl Div<f64> for Spectrum4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        self * (1.0 / rhs)
    }
}

/// Wyman, Sloan, and Shirley (JCGT 2013) piecewise-Gaussian CIE 1931 2-degree observer fit.
#[inline]
pub fn cie_xyz_1931(lambda_nm: f64) -> (f64, f64, f64) {
    #[inline]
    fn lobe(x: f64, mu: f64, s_lo: f64, s_hi: f64) -> f64 {
        let t = (x - mu) * if x < mu { s_lo } else { s_hi };
        (-0.5 * t * t).exp()
    }
    let x = 0.362 * lobe(lambda_nm, 442.0, 0.0624, 0.0374)
        + 1.056 * lobe(lambda_nm, 599.8, 0.0264, 0.0323)
        - 0.065 * lobe(lambda_nm, 501.1, 0.0490, 0.0382);
    let y = 0.821 * lobe(lambda_nm, 568.8, 0.0213, 0.0247)
        + 0.286 * lobe(lambda_nm, 530.9, 0.0613, 0.0322);
    let z = 1.217 * lobe(lambda_nm, 437.0, 0.0845, 0.0278)
        + 0.681 * lobe(lambda_nm, 459.0, 0.0385, 0.0725);
    (x.max(0.0), y.max(0.0), z.max(0.0))
}

/// Smooth partition-of-unity spectral upsampling curve (strictly energy-conserving: `w_b + w_g + w_r == 1.0`).
#[derive(Clone, Copy, Debug)]
pub enum SpectralCurve {
    /// Smooth RGB reflectance/absorption/emission curve evaluated continuously in [380, 750] nm.
    SmoothRgb(Vec3),
    /// Gas-discharge neon spectral line emitter: smooth continuum + narrowband Gaussian spectral spikes.
    NeonDischarge {
        continuum: Vec3,
        peak1_nm: f64,
        sigma1_nm: f64,
        amp1: f64,
        peak2_nm: f64,
        sigma2_nm: f64,
        amp2: f64,
    },
}

impl SpectralCurve {
    #[inline]
    pub const fn rgb(r: f64, g: f64, b: f64) -> Self {
        Self::SmoothRgb(Vec3::new(r, g, b))
    }

    #[inline]
    pub fn eval_at(&self, lambda_nm: f64) -> f64 {
        let w_b = 1.0 / (1.0 + ((lambda_nm - 492.0) / 14.0).exp());
        let w_r = 1.0 / (1.0 + (-(lambda_nm - 586.0) / 14.0).exp());
        let w_g = (1.0 - w_b - w_r).max(0.0);
        match *self {
            Self::SmoothRgb(c) => c.x * w_r + c.y * w_g + c.z * w_b,
            Self::NeonDischarge {
                continuum,
                peak1_nm,
                sigma1_nm,
                amp1,
                peak2_nm,
                sigma2_nm,
                amp2,
            } => {
                let base = continuum.x * w_r + continuum.y * w_g + continuum.z * w_b;
                let g1 = amp1 * (-0.5 * ((lambda_nm - peak1_nm) / sigma1_nm).powi(2)).exp();
                let g2 = amp2 * (-0.5 * ((lambda_nm - peak2_nm) / sigma2_nm).powi(2)).exp();
                base + g1 + g2
            }
        }
    }

    #[inline]
    pub fn eval4(&self, wl: &Wavelengths4) -> Spectrum4 {
        Spectrum4 {
            c: [
                self.eval_at(wl.lambda[0]),
                self.eval_at(wl.lambda[1]),
                self.eval_at(wl.lambda[2]),
                self.eval_at(wl.lambda[3]),
            ],
        }
    }
}

/// Stephen Hill's ACES Filmic Tone Mapping (sRGB -> AP1/RRT+ODT -> sRGB) + IEC 61966-2-1 sRGB OETF.
#[inline]
pub fn aces_filmic_tonemap(linear_rgb: Vec3, exposure: f64) -> Vec3 {
    let v = (linear_rgb * exposure).max_elem(Vec3::ZERO);

    // sRGB (Rec.709 D65) to ACES Input Matrix (Stephen Hill)
    let a_r = 0.59719 * v.x + 0.35458 * v.y + 0.04823 * v.z;
    let a_g = 0.07600 * v.x + 0.90834 * v.y + 0.01566 * v.z;
    let a_b = 0.02840 * v.x + 0.13383 * v.y + 0.83777 * v.z;

    #[inline]
    fn rrt_and_odt_fit(x: f64) -> f64 {
        let a = x * (x + 0.0245786) - 0.000090537;
        let b = x * (0.983729 * x + 0.4329510) + 0.238081;
        a / b
    }

    let f_r = rrt_and_odt_fit(a_r);
    let f_g = rrt_and_odt_fit(a_g);
    let f_b = rrt_and_odt_fit(a_b);

    // ACES Output Matrix back to linear sRGB
    let out_r = 1.60475 * f_r - 0.53108 * f_g - 0.07367 * f_b;
    let out_g = -0.10208 * f_r + 1.10813 * f_g - 0.00605 * f_b;
    let out_b = -0.00327 * f_r - 0.07276 * f_g + 1.07602 * f_b;

    Vec3::new(
        linear_to_srgb_channel(out_r.clamp(0.0, 1.0)),
        linear_to_srgb_channel(out_g.clamp(0.0, 1.0)),
        linear_to_srgb_channel(out_b.clamp(0.0, 1.0)),
    )
}

#[inline]
pub fn linear_to_srgb_channel(x: f64) -> f64 {
    let c = x.clamp(0.0, 1.0);
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat4_inverse_and_quat_consistency() {
        let q = Quat::from_euler_yxz(0.73, -0.41, 1.12);
        let v = Vec3::new(3.2, -1.7, 4.5);
        let r_quat = q.rotate_vec3(v);
        let r_mat = q.to_mat3().mul_vec3(v);
        assert!((r_quat - r_mat).length() < 1e-12);

        let q_inv = q.inverse();
        let back = q_inv.rotate_vec3(r_quat);
        assert!((back - v).length() < 1e-12);

        let m = Mat4::translation(Vec3::new(4.0, -2.5, 9.0))
            * q.to_mat4()
            * Mat4::scale(Vec3::new(2.0, 0.5, 1.5));
        let m_inv = m.inverse().expect("Transform should be invertible");
        let prod = m * m_inv;
        for r in 0..4 {
            for c in 0..4 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!((prod.m[r][c] - expected).abs() < 1e-11);
            }
        }
    }

    #[test]
    fn test_onb_orthogonality() {
        for n in [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.37, -0.82, 0.43).normalize(),
        ] {
            let onb = Onb::from_normal(n);
            assert!((onb.u.length() - 1.0).abs() < 1e-12);
            assert!((onb.v.length() - 1.0).abs() < 1e-12);
            assert!((onb.w.length() - 1.0).abs() < 1e-12);
            assert!(onb.u.dot(onb.v).abs() < 1e-12);
            assert!(onb.u.dot(onb.w).abs() < 1e-12);
            assert!(onb.v.dot(onb.w).abs() < 1e-12);
        }
    }

    #[test]
    fn test_spectral_whitepoint_and_energy_bounds() {
        // Verify smooth spectral reflectance stays strictly inside [0, 1] across [380, 750]nm
        let curve = SpectralCurve::rgb(0.85, 0.22, 0.91);
        for nm in 380..=750 {
            let val = curve.eval_at(nm as f64);
            assert!((0.0..=1.0).contains(&val));
        }

        // Verify stratified flat white spectrum integrates to neutral linear sRGB (1.0, 1.0, 1.0)
        let flat = SpectralCurve::rgb(1.0, 1.0, 1.0);
        let n = 2000;
        let mut accum = Vec3::ZERO;
        for i in 0..n {
            let u = ((i as f64) + 0.5) / (n as f64);
            let wl = Wavelengths4::sample(u);
            let spec = flat.eval4(&wl);
            accum += spec.to_linear_srgb(&wl);
        }
        let rgb = accum / (n as f64);
        assert!((rgb.x - 1.0).abs() < 0.005, "R={} expected ~1.0", rgb.x);
        assert!((rgb.y - 1.0).abs() < 0.005, "G={} expected ~1.0", rgb.y);
        assert!((rgb.z - 1.0).abs() < 0.005, "B={} expected ~1.0", rgb.z);
    }
}
