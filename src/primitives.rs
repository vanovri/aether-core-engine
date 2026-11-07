use std::f64::consts::{PI, TAU};

use crate::math::{Aabb, Mat4, Ray, Vec3};
use crate::rng::Rng;

#[derive(Clone, Copy, Debug)]
pub struct HitRecord {
    pub t: f64,
    pub point: Vec3,
    /// Outward-pointing surface normal (unflipped).
    pub outward_normal: Vec3,
    /// Face-forwarded shading normal (`normal.dot(ray.dir) <= 0`).
    pub normal: Vec3,
    /// True when ray strikes the outside of the surface (`ray.dir.dot(outward_normal) < 0`).
    pub front_face: bool,
    pub uv: (f64, f64),
    pub material_id: usize,
    pub primitive_id: usize,
}

impl HitRecord {
    #[inline]
    pub fn new(
        t: f64,
        point: Vec3,
        ray_dir: Vec3,
        outward_normal: Vec3,
        shading_outward: Vec3,
        uv: (f64, f64),
        material_id: usize,
        primitive_id: usize,
    ) -> Self {
        let front_face = ray_dir.dot(outward_normal) < 0.0;
        let normal = if front_face {
            shading_outward
        } else {
            -shading_outward
        };
        Self {
            t,
            point,
            outward_normal,
            normal,
            front_face,
            uv,
            material_id,
            primitive_id,
        }
    }
}

/// Surface point sample used by Next Event Estimation (area light sampling).
#[derive(Clone, Copy, Debug)]
pub struct SurfaceSample {
    pub point: Vec3,
    pub normal: Vec3,
    pub pdf_area: f64,
}

// ============================================================================
// Triangle Primitive (Moller-Trumbore)
// ============================================================================

#[derive(Clone, Debug)]
pub struct Triangle {
    pub v0: Vec3,
    pub e1: Vec3,
    pub e2: Vec3,
    pub n0: Vec3,
    pub n1: Vec3,
    pub n2: Vec3,
    pub geom_normal: Vec3,
    pub uv0: (f64, f64),
    pub uv1: (f64, f64),
    pub uv2: (f64, f64),
    pub area: f64,
    pub bounds: Aabb,
    pub material_id: usize,
}

impl Triangle {
    pub fn new(v0: Vec3, v1: Vec3, v2: Vec3, material_id: usize) -> Self {
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let cross = e1.cross(e2);
        let area = 0.5 * cross.length();
        let geom_normal = cross.normalize();
        Self::with_normals_uvs(
            v0,
            v1,
            v2,
            geom_normal,
            geom_normal,
            geom_normal,
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            area,
            geom_normal,
            material_id,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_normals_uvs(
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        n0: Vec3,
        n1: Vec3,
        n2: Vec3,
        uv0: (f64, f64),
        uv1: (f64, f64),
        uv2: (f64, f64),
        area: f64,
        geom_normal: Vec3,
        material_id: usize,
    ) -> Self {
        let bounds = Aabb::EMPTY
            .union_point(v0)
            .union_point(v1)
            .union_point(v2)
            .pad(1e-5);
        Self {
            v0,
            e1: v1 - v0,
            e2: v2 - v0,
            n0: n0.normalize(),
            n1: n1.normalize(),
            n2: n2.normalize(),
            geom_normal,
            uv0,
            uv1,
            uv2,
            area: area.max(1e-12),
            bounds,
            material_id,
        }
    }

    /// Helper to append two triangles forming a quad `(p0, p1, p2, p3)` in counter-clockwise order.
    pub fn quad(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, material_id: usize) -> [Self; 2] {
        let e1 = p1 - p0;
        let e2 = p3 - p0;
        let n = e1.cross(e2).normalize();
        let a1 = 0.5 * (p1 - p0).cross(p2 - p0).length();
        let a2 = 0.5 * (p2 - p0).cross(p3 - p0).length();
        [
            Self::with_normals_uvs(
                p0,
                p1,
                p2,
                n,
                n,
                n,
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                a1,
                n,
                material_id,
            ),
            Self::with_normals_uvs(
                p0,
                p2,
                p3,
                n,
                n,
                n,
                (0.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                a2,
                n,
                material_id,
            ),
        ]
    }

    #[inline]
    pub fn intersect(&self, ray: &Ray, t_min: f64, t_max: f64, prim_id: usize) -> Option<HitRecord> {
        let pvec = ray.dir.cross(self.e2);
        let det = self.e1.dot(pvec);
        if det.abs() < 1e-13 {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = ray.origin - self.v0;
        let u = tvec.dot(pvec) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let qvec = tvec.cross(self.e1);
        let v = ray.dir.dot(qvec) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = self.e2.dot(qvec) * inv_det;
        if t < t_min || t > t_max {
            return None;
        }

        let w = 1.0 - u - v;
        let shading_outward = (self.n0 * w + self.n1 * u + self.n2 * v).normalize();
        let uv = (
            self.uv0.0 * w + self.uv1.0 * u + self.uv2.0 * v,
            self.uv0.1 * w + self.uv1.1 * u + self.uv2.1 * v,
        );
        let point = ray.at(t);
        Some(HitRecord::new(
            t,
            point,
            ray.dir,
            self.geom_normal,
            shading_outward,
            uv,
            self.material_id,
            prim_id,
        ))
    }

    #[inline]
    pub fn sample_surface(&self, rng: &mut Rng) -> SurfaceSample {
        let su0 = rng.next_f64().sqrt();
        let u = 1.0 - su0;
        let v = rng.next_f64() * su0;
        let point = self.v0 + self.e1 * u + self.e2 * v;
        SurfaceSample {
            point,
            normal: self.geom_normal,
            pdf_area: 1.0 / self.area,
        }
    }
}

// ============================================================================
// Analytic Sphere Primitive
// ============================================================================

#[derive(Clone, Debug)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
    pub bounds: Aabb,
    pub material_id: usize,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64, material_id: usize) -> Self {
        let r = radius.abs().max(1e-6);
        let rv = Vec3::splat(r);
        Self {
            center,
            radius: r,
            bounds: Aabb::new(center - rv, center + rv),
            material_id,
        }
    }

    #[inline]
    pub fn area(&self) -> f64 {
        4.0 * PI * self.radius * self.radius
    }

    #[inline]
    pub fn intersect(&self, ray: &Ray, t_min: f64, t_max: f64, prim_id: usize) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let half_b = oc.dot(ray.dir);
        let c = oc.length_squared() - self.radius * self.radius;
        let disc = half_b * half_b - c;
        if disc < 0.0 {
            return None;
        }
        let sqrtd = disc.sqrt();
        let mut root = -half_b - sqrtd;
        if root < t_min || root > t_max {
            root = -half_b + sqrtd;
            if root < t_min || root > t_max {
                return None;
            }
        }
        let point = ray.at(root);
        let outward = ((point - self.center) / self.radius).normalize();
        let theta = (-outward.y).clamp(-1.0, 1.0).acos();
        let phi = (-outward.z).atan2(outward.x) + PI;
        let uv = (phi / TAU, theta / PI);
        Some(HitRecord::new(
            root,
            point,
            ray.dir,
            outward,
            outward,
            uv,
            self.material_id,
            prim_id,
        ))
    }

    #[inline]
    pub fn sample_surface(&self, rng: &mut Rng) -> SurfaceSample {
        let (dx, dy, dz) = rng.uniform_sphere();
        let n = Vec3::new(dx, dy, dz);
        SurfaceSample {
            point: self.center + n * self.radius,
            normal: n,
            pdf_area: 1.0 / self.area(),
        }
    }
}

// ============================================================================
// CSG Signed-Distance Primitives (Composable Exact Euclidean SDF Tree)
// ============================================================================

#[derive(Clone, Debug)]
pub enum SdfNode {
    Sphere {
        center: Vec3,
        radius: f64,
    },
    RoundBox {
        center: Vec3,
        half_extents: Vec3,
        corner_radius: f64,
    },
    Torus {
        center: Vec3,
        major_radius: f64,
        minor_radius: f64,
    },
    Cylinder {
        center: Vec3,
        radius: f64,
        half_height: f64,
        /// 0 = X axis, 1 = Y axis, 2 = Z axis.
        axis: u8,
    },
    Octahedron {
        center: Vec3,
        size: f64,
    },
    Transform {
        inv_matrix: Mat4,
        uniform_scale: f64,
        child: Box<SdfNode>,
    },
    Union(Box<SdfNode>, Box<SdfNode>),
    SmoothUnion {
        a: Box<SdfNode>,
        b: Box<SdfNode>,
        k: f64,
    },
    Intersection(Box<SdfNode>, Box<SdfNode>),
    Difference(Box<SdfNode>, Box<SdfNode>),
}

impl SdfNode {
    #[inline]
    pub fn eval(&self, p: Vec3) -> f64 {
        match self {
            Self::Sphere { center, radius } => (p - *center).length() - *radius,
            Self::RoundBox {
                center,
                half_extents,
                corner_radius,
            } => {
                let q = (p - *center).abs() - *half_extents;
                q.max_elem(Vec3::ZERO).length() + q.max_comp().min(0.0) - *corner_radius
            }
            Self::Torus {
                center,
                major_radius,
                minor_radius,
            } => {
                let q = p - *center;
                let xz = (q.x * q.x + q.z * q.z).sqrt() - *major_radius;
                (xz * xz + q.y * q.y).sqrt() - *minor_radius
            }
            Self::Cylinder {
                center,
                radius,
                half_height,
                axis,
            } => {
                let q = p - *center;
                let (radial, axial) = match axis {
                    0 => ((q.y * q.y + q.z * q.z).sqrt(), q.x.abs()),
                    1 => ((q.x * q.x + q.z * q.z).sqrt(), q.y.abs()),
                    _ => ((q.x * q.x + q.y * q.y).sqrt(), q.z.abs()),
                };
                let dx = radial - *radius;
                let dy = axial - *half_height;
                dx.max(dy).min(0.0) + (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt()
            }
            Self::Octahedron { center, size } => {
                let q = (p - *center).abs();
                let m = q.x + q.y + q.z - *size;
                let (qx, qy, qz) = if 3.0 * q.x - m < 0.0 {
                    (q.x, q.y, q.z)
                } else if 3.0 * q.y - m < 0.0 {
                    (q.y, q.z, q.x)
                } else if 3.0 * q.z - m < 0.0 {
                    (q.z, q.x, q.y)
                } else {
                    return m * 0.577_350_269_189_625_8;
                };
                let k_val = (0.5 * (qz - qy + *size)).clamp(0.0, *size);
                Vec3::new(qx, qy - *size + k_val, qz - k_val).length()
            }
            Self::Transform {
                inv_matrix,
                uniform_scale,
                child,
            } => child.eval(inv_matrix.transform_point(p)) * *uniform_scale,
            Self::Union(a, b) => a.eval(p).min(b.eval(p)),
            Self::SmoothUnion { a, b, k } => {
                let d1 = a.eval(p);
                let d2 = b.eval(p);
                let h = (0.5 + 0.5 * (d2 - d1) / k.max(1e-6)).clamp(0.0, 1.0);
                d2 * (1.0 - h) + d1 * h - k * h * (1.0 - h)
            }
            Self::Intersection(a, b) => a.eval(p).max(b.eval(p)),
            Self::Difference(a, b) => a.eval(p).max(-b.eval(p)),
        }
    }

    /// 4-tap Tetrahedral Central-Difference SDF Gradient (Paul Malin / Inigo Quilez).
    #[inline]
    pub fn normal(&self, p: Vec3) -> Vec3 {
        const H: f64 = 1e-5;
        let k0 = Vec3::new(1.0, -1.0, -1.0);
        let k1 = Vec3::new(-1.0, -1.0, 1.0);
        let k2 = Vec3::new(-1.0, 1.0, -1.0);
        let k3 = Vec3::new(1.0, 1.0, 1.0);
        let grad = k0 * self.eval(p + k0 * H)
            + k1 * self.eval(p + k1 * H)
            + k2 * self.eval(p + k2 * H)
            + k3 * self.eval(p + k3 * H);
        grad.normalize()
    }
}

#[derive(Clone, Debug)]
pub struct CsgPrimitive {
    pub root: SdfNode,
    pub bounds: Aabb,
    pub material_id: usize,
}

impl CsgPrimitive {
    pub fn new(root: SdfNode, bounds: Aabb, material_id: usize) -> Self {
        Self {
            root,
            bounds: bounds.pad(1e-3),
            material_id,
        }
    }

    /// Enhanced Sphere-Tracing with Bisection Root Polishing (supports both exterior and interior rays).
    pub fn intersect(&self, ray: &Ray, t_min: f64, t_max: f64, prim_id: usize) -> Option<HitRecord> {
        let (box_enter, box_exit) = self.bounds.intersect(ray, t_min, t_max)?;
        let mut t = box_enter.max(t_min);
        let limit = box_exit.min(t_max);
        if t > limit {
            return None;
        }

        const HIT_EPS: f64 = 2.5e-5;
        const MAX_STEPS: usize = 160;

        let f_start = self.root.eval(ray.at(t));
        // `sign`: +1.0 if ray starts outside the CSG solid, -1.0 if ray starts inside (e.g. refracted ray).
        let sign = if f_start < 0.0 { -1.0 } else { 1.0 };

        let mut prev_t = t;
        for _ in 0..MAX_STEPS {
            if t > limit {
                break;
            }
            let d = sign * self.root.eval(ray.at(t));
            if d > (limit - t) + HIT_EPS {
                return None;
            }
            if d <= HIT_EPS {
                // Polish zero-crossing with 6 bisection iterations when sign flipped
                let mut lo = prev_t;
                let mut hi = t;
                if d < 0.0 && hi > lo {
                    for _ in 0..6 {
                        let mid = 0.5 * (lo + hi);
                        if sign * self.root.eval(ray.at(mid)) <= 0.0 {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    t = 0.5 * (lo + hi);
                }
                if t >= t_min && t <= limit {
                    let hit_p = ray.at(t);
                    let outward = self.root.normal(hit_p);
                    let uv = (
                        0.5 + outward.z.atan2(outward.x) / TAU,
                        0.5 - outward.y.clamp(-1.0, 1.0).asin() / PI,
                    );
                    return Some(HitRecord::new(
                        t,
                        hit_p,
                        ray.dir,
                        outward,
                        outward,
                        uv,
                        self.material_id,
                        prim_id,
                    ));
                }
                return None;
            }
            prev_t = t;
            t += d * 0.96;
        }
        None
    }
}

// ============================================================================
// Unified Scene Primitive Wrapper
// ============================================================================

#[derive(Clone, Debug)]
pub enum Primitive {
    Triangle(Triangle),
    Sphere(Sphere),
    Csg(CsgPrimitive),
}

impl Primitive {
    #[inline]
    pub fn bounds(&self) -> Aabb {
        match self {
            Self::Triangle(t) => t.bounds,
            Self::Sphere(s) => s.bounds,
            Self::Csg(c) => c.bounds,
        }
    }

    #[inline]
    pub fn material_id(&self) -> usize {
        match self {
            Self::Triangle(t) => t.material_id,
            Self::Sphere(s) => s.material_id,
            Self::Csg(c) => c.material_id,
        }
    }

    #[inline]
    pub fn intersect(&self, ray: &Ray, t_min: f64, t_max: f64, prim_id: usize) -> Option<HitRecord> {
        match self {
            Self::Triangle(t) => t.intersect(ray, t_min, t_max, prim_id),
            Self::Sphere(s) => s.intersect(ray, t_min, t_max, prim_id),
            Self::Csg(c) => c.intersect(ray, t_min, t_max, prim_id),
        }
    }

    #[inline]
    pub fn area(&self) -> f64 {
        match self {
            Self::Triangle(t) => t.area,
            Self::Sphere(s) => s.area(),
            Self::Csg(c) => c.bounds.surface_area() * 0.5,
        }
    }

    #[inline]
    pub fn sample_surface(&self, rng: &mut Rng) -> Option<SurfaceSample> {
        match self {
            Self::Triangle(t) => Some(t.sample_surface(rng)),
            Self::Sphere(s) => Some(s.sample_surface(rng)),
            Self::Csg(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csg_boolean_difference_ray_intersection() {
        // Box with a cylindrical hole carved completely through Z axis:
        // A ray down the center (0,0,-5) -> +Z must pass cleanly through the carved hole without hitting,
        // while an off-axis ray (0.7, 0, -5) -> +Z must hit the front face of the rounded box at z ~ -1.0!
        let box_node = SdfNode::RoundBox {
            center: Vec3::ZERO,
            half_extents: Vec3::splat(0.95),
            corner_radius: 0.05,
        };
        let hole_node = SdfNode::Cylinder {
            center: Vec3::ZERO,
            radius: 0.4,
            half_height: 2.0,
            axis: 2,
        };
        let csg = CsgPrimitive::new(
            SdfNode::Difference(Box::new(box_node), Box::new(hole_node)),
            Aabb::new(Vec3::splat(-1.2), Vec3::splat(1.2)),
            0,
        );

        let ray_through_hole = Ray::new(Vec3::new(0.0, 0.0, -5.0), Vec3::Z, 1e-4, 100.0);
        assert!(csg.intersect(&ray_through_hole, 1e-4, 100.0, 0).is_none());

        let ray_hit_wall = Ray::new(Vec3::new(0.7, 0.0, -5.0), Vec3::Z, 1e-4, 100.0);
        let hit = csg
            .intersect(&ray_hit_wall, 1e-4, 100.0, 0)
            .expect("Off-axis ray should hit CSG frame");
        assert!((hit.t - 4.0).abs() < 1e-3, "Expected t ~ 4.0, got {}", hit.t);
        assert!((hit.outward_normal - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-2);
    }
}
