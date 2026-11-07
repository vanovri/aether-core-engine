use crate::bsdf::Material;
use crate::bvh::Bvh;
use crate::fluid::FluidMedium;
use crate::math::{Aabb, Mat4, Quat, Ray, SpectralCurve, Spectrum4, Vec3, Wavelengths4};
use crate::primitives::{CsgPrimitive, HitRecord, Primitive, SdfNode, Sphere, Triangle};
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Camera {
    pub origin: Vec3,
    pub lower_left: Vec3,
    pub horizontal: Vec3,
    pub vertical: Vec3,
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
    pub lens_radius: f64,
}

impl Camera {
    pub fn new(
        look_from: Vec3,
        look_at: Vec3,
        vup: Vec3,
        vfov_deg: f64,
        aspect_ratio: f64,
        aperture: f64,
        focus_dist: f64,
    ) -> Self {
        let theta = vfov_deg.to_radians();
        let h = (0.5 * theta).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let w = (look_from - look_at).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);

        let horizontal = u * (focus_dist * viewport_width);
        let vertical = v * (focus_dist * viewport_height);
        let lower_left = look_from - horizontal * 0.5 - vertical * 0.5 - w * focus_dist;

        Self {
            origin: look_from,
            lower_left,
            horizontal,
            vertical,
            u,
            v,
            w,
            lens_radius: 0.5 * aperture,
        }
    }

    #[inline]
    pub fn generate_ray(&self, s: f64, t: f64, rng: &mut Rng) -> Ray {
        let offset = if self.lens_radius > 1e-6 {
            let (dx, dy) = rng.concentric_disk();
            self.u * (dx * self.lens_radius) + self.v * (dy * self.lens_radius)
        } else {
            Vec3::ZERO
        };
        let orig = self.origin + offset;
        let target = self.lower_left + self.horizontal * s + self.vertical * t;
        Ray::new(orig, target - orig, 1e-4, 1e6)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LightEntry {
    pub primitive_id: usize,
    pub prob: f64,
    pub cdf: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct DirectLightSample {
    pub wi: Vec3,
    pub dist: f64,
    pub emission: Spectrum4,
    pub pdf_solid_angle: f64,
}

pub struct Scene {
    pub bvh: Bvh,
    pub materials: Vec<Material>,
    pub lights: Vec<LightEntry>,
    pub prim_light_prob: Vec<f64>,
    pub medium: Option<FluidMedium>,
    pub camera: Camera,
}

impl Scene {
    pub fn new(
        primitives: Vec<Primitive>,
        materials: Vec<Material>,
        medium: Option<FluidMedium>,
        camera: Camera,
    ) -> Self {
        let bvh = Bvh::build(primitives);
        let mut raw_weights = Vec::new();
        let mut total_weight = 0.0;

        for (pid, prim) in bvh.primitives.iter().enumerate() {
            if prim.sample_surface(&mut Rng::from_seed(1)).is_none() {
                continue;
            }
            let mat = &materials[prim.material_id()];
            let power = mat.mean_emission_power() * prim.area();
            if power > 1e-6 {
                raw_weights.push((pid, power));
                total_weight += power;
            }
        }

        let mut lights = Vec::with_capacity(raw_weights.len());
        let mut prim_light_prob = vec![0.0; bvh.primitives.len()];
        if total_weight > 0.0 {
            let mut accum = 0.0;
            for (pid, w) in raw_weights {
                let p = w / total_weight;
                accum += p;
                prim_light_prob[pid] = p;
                lights.push(LightEntry {
                    primitive_id: pid,
                    prob: p,
                    cdf: accum.min(1.0),
                });
            }
            if let Some(last) = lights.last_mut() {
                last.cdf = 1.0;
            }
        }

        Self {
            bvh,
            materials,
            lights,
            prim_light_prob,
            medium,
            camera,
        }
    }

    /// Sample one point on an emissive primitive for Next Event Estimation (returns solid-angle PDF).
    pub fn sample_direct_light(
        &self,
        ref_point: Vec3,
        wl: &Wavelengths4,
        rng: &mut Rng,
    ) -> Option<DirectLightSample> {
        if self.lights.is_empty() {
            return None;
        }
        let u = rng.next_f64();
        let idx = self
            .lights
            .partition_point(|entry| entry.cdf < u)
            .min(self.lights.len() - 1);
        let entry = self.lights[idx];
        let prim = &self.bvh.primitives[entry.primitive_id];
        let surf = prim.sample_surface(rng)?;

        let to_light = surf.point - ref_point;
        let dist2 = to_light.length_squared();
        if dist2 < 1e-10 {
            return None;
        }
        let dist = dist2.sqrt();
        let wi = to_light / dist;
        let cos_light = surf.normal.dot(-wi).abs();
        if cos_light <= 1e-6 {
            return None;
        }

        let pdf_solid_angle = entry.prob * surf.pdf_area * dist2 / cos_light;
        if pdf_solid_angle <= 1e-14 || !pdf_solid_angle.is_finite() {
            return None;
        }

        let emission = self.materials[prim.material_id()].emission(wl);
        Some(DirectLightSample {
            wi,
            dist,
            emission,
            pdf_solid_angle,
        })
    }

    /// Solid-angle PDF with which `sample_direct_light(origin)` would have generated `hit` along `ray`.
    #[inline]
    pub fn pdf_direct_light(&self, ray: &Ray, hit: &HitRecord) -> f64 {
        let choose_prob = self.prim_light_prob[hit.primitive_id];
        if choose_prob <= 0.0 {
            return 0.0;
        }
        let area = self.bvh.primitives[hit.primitive_id].area();
        let cos_light = hit.outward_normal.dot(-ray.dir).abs();
        if cos_light <= 1e-7 || area <= 1e-12 {
            return 0.0;
        }
        choose_prob * (hit.t * hit.t) / (area * cos_light)
    }

    /// Procedural spectral cyberpunk night atmosphere sky (horizon neon haze + zenith deep indigo).
    #[inline]
    pub fn eval_environment(&self, dir: Vec3, wl: &Wavelengths4) -> Spectrum4 {
        let up = dir.y.clamp(-1.0, 1.0);
        let horizon = (1.0 - up.abs()).powi(4);
        let azimuth_glow = (0.5 + 0.5 * (dir.x * 1.8 - dir.z * 0.9).sin()).powi(2);

        let zenith_curve = SpectralCurve::rgb(0.012, 0.018, 0.052);
        let cyan_haze = SpectralCurve::rgb(0.04, 0.26, 0.42);
        let magenta_haze = SpectralCurve::rgb(0.38, 0.05, 0.28);

        zenith_curve.eval4(wl)
            + cyan_haze.eval4(wl) * (horizon * 0.55)
            + magenta_haze.eval4(wl) * (horizon * azimuth_glow * 0.48)
    }

    /// Helper to append an axis-aligned box made of 12 triangles.
    fn push_box_triangles(prims: &mut Vec<Primitive>, min: Vec3, max: Vec3, mat_id: usize) {
        let p000 = Vec3::new(min.x, min.y, min.z);
        let p100 = Vec3::new(max.x, min.y, min.z);
        let p110 = Vec3::new(max.x, max.y, min.z);
        let p010 = Vec3::new(min.x, max.y, min.z);
        let p001 = Vec3::new(min.x, min.y, max.z);
        let p101 = Vec3::new(max.x, min.y, max.z);
        let p111 = Vec3::new(max.x, max.y, max.z);
        let p011 = Vec3::new(min.x, max.y, max.z);

        // +Z front, -Z back, +X right, -X left, +Y top, -Y bottom
        for [t0, t1] in [
            Triangle::quad(p001, p101, p111, p011, mat_id),
            Triangle::quad(p100, p000, p010, p110, mat_id),
            Triangle::quad(p101, p100, p110, p111, mat_id),
            Triangle::quad(p000, p001, p011, p010, mat_id),
            Triangle::quad(p011, p111, p110, p010, mat_id),
            Triangle::quad(p000, p100, p101, p001, mat_id),
        ] {
            prims.push(Primitive::Triangle(t0));
            prims.push(Primitive::Triangle(t1));
        }
    }

    /// Construct the flagship Cyberpunk Scene coupling all BSDFs, CSG SDFs, Triangles, Spheres, and 3D Fluid Smoke.
    pub fn build_cyberpunk_scene(aspect_ratio: f64, medium: Option<FluidMedium>) -> Self {
        let mut materials = Vec::new();

        // 0: Wet Rain-Slicked Cyberpunk Street Floor (Coated Diffuse with procedural mirror puddles)
        let mat_wet_street = materials.len();
        materials.push(Material::CoatedDiffuse {
            base_albedo: SpectralCurve::rgb(0.055, 0.062, 0.082),
            coat_ior: 1.48,
            roughness: 0.16,
            puddle_modulation: true,
        });

        // 1: Brushed Dark Gunmetal Cyber-Architecture (GGX Conductor, complex Titanium-Carbide IOR)
        let mat_gunmetal = materials.len();
        materials.push(Material::Conductor {
            eta: SpectralCurve::rgb(2.45, 2.35, 2.15),
            k: SpectralCurve::rgb(3.25, 3.05, 2.85),
            tint: SpectralCurve::rgb(0.68, 0.74, 0.86),
            roughness_x: 0.14,
            roughness_y: 0.22,
        });

        // 2: Dispersive Crown-Flint Crystal Glass Orb (Exact Fresnel + Snell + Cauchy Dispersion + Beer-Lambert Absorption)
        let mat_dispersive_glass = materials.len();
        materials.push(Material::Dielectric {
            base_ior: 1.53,
            dispersion_b: 0.0135,
            sigma_a: SpectralCurve::rgb(0.24, 0.05, 0.08), // Subtle cyan-aquamarine Beer-Lambert tint
        });

        // 3: Sculpted CSG Gold-Copper Cyber-Relic (Anisotropic GGX Conductor with real Gold complex IOR)
        let mat_gold_conductor = materials.len();
        materials.push(Material::Conductor {
            eta: SpectralCurve::rgb(0.18, 0.42, 1.35),
            k: SpectralCurve::rgb(3.42, 2.45, 1.86),
            tint: SpectralCurve::rgb(1.0, 0.88, 0.66),
            roughness_x: 0.075,
            roughness_y: 0.075,
        });

        // 4: Subsurface Isotropic Volumetric Scattering Cyber-Jade / Bioluminescent Orb
        let mat_subsurface_jade = materials.len();
        materials.push(Material::Subsurface {
            ior: 1.36,
            sigma_s: SpectralCurve::rgb(2.6, 3.8, 3.4),
            sigma_a: SpectralCurve::rgb(0.45, 0.025, 0.12), // Rich emerald-cyan subsurface translucency
            surface_roughness: 0.02,
        });

        // 5: Mirror Chrome Accent Spheres / Rings (Low-roughness Silver-Chrome GGX Conductor)
        let mat_chrome = materials.len();
        materials.push(Material::Conductor {
            eta: SpectralCurve::rgb(0.16, 0.14, 0.13),
            k: SpectralCurve::rgb(4.10, 3.35, 2.75),
            tint: SpectralCurve::rgb(0.95, 0.96, 0.99),
            roughness_x: 0.025,
            roughness_y: 0.025,
        });

        // 6: Electric Cyan Gas-Discharge Neon Area Light (486nm spectral spike)
        let mat_neon_cyan = materials.len();
        materials.push(Material::Emissive {
            emission: SpectralCurve::NeonDischarge {
                continuum: Vec3::new(0.08, 0.78, 1.0),
                peak1_nm: 486.0,
                sigma1_nm: 12.0,
                amp1: 1.8,
                peak2_nm: 442.0,
                sigma2_nm: 10.0,
                amp2: 0.9,
            },
            strength: 11.5,
        });

        // 7: Hot Cyberpunk Magenta-Pink Neon Area Light (618nm + 435nm spectral spikes)
        let mat_neon_magenta = materials.len();
        materials.push(Material::Emissive {
            emission: SpectralCurve::NeonDischarge {
                continuum: Vec3::new(1.0, 0.10, 0.62),
                peak1_nm: 618.0,
                sigma1_nm: 13.0,
                amp1: 2.0,
                peak2_nm: 438.0,
                sigma2_nm: 11.0,
                amp2: 1.2,
            },
            strength: 12.0,
        });

        // 8: Overhead Warm Amber-White Key Light Canopy (illuminating smoke plume & caustics)
        let mat_key_amber = materials.len();
        materials.push(Material::Emissive {
            emission: SpectralCurve::rgb(1.0, 0.84, 0.64),
            strength: 8.8,
        });

        // 9: Reactor Vent Core Beacon inside CSG & Floor Grate
        let mat_core_beacon = materials.len();
        materials.push(Material::Emissive {
            emission: SpectralCurve::rgb(0.25, 0.95, 1.0),
            strength: 15.0,
        });

        // 10: Matte Carbon Pedestal Trim
        let mat_matte_trim = materials.len();
        materials.push(Material::Diffuse {
            albedo: SpectralCurve::rgb(0.14, 0.15, 0.18),
            checker_alt: Some((SpectralCurve::rgb(0.06, 0.07, 0.09), 2.5)),
        });

        let mut prims: Vec<Primitive> = Vec::new();

        // 1. Wide Wet Street Ground Quad (y = 0.0)
        for tri in Triangle::quad(
            Vec3::new(-18.0, 0.0, 14.0),
            Vec3::new(18.0, 0.0, 14.0),
            Vec3::new(18.0, 0.0, -18.0),
            Vec3::new(-18.0, 0.0, -18.0),
            mat_wet_street,
        ) {
            prims.push(Primitive::Triangle(tri));
        }

        // 2. Back Brutalist Cyber-Wall & Reactor Plinth
        Self::push_box_triangles(
            &mut prims,
            Vec3::new(-9.5, -0.05, -6.2),
            Vec3::new(9.5, 7.5, -5.6),
            mat_gunmetal,
        );
        // Central Reactor Vent Octagonal/Box Collar beneath the 3D Fluid Smoke Plume
        Self::push_box_triangles(
            &mut prims,
            Vec3::new(-1.35, -0.05, -1.35),
            Vec3::new(1.35, 0.22, 1.35),
            mat_matte_trim,
        );
        // Glowing Vent Core Strip illuminating base of the rising Navier-Stokes plume
        for tri in Triangle::quad(
            Vec3::new(-0.85, 0.23, 0.85),
            Vec3::new(0.85, 0.23, 0.85),
            Vec3::new(0.85, 0.23, -0.85),
            Vec3::new(-0.85, 0.23, -0.85),
            mat_core_beacon,
        ) {
            prims.push(Primitive::Triangle(tri));
        }

        // 3. Flanking Cyberpunk Monolith Pillars + Vertical Neon Light Strips
        // Left Cyan Monolith
        Self::push_box_triangles(
            &mut prims,
            Vec3::new(-4.8, -0.05, -3.6),
            Vec3::new(-3.7, 5.8, -1.8),
            mat_gunmetal,
        );
        for tri in Triangle::quad(
            Vec3::new(-3.68, 0.35, -2.05),
            Vec3::new(-3.68, 0.35, -3.35),
            Vec3::new(-3.68, 5.25, -3.35),
            Vec3::new(-3.68, 5.25, -2.05),
            mat_neon_cyan,
        ) {
            prims.push(Primitive::Triangle(tri));
        }
        // Front-facing accent strip on left monolith
        for tri in Triangle::quad(
            Vec3::new(-4.45, 0.45, -1.78),
            Vec3::new(-3.95, 0.45, -1.78),
            Vec3::new(-3.95, 5.10, -1.78),
            Vec3::new(-4.45, 5.10, -1.78),
            mat_neon_cyan,
        ) {
            prims.push(Primitive::Triangle(tri));
        }

        // Right Magenta Monolith
        Self::push_box_triangles(
            &mut prims,
            Vec3::new(3.7, -0.05, -3.6),
            Vec3::new(4.8, 5.8, -1.8),
            mat_gunmetal,
        );
        for tri in Triangle::quad(
            Vec3::new(3.68, 0.35, -3.35),
            Vec3::new(3.68, 0.35, -2.05),
            Vec3::new(3.68, 5.25, -2.05),
            Vec3::new(3.68, 5.25, -3.35),
            mat_neon_magenta,
        ) {
            prims.push(Primitive::Triangle(tri));
        }
        for tri in Triangle::quad(
            Vec3::new(3.95, 0.45, -1.78),
            Vec3::new(4.45, 0.45, -1.78),
            Vec3::new(4.45, 5.10, -1.78),
            Vec3::new(3.95, 5.10, -1.78),
            mat_neon_magenta,
        ) {
            prims.push(Primitive::Triangle(tri));
        }

        // Background Horizon Neon Portal Bars on the back wall (creates crisp reflections in glass & wet puddles)
        for (y_bar, mat_bar) in [(1.4, mat_neon_cyan), (2.9, mat_neon_magenta), (4.4, mat_neon_cyan)]
        {
            for tri in Triangle::quad(
                Vec3::new(-3.1, y_bar, -5.56),
                Vec3::new(3.1, y_bar, -5.56),
                Vec3::new(3.1, y_bar + 0.18, -5.56),
                Vec3::new(-3.1, y_bar + 0.18, -5.56),
                mat_bar,
            ) {
                prims.push(Primitive::Triangle(tri));
            }
        }

        // Overhead Drone / Streetlight Canopy Quad (facing downward -Y)
        for tri in Triangle::quad(
            Vec3::new(-2.0, 5.9, -1.2),
            Vec3::new(-2.0, 5.9, 1.6),
            Vec3::new(2.0, 5.9, 1.6),
            Vec3::new(2.0, 5.9, -1.2),
            mat_key_amber,
        ) {
            prims.push(Primitive::Triangle(tri));
        }

        // 4. Hero Primitive A (Left Foreground): Dispersive Dielectric Crystal Sphere (`Cauchy + Beer-Lambert`)
        prims.push(Primitive::Sphere(Sphere::new(
            Vec3::new(-1.62, 0.88, 1.65),
            0.88,
            mat_dispersive_glass,
        )));

        // 5. Hero Primitive B (Right Foreground): Subsurface Isotropic Volumetric Scattering Jade Sphere on Pedestal
        Self::push_box_triangles(
            &mut prims,
            Vec3::new(1.18, -0.05, 1.15),
            Vec3::new(2.22, 0.34, 2.19),
            mat_matte_trim,
        );
        prims.push(Primitive::Sphere(Sphere::new(
            Vec3::new(1.70, 1.06, 1.67),
            0.72,
            mat_subsurface_jade,
        )));

        // 6. Hero Primitive C (Center-Right Midground): Quaternion-Rotated CSG Signed-Distance Sculpture!
        //    Boolean Difference(SmoothUnion(RoundBox, Torus), Union(CylX, Union(CylY, CylZ)))
        let csg_center = Vec3::new(0.0, 1.28, 2.05);
        let rot = Quat::from_euler_yxz(0.58, 0.34, -0.24);
        let world_from_local = Mat4::translation(csg_center) * rot.to_mat4();
        let local_from_world = world_from_local
            .inverse()
            .expect("Rigid CSG transform must be invertible");

        let outer_body = SdfNode::SmoothUnion {
            a: Box::new(SdfNode::RoundBox {
                center: Vec3::ZERO,
                half_extents: Vec3::splat(0.52),
                corner_radius: 0.08,
            }),
            b: Box::new(SdfNode::Torus {
                center: Vec3::ZERO,
                major_radius: 0.68,
                minor_radius: 0.11,
            }),
            k: 0.09,
        };
        let cross_holes = SdfNode::Union(
            Box::new(SdfNode::Cylinder {
                center: Vec3::ZERO,
                radius: 0.28,
                half_height: 1.2,
                axis: 0,
            }),
            Box::new(SdfNode::Union(
                Box::new(SdfNode::Cylinder {
                    center: Vec3::ZERO,
                    radius: 0.28,
                    half_height: 1.2,
                    axis: 1,
                }),
                Box::new(SdfNode::Cylinder {
                    center: Vec3::ZERO,
                    radius: 0.28,
                    half_height: 1.2,
                    axis: 2,
                }),
            )),
        );
        let csg_sdf = SdfNode::Transform {
            inv_matrix: local_from_world,
            uniform_scale: 1.0,
            child: Box::new(SdfNode::Difference(
                Box::new(outer_body),
                Box::new(cross_holes),
            )),
        };
        prims.push(Primitive::Csg(CsgPrimitive::new(
            csg_sdf,
            Aabb::new(
                csg_center - Vec3::splat(0.98),
                csg_center + Vec3::splat(0.98),
            ),
            mat_gold_conductor,
        )));

        // Suspended glowing plasma pearl inside the carved hollow CSG chamber!
        prims.push(Primitive::Sphere(Sphere::new(
            csg_center,
            0.17,
            mat_core_beacon,
        )));

        // Accent Mirror-Chrome Spheres flanking the courtyard
        prims.push(Primitive::Sphere(Sphere::new(
            Vec3::new(-2.85, 0.52, 0.35),
            0.52,
            mat_chrome,
        )));
        prims.push(Primitive::Sphere(Sphere::new(
            Vec3::new(2.85, 0.52, 0.35),
            0.52,
            mat_chrome,
        )));

        let look_from = Vec3::new(0.0, 2.18, 6.65);
        let look_at = Vec3::new(0.0, 1.62, 0.45);
        let focus_dist = (look_from - csg_center).length();
        let camera = Camera::new(
            look_from,
            look_at,
            Vec3::Y,
            39.0,
            aspect_ratio,
            0.024, // Subtle physically based thin-lens depth of field
            focus_dist,
        );

        Self::new(prims, materials, medium, camera)
    }
}
