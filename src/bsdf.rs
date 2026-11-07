use std::f64::consts::PI;

use crate::math::{Onb, SpectralCurve, Spectrum4, Vec3, Wavelengths4};
use crate::primitives::HitRecord;
use crate::rng::Rng;

// ============================================================================
// Exact Dielectric & Complex Conductor Fresnel Equations
// ============================================================================

/// Exact unpolarized Fresnel reflectance for a smooth dielectric interface (`eta_i` -> `eta_t`).
#[inline]
pub fn fresnel_dielectric(cos_theta_i: f64, eta_i: f64, eta_t: f64) -> f64 {
    let cos_i = cos_theta_i.clamp(0.0, 1.0);
    let sin2_i = (1.0 - cos_i * cos_i).max(0.0);
    let eta = eta_i / eta_t;
    let sin2_t = eta * eta * sin2_i;
    if sin2_t >= 1.0 {
        return 1.0; // Total Internal Reflection
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    let r_parl = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t).max(1e-18);
    let r_perp = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t).max(1e-18);
    0.5 * (r_parl * r_parl + r_perp * r_perp)
}

/// Exact Born & Wolf unpolarized Fresnel reflectance for a complex conductor `(eta + i*k)`.
#[inline]
pub fn fresnel_conductor(cos_theta_i: f64, eta: f64, k: f64) -> f64 {
    let cos_i = cos_theta_i.clamp(0.0, 1.0);
    let cos2 = cos_i * cos_i;
    let sin2 = (1.0 - cos2).max(0.0);
    let eta2 = eta * eta;
    let k2 = k * k;

    let t0 = eta2 - k2 - sin2;
    let a2plusb2 = (t0 * t0 + 4.0 * eta2 * k2).max(0.0).sqrt();
    let t1 = a2plusb2 + cos2;
    let a = (0.5 * (a2plusb2 + t0).max(0.0)).sqrt();
    let t2 = 2.0 * cos_i * a;
    let rs = (t1 - t2) / (t1 + t2).max(1e-18);

    let t3 = cos2 * a2plusb2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let rp = rs * (t3 - t4) / (t3 + t4).max(1e-18);
    (0.5 * (rp + rs)).clamp(0.0, 1.0)
}

// ============================================================================
// Trowbridge-Reitz GGX Microfacet Distribution + Smith G2 + Heitz VNDF
// ============================================================================

#[inline]
pub fn ggx_d(m: Vec3, ax: f64, ay: f64) -> f64 {
    if m.z <= 0.0 {
        return 0.0;
    }
    let term = (m.x / ax).powi(2) + (m.y / ay).powi(2) + m.z * m.z;
    1.0 / (PI * ax * ay * term * term).max(1e-24)
}

#[inline]
pub fn smith_lambda(w: Vec3, ax: f64, ay: f64) -> f64 {
    let abs_tan2 = (ax * ax * w.x * w.x + ay * ay * w.y * w.y) / (w.z * w.z).max(1e-18);
    0.5 * ((1.0 + abs_tan2).sqrt() - 1.0)
}

#[inline]
pub fn smith_g1(w: Vec3, ax: f64, ay: f64) -> f64 {
    if w.z <= 0.0 {
        return 0.0;
    }
    1.0 / (1.0 + smith_lambda(w, ax, ay))
}

/// Height-correlated Smith shadowing-masking G2(wo, wi).
#[inline]
pub fn smith_g2(wo: Vec3, wi: Vec3, ax: f64, ay: f64) -> f64 {
    if wo.z <= 0.0 || wi.z <= 0.0 {
        return 0.0;
    }
    1.0 / (1.0 + smith_lambda(wo, ax, ay) + smith_lambda(wi, ax, ay))
}

/// Eric Heitz (JCGT 2018) Exact Visible Normal Distribution Function (VNDF) sampler.
#[inline]
pub fn sample_ggx_vndf(wo: Vec3, ax: f64, ay: f64, u1: f64, u2: f64) -> Vec3 {
    // Stretch view vector to hemispherical configuration
    let vh = Vec3::new(ax * wo.x, ay * wo.y, wo.z.max(1e-7)).normalize();

    // Orthonormal basis around vh
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 1e-14 {
        Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::X
    };
    let t2 = vh.cross(t1);

    // Uniformly sample projected disk area
    let r = u1.sqrt();
    let phi = 2.0 * PI * u2;
    let p1 = r * phi.cos();
    let mut p2 = r * phi.sin();
    let s = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2;

    let nh = t1 * p1 + t2 * p2 + vh * (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    Vec3::new(ax * nh.x, ay * nh.y, nh.z.max(1e-7)).normalize()
}

// ============================================================================
// Material Definitions & BSDF Sampling / Evaluation
// ============================================================================

#[derive(Clone, Debug)]
pub enum Material {
    /// Matte spectral Lambertian reflector (`albedo(lambda) in [0, 1]`).
    Diffuse {
        albedo: SpectralCurve,
        checker_alt: Option<(SpectralCurve, f64)>,
    },
    /// Physically based GGX Microfacet Conductor with complex spectral IOR `(eta(lambda), k(lambda))`.
    Conductor {
        eta: SpectralCurve,
        k: SpectralCurve,
        tint: SpectralCurve,
        roughness_x: f64,
        roughness_y: f64,
    },
    /// Smooth Specular Dielectric with exact Fresnel + Snell refraction + Cauchy dispersion + Beer-Lambert absorption.
    Dielectric {
        base_ior: f64,
        /// Cauchy B coefficient in um^2 (`n(lambda) = base_ior + dispersion_b / lambda_um^2`).
        dispersion_b: f64,
        /// Spectral Beer-Lambert absorption coefficient `sigma_a` [m^-1].
        sigma_a: SpectralCurve,
    },
    /// Subsurface Isotropic Volumetric Scattering Solid (`sigma_s`, `sigma_a`, dielectric boundary `ior`).
    Subsurface {
        ior: f64,
        sigma_s: SpectralCurve,
        sigma_a: SpectralCurve,
        surface_roughness: f64,
    },
    /// Glossy Dielectric-Coated Diffuse Substrate (e.g. wet rain-slicked cyberpunk asphalt).
    CoatedDiffuse {
        base_albedo: SpectralCurve,
        coat_ior: f64,
        roughness: f64,
        puddle_modulation: bool,
    },
    /// Spectral Area Light Emitter.
    Emissive {
        emission: SpectralCurve,
        strength: f64,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct BsdfSample {
    pub wi: Vec3,
    /// Pre-multiplied importance weight `f(wo, wi) * |cos_theta_i| / pdf` (exact cancellation for VNDF & delta).
    pub weight: Spectrum4,
    pub pdf: f64,
    pub is_delta: bool,
    /// True if ray just entered a Subsurface Isotropic Volumetric Scattering interior.
    pub entered_subsurface: bool,
}

impl Material {
    #[inline]
    pub fn is_emissive(&self) -> bool {
        matches!(self, Self::Emissive { .. })
    }

    #[inline]
    pub fn is_delta(&self) -> bool {
        match self {
            Self::Dielectric { .. } => true,
            Self::Subsurface { .. } => true,
            Self::Conductor {
                roughness_x,
                roughness_y,
                ..
            } => *roughness_x < 1e-3 && *roughness_y < 1e-3,
            _ => false,
        }
    }

    #[inline]
    pub fn emission(&self, wl: &Wavelengths4) -> Spectrum4 {
        match self {
            Self::Emissive { emission, strength } => emission.eval4(wl) * *strength,
            _ => Spectrum4::ZERO,
        }
    }

    #[inline]
    pub fn mean_emission_power(&self) -> f64 {
        match self {
            Self::Emissive { emission, strength } => {
                let avg = 0.25
                    * (emission.eval_at(430.0)
                        + emission.eval_at(510.0)
                        + emission.eval_at(580.0)
                        + emission.eval_at(650.0));
                avg * *strength
            }
            _ => 0.0,
        }
    }

    #[inline]
    pub fn dielectric_absorption(&self, wl: &Wavelengths4) -> Option<Spectrum4> {
        match self {
            Self::Dielectric { sigma_a, .. } => Some(sigma_a.eval4(wl)),
            _ => None,
        }
    }

    #[inline]
    pub fn subsurface_params(&self, wl: &Wavelengths4) -> Option<(f64, Spectrum4, Spectrum4)> {
        match self {
            Self::Subsurface {
                ior,
                sigma_s,
                sigma_a,
                ..
            } => Some((*ior, sigma_s.eval4(wl), sigma_a.eval4(wl))),
            _ => None,
        }
    }

    #[inline]
    fn coated_roughness(&self, hit: &HitRecord) -> f64 {
        match self {
            Self::CoatedDiffuse {
                roughness,
                puddle_modulation,
                ..
            } => {
                if *puddle_modulation {
                    // Procedural rain puddle modulation on cyberpunk street floor
                    let pattern = (hit.point.x * 1.35).sin() * (hit.point.z * 1.15).cos()
                        + 0.5 * (hit.point.x * 3.1 - hit.point.z * 2.7).sin();
                    if pattern > 0.15 {
                        0.025 // Mirror-like wet puddle
                    } else {
                        *roughness
                    }
                } else {
                    *roughness
                }
            }
            _ => 0.1,
        }
    }

    /// Evaluate `f(wo, wi)` [sr^-1] for non-delta BSDFs (used by Next Event Estimation).
    pub fn eval(&self, hit: &HitRecord, wo_world: Vec3, wi_world: Vec3, wl: &Wavelengths4) -> Spectrum4 {
        let onb = Onb::from_normal(hit.normal);
        let wo = onb.to_local(wo_world);
        let wi = onb.to_local(wi_world);
        if wo.z <= 1e-7 || wi.z <= 1e-7 {
            return Spectrum4::ZERO;
        }

        match self {
            Self::Diffuse {
                albedo,
                checker_alt,
            } => {
                let rho = if let Some((alt, scale)) = checker_alt {
                    let cx = (hit.point.x * *scale).floor() as i64;
                    let cz = (hit.point.z * *scale).floor() as i64;
                    if (cx + cz).rem_euclid(2) == 0 {
                        albedo.eval4(wl)
                    } else {
                        alt.eval4(wl)
                    }
                } else {
                    albedo.eval4(wl)
                };
                rho * (1.0 / PI)
            }
            Self::Conductor {
                eta,
                k,
                tint,
                roughness_x,
                roughness_y,
            } => {
                let ax = roughness_x.max(0.002);
                let ay = roughness_y.max(0.002);
                let h_unnorm = wo + wi;
                if h_unnorm.length_squared() < 1e-16 {
                    return Spectrum4::ZERO;
                }
                let m = h_unnorm.normalize();
                let cos_d = wo.dot(m).clamp(0.0, 1.0);
                let d = ggx_d(m, ax, ay);
                let g2 = smith_g2(wo, wi, ax, ay);
                let denom = (4.0 * wo.z * wi.z).max(1e-16);
                let eta4 = eta.eval4(wl);
                let k4 = k.eval4(wl);
                let tint4 = tint.eval4(wl);
                let mut f4 = Spectrum4::ZERO;
                for i in 0..4 {
                    f4.c[i] = fresnel_conductor(cos_d, eta4.c[i], k4.c[i]) * tint4.c[i];
                }
                f4 * (d * g2 / denom)
            }
            Self::CoatedDiffuse {
                base_albedo,
                coat_ior,
                ..
            } => {
                let alpha = self.coated_roughness(hit).max(0.01);
                let m = (wo + wi).normalize();
                let cos_d = wo.dot(m).clamp(0.0, 1.0);
                let f_spec = fresnel_dielectric(cos_d, 1.0, *coat_ior);
                let d = ggx_d(m, alpha, alpha);
                let g2 = smith_g2(wo, wi, alpha, alpha);
                let spec_val = f_spec * d * g2 / (4.0 * wo.z * wi.z).max(1e-16);

                let t_out = 1.0 - fresnel_dielectric(wo.z, 1.0, *coat_ior);
                let t_in = 1.0 - fresnel_dielectric(wi.z, 1.0, *coat_ior);
                let diff = base_albedo.eval4(wl) * (t_out * t_in / PI);
                diff + Spectrum4::splat(spec_val)
            }
            Self::Dielectric { .. } | Self::Subsurface { .. } | Self::Emissive { .. } => {
                Spectrum4::ZERO
            }
        }
    }

    /// Evaluate solid-angle sampling PDF `p(wi)` [sr^-1] matching `sample()`.
    pub fn pdf(&self, hit: &HitRecord, wo_world: Vec3, wi_world: Vec3) -> f64 {
        let onb = Onb::from_normal(hit.normal);
        let wo = onb.to_local(wo_world);
        let wi = onb.to_local(wi_world);
        if wo.z <= 1e-7 || wi.z <= 1e-7 {
            return 0.0;
        }

        match self {
            Self::Diffuse { .. } => wi.z / PI,
            Self::Conductor {
                roughness_x,
                roughness_y,
                ..
            } => {
                let ax = roughness_x.max(0.002);
                let ay = roughness_y.max(0.002);
                let h_unnorm = wo + wi;
                if h_unnorm.length_squared() < 1e-16 {
                    return 0.0;
                }
                let m = h_unnorm.normalize();
                ggx_d(m, ax, ay) * smith_g1(wo, ax, ay) / (4.0 * wo.z).max(1e-16)
            }
            Self::CoatedDiffuse { coat_ior, .. } => {
                let alpha = self.coated_roughness(hit).max(0.01);
                let f_view = fresnel_dielectric(wo.z, 1.0, *coat_ior);
                let w_spec = f_view.clamp(0.15, 0.85);
                let m = (wo + wi).normalize();
                let pdf_spec =
                    ggx_d(m, alpha, alpha) * smith_g1(wo, alpha, alpha) / (4.0 * wo.z).max(1e-16);
                let pdf_diff = wi.z / PI;
                w_spec * pdf_spec + (1.0 - w_spec) * pdf_diff
            }
            Self::Dielectric { .. } | Self::Subsurface { .. } | Self::Emissive { .. } => 0.0,
        }
    }

    /// Importance-sample BSDF direction `wi` given outgoing direction `wo_world` (pointing toward viewer).
    pub fn sample(
        &self,
        hit: &HitRecord,
        wo_world: Vec3,
        wl: &mut Wavelengths4,
        throughput: &mut Spectrum4,
        rng: &mut Rng,
    ) -> Option<BsdfSample> {
        let onb = Onb::from_normal(hit.normal);
        let wo = onb.to_local(wo_world);
        if wo.z <= 1e-9 {
            return None;
        }

        match self {
            Self::Diffuse { .. } => {
                let (lx, ly, lz) = rng.cosine_hemisphere();
                if lz <= 1e-7 {
                    return None;
                }
                let wi_world = onb.to_world(Vec3::new(lx, ly, lz));
                let pdf = lz / PI;
                let f = self.eval(hit, wo_world, wi_world, wl);
                Some(BsdfSample {
                    wi: wi_world,
                    weight: f * (lz / pdf),
                    pdf,
                    is_delta: false,
                    entered_subsurface: false,
                })
            }
            Self::Conductor {
                eta,
                k,
                tint,
                roughness_x,
                roughness_y,
            } => {
                if *roughness_x < 1e-3 && *roughness_y < 1e-3 {
                    // Exact mirror conductor
                    let wi_local = Vec3::new(-wo.x, -wo.y, wo.z);
                    let eta4 = eta.eval4(wl);
                    let k4 = k.eval4(wl);
                    let tint4 = tint.eval4(wl);
                    let mut weight = Spectrum4::ZERO;
                    for i in 0..4 {
                        weight.c[i] = fresnel_conductor(wo.z, eta4.c[i], k4.c[i]) * tint4.c[i];
                    }
                    return Some(BsdfSample {
                        wi: onb.to_world(wi_local),
                        weight,
                        pdf: 1.0,
                        is_delta: true,
                        entered_subsurface: false,
                    });
                }
                let ax = roughness_x.max(0.002);
                let ay = roughness_y.max(0.002);
                let m = sample_ggx_vndf(wo, ax, ay, rng.next_f64(), rng.next_f64());
                let cos_d = wo.dot(m);
                if cos_d <= 1e-8 {
                    return None;
                }
                let wi = m * (2.0 * cos_d) - wo;
                if wi.z <= 1e-7 {
                    return None;
                }
                let g1 = smith_g1(wo, ax, ay).max(1e-16);
                let g2 = smith_g2(wo, wi, ax, ay);
                let eta4 = eta.eval4(wl);
                let k4 = k.eval4(wl);
                let tint4 = tint.eval4(wl);
                let mut weight = Spectrum4::ZERO;
                for i in 0..4 {
                    let f = fresnel_conductor(cos_d, eta4.c[i], k4.c[i]) * tint4.c[i];
                    weight.c[i] = f * (g2 / g1);
                }
                let pdf = ggx_d(m, ax, ay) * g1 / (4.0 * wo.z).max(1e-16);
                Some(BsdfSample {
                    wi: onb.to_world(wi),
                    weight,
                    pdf,
                    is_delta: false,
                    entered_subsurface: false,
                })
            }
            Self::Dielectric {
                base_ior,
                dispersion_b,
                ..
            } => {
                // Hero wavelength IOR via Cauchy's equation: n(lambda) = n0 + B / lambda_um^2
                let ior_at = |nm: f64| -> f64 {
                    let um = nm * 1e-3;
                    *base_ior + *dispersion_b / (um * um)
                };
                let n_hero = ior_at(wl.hero());
                let (eta_i, eta_t) = if hit.front_face {
                    (1.0, n_hero)
                } else {
                    (n_hero, 1.0)
                };
                let f_hero = fresnel_dielectric(wo.z, eta_i, eta_t);
                let incident = -wo_world;

                if rng.next_f64() < f_hero {
                    // Specular reflection: geometry is wavelength-independent!
                    let wi_world = incident.reflect(hit.normal).normalize();
                    let mut weight = Spectrum4::ONE;
                    if *dispersion_b > 0.0 && !wl.hero_only && f_hero > 1e-9 {
                        for i in 0..4 {
                            let ni = ior_at(wl.lambda[i]);
                            let (ei, et) = if hit.front_face { (1.0, ni) } else { (ni, 1.0) };
                            weight.c[i] = fresnel_dielectric(wo.z, ei, et) / f_hero;
                        }
                    }
                    Some(BsdfSample {
                        wi: wi_world,
                        weight,
                        pdf: 1.0,
                        is_delta: true,
                        entered_subsurface: false,
                    })
                } else {
                    // Snell refraction: when dispersive, collapse packet to hero wavelength (Wilkie 2014 MIS)
                    if *dispersion_b > 0.0 {
                        throughput.collapse_to_hero(wl);
                    }
                    let eta_ratio = eta_i / eta_t;
                    if let Some(refracted) = Vec3::refract(incident, hit.normal, eta_ratio) {
                        Some(BsdfSample {
                            wi: refracted,
                            weight: Spectrum4::ONE,
                            pdf: 1.0,
                            is_delta: true,
                            entered_subsurface: false,
                        })
                    } else {
                        // Fallback TIR guard
                        let wi_world = incident.reflect(hit.normal).normalize();
                        Some(BsdfSample {
                            wi: wi_world,
                            weight: Spectrum4::ONE,
                            pdf: 1.0,
                            is_delta: true,
                            entered_subsurface: false,
                        })
                    }
                }
            }
            Self::Subsurface {
                ior,
                surface_roughness,
                ..
            } => {
                // Smooth/slightly-rough dielectric boundary gating entry into Subsurface Isotropic Random Walk
                let alpha = surface_roughness.max(0.0);
                let micro_n = if alpha > 1e-3 {
                    let m_local = sample_ggx_vndf(wo, alpha, alpha, rng.next_f64(), rng.next_f64());
                    onb.to_world(m_local)
                } else {
                    hit.normal
                };
                let cos_i = wo_world.dot(micro_n).clamp(0.0, 1.0);
                let (eta_i, eta_t) = if hit.front_face {
                    (1.0, *ior)
                } else {
                    (*ior, 1.0)
                };
                let f = if (*ior - 1.0).abs() < 1e-4 {
                    0.0
                } else {
                    fresnel_dielectric(cos_i, eta_i, eta_t)
                };
                let incident = -wo_world;

                if rng.next_f64() < f {
                    let wi_world = incident.reflect(micro_n).normalize();
                    if wi_world.dot(hit.normal) <= 0.0 {
                        return None;
                    }
                    Some(BsdfSample {
                        wi: wi_world,
                        weight: Spectrum4::ONE,
                        pdf: 1.0,
                        is_delta: true,
                        entered_subsurface: false,
                    })
                } else if let Some(refracted) = Vec3::refract(incident, micro_n, eta_i / eta_t) {
                    Some(BsdfSample {
                        wi: refracted,
                        weight: Spectrum4::ONE,
                        pdf: 1.0,
                        is_delta: true,
                        entered_subsurface: hit.front_face,
                    })
                } else {
                    let wi_world = incident.reflect(micro_n).normalize();
                    Some(BsdfSample {
                        wi: wi_world,
                        weight: Spectrum4::ONE,
                        pdf: 1.0,
                        is_delta: true,
                        entered_subsurface: false,
                    })
                }
            }
            Self::CoatedDiffuse { coat_ior, .. } => {
                let alpha = self.coated_roughness(hit).max(0.01);
                let f_view = fresnel_dielectric(wo.z, 1.0, *coat_ior);
                let w_spec = f_view.clamp(0.15, 0.85);

                let wi_local = if rng.next_f64() < w_spec {
                    let m = sample_ggx_vndf(wo, alpha, alpha, rng.next_f64(), rng.next_f64());
                    let cos_d = wo.dot(m);
                    if cos_d <= 1e-8 {
                        return None;
                    }
                    m * (2.0 * cos_d) - wo
                } else {
                    let (lx, ly, lz) = rng.cosine_hemisphere();
                    Vec3::new(lx, ly, lz)
                };
                if wi_local.z <= 1e-7 {
                    return None;
                }
                let wi_world = onb.to_world(wi_local);
                let pdf = self.pdf(hit, wo_world, wi_world);
                if pdf <= 1e-14 {
                    return None;
                }
                let f = self.eval(hit, wo_world, wi_world, wl);
                Some(BsdfSample {
                    wi: wi_world,
                    weight: f * (wi_local.z / pdf),
                    pdf,
                    is_delta: false,
                    entered_subsurface: false,
                })
            }
            Self::Emissive { .. } => None,
        }
    }
}

/// Veach's Power Heuristic (`beta = 2`) for Multiple Importance Sampling.
#[inline]
pub fn mis_power_heuristic(pdf_a: f64, pdf_b: f64) -> f64 {
    let a2 = pdf_a * pdf_a;
    let b2 = pdf_b * pdf_b;
    let denom = a2 + b2;
    if denom > 1e-30 {
        a2 / denom
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dielectric_fresnel_brewster_and_tir() {
        // Normal incidence for glass n = 1.5 -> R = ((1 - 1.5)/(1 + 1.5))^2 = 0.04
        let r0 = fresnel_dielectric(1.0, 1.0, 1.5);
        assert!((r0 - 0.04).abs() < 1e-12);

        // Total Internal Reflection from n=1.5 -> n=1.0 beyond critical angle asin(1/1.5) ~ 41.81 deg
        let cos_tir = 60.0_f64.to_radians().cos();
        let r_tir = fresnel_dielectric(cos_tir, 1.5, 1.0);
        assert!((r_tir - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_conductor_white_furnace_and_reciprocity() {
        // 100% reflective ideal conductor (`k` large -> F -> 1.0): directional albedo <= 1.0 (White Furnace Test)
        let mat = Material::Conductor {
            eta: SpectralCurve::rgb(0.15, 0.15, 0.15),
            k: SpectralCurve::rgb(8.0, 8.0, 8.0),
            tint: SpectralCurve::rgb(1.0, 1.0, 1.0),
            roughness_x: 0.28,
            roughness_y: 0.28,
        };
        let hit = HitRecord::new(
            1.0,
            Vec3::ZERO,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::Y,
            Vec3::Y,
            (0.0, 0.0),
            0,
            0,
        );
        let mut rng = Rng::from_seed(1337);
        let mut wl = Wavelengths4::sample(0.35);
        let wo = Vec3::new(0.45, 0.85, 0.27).normalize();

        // Check Helmholtz reciprocity: f(wo, wi) == f(wi, wo)
        let wi = Vec3::new(-0.31, 0.78, -0.54).normalize();
        let f_ab = mat.eval(&hit, wo, wi, &wl);
        let f_ba = mat.eval(&hit, wi, wo, &wl);
        assert!((f_ab.c[0] - f_ba.c[0]).abs() < 1e-12);

        // White furnace integral E[f(wo, wi) * cos(theta_i) / pdf(wi)] <= 1.0
        let n = 20_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let mut tp = Spectrum4::ONE;
            if let Some(s) = mat.sample(&hit, wo, &mut wl, &mut tp, &mut rng) {
                assert!(s.weight.c[0] <= 1.0 + 1e-9);
                sum += s.weight.c[0];
            }
        }
        let furnace = sum / (n as f64);
        assert!(
            furnace > 0.85 && furnace <= 1.000_001,
            "White furnace energy = {} out of physical bounds",
            furnace
        );
    }
}
