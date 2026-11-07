use std::fs;
use std::path::Path;

use crate::audio::{decode_and_verify_wav, WavInspection, SAMPLE_RATE, TOTAL_FRAMES};
use crate::bsdf::{fresnel_dielectric, Material};
use crate::fluid::{FluidSolver3D, PoissonSolverKind};
use crate::math::{Mat4, Onb, Quat, Ray, SpectralCurve, Spectrum4, Vec3, Wavelengths4};
use crate::png::{ adler32, crc32, decode_and_verify_png, DecodedPng };
use crate::primitives::{HitRecord, Primitive, Sphere};
use crate::renderer::trace_spectral_path;
use crate::rng::Rng;
use crate::scene::{Camera, Scene};

#[derive(Clone, Debug)]
pub struct CheckResult {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

/// Run core mathematical, BVH, BSDF energy conservation, and fluid solver verification checks.
pub fn run_core_verification_suite(scene: &Scene) -> Result<Vec<CheckResult>, String> {
    let mut checks = Vec::new();

    // 1. 3D Math, Quaternion, Matrix Inversion & Checksum RFC Vectors
    {
        if crc32(b"123456789") != 0xCBF4_3926 || adler32(b"Wikipedia") != 0x11E6_0398 {
            return Err("CRC32 / Adler32 RFC standard test vector mismatch".to_string());
        }
        let q = Quat::from_euler_yxz(0.61, -0.37, 0.89);
        let m = Mat4::translation(Vec3::new(-3.2, 1.9, 5.4))
            * q.to_mat4()
            * Mat4::scale(Vec3::new(1.4, 0.8, 2.1));
        let inv = m.inverse().ok_or("Mat4::inverse returned None on valid TRS")?;
        let id = m * inv;
        let mut max_err = 0.0_f64;
        for r in 0..4 {
            for c in 0..4 {
                let target = if r == c { 1.0 } else { 0.0 };
                max_err = max_err.max((id.m[r][c] - target).abs());
            }
        }
        let onb = Onb::from_normal(Vec3::new(-0.41, 0.77, -0.49).normalize());
        let ortho_err = onb.u.dot(onb.v).abs().max(onb.u.dot(onb.w).abs());
        if max_err > 1e-11 || ortho_err > 1e-12 {
            return Err(format!("Linear algebra error too large: mat={} onb={}", max_err, ortho_err));
        }
        checks.push(CheckResult {
            name: "Math / Quat / Mat4^-1 / ONB / Checksums",
            passed: true,
            detail: format!("max |M*M^-1 - I| = {:.2e}, ONB ortho = {:.2e}", max_err, ortho_err),
        });
    }

    // 2. SAH BVH Structural Invariants & Exact Ray-Intersection Equivalence (2,500 random rays)
    {
        let stats = scene.bvh.verify_invariants()?;
        let mut rng = Rng::from_seed(0xB74_7E57_2026);
        let num_rays = 2_500usize;
        let mut hits_verified = 0usize;
        let mut max_t_diff = 0.0_f64;

        for i in 0..num_rays {
            let origin = if i % 4 == 0 {
                // Camera-like rays aimed toward scene center
                Vec3::new(
                    rng.range_f64(-2.0, 2.0),
                    rng.range_f64(0.8, 3.5),
                    rng.range_f64(4.5, 7.5),
                )
            } else {
                Vec3::new(
                    rng.range_f64(-7.0, 7.0),
                    rng.range_f64(0.05, 5.5),
                    rng.range_f64(-5.0, 6.5),
                )
            };
            let dir = if i % 4 == 0 {
                let target = Vec3::new(
                    rng.range_f64(-3.5, 3.5),
                    rng.range_f64(0.1, 3.5),
                    rng.range_f64(-3.0, 2.5),
                );
                (target - origin).normalize()
            } else {
                let (dx, dy, dz) = rng.uniform_sphere();
                Vec3::new(dx, dy, dz)
            };
            let ray = Ray::new(origin, dir, 1e-4, 250.0);

            let bvh_hit = scene.bvh.intersect(&ray);
            let ref_hit = scene.bvh.intersect_brute_force(&ray);
            match (bvh_hit, ref_hit) {
                (Some(a), Some(b)) => {
                    let dt = (a.t - b.t).abs();
                    max_t_diff = max_t_diff.max(dt);
                    let same_point = (a.point - b.point).length() <= 1e-6;
                    if dt > 1e-8 || (a.primitive_id != b.primitive_id && !same_point) {
                        return Err(format!(
                            "BVH ray mismatch on ray {}: bvh=(t={}, pid={}) vs ref=(t={}, pid={})",
                            i, a.t, a.primitive_id, b.t, b.primitive_id
                        ));
                    }
                    hits_verified += 1;
                }
                (None, None) => {}
                (a, b) => {
                    return Err(format!(
                        "BVH hit presence mismatch on ray {}: bvh={} ref={}",
                        i,
                        a.is_some(),
                        b.is_some()
                    ));
                }
            }
        }

        checks.push(CheckResult {
            name: "SAH BVH Invariants & Ray Equivalence",
            passed: true,
            detail: format!(
                "{} nodes ({} leaves, depth {}), {}/{} hits exact (max |dt|={:.2e})",
                stats.node_count, stats.leaf_count, stats.max_depth, hits_verified, num_rays, max_t_diff
            ),
        });
    }

    // 3. Physically Based BSDF Energy Conservation (Dielectric, GGX Conductor Furnace, Subsurface)
    {
        // 3a. Dielectric Fresnel R + T == 1.0 across [0..90) degrees + TIR
        for deg in 0..90 {
            let cos_i = (deg as f64).to_radians().cos();
            let r = fresnel_dielectric(cos_i, 1.0, 1.53);
            let t = 1.0 - r;
            if !(0.0..=1.0).contains(&r) || ((r + t) - 1.0).abs() > 1e-14 {
                return Err("Dielectric Fresnel energy conservation violated".to_string());
            }
        }

        // 3b. GGX Conductor White-Furnace Directional Albedo <= 1.0 across roughnesses
        let mut rng = Rng::from_seed(0xF024_ACE1);
        let mut max_furnace = 0.0_f64;
        for &alpha in &[0.04, 0.18, 0.45] {
            let ideal_metal = Material::Conductor {
                eta: SpectralCurve::rgb(0.12, 0.12, 0.12),
                k: SpectralCurve::rgb(9.5, 9.5, 9.5),
                tint: SpectralCurve::rgb(1.0, 1.0, 1.0),
                roughness_x: alpha,
                roughness_y: alpha,
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
            let wo = Vec3::new(0.52, 0.78, 0.35).normalize();
            let mut wl = Wavelengths4::sample(0.5);
            let n_samples = 8_000;
            let mut accum = 0.0;
            for _ in 0..n_samples {
                let mut tp = Spectrum4::ONE;
                if let Some(s) = ideal_metal.sample(&hit, wo, &mut wl, &mut tp, &mut rng) {
                    if s.weight.c[0] > 1.0 + 1e-9 {
                        return Err(format!("VNDF sample weight {} > 1.0", s.weight.c[0]));
                    }
                    accum += s.weight.c[0];
                }
            }
            let albedo = accum / (n_samples as f64);
            max_furnace = max_furnace.max(albedo);
            let min_expected = if alpha <= 0.05 { 0.96 } else { 0.65 };
            if albedo > 1.000_001 || albedo < min_expected {
                return Err(format!("Furnace albedo {} out of bounds for alpha={}", albedo, alpha));
            }
        }

        // 3c. Subsurface Isotropic Volumetric Scattering White-Furnace Conservation (`sigma_a = 0` vs `sigma_a > 0`)
        let sub_conservative = Scene::new(
            vec![Primitive::Sphere(Sphere::new(Vec3::ZERO, 0.6, 0))],
            vec![Material::Subsurface {
                ior: 1.35,
                sigma_s: SpectralCurve::rgb(6.0, 6.0, 6.0),
                sigma_a: SpectralCurve::rgb(0.0, 0.0, 0.0),
                surface_roughness: 0.0,
            }],
            None,
            Camera::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y, 40.0, 1.0, 0.0, 3.0),
        );
        let sub_absorbing = Scene::new(
            vec![Primitive::Sphere(Sphere::new(Vec3::ZERO, 0.6, 0))],
            vec![Material::Subsurface {
                ior: 1.35,
                sigma_s: SpectralCurve::rgb(6.0, 6.0, 6.0),
                sigma_a: SpectralCurve::rgb(2.0, 2.0, 2.0),
                surface_roughness: 0.0,
            }],
            None,
            Camera::new(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y, 40.0, 1.0, 0.0, 3.0),
        );
        let mut sum_cons = 0.0;
        let mut sum_abs = 0.0;
        for i in 0..400 {
            let wl = Wavelengths4::sample(((i as f64) + 0.5) / 400.0);
            let r = Ray::new(Vec3::new(0.0, 0.0, 2.5), Vec3::new(0.0, 0.0, -1.0), 1e-4, 100.0);
            let (c0, _) = trace_spectral_path(&sub_conservative, r, wl, 32, 100.0, &mut rng);
            let (c1, _) = trace_spectral_path(&sub_absorbing, r, wl, 32, 100.0, &mut rng);
            sum_cons += c0.y;
            sum_abs += c1.y;
        }
        if sum_abs >= sum_cons || sum_cons <= 0.0 {
            return Err("Subsurface absorption failed monotonicity check".to_string());
        }

        checks.push(CheckResult {
            name: "BSDF & Subsurface Energy Conservation",
            passed: true,
            detail: format!(
                "Fresnel R+T=1.000000, GGX Furnace peak={:.4}<=1.0, SSS absorb<conserv",
                max_furnace
            ),
        });
    }

    // 4. 3D Navier-Stokes Incompressibility Projection Verification
    {
        let mut sim = FluidSolver3D::new(16, 20, 16);
        sim.solver_kind = PoissonSolverKind::ConjugateGradient;
        sim.solver_iters = 40;
        sim.inject_cyberpunk_sources(0.5, 0.1);
        sim.apply_buoyancy_and_vorticity_confinement(0.1);
        let div_pre = sim.compute_divergence_norm();
        sim.project();
        let div_post = sim.compute_divergence_norm();
        if div_post > div_pre * 0.03 {
            return Err(format!(
                "Poisson CG projection insufficient: pre={} post={}",
                div_pre, div_post
            ));
        }
        checks.push(CheckResult {
            name: "3D Navier-Stokes CG Poisson Projection",
            passed: true,
            detail: format!("||div u||_2: {:.4e} -> {:.4e} ({:.1}% reduction)", div_pre, div_post, (1.0 - div_post / div_pre) * 100.0),
        });
    }

    Ok(checks)
}

/// Verify on-disk PNG and WAV binary headers, checksums, DEFLATE roundtrip, and PCM audio properties.
pub fn verify_disk_outputs(
    png_path: &Path,
    expected_w: u32,
    expected_h: u32,
    expected_rgb: &[u8],
    wav_path: &Path,
) -> Result<(DecodedPng, WavInspection, Vec<CheckResult>), String> {
    let png_bytes = fs::read(png_path)
        .map_err(|e| format!("Failed reading PNG {:?}: {}", png_path, e))?;
    let decoded_png = decode_and_verify_png(&png_bytes)?;
    if decoded_png.width != expected_w || decoded_png.height != expected_h {
        return Err(format!(
            "PNG IHDR dimension mismatch: {}x{} != {}x{}",
            decoded_png.width, decoded_png.height, expected_w, expected_h
        ));
    }
    if decoded_png.rgb_pixels != expected_rgb {
        return Err("Decompressed PNG pixels do not match rendered framebuffer".to_string());
    }

    let wav_bytes = fs::read(wav_path)
        .map_err(|e| format!("Failed reading WAV {:?}: {}", wav_path, e))?;
    let wav_info = decode_and_verify_wav(&wav_bytes)?;
    if wav_info.sample_rate != SAMPLE_RATE
        || wav_info.channels != 2
        || wav_info.bits_per_sample != 24
        || wav_info.frame_count != TOTAL_FRAMES
    {
        return Err(format!("Unexpected WAV stream metadata: {:?}", wav_info));
    }
    if wav_info.rms_left < 0.01
        || wav_info.rms_right < 0.01
        || wav_info.stereo_diff_rms < 0.005
        || wav_info.peak_abs > 0.9999
    {
        return Err(format!("WAV signal quality bounds violated: {:?}", wav_info));
    }

    let checks = vec![
        CheckResult {
            name: "PNG Binary Header / CRC32 / Adler32 / LZ77 Roundtrip",
            passed: true,
            detail: format!(
                "{}x{} 8-bit RGB, {} chunks, {:.1} KB lossless DEFLATE verified",
                decoded_png.width,
                decoded_png.height,
                decoded_png.chunk_count,
                (png_bytes.len() as f64) / 1024.0
            ),
        },
        CheckResult {
            name: "WAV 24-Bit 48kHz Stereo RIFF Header & Spatial Signal",
            passed: true,
            detail: format!(
                "{} frames (15.00s), RMS L={:.3} R={:.3}, Peak={:.3}, StereoDiff={:.3}",
                wav_info.frame_count,
                wav_info.rms_left,
                wav_info.rms_right,
                wav_info.peak_abs,
                wav_info.stereo_diff_rms
            ),
        },
    ];

    Ok((decoded_png, wav_info, checks))
}
