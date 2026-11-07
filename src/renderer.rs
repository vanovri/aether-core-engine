use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use std::thread;
use std::time::Instant;

use crate::bsdf::mis_power_heuristic;
use crate::math::{aces_filmic_tonemap, Ray, Spectrum4, Vec3, Wavelengths4};
use crate::rng::Rng;
use crate::scene::Scene;

#[derive(Clone, Copy, Debug)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    pub spp: u32,
    pub max_bounces: u32,
    pub tile_size: u32,
    pub exposure: f64,
    pub indirect_clamp: f64,
    pub num_threads: usize,
}

impl Default for RenderConfig {
    fn default() -> Self {
        let threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            width: 1280,
            height: 720,
            spp: 64,
            max_bounces: 10,
            tile_size: 16,
            exposure: 1.22,
            indirect_clamp: 18.0,
            num_threads: threads,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TileJob {
    pub tile_id: usize,
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct TileMetric {
    pub tile_id: usize,
    pub worker_id: usize,
    pub stolen: bool,
    pub duration_ms: f64,
    pub rays_traced: u64,
    pub samples: u64,
    pub mrays_per_sec: f64,
}

#[derive(Clone, Debug)]
pub struct RenderReport {
    pub width: u32,
    pub height: u32,
    pub spp: u32,
    pub num_threads: usize,
    pub tile_count: usize,
    pub stolen_tiles: usize,
    pub total_rays: u64,
    pub wall_time_sec: f64,
    pub aggregate_mrays_per_sec: f64,
    pub aggregate_msamples_per_sec: f64,
    pub tile_latency_p50_ms: f64,
    pub tile_latency_p95_ms: f64,
    pub tile_latency_p99_ms: f64,
    pub tile_throughput_p50_mrays: f64,
    pub tile_throughput_p95_mrays: f64,
    pub tile_throughput_p99_mrays: f64,
    pub rgb8_pixels: Vec<u8>,
    pub linear_hdr_pixels: Vec<Vec3>,
}

#[inline]
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx_f = (p.clamp(0.0, 1.0)) * ((sorted.len() - 1) as f64);
    let lo = idx_f.floor() as usize;
    let hi = idx_f.ceil() as usize;
    let frac = idx_f - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Trace a single spectral Monte Carlo path (`Wavelengths4` hero-wavelength packet), returning `(linear_srgb, ray_count)`.
pub fn trace_spectral_path(
    scene: &Scene,
    initial_ray: Ray,
    mut wl: Wavelengths4,
    max_bounces: u32,
    indirect_clamp: f64,
    rng: &mut Rng,
) -> (Vec3, u64) {
    let mut ray = initial_ray;
    let mut throughput = Spectrum4::ONE;
    let mut radiance = Spectrum4::ZERO;
    let mut rays_traced = 0u64;

    let mut prev_bsdf_pdf = 0.0_f64;
    let mut prev_was_delta = true;
    // `Some((primitive_id, material_id))` while executing internal Subsurface Isotropic Volumetric Scattering random walk.
    let mut active_subsurface: Option<(usize, usize)> = None;
    let mut bounce = 0u32;
    let mut subsurface_steps = 0u32;

    while bounce < max_bounces && subsurface_steps < 48 {
        // ====================================================================
        // Branch A: Internal Subsurface Isotropic Volumetric Random Walk
        // ====================================================================
        if let Some((sub_pid, sub_mid)) = active_subsurface {
            subsurface_steps += 1;
            let Some((_, sigma_s, sigma_a)) = scene.materials[sub_mid].subsurface_params(&wl) else {
                active_subsurface = None;
                continue;
            };
            let sigma_t = Spectrum4 {
                c: [
                    (sigma_s.c[0] + sigma_a.c[0]).max(1e-5),
                    (sigma_s.c[1] + sigma_a.c[1]).max(1e-5),
                    (sigma_s.c[2] + sigma_a.c[2]).max(1e-5),
                    (sigma_s.c[3] + sigma_a.c[3]).max(1e-5),
                ],
            };
            let active_ch = if wl.hero_only { 1 } else { 4 };
            let ch = if active_ch == 1 { 0 } else { rng.next_usize(4) };
            let u = (1.0 - rng.next_f64()).max(1e-15);
            let s_free = -u.ln() / sigma_t.c[ch];

            rays_traced += 1;
            let exit_hit = scene.bvh.primitives[sub_pid]
                .intersect(&ray, ray.t_min, 1e5, sub_pid)
                .or_else(|| scene.bvh.intersect(&ray));

            let Some(hit_out) = exit_hit else {
                break;
            };

            if s_free < hit_out.t {
                // Internal isotropic scattering event inside the subsurface solid!
                let mut pdf_s = 0.0;
                for k in 0..active_ch {
                    pdf_s += sigma_t.c[k] * (-sigma_t.c[k] * s_free).exp();
                }
                pdf_s /= active_ch as f64;
                if pdf_s <= 1e-20 {
                    break;
                }
                for k in 0..active_ch {
                    throughput.c[k] *= sigma_s.c[k] * (-sigma_t.c[k] * s_free).exp() / pdf_s;
                }
                let scatter_p = ray.at(s_free);
                let (dx, dy, dz) = rng.uniform_sphere();
                ray = Ray::new(scatter_p, Vec3::new(dx, dy, dz), 1e-5, 1e5);
                prev_was_delta = false;
                prev_bsdf_pdf = 1.0 / (4.0 * std::f64::consts::PI);
            } else {
                // Ray reaches the boundary from inside the subsurface medium!
                let mut prob_surf = 0.0;
                for k in 0..active_ch {
                    prob_surf += (-sigma_t.c[k] * hit_out.t).exp();
                }
                prob_surf /= active_ch as f64;
                if prob_surf <= 1e-20 {
                    break;
                }
                for k in 0..active_ch {
                    throughput.c[k] *= (-sigma_t.c[k] * hit_out.t).exp() / prob_surf;
                }

                let mut hit_exit = hit_out;
                // Ensure exit normal treats ray as interior-to-exterior (`front_face = false`)
                if hit_exit.front_face {
                    hit_exit.front_face = false;
                    hit_exit.normal = -hit_exit.outward_normal;
                }
                let mat = &scene.materials[sub_mid];
                let Some(bsdf_s) = mat.sample(&hit_exit, -ray.dir, &mut wl, &mut throughput, rng) else {
                    break;
                };
                throughput *= bsdf_s.weight;
                // Did the ray transmit out into exterior space (`wi` points along `outward_normal`)?
                if bsdf_s.wi.dot(hit_exit.outward_normal) > 0.0 {
                    active_subsurface = None;
                    let orig = hit_exit.point + hit_exit.outward_normal * 1.5e-4;
                    ray = Ray::new(orig, bsdf_s.wi, 1e-4, 1e6);
                } else {
                    let orig = hit_exit.point - hit_exit.outward_normal * 1.5e-4;
                    ray = Ray::new(orig, bsdf_s.wi, 1e-5, 1e5);
                }
                prev_was_delta = true;
                prev_bsdf_pdf = 1.0;
            }
        } else {
            // ================================================================
            // Branch B: Exterior Scene Ray + Coupled 3D Fluid Medium + SAH BVH
            // ================================================================
            let cur_bounce = bounce;
            bounce += 1;
            rays_traced += 1;
            let surface_hit = scene.bvh.intersect(&ray);
            let surface_t = surface_hit.map_or(f64::INFINITY, |h| h.t);

            // Check if ray is currently propagating inside a Dielectric solid (`!front_face`)
            let inside_dielectric = surface_hit.is_some_and(|h| {
                !h.front_face && scene.materials[h.material_id].dielectric_absorption(&wl).is_some()
            });

            if inside_dielectric {
                if let Some(h) = surface_hit {
                    if let Some(sigma_a) = scene.materials[h.material_id].dielectric_absorption(&wl) {
                        // Exact Spectral Beer-Lambert Volumetric Attenuation inside Dielectric!
                        let beer = (sigma_a * (-h.t)).exp();
                        throughput *= beer;
                    }
                }
            }

            // Sample heterogeneous 3D Navier-Stokes smoke/plasma medium when in exterior air
            let med_event = if !inside_dielectric {
                scene
                    .medium
                    .as_ref()
                    .and_then(|m| m.sample_interaction(&ray, surface_t, &wl, rng))
            } else {
                None
            };

            if let Some(med_hit) = med_event {
                let medium = scene.medium.as_ref().unwrap();
                if !med_hit.emission.is_zero() {
                    let contrib = throughput * med_hit.emission;
                    radiance += if cur_bounce == 0 {
                        contrib
                    } else {
                        contrib.clamp(0.0, indirect_clamp)
                    };
                }
                throughput *= med_hit.albedo;

                // Next Event Estimation (MIS) at Volumetric Scattering Vertex
                if let Some(dl) = scene.sample_direct_light(med_hit.point, &wl, rng) {
                    let shadow_ray = Ray::new(med_hit.point, dl.wi, 1e-4, dl.dist - 2e-4);
                    rays_traced += 1;
                    if !scene.bvh.occluded(&shadow_ray, dl.dist - 2e-4) {
                        let tr = medium.transmittance(&shadow_ray, dl.dist - 2e-4, rng);
                        if tr > 1e-6 {
                            let phase_val = medium.phase_hg_pdf(-ray.dir, dl.wi);
                            let w_mis = mis_power_heuristic(dl.pdf_solid_angle, phase_val);
                            let contrib = throughput
                                * dl.emission
                                * (phase_val * tr * w_mis / dl.pdf_solid_angle);
                            radiance += contrib.clamp(0.0, indirect_clamp);
                        }
                    }
                }

                let (wi, pdf_phase) = medium.sample_phase_hg(-ray.dir, rng);
                prev_bsdf_pdf = pdf_phase;
                prev_was_delta = false;
                ray = Ray::new(med_hit.point, wi, 1e-4, 1e6);
            } else if let Some(hit) = surface_hit {
                let mat = &scene.materials[hit.material_id];

                // Direct or BSDF-sampled hit on an Emissive Area Light
                if mat.is_emissive() {
                    let mis_w = if cur_bounce == 0 || prev_was_delta {
                        1.0
                    } else {
                        let pdf_light = scene.pdf_direct_light(&ray, &hit);
                        mis_power_heuristic(prev_bsdf_pdf, pdf_light)
                    };
                    let contrib = throughput * mat.emission(&wl) * mis_w;
                    radiance += if cur_bounce == 0 {
                        contrib
                    } else {
                        contrib.clamp(0.0, indirect_clamp)
                    };
                    break;
                }

                // Surface Next Event Estimation (Multiple Importance Sampling: Direct Light + BSDF)
                if !mat.is_delta() {
                    let shadow_orig = hit.point + hit.normal * 1.5e-4;
                    if let Some(dl) = scene.sample_direct_light(shadow_orig, &wl, rng) {
                        let cos_surf = hit.normal.dot(dl.wi);
                        if cos_surf > 1e-6 {
                            let shadow_ray = Ray::new(shadow_orig, dl.wi, 1e-4, dl.dist - 2e-4);
                            rays_traced += 1;
                            if !scene.bvh.occluded(&shadow_ray, dl.dist - 2e-4) {
                                let tr = scene
                                    .medium
                                    .as_ref()
                                    .map_or(1.0, |m| m.transmittance(&shadow_ray, dl.dist - 2e-4, rng));
                                if tr > 1e-6 {
                                    let f_bsdf = mat.eval(&hit, -ray.dir, dl.wi, &wl);
                                    let pdf_bsdf = mat.pdf(&hit, -ray.dir, dl.wi);
                                    let w_mis = mis_power_heuristic(dl.pdf_solid_angle, pdf_bsdf);
                                    let contrib = throughput
                                        * f_bsdf
                                        * dl.emission
                                        * (cos_surf * tr * w_mis / dl.pdf_solid_angle);
                                    radiance += contrib.clamp(0.0, indirect_clamp);
                                }
                            }
                        }
                    }
                }

                // Sample BSDF for Indirect Continuation
                let Some(bsdf_s) = mat.sample(&hit, -ray.dir, &mut wl, &mut throughput, rng) else {
                    break;
                };
                throughput *= bsdf_s.weight;
                prev_bsdf_pdf = bsdf_s.pdf;
                prev_was_delta = bsdf_s.is_delta;

                if bsdf_s.entered_subsurface {
                    active_subsurface = Some((hit.primitive_id, hit.material_id));
                }

                let offset_sign = if bsdf_s.wi.dot(hit.outward_normal) >= 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let next_orig = hit.point + hit.outward_normal * (offset_sign * 1.5e-4);
                ray = Ray::new(next_orig, bsdf_s.wi, 1e-4, 1e6);
            } else {
                // Escaped scene -> sample spectral cyberpunk atmosphere
                let contrib = throughput * scene.eval_environment(ray.dir, &wl);
                radiance += if cur_bounce == 0 {
                    contrib
                } else {
                    contrib.clamp(0.0, indirect_clamp)
                };
                break;
            }
        }

        // Unbiased Russian Roulette path termination after bounce 3 (or after 10 internal SSS steps)
        if (active_subsurface.is_none() && bounce >= 4)
            || (active_subsurface.is_some() && subsurface_steps >= 10)
        {
            let max_tp = if wl.hero_only {
                throughput.c[0] * 0.25
            } else {
                throughput.max_comp()
            };
            if max_tp <= 1e-8 {
                break;
            }
            let survival = max_tp.clamp(0.05, 0.95);
            if rng.next_f64() > survival {
                break;
            }
            throughput = throughput / survival;
        }
    }

    let rgb = radiance.to_linear_srgb(&wl).max_elem(Vec3::ZERO);
    (rgb, rays_traced)
}

/// Execute the multithreaded tile-based work-stealing spectral path tracer.
pub fn render_scene_work_stealing(scene: &Scene, cfg: &RenderConfig) -> RenderReport {
    let start_wall = Instant::now();
    let num_threads = cfg.num_threads.max(1);
    let tiles_x = cfg.width.div_ceil(cfg.tile_size);
    let tiles_y = cfg.height.div_ceil(cfg.tile_size);

    let mut all_tiles = Vec::with_capacity((tiles_x * tiles_y) as usize);
    let cx = (cfg.width as f64) * 0.5;
    let cy = (cfg.height as f64) * 0.5;

    let mut tid_seq = 0usize;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x0 = tx * cfg.tile_size;
            let y0 = ty * cfg.tile_size;
            let x1 = (x0 + cfg.tile_size).min(cfg.width);
            let y1 = (y0 + cfg.tile_size).min(cfg.height);
            all_tiles.push(TileJob {
                tile_id: tid_seq,
                x0,
                y0,
                x1,
                y1,
            });
            tid_seq += 1;
        }
    }

    // Order tiles center-outward so computationally dense volumetric/CSG/refraction tiles cluster
    // unevenly across worker deques, actively engaging cross-thread work-stealing!
    all_tiles.sort_by(|a, b| {
        let ax = 0.5 * ((a.x0 + a.x1) as f64) - cx;
        let ay = 0.5 * ((a.y0 + a.y1) as f64) - cy;
        let bx = 0.5 * ((b.x0 + b.x1) as f64) - cx;
        let by = 0.5 * ((b.y0 + b.y1) as f64) - cy;
        (ax * ax + ay * ay)
            .partial_cmp(&(bx * bx + by * by))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_tiles = all_tiles.len();
    // Contiguous chunk partitioning per worker so Worker 0 gets the heavy core tiles and outer workers steal!
    let worker_deques: Vec<Mutex<VecDeque<TileJob>>> = (0..num_threads)
        .map(|_| Mutex::new(VecDeque::new()))
        .collect();
    for (i, job) in all_tiles.into_iter().enumerate() {
        let owner = (i * num_threads) / total_tiles.max(1);
        worker_deques[owner].lock().unwrap().push_back(job);
    }

    let completed_tiles = AtomicUsize::new(0);
    let stolen_count = AtomicUsize::new(0);

    struct FinishedTile {
        job: TileJob,
        pixels_hdr: Vec<Vec3>,
        metric: TileMetric,
    }

    let worker_outputs: Vec<Vec<FinishedTile>> = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(num_threads);
        for worker_id in 0..num_threads {
            let deques_ref = &worker_deques;
            let done_ref = &completed_tiles;
            let stolen_ref = &stolen_count;

            handles.push(scope.spawn(move || {
                let mut local_results = Vec::new();

                loop {
                    if done_ref.load(Ordering::Relaxed) >= total_tiles {
                        break;
                    }

                    // 1. Pop from own deque front (LIFO/FIFO local fast path)
                    let mut next_job = deques_ref[worker_id].lock().unwrap().pop_front();
                    let mut was_stolen = false;

                    // 2. Work-stealing path: steal from the back of the busiest victim deque
                    if next_job.is_none() {
                        let mut best_victim = None;
                        let mut best_len = 0usize;
                        for offset in 1..num_threads {
                            let cand = (worker_id + offset) % num_threads;
                            let len = deques_ref[cand].lock().unwrap().len();
                            if len > best_len {
                                best_len = len;
                                best_victim = Some(cand);
                            }
                        }
                        if let Some(v) = best_victim {
                            if let Some(stolen_job) = deques_ref[v].lock().unwrap().pop_back() {
                                next_job = Some(stolen_job);
                                was_stolen = true;
                                stolen_ref.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }

                    let Some(job) = next_job else {
                        if done_ref.load(Ordering::Acquire) >= total_tiles {
                            break;
                        }
                        thread::yield_now();
                        continue;
                    };

                    let t0 = Instant::now();
                    let tw = (job.x1 - job.x0) as usize;
                    let th = (job.y1 - job.y0) as usize;
                    let mut tile_hdr = Vec::with_capacity(tw * th);
                    let mut tile_rays = 0u64;
                    let inv_spp = 1.0 / (cfg.spp.max(1) as f64);
                    let w_f = cfg.width as f64;
                    let h_f = cfg.height as f64;

                    // Stratified 2D subpixel grid dimensions (e.g. 8x8 for 64 SPP)
                    let strata_side = (cfg.spp as f64).sqrt().floor().max(1.0) as u32;
                    let strata_total = strata_side * strata_side;

                    for py in job.y0..job.y1 {
                        for px in job.x0..job.x1 {
                            let mut accum_rgb = Vec3::ZERO;
                            for s in 0..cfg.spp {
                                let mut rng = Rng::from_pixel_sample(px, py, s, 0xAE74_E901);
                                let (jx, jy) = if s < strata_total {
                                    let sx = s % strata_side;
                                    let sy = s / strata_side;
                                    (
                                        ((sx as f64) + rng.next_f64()) / (strata_side as f64),
                                        ((sy as f64) + rng.next_f64()) / (strata_side as f64),
                                    )
                                } else {
                                    (rng.next_f64(), rng.next_f64())
                                };

                                // Image Y=0 is top scanline -> camera v=1.0 is top
                                let u_cam = ((px as f64) + jx) / w_f;
                                let v_cam = (((cfg.height - 1 - py) as f64) + jy) / h_f;
                                let cam_ray = scene.camera.generate_ray(u_cam, v_cam, &mut rng);

                                // Stratified hero-wavelength sampling across samples `0..spp`
                                let u_lambda = ((s as f64) + rng.next_f64()) * inv_spp;
                                let wl = Wavelengths4::sample(u_lambda);

                                let (sample_rgb, r_cnt) = trace_spectral_path(
                                    scene,
                                    cam_ray,
                                    wl,
                                    cfg.max_bounces,
                                    cfg.indirect_clamp,
                                    &mut rng,
                                );
                                tile_rays += r_cnt;
                                if sample_rgb.is_finite() {
                                    accum_rgb += sample_rgb;
                                }
                            }
                            tile_hdr.push(accum_rgb * inv_spp);
                        }
                    }

                    let dt_sec = t0.elapsed().as_secs_f64().max(1e-9);
                    let samples = ((tw * th) as u64) * (cfg.spp as u64);
                    let metric = TileMetric {
                        tile_id: job.tile_id,
                        worker_id,
                        stolen: was_stolen,
                        duration_ms: dt_sec * 1e3,
                        rays_traced: tile_rays,
                        samples,
                        mrays_per_sec: (tile_rays as f64) / (dt_sec * 1e6),
                    };

                    local_results.push(FinishedTile {
                        job,
                        pixels_hdr: tile_hdr,
                        metric,
                    });
                    done_ref.fetch_add(1, Ordering::Release);
                }
                local_results
            }));
        }

        handles
            .into_iter()
            .map(|h| h.join().expect("Render worker thread panicked"))
            .collect()
    });

    let total_pixels = (cfg.width * cfg.height) as usize;
    let mut linear_hdr_pixels = vec![Vec3::ZERO; total_pixels];
    let mut rgb8_pixels = vec![0u8; total_pixels * 3];
    let mut metrics = Vec::with_capacity(total_tiles);
    let mut total_rays = 0u64;

    for batch in worker_outputs {
        for fin in batch {
            total_rays += fin.metric.rays_traced;
            metrics.push(fin.metric);
            let tw = (fin.job.x1 - fin.job.x0) as usize;
            for (ly, py) in (fin.job.y0..fin.job.y1).enumerate() {
                for (lx, px) in (fin.job.x0..fin.job.x1).enumerate() {
                    let hdr = fin.pixels_hdr[ly * tw + lx];
                    let img_idx = (py as usize) * (cfg.width as usize) + (px as usize);
                    linear_hdr_pixels[img_idx] = hdr;

                    let mapped = aces_filmic_tonemap(hdr, cfg.exposure);
                    rgb8_pixels[img_idx * 3] = (mapped.x * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    rgb8_pixels[img_idx * 3 + 1] = (mapped.y * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    rgb8_pixels[img_idx * 3 + 2] = (mapped.z * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    let wall_time_sec = start_wall.elapsed().as_secs_f64().max(1e-9);
    let total_samples = (total_pixels as u64) * (cfg.spp as u64);

    let mut latencies: Vec<f64> = metrics.iter().map(|m| m.duration_ms).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut throughputs: Vec<f64> = metrics.iter().map(|m| m.mrays_per_sec).collect();
    throughputs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    RenderReport {
        width: cfg.width,
        height: cfg.height,
        spp: cfg.spp,
        num_threads,
        tile_count: total_tiles,
        stolen_tiles: stolen_count.load(Ordering::Relaxed),
        total_rays,
        wall_time_sec,
        aggregate_mrays_per_sec: (total_rays as f64) / (wall_time_sec * 1e6),
        aggregate_msamples_per_sec: (total_samples as f64) / (wall_time_sec * 1e6),
        tile_latency_p50_ms: percentile_sorted(&latencies, 0.50),
        tile_latency_p95_ms: percentile_sorted(&latencies, 0.95),
        tile_latency_p99_ms: percentile_sorted(&latencies, 0.99),
        tile_throughput_p50_mrays: percentile_sorted(&throughputs, 0.50),
        tile_throughput_p95_mrays: percentile_sorted(&throughputs, 0.95),
        tile_throughput_p99_mrays: percentile_sorted(&throughputs, 0.99),
        rgb8_pixels,
        linear_hdr_pixels,
    }
}
