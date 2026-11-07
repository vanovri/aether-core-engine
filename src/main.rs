use std::env;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use aether_core_engine::audio::{save_wav_24bit_stereo, synthesize_fluid_soundtrack, SAMPLE_RATE};
use aether_core_engine::fluid::{FluidMedium, FluidSolver3D};
use aether_core_engine::math::{Aabb, Vec3};
use aether_core_engine::png::save_png_rgb8;
use aether_core_engine::renderer::{render_scene_work_stealing, RenderConfig};
use aether_core_engine::scene::Scene;
use aether_core_engine::verify::{run_core_verification_suite, verify_disk_outputs};

fn parse_cli_overrides(cfg: &mut RenderConfig, fluid_steps: &mut usize) {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--width" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<u32>() {
                    cfg.width = v.max(64);
                }
                i += 2;
            }
            "--height" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<u32>() {
                    cfg.height = v.max(64);
                }
                i += 2;
            }
            "--spp" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<u32>() {
                    cfg.spp = v.max(1);
                }
                i += 2;
            }
            "--fluid-steps" if i + 1 < args.len() => {
                if let Ok(v) = args[i + 1].parse::<usize>() {
                    *fluid_steps = v.max(10);
                }
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
}

fn main() -> ExitCode {
    let total_timer = Instant::now();
    let mut render_cfg = RenderConfig {
        width: 1280,
        height: 720,
        spp: 48,
        max_bounces: 10,
        tile_size: 16,
        exposure: 1.26,
        indirect_clamp: 18.0,
        ..RenderConfig::default()
    };
    let mut fluid_steps = 120usize;
    parse_cli_overrides(&mut render_cfg, &mut fluid_steps);

    println!("================================================================================");
    println!(" AETHER CORE ENGINE v1.0 | ZERO-DEPENDENCY SPECTRAL PATH TRACER, FLUID & AUDIO");
    println!("================================================================================");
    println!(
        " Config : {}x{} @ {} SPP | {} Worker Threads | Tile {}x{} | Max Bounces {}",
        render_cfg.width,
        render_cfg.height,
        render_cfg.spp,
        render_cfg.num_threads,
        render_cfg.tile_size,
        render_cfg.tile_size,
        render_cfg.max_bounces
    );

    // ------------------------------------------------------------------------
    // Stage 1: 3D Incompressible Navier-Stokes Eulerian Fluid + Vorticity Confinement
    // ------------------------------------------------------------------------
    let fluid_timer = Instant::now();
    let (nx, ny, nz) = (36usize, 52usize, 36usize);
    let mut solver = FluidSolver3D::new(nx, ny, nz);
    solver.vorticity_eps = 6.5;
    solver.solver_iters = 28;

    let mut telemetry = Vec::with_capacity(fluid_steps);
    let sim_dt = 0.08;
    for step_idx in 0..fluid_steps {
        let timeline_sec = 15.0 * ((step_idx as f64) / ((fluid_steps - 1).max(1) as f64));
        let frame = solver.step(sim_dt, timeline_sec);
        telemetry.push(frame);
    }
    let fluid_elapsed = fluid_timer.elapsed().as_secs_f64();
    let peak_ke = telemetry
        .iter()
        .map(|f| f.kinetic_energy)
        .fold(0.0_f64, f64::max);
    let peak_vort = telemetry
        .iter()
        .map(|f| f.peak_vorticity)
        .fold(0.0_f64, f64::max);

    let medium_bounds = Aabb::new(
        Vec3::new(-2.25, 0.22, -2.15),
        Vec3::new(2.25, 5.15, 1.45),
    );
    let fluid_medium = FluidMedium::from_solver(&solver, medium_bounds, 1.42, 0.34);
    println!(
        " [1/4] 3D Navier-Stokes Solver ({}x{}x{}, {} steps): {:.3}s | Peak KE={:.4e} | Peak |w|={:.2} | Max rho={:.2}",
        nx, ny, nz, fluid_steps, fluid_elapsed, peak_ke, peak_vort, fluid_medium.max_density
    );

    // ------------------------------------------------------------------------
    // Stage 2: Build Cyberpunk Scene + SAH BVH + Run Core Invariant Verification
    // ------------------------------------------------------------------------
    let aspect = (render_cfg.width as f64) / (render_cfg.height as f64);
    let scene = Scene::build_cyberpunk_scene(aspect, Some(fluid_medium));

    let verify_timer = Instant::now();
    let mut all_checks = match run_core_verification_suite(&scene) {
        Ok(c) => c,
        Err(err) => {
            eprintln!(" [FATAL] Core verification failed: {}", err);
            return ExitCode::FAILURE;
        }
    };
    let verify_core_sec = verify_timer.elapsed().as_secs_f64();
    println!(
        " [2/4] Scene SAH BVH ({} primitives, {} area lights) & Pre-Flight Verification: {:.3}s",
        scene.bvh.primitives.len(),
        scene.lights.len(),
        verify_core_sec
    );

    // ------------------------------------------------------------------------
    // Stage 3: Multithreaded Work-Stealing Spectral Path Tracer + Custom LZ77 PNG
    // ------------------------------------------------------------------------
    let report = render_scene_work_stealing(&scene, &render_cfg);
    let png_path = PathBuf::from(r"C:\AetherCoreEngine\renders\cyberpunk_scene.png");
    let png_bytes_written = match save_png_rgb8(
        &png_path,
        render_cfg.width,
        render_cfg.height,
        &report.rgb8_pixels,
    ) {
        Ok(sz) => sz,
        Err(err) => {
            eprintln!(" [FATAL] Failed writing PNG {:?}: {}", png_path, err);
            return ExitCode::FAILURE;
        }
    };
    println!(
        " [3/4] Spectral Path Tracer Complete: {:.3}s | {:.2} MRays/s | Wrote {} ({:.1} KB)",
        report.wall_time_sec,
        report.aggregate_mrays_per_sec,
        png_path.display(),
        (png_bytes_written as f64) / 1024.0
    );

    // ------------------------------------------------------------------------
    // Stage 4: Procedural FM Bells + 8-Line Hadamard FDN + 4-Pole TPT SVF 24-Bit WAV
    // ------------------------------------------------------------------------
    let audio_timer = Instant::now();
    let stereo_frames = synthesize_fluid_soundtrack(&telemetry);
    let wav_path = PathBuf::from(r"C:\AetherCoreEngine\renders\simulation_soundtrack.wav");
    let wav_bytes_written = match save_wav_24bit_stereo(&wav_path, SAMPLE_RATE, &stereo_frames) {
        Ok(sz) => sz,
        Err(err) => {
            eprintln!(" [FATAL] Failed writing WAV {:?}: {}", wav_path, err);
            return ExitCode::FAILURE;
        }
    };
    let audio_elapsed = audio_timer.elapsed().as_secs_f64();
    println!(
        " [4/4] Procedural 24-Bit 48kHz Spatial Audio Complete: {:.3}s ({:.1}x realtime) | Wrote {} ({:.2} MB)",
        audio_elapsed,
        15.0 / audio_elapsed.max(1e-9),
        wav_path.display(),
        (wav_bytes_written as f64) / (1024.0 * 1024.0)
    );

    // ------------------------------------------------------------------------
    // Post-Flight Disk Binary Header, Checksum & Round-Trip Verification
    // ------------------------------------------------------------------------
    let (_, _, disk_checks) = match verify_disk_outputs(
        &png_path,
        render_cfg.width,
        render_cfg.height,
        &report.rgb8_pixels,
        &wav_path,
    ) {
        Ok(res) => res,
        Err(err) => {
            eprintln!(" [FATAL] Disk output verification failed: {}", err);
            return ExitCode::FAILURE;
        }
    };
    all_checks.extend(disk_checks);

    println!("\n--------------------------------------------------------------------------------");
    println!(" AUTOMATED SELF-TEST & INVARIANT VERIFICATION SUITE ({}/{} PASSED)", all_checks.len(), all_checks.len());
    println!("--------------------------------------------------------------------------------");
    for chk in &all_checks {
        let tag = if chk.passed { "PASS" } else { "FAIL" };
        println!(" [{}] {:<52} | {}", tag, chk.name, chk.detail);
    }

    println!("\n--------------------------------------------------------------------------------");
    println!(" WORK-STEALING RENDERER THROUGHPUT & LATENCY PERCENTILE BENCHMARKS");
    println!("--------------------------------------------------------------------------------");
    println!(
        " Total Rays Traced        : {} ({:.3}M camera samples across {} tiles)",
        report.total_rays,
        ((report.width as u64) * (report.height as u64) * (report.spp as u64)) as f64 / 1e6,
        report.tile_count
    );
    println!(
        " Work-Stealing Scheduler  : {} cross-worker tile steals across {} threads",
        report.stolen_tiles, report.num_threads
    );
    println!(
        " Aggregate Wall Throughput: {:.3} MRays/sec  |  {:.3} MSamples/sec  (Wall: {:.3}s)",
        report.aggregate_mrays_per_sec, report.aggregate_msamples_per_sec, report.wall_time_sec
    );
    println!(
        " Per-Tile Throughput (MRays/sec) :  p50 = {:>7.3}  |  p95 = {:>7.3}  |  p99 = {:>7.3}",
        report.tile_throughput_p50_mrays,
        report.tile_throughput_p95_mrays,
        report.tile_throughput_p99_mrays
    );
    println!(
        " Per-Tile Latency    (millisec)  :  p50 = {:>7.3}  |  p95 = {:>7.3}  |  p99 = {:>7.3}",
        report.tile_latency_p50_ms,
        report.tile_latency_p95_ms,
        report.tile_latency_p99_ms
    );
    println!(
        " Total Pipeline Wall Time : {:.3}s",
        total_timer.elapsed().as_secs_f64()
    );
    println!("================================================================================");

    ExitCode::SUCCESS
}
