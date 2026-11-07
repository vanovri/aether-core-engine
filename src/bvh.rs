use std::cmp::Ordering;

use crate::math::{Aabb, Ray};
use crate::primitives::{HitRecord, Primitive};

const SAH_BINS: usize = 16;
const TRAVERSAL_COST: f64 = 0.125;
const MAX_LEAF_PRIMS: usize = 4;

#[derive(Clone, Copy, Debug)]
pub struct LinearBvhNode {
    pub bounds: Aabb,
    /// For leaves (`count > 0`): first primitive index.
    /// For interior nodes (`count == 0`): right child node index (left child is always `node_idx + 1`).
    pub right_or_first: u32,
    pub count: u16,
    pub axis: u8,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BvhStats {
    pub node_count: usize,
    pub leaf_count: usize,
    pub max_depth: usize,
    pub primitive_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct SahBin {
    bounds: Aabb,
    count: usize,
}

impl Default for SahBin {
    fn default() -> Self {
        Self {
            bounds: Aabb::EMPTY,
            count: 0,
        }
    }
}

pub struct Bvh {
    pub nodes: Vec<LinearBvhNode>,
    pub primitives: Vec<Primitive>,
}

impl Bvh {
    pub fn build(mut primitives: Vec<Primitive>) -> Self {
        let n = primitives.len();
        if n == 0 {
            return Self {
                nodes: Vec::new(),
                primitives,
            };
        }
        let mut nodes = Vec::with_capacity(2 * n);
        Self::build_recursive(&mut nodes, &mut primitives, 0, n);
        Self { nodes, primitives }
    }

    fn build_recursive(
        nodes: &mut Vec<LinearBvhNode>,
        prims: &mut [Primitive],
        start: usize,
        end: usize,
    ) -> usize {
        let node_idx = nodes.len();
        nodes.push(LinearBvhNode {
            bounds: Aabb::EMPTY,
            right_or_first: 0,
            count: 0,
            axis: 0,
        });

        let mut bounds = Aabb::EMPTY;
        let mut centroid_bounds = Aabb::EMPTY;
        for p in &prims[start..end] {
            let pb = p.bounds();
            bounds = bounds.union(pb);
            centroid_bounds = centroid_bounds.union_point(pb.centroid());
        }

        let count = end - start;
        let parent_sa = bounds.surface_area();
        if count <= 2 || parent_sa <= 1e-18 {
            nodes[node_idx] = LinearBvhNode {
                bounds,
                right_or_first: start as u32,
                count: count as u16,
                axis: 0,
            };
            return node_idx;
        }

        // Binned Surface Area Heuristic across X, Y, Z axes
        let mut best_cost = f64::INFINITY;
        let mut best_axis = centroid_bounds.longest_axis();
        let mut best_split_bin = None;

        for axis in 0..3 {
            let c_min = centroid_bounds.min[axis];
            let c_max = centroid_bounds.max[axis];
            let extent = c_max - c_min;
            if extent <= 1e-12 {
                continue;
            }

            let mut bins = [SahBin::default(); SAH_BINS];
            let scale = (SAH_BINS as f64) / extent;
            for p in &prims[start..end] {
                let pb = p.bounds();
                let c = pb.centroid()[axis];
                let b = (((c - c_min) * scale) as usize).min(SAH_BINS - 1);
                bins[b].count += 1;
                bins[b].bounds = bins[b].bounds.union(pb);
            }

            let mut left_area = [0.0; SAH_BINS - 1];
            let mut left_count = [0usize; SAH_BINS - 1];
            let mut cur_box = Aabb::EMPTY;
            let mut cur_cnt = 0;
            for i in 0..(SAH_BINS - 1) {
                cur_box = cur_box.union(bins[i].bounds);
                cur_cnt += bins[i].count;
                left_area[i] = cur_box.surface_area();
                left_count[i] = cur_cnt;
            }

            let mut right_box = Aabb::EMPTY;
            let mut right_cnt = 0;
            for i in (0..(SAH_BINS - 1)).rev() {
                right_box = right_box.union(bins[i + 1].bounds);
                right_cnt += bins[i + 1].count;
                if left_count[i] > 0 && right_cnt > 0 {
                    let cost = TRAVERSAL_COST
                        + ((left_count[i] as f64) * left_area[i]
                            + (right_cnt as f64) * right_box.surface_area())
                            / parent_sa;
                    if cost < best_cost {
                        best_cost = cost;
                        best_axis = axis;
                        best_split_bin = Some(i);
                    }
                }
            }
        }

        let leaf_cost = count as f64;
        if best_cost >= leaf_cost && count <= MAX_LEAF_PRIMS {
            nodes[node_idx] = LinearBvhNode {
                bounds,
                right_or_first: start as u32,
                count: count as u16,
                axis: 0,
            };
            return node_idx;
        }

        // Partition slice `[start..end]` by winning SAH split bin, or median fallback
        let mid = if let Some(split_bin) = best_split_bin {
            let c_min = centroid_bounds.min[best_axis];
            let extent = centroid_bounds.max[best_axis] - c_min;
            let scale = (SAH_BINS as f64) / extent;
            let mut i = start;
            let mut j = end;
            while i < j {
                let c = prims[i].bounds().centroid()[best_axis];
                let b = (((c - c_min) * scale) as usize).min(SAH_BINS - 1);
                if b <= split_bin {
                    i += 1;
                } else {
                    j -= 1;
                    prims.swap(i, j);
                }
            }
            if i == start || i == end {
                let m = start + count / 2;
                prims[start..end].select_nth_unstable_by(m - start, |a, b| {
                    a.bounds().centroid()[best_axis]
                        .partial_cmp(&b.bounds().centroid()[best_axis])
                        .unwrap_or(Ordering::Equal)
                });
                m
            } else {
                i
            }
        } else {
            let m = start + count / 2;
            prims[start..end].select_nth_unstable_by(m - start, |a, b| {
                a.bounds().centroid()[best_axis]
                    .partial_cmp(&b.bounds().centroid()[best_axis])
                    .unwrap_or(Ordering::Equal)
            });
            m
        };

        // Left child is immediately at `node_idx + 1`
        Self::build_recursive(nodes, prims, start, mid);
        let right_child = Self::build_recursive(nodes, prims, mid, end);

        nodes[node_idx] = LinearBvhNode {
            bounds,
            right_or_first: right_child as u32,
            count: 0,
            axis: best_axis as u8,
        };
        node_idx
    }

    /// Fast stack-based front-to-back SAH BVH closest-hit ray traversal.
    #[inline]
    pub fn intersect(&self, ray: &Ray) -> Option<HitRecord> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut closest_t = ray.t_max;
        let mut best_hit: Option<HitRecord> = None;

        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut cur = 0u32;

        loop {
            let node = &self.nodes[cur as usize];
            if node.bounds.intersect(ray, ray.t_min, closest_t).is_some() {
                if node.count > 0 {
                    let first = node.right_or_first as usize;
                    let end = first + (node.count as usize);
                    for pid in first..end {
                        if let Some(hit) = self.primitives[pid].intersect(ray, ray.t_min, closest_t, pid) {
                            if hit.t < closest_t
                                || (hit.t == closest_t
                                    && best_hit.is_none_or(|b| pid < b.primitive_id))
                            {
                                closest_t = hit.t;
                                best_hit = Some(hit);
                            }
                        }
                    }
                    if sp == 0 {
                        break;
                    }
                    sp -= 1;
                    cur = stack[sp];
                } else {
                    let left = cur + 1;
                    let right = node.right_or_first;
                    if ray.dir[node.axis as usize] < 0.0 {
                        stack[sp] = left;
                        sp += 1;
                        cur = right;
                    } else {
                        stack[sp] = right;
                        sp += 1;
                        cur = left;
                    }
                }
            } else {
                if sp == 0 {
                    break;
                }
                sp -= 1;
                cur = stack[sp];
            }
        }
        best_hit
    }

    /// Early-exit shadow ray visibility test (`true` if any primitive blocks `[ray.t_min, max_dist]`).
    #[inline]
    pub fn occluded(&self, ray: &Ray, max_dist: f64) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut stack = [0u32; 64];
        let mut sp = 0usize;
        let mut cur = 0u32;

        loop {
            let node = &self.nodes[cur as usize];
            if node.bounds.intersect(ray, ray.t_min, max_dist).is_some() {
                if node.count > 0 {
                    let first = node.right_or_first as usize;
                    let end = first + (node.count as usize);
                    for pid in first..end {
                        if self.primitives[pid].intersect(ray, ray.t_min, max_dist, pid).is_some() {
                            return true;
                        }
                    }
                    if sp == 0 {
                        break;
                    }
                    sp -= 1;
                    cur = stack[sp];
                } else {
                    let left = cur + 1;
                    let right = node.right_or_first;
                    if ray.dir[node.axis as usize] < 0.0 {
                        stack[sp] = left;
                        sp += 1;
                        cur = right;
                    } else {
                        stack[sp] = right;
                        sp += 1;
                        cur = left;
                    }
                }
            } else {
                if sp == 0 {
                    break;
                }
                sp -= 1;
                cur = stack[sp];
            }
        }
        false
    }

    /// Reference O(N) linear scan used by automated verification tests.
    pub fn intersect_brute_force(&self, ray: &Ray) -> Option<HitRecord> {
        let mut closest_t = ray.t_max;
        let mut best_hit: Option<HitRecord> = None;
        for (pid, prim) in self.primitives.iter().enumerate() {
            if let Some(hit) = prim.intersect(ray, ray.t_min, closest_t, pid) {
                if hit.t < closest_t
                    || (hit.t == closest_t && best_hit.is_none_or(|b| pid < b.primitive_id))
                {
                    closest_t = hit.t;
                    best_hit = Some(hit);
                }
            }
        }
        best_hit
    }

    /// Verifies structural SAH BVH invariants: bounding box containment, leaf partition completeness, and tree acyclicity.
    pub fn verify_invariants(&self) -> Result<BvhStats, String> {
        if self.nodes.is_empty() {
            return Ok(BvhStats::default());
        }
        let mut covered = vec![false; self.primitives.len()];
        let mut stats = BvhStats {
            node_count: self.nodes.len(),
            leaf_count: 0,
            max_depth: 0,
            primitive_count: self.primitives.len(),
        };
        self.check_node(0, 1, &mut covered, &mut stats)?;
        for (i, &c) in covered.iter().enumerate() {
            if !c {
                return Err(format!("Primitive {} missing from BVH leaves", i));
            }
        }
        Ok(stats)
    }

    fn check_node(
        &self,
        idx: usize,
        depth: usize,
        covered: &mut [bool],
        stats: &mut BvhStats,
    ) -> Result<(), String> {
        if idx >= self.nodes.len() {
            return Err(format!("Out-of-bounds BVH node index {}", idx));
        }
        stats.max_depth = stats.max_depth.max(depth);
        let node = &self.nodes[idx];
        let eps = 1e-7;

        if node.count > 0 {
            stats.leaf_count += 1;
            let first = node.right_or_first as usize;
            let end = first + (node.count as usize);
            if end > self.primitives.len() {
                return Err(format!("Leaf primitive range {}..{} out of bounds", first, end));
            }
            for pid in first..end {
                if covered[pid] {
                    return Err(format!("Primitive {} referenced by multiple leaves", pid));
                }
                covered[pid] = true;
                let pb = self.primitives[pid].bounds();
                if pb.min.x < node.bounds.min.x - eps
                    || pb.min.y < node.bounds.min.y - eps
                    || pb.min.z < node.bounds.min.z - eps
                    || pb.max.x > node.bounds.max.x + eps
                    || pb.max.y > node.bounds.max.y + eps
                    || pb.max.z > node.bounds.max.z + eps
                {
                    return Err(format!("Leaf {} AABB fails to enclose primitive {}", idx, pid));
                }
            }
        } else {
            let left = idx + 1;
            let right = node.right_or_first as usize;
            for child in [left, right] {
                if child >= self.nodes.len() {
                    return Err(format!("Child index {} out of bounds", child));
                }
                let cb = self.nodes[child].bounds;
                if cb.min.x < node.bounds.min.x - eps
                    || cb.min.y < node.bounds.min.y - eps
                    || cb.min.z < node.bounds.min.z - eps
                    || cb.max.x > node.bounds.max.x + eps
                    || cb.max.y > node.bounds.max.y + eps
                    || cb.max.z > node.bounds.max.z + eps
                {
                    return Err(format!(
                        "Parent node {} AABB fails to enclose child {}",
                        idx, child
                    ));
                }
                self.check_node(child, depth + 1, covered, stats)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::primitives::{CsgPrimitive, SdfNode, Sphere, Triangle};
    use crate::rng::Rng;

    #[test]
    fn test_bvh_invariants_and_exact_equivalence() {
        let mut rng = Rng::from_seed(2026);
        let mut prims = Vec::new();

        // Mix of spheres, triangles, and a CSG primitive
        for i in 0..45 {
            let center = Vec3::new(
                rng.range_f64(-6.0, 6.0),
                rng.range_f64(-4.0, 4.0),
                rng.range_f64(-6.0, 6.0),
            );
            let r = rng.range_f64(0.2, 0.9);
            prims.push(Primitive::Sphere(Sphere::new(center, r, i % 4)));
        }
        for i in 0..40 {
            let p0 = Vec3::new(
                rng.range_f64(-5.0, 5.0),
                rng.range_f64(-3.0, 3.0),
                rng.range_f64(-5.0, 5.0),
            );
            let p1 = p0 + Vec3::new(rng.range_f64(-1.2, 1.2), rng.range_f64(-1.2, 1.2), rng.range_f64(-1.2, 1.2));
            let p2 = p0 + Vec3::new(rng.range_f64(-1.2, 1.2), rng.range_f64(-1.2, 1.2), rng.range_f64(-1.2, 1.2));
            if (p1 - p0).cross(p2 - p0).length() > 1e-3 {
                prims.push(Primitive::Triangle(Triangle::new(p0, p1, p2, i % 3)));
            }
        }
        prims.push(Primitive::Csg(CsgPrimitive::new(
            SdfNode::Difference(
                Box::new(SdfNode::RoundBox {
                    center: Vec3::new(0.0, 0.5, 0.0),
                    half_extents: Vec3::splat(0.8),
                    corner_radius: 0.06,
                }),
                Box::new(SdfNode::Sphere {
                    center: Vec3::new(0.0, 0.5, 0.0),
                    radius: 0.95,
                }),
            ),
            Aabb::new(Vec3::new(-1.0, -0.5, -1.0), Vec3::new(1.0, 1.5, 1.0)),
            1,
        )));

        let bvh = Bvh::build(prims);
        let stats = bvh.verify_invariants().expect("BVH invariants must hold");
        assert!(stats.node_count > 1);
        assert!(stats.max_depth < 64);

        for _ in 0..1000 {
            let orig = Vec3::new(
                rng.range_f64(-10.0, 10.0),
                rng.range_f64(-8.0, 8.0),
                rng.range_f64(-10.0, 10.0),
            );
            let (dx, dy, dz) = rng.uniform_sphere();
            let ray = Ray::new(orig, Vec3::new(dx, dy, dz), 1e-4, 200.0);
            let h_bvh = bvh.intersect(&ray);
            let h_ref = bvh.intersect_brute_force(&ray);
            match (h_bvh, h_ref) {
                (Some(a), Some(b)) => {
                    assert!(
                        (a.t - b.t).abs() < 1e-9,
                        "BVH t={} != BruteForce t={}",
                        a.t,
                        b.t
                    );
                    assert_eq!(a.primitive_id, b.primitive_id);
                }
                (None, None) => {}
                (a, b) => panic!("Mismatch between BVH {:?} and BruteForce {:?}", a.is_some(), b.is_some()),
            }
        }
    }
}
