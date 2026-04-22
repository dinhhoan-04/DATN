import json
import math
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Any, Dict, Tuple, Optional

import numpy as np
from numba import njit, prange
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────────────────────────
# ACOUSTIC CHANNEL — cached pure functions
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def _thorp_absorption_db(fc_khz: float) -> float:
    f2 = fc_khz * fc_khz
    return (
        0.11 * f2 / (1.0 + f2)
        + 44.0 * f2 / (4100.0 + f2)
        + 2.75e-4 * f2
        + 0.003
    )


@lru_cache(maxsize=256)
def _absorption_linear(fc_khz: float) -> float:
    return 10.0 ** (_thorp_absorption_db(fc_khz) / 10.0)


@lru_cache(maxsize=256)
def _total_noise(fc_khz: float, shipping_activity: float, wind_speed: float) -> float:
    f2 = fc_khz * fc_khz
    nt = 10.0 ** ((17.0 - 30.0 * math.log10(fc_khz)) / 10.0)
    ns_db = (
        40.0 + 20.0 * (shipping_activity - 0.5)
        + 26.0 * math.log10(fc_khz)
        - 60.0 * math.log10(fc_khz + 0.03)
    )
    ns = 10.0 ** (ns_db / 10.0)
    nw_db = (
        50.0 + 7.5 * math.sqrt(max(wind_speed, 0.0))
        + 20.0 * math.log10(fc_khz)
        - 40.0 * math.log10(fc_khz + 0.4)
    )
    nw = 10.0 ** (nw_db / 10.0)
    nth = 10.0 ** ((-15.0 + 20.0 * math.log10(fc_khz)) / 10.0)
    return nt + ns + nw + nth


def _get_channel_key(eparams) -> tuple:
    return (
        eparams.fc_khz,
        eparams.PSL_tr_db,
        eparams.B,
        eparams.spreading_factor,
        eparams.shipping_activity,
        eparams.wind_speed,
        eparams.A0,
        eparams.v_sound,
    )


# ── cache limits ─────────────────────────────────────────────────────────────

_RATE_CACHE_MAXSIZE = 32768
_ENERGY_CACHE_MAXSIZE = 32768

_RATE_CACHE: Dict[Tuple, Tuple] = {}
_ENERGY_CACHE: Dict[Tuple, Dict] = {}

_rate_cache_keys: list = []
_energy_cache_keys: list = []


def _get_rate_cached(distance_m: float, eparams) -> Tuple[float, float]:
    key = (round(distance_m, 4),) + _get_channel_key(eparams)
    if key in _RATE_CACHE:
        return _RATE_CACHE[key]

    d_km = max(distance_m / 1000.0, 1e-9)
    a_fc = _absorption_linear(eparams.fc_khz)
    A_df = eparams.A0 * (d_km ** eparams.spreading_factor) * (a_fc ** d_km)
    N_fc = _total_noise(eparams.fc_khz, eparams.shipping_activity, eparams.wind_speed)
    gamma = 1.0 / max(A_df * N_fc, 1e-30)
    snr_term = (10.0 ** (eparams.PSL_tr_db / 10.0)) * gamma / max(eparams.B, 1e-9)
    rate = max(eparams.B * math.log2(1.0 + snr_term), 1e-9)
    t_up = (eparams.G * eparams.L) / rate + distance_m / max(eparams.v_sound, 1e-9)
    result = (rate, t_up)

    if len(_RATE_CACHE) >= _RATE_CACHE_MAXSIZE:
        old = _rate_cache_keys.pop(0)
        _RATE_CACHE.pop(old, None)
    _RATE_CACHE[key] = result
    _rate_cache_keys.append(key)
    return result


def _get_energy_cached(distance_m: float, T_total: float, eparams) -> Dict:
    key = (round(distance_m, 4), round(T_total, 4)) + _get_channel_key(eparams) + (
        eparams.G,
        eparams.L,
        eparams.P_c,
        eparams.P_trans,
        eparams.P_idle,
    )
    if key in _ENERGY_CACHE:
        return _ENERGY_CACHE[key]

    rate, t_up = _get_rate_cached(distance_m, eparams)
    collect_duration = (eparams.G * eparams.L) / max(rate, 1e-9)
    e_collect = eparams.G * eparams.P_c * eparams.L / max(rate, 1e-9)
    e_trans = eparams.P_trans * t_up
    idle_time = max(T_total - collect_duration - t_up, 0.0)
    e_idle = idle_time * eparams.P_idle

    result = {
        "distance_to_hp": float(distance_m),
        "R_mn": float(rate),
        "collect_duration": float(collect_duration),
        "t_upload": float(t_up),
        "E_collect": float(e_collect),
        "E_trans": float(e_trans),
        "E_idle": float(e_idle),
        "E_sensor_total": float(e_collect + e_trans + e_idle),
    }

    if len(_ENERGY_CACHE) >= _ENERGY_CACHE_MAXSIZE:
        old = _energy_cache_keys.pop(0)
        _ENERGY_CACHE.pop(old, None)
    _ENERGY_CACHE[key] = result
    _energy_cache_keys.append(key)
    return result


def clear_round_energy_cache():
    _ENERGY_CACHE.clear()
    _energy_cache_keys.clear()


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def thorp_absorption_db_per_km(fc_khz: float) -> float:
    return _thorp_absorption_db(fc_khz)


def absorption_linear_per_km(fc_khz: float) -> float:
    return _absorption_linear(fc_khz)


def total_noise(fc_khz: float, shipping_activity: float, wind_speed: float) -> float:
    return _total_noise(fc_khz, shipping_activity, wind_speed)


def rate_R(distance_m, eparams):
    return _get_rate_cached(distance_m, eparams)[0]


def t_upload(distance_m, eparams):
    return _get_rate_cached(distance_m, eparams)[1]


def sensor_energy_components(distance, T_total, eparams):
    return _get_energy_cached(distance, T_total, eparams)


# ─────────────────────────────────────────────────────────────────────────────
# NUMBA JIT — time matrix
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def compute_vs(p1, p2, v_f, v_AUV):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        return v_AUV
    cosb = dz / L
    cosb = max(-1.0, min(1.0, cosb))
    sin2 = 1.0 - cosb * cosb
    sqrt_term = v_AUV * v_AUV - v_f * v_f * sin2
    if sqrt_term < 0.0:
        sqrt_term = 0.0
    v_s = v_f * cosb + math.sqrt(sqrt_term)
    return max(v_s, 1e-9)


@njit(cache=True, parallel=True)
def _build_time_matrix_njit(coords, v_f, v_AUV):
    n = coords.shape[0]
    T = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        for j in range(n):
            if i != j:
                dx = coords[j, 0] - coords[i, 0]
                dy = coords[j, 1] - coords[i, 1]
                dz = coords[j, 2] - coords[i, 2]
                L = math.sqrt(dx * dx + dy * dy + dz * dz)
                if L < 1e-12:
                    continue
                cosb = dz / L
                cosb = max(-1.0, min(1.0, cosb))
                sin2 = 1.0 - cosb * cosb
                sqrt_term = v_AUV * v_AUV - v_f * v_f * sin2
                if sqrt_term < 0.0:
                    sqrt_term = 0.0
                v_s = v_f * cosb + math.sqrt(sqrt_term)
                v_s = max(v_s, 1e-9)
                T[i, j] = L / v_s
    return T


def build_time_matrix(coords, v_f=1.2, v_AUV=3.0):
    return _build_time_matrix_njit(np.asarray(coords, dtype=np.float64), v_f, v_AUV)


# ─────────────────────────────────────────────────────────────────────────────
# NUMBA JIT — energy evaluation
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _eval_sensor_energy_njit(
    dist_arr,
    T_total,
    G, L, B,
    P_c, P_trans, P_idle,
    fc_khz, PSL_tr_db, spreading_factor,
    shipping_activity, wind_speed, A0, v_sound,
    a_fc_lin,
    N_fc,
) -> float:
    total = 0.0
    snr_psl = 10.0 ** (PSL_tr_db / 10.0)
    for k in range(dist_arr.shape[0]):
        d = dist_arr[k]
        d_km = d / 1000.0
        if d_km < 1e-9:
            d_km = 1e-9
        A_df = A0 * (d_km ** spreading_factor) * (a_fc_lin ** d_km)
        gamma = 1.0 / max(A_df * N_fc, 1e-30)
        snr_term = snr_psl * gamma / max(B, 1e-9)
        rate = max(B * math.log2(1.0 + snr_term), 1e-9)
        t_up = (G * L) / rate + d / max(v_sound, 1e-9)
        collect_dur = (G * L) / max(rate, 1e-9)
        e_collect = G * P_c * L / max(rate, 1e-9)
        e_trans = P_trans * t_up
        idle_time = T_total - collect_dur - t_up
        if idle_time < 0.0:
            idle_time = 0.0
        e_idle = idle_time * P_idle
        total += e_collect + e_trans + e_idle
    return total


@njit(cache=True)
def _eval_hover_energy_njit(upload_arr, P_hover) -> float:
    return np.sum(upload_arr) * P_hover


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTED ROUND DATA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoundPrecomputed:
    dist_arrays: list
    upload_arrays: list
    rate_arrays: list
    a_fc_lin: float
    N_fc: float


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

class Clustering:
    def __init__(self, space_size=400, r_sen=50, max_cluster_size=20, min_cluster_size=5):
        self.space_size = space_size
        self.r_sen = r_sen
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size

    def estimate_optimal_k(self, nodes, base_station=(200, 200, 400)):
        N = len(nodes)
        if N == 0:
            return 0
        base_pos = np.array(base_station, dtype=float)
        distances = np.linalg.norm(nodes - base_pos, axis=1)
        d_tobs = float(np.mean(distances)) if len(distances) > 0 else 1.0
        d_tobs = max(d_tobs, 1e-9)
        k_optimal = np.sqrt(N * self.space_size / (np.pi * d_tobs))
        k_optimal = max(2, int(np.round(k_optimal)))
        k_min = int(np.ceil(N / self.max_cluster_size))
        return max(k_optimal, k_min)

    def check_cluster_validity(self, cluster_nodes):
        size = len(cluster_nodes)
        if size < self.min_cluster_size or size > self.max_cluster_size:
            return False, 0.0, size
        if size > 1:
            distances = pdist(cluster_nodes)
            max_dist = float(np.max(distances)) if len(distances) > 0 else 0.0
            if max_dist > self.r_sen:
                return False, max_dist, size
            return True, max_dist, size
        return True, 0.0, size

    def split_invalid_cluster(self, cluster_nodes, cluster_ids):
        if len(cluster_nodes) < 2:
            return [(cluster_nodes, cluster_ids)]
        kmeans = KMeans(n_clusters=2, n_init=10, init="k-means++", random_state=42)
        labels = kmeans.fit_predict(cluster_nodes)
        sub_clusters = []
        for i in range(2):
            sub_nodes = cluster_nodes[labels == i]
            sub_ids = [cluster_ids[j] for j in range(len(cluster_ids)) if labels[j] == i]
            if len(sub_nodes) > 0:
                sub_clusters.append((sub_nodes, sub_ids))
        return sub_clusters

    def merge_small_clusters(self, clusters_data):
        def max_pairwise_dist(arr):
            if len(arr) <= 1:
                return 0.0
            return float(np.max(pdist(arr)))

        if len(clusters_data) <= 1:
            return clusters_data

        merged, smalls = [], []
        for nodes, ids in clusters_data:
            (smalls if len(nodes) < self.min_cluster_size else merged).append((nodes, ids))

        i = 0
        while i < len(smalls):
            small_nodes, small_ids = smalls[i]
            if len(small_nodes) == 0:
                smalls.pop(i)
                continue
            merged_success = False
            if merged:
                small_center = np.mean(small_nodes, axis=0)
                centers = np.array([np.mean(n, axis=0) for n, _ in merged if len(n) > 0])
                if len(centers) > 0:
                    dists = np.linalg.norm(centers - small_center, axis=1)
                    for idx in np.argsort(dists):
                        target_nodes, target_ids = merged[idx]
                        if len(target_nodes) + len(small_nodes) > self.max_cluster_size:
                            continue
                        combined = np.vstack([target_nodes, small_nodes])
                        if max_pairwise_dist(combined) <= self.r_sen:
                            merged[idx] = (combined, target_ids + small_ids)
                            merged_success = True
                            break
            if merged_success:
                smalls.pop(i)
                continue

            paired = False
            j = i + 1
            while j < len(smalls):
                other_nodes, other_ids = smalls[j]
                if len(other_nodes) == 0:
                    smalls.pop(j)
                    continue
                if len(other_nodes) + len(small_nodes) > self.max_cluster_size:
                    j += 1
                    continue
                combined = np.vstack([other_nodes, small_nodes])
                if max_pairwise_dist(combined) <= self.r_sen:
                    merged.append((combined, other_ids + small_ids))
                    smalls.pop(j)
                    smalls.pop(i)
                    paired = True
                    break
                j += 1
            if paired:
                continue
            merged.append((small_nodes, small_ids))
            smalls.pop(i)

        return [(n, ids) for n, ids in merged if len(n) > 0]

    def balance_clusters(self, clusters):
        def max_pairwise_dist(arr):
            if len(arr) <= 1:
                return 0.0
            return float(np.max(pdist(arr)))

        if len(clusters) <= 1:
            return clusters

        improved = True
        while improved:
            improved = False
            sizes = [len(nodes) for nodes, _ in clusters]
            max_idx, min_idx = int(np.argmax(sizes)), int(np.argmin(sizes))
            if sizes[max_idx] - sizes[min_idx] <= 1:
                break
            big_nodes, big_ids = clusters[max_idx]
            small_nodes, small_ids = clusters[min_idx]
            moved = False
            for i in range(len(big_nodes)):
                candidate = big_nodes[i].reshape(1, -1)
                if len(small_nodes) + 1 > self.max_cluster_size:
                    continue
                new_small = np.vstack([small_nodes, candidate])
                if max_pairwise_dist(new_small) > self.r_sen:
                    continue
                clusters[min_idx] = (new_small, small_ids + [big_ids[i]])
                clusters[max_idx] = (
                    np.delete(big_nodes, i, axis=0),
                    big_ids[:i] + big_ids[i+1:]
                )
                moved = improved = True
                break
            if not moved:
                break
        return [(n, ids) for n, ids in clusters if len(n) > 0]

    def cluster_with_constraints(self, nodes, node_ids, k=None, max_iterations=10):
        if len(nodes) == 0:
            return []
        if k is None:
            k = self.estimate_optimal_k(nodes)
        k = max(1, min(int(k), len(nodes)))
        kmeans = KMeans(n_clusters=k, n_init=10, init="k-means++", random_state=42)
        labels = kmeans.fit_predict(nodes)
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        iteration = 0

        while iteration < max_iterations:
            valid_clusters, invalid_clusters = [], []
            for i in range(k):
                cluster_nodes = nodes[labels == i]
                cluster_ids = [node_ids[j] for j in range(len(node_ids)) if labels[j] == i]
                if len(cluster_nodes) == 0:
                    continue
                is_valid, _, _ = self.check_cluster_validity(cluster_nodes)
                (valid_clusters if is_valid else invalid_clusters).append((cluster_nodes, cluster_ids))

            if not invalid_clusters:
                break

            for cn, ci in invalid_clusters:
                valid_clusters.extend(self.split_invalid_cluster(cn, ci))
            k = len(valid_clusters)
            labels = np.zeros(len(nodes), dtype=int)
            for ci, (_, c_ids) in enumerate(valid_clusters):
                for nid in c_ids:
                    labels[id_to_idx[nid]] = ci
            iteration += 1

        final = self.merge_small_clusters(valid_clusters)
        return self.balance_clusters(final)

    def compute_hovering_point(self, cluster_nodes):
        arr = np.asarray(cluster_nodes, dtype=np.float64)
        return np.mean(arr, axis=0) if len(arr) > 0 else np.zeros(3, dtype=np.float64)

    def compute_hovering_point_stats(self, cluster_nodes, hovering_point=None):
        arr = np.asarray(cluster_nodes, dtype=np.float64)
        if len(arr) == 0:
            return {"avg_distance": 0.0, "max_distance": 0.0}
        if hovering_point is None:
            hovering_point = self.compute_hovering_point(arr)
        distances = np.linalg.norm(arr - hovering_point, axis=1)
        return {
            "avg_distance": float(np.mean(distances)),
            "max_distance": float(np.max(distances)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PARAMS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NSGAIIParams:
    num_auvs: int = 3
    population_size: int = 60
    neighborhood_size: int = 10  # compatibility with old MOEA/D runner; unused in NSGA-II
    generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25
    service_times: list = None
    aoi_uploads: list = None
    aoi_bases: list = None
    sensor_distances: list = None
    sensor_rates: list = None
    sensor_uploads: list = None
    P_c: float = 0.0008
    P_trans: float = 0.0016
    P_idle: float = 0.00001
    P_hover: float = 0.01
    P_move: float = 0.02
    P_receive: float = 0.0016
    G: float = 100.0
    L: float = 1024.0
    B: float = 16000.0
    v_sound: float = 1500.0
    fc_khz: float = 26.0
    PSL_tr_db: float = 154.0
    spreading_factor: float = 1.5
    shipping_activity: float = 0.5
    wind_speed: float = 0.0
    A0: float = 1.0
    v_f: float = 1.2
    v_AUV: float = 3.0
    random_seed: int = 42


@dataclass
class EnergyParams:
    G: float = 100.0
    L: float = 1024.0
    B: float = 16000.0
    P_c: float = 0.0008
    P_trans: float = 0.0016
    P_idle: float = 0.00001
    P_hover: float = 0.01
    P_move: float = 0.02
    P_receive: float = 0.0016
    v_sound: float = 1500.0
    fc_khz: float = 26.0
    PSL_tr_db: float = 154.0
    spreading_factor: float = 1.5
    shipping_activity: float = 0.5
    wind_speed: float = 0.0
    A0: float = 1.0
    sensor_death_threshold: float = 0.5
    auv_initial_energy: float = 1000.0
    max_rounds: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# NSGA-II SOLVER
# ─────────────────────────────────────────────────────────────────────────────

class NSGAIIMultiAUV:
    def __init__(self, coords: np.ndarray, visit_ids: List[Any], params: NSGAIIParams,
                 precomputed: Optional[RoundPrecomputed] = None):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.visit_ids = list(visit_ids)
        self.params = params
        self.num_nodes = len(self.coords) - 1
        self.num_auvs = params.num_auvs
        self.rng = random.Random(params.random_seed)
        np.random.seed(params.random_seed)

        if self.num_nodes < self.num_auvs:
            raise ValueError("Not enough visiting points for all AUVs")

        self.time_matrix = build_time_matrix(self.coords, params.v_f, params.v_AUV)
        self.precomputed = precomputed

        n_coords = len(self.coords)
        if precomputed is not None:
            self._dist_arrays = precomputed.dist_arrays
            self._upload_arrays = precomputed.upload_arrays
            self._a_fc_lin = precomputed.a_fc_lin
            self._N_fc = precomputed.N_fc
        else:
            self._dist_arrays = [
                np.array(params.sensor_distances[i], dtype=np.float64)
                if params.sensor_distances and i < len(params.sensor_distances)
                else np.array([], dtype=np.float64)
                for i in range(n_coords)
            ]
            self._upload_arrays = [
                np.array(params.sensor_uploads[i], dtype=np.float64)
                if params.sensor_uploads and i < len(params.sensor_uploads)
                else np.array([], dtype=np.float64)
                for i in range(n_coords)
            ]
            self._a_fc_lin = _absorption_linear(params.fc_khz)
            self._N_fc = _total_noise(params.fc_khz, params.shipping_activity, params.wind_speed)

        if params.service_times is None:
            self.service_times = [0.0] * n_coords
        else:
            assert len(params.service_times) == n_coords
            self.service_times = [float(x) for x in params.service_times]

        self.aoi_uploads = params.aoi_uploads or [[] for _ in range(n_coords)]
        self.aoi_bases = params.aoi_bases or [[] for _ in range(n_coords)]
        self.sensor_uploads = params.sensor_uploads or [[] for _ in range(n_coords)]

    def _is_valid_perm(self, perm) -> bool:
        n = self.num_nodes
        return len(perm) == n and len(set(perm)) == n and sum(perm) == n * (n + 1) // 2

    def _repair_perm(self, perm):
        n = self.num_nodes
        seen = bytearray(n + 2)
        result = list(perm)
        slots = []
        for i, x in enumerate(result):
            if 1 <= x <= n and not seen[x]:
                seen[x] = 1
            else:
                result[i] = 0
                slots.append(i)
        mi = 0
        for x in range(1, n + 1):
            if not seen[x]:
                result[slots[mi]] = x
                mi += 1
                if mi == len(slots):
                    break
        result = result[:n]
        while len(result) < n:
            for x in range(1, n + 1):
                if x not in result:
                    result.append(x)
                    break
        if not self._is_valid_perm(result):
            raise RuntimeError(f"_repair_perm failed. Input={perm}, Output={result}")
        return result

    def _valid_random_cuts(self):
        return sorted(self.rng.sample(range(1, self.num_nodes), self.num_auvs - 1))

    def _random_solution(self):
        perm = list(range(1, self.num_nodes + 1))
        self.rng.shuffle(perm)
        return {"perm": perm, "cuts": self._valid_random_cuts()}

    def _repair(self, sol):
        n = self.num_nodes
        fixed_perm = self._repair_perm(list(sol["perm"]))
        raw_cuts = [int(c) for c in sol["cuts"]]
        valid_cuts = sorted(set(c for c in raw_cuts if 1 <= c <= n - 1))
        needed = self.num_auvs - 1 - len(valid_cuts)
        if needed > 0:
            pool = [c for c in range(1, n) if c not in set(valid_cuts)]
            valid_cuts += sorted(self.rng.sample(pool, needed))
        return {"perm": fixed_perm, "cuts": sorted(valid_cuts[:self.num_auvs - 1])}

    def _decode(self, sol):
        perm, cuts = sol["perm"], sol["cuts"]
        routes, start = [], 0
        for c in cuts:
            routes.append([0] + perm[start:c] + [0])
            start = c
        routes.append([0] + perm[start:] + [0])
        return routes

    def evaluate(self, sol):
        perm = sol["perm"]
        cuts = sol["cuts"]
        T = self.time_matrix
        S = self.service_times
        U = self.aoi_uploads
        B0 = self.aoi_bases
        N = self.num_nodes

        boundaries = [0] + cuts + [N]
        route_times, move_times, loads = [], [], []
        total_aoi = 0.0
        total_sensors = 0
        visited_hp_indices = []
        route_total_times = []

        p = self.params

        for k in range(self.num_auvs):
            s, e = boundaries[k], boundaries[k + 1]
            n = e - s
            loads.append(n)
            if n == 0:
                route_times.append(0.0)
                move_times.append(0.0)
                visited_hp_indices.append([])
                route_total_times.append(0.0)
                continue

            cluster_route = perm[s:e]
            visited_hp_indices.append(cluster_route)

            total_time = T[0, cluster_route[0]]
            move_time = T[0, cluster_route[0]]
            total_time += S[cluster_route[0]]

            for i in range(len(cluster_route) - 1):
                cur, nxt = cluster_route[i], cluster_route[i + 1]
                dt = T[cur, nxt]
                move_time += dt
                total_time += dt + S[nxt]

            last = cluster_route[-1]
            move_time += T[last, 0]
            total_time += T[last, 0]
            move_times.append(move_time)
            route_times.append(total_time)
            route_total_times.append(total_time)

            delivery_time = total_time
            prefix_move = [0.0] * len(cluster_route)
            prefix_service = [0.0] * len(cluster_route)
            prefix_move[0] = float(T[0, cluster_route[0]])
            for i in range(1, len(cluster_route)):
                prefix_move[i] = prefix_move[i-1] + float(T[cluster_route[i-1], cluster_route[i]])
                prefix_service[i] = prefix_service[i-1] + float(S[cluster_route[i-1]])

            for m, hp_idx in enumerate(cluster_route):
                upload_list = U[hp_idx] if hp_idx < len(U) else []
                base_list = B0[hp_idx] if hp_idx < len(B0) else []
                if not upload_list:
                    continue
                arrival = prefix_move[m] + prefix_service[m]
                running = 0.0
                for j, ut in enumerate(upload_list):
                    usg = arrival + running
                    running += ut
                    base_age = base_list[j] if j < len(base_list) else 0.0
                    aoi = base_age + (delivery_time - usg)
                    if aoi < 0.0:
                        aoi = 0.0
                    total_aoi += aoi
                    total_sensors += 1

        total_sensor_energy = 0.0
        total_hover_energy = 0.0

        for auv_idx, route in enumerate(visited_hp_indices):
            T_total_auv = float(route_total_times[auv_idx])
            for hp_idx in route:
                dist_arr = self._dist_arrays[hp_idx] if hp_idx < len(self._dist_arrays) else np.array([], dtype=np.float64)
                upload_arr = self._upload_arrays[hp_idx] if hp_idx < len(self._upload_arrays) else np.array([], dtype=np.float64)
                if len(dist_arr) > 0:
                    total_sensor_energy += _eval_sensor_energy_njit(
                        dist_arr, T_total_auv,
                        p.G, p.L, p.B,
                        p.P_c, p.P_trans, p.P_idle,
                        p.fc_khz, p.PSL_tr_db, p.spreading_factor,
                        p.shipping_activity, p.wind_speed, p.A0, p.v_sound,
                        self._a_fc_lin, self._N_fc,
                    )
                if len(upload_arr) > 0:
                    total_hover_energy += float(np.sum(upload_arr)) * p.P_hover

        total_move_energy = float(sum(move_times)) * p.P_move

        f1 = float(total_sensor_energy + total_hover_energy + total_move_energy)
        f2 = float(total_aoi / max(total_sensors, 1))
        f3 = float(max(loads) - min(loads)) if loads else 0.0
        non_zero = [x for x in route_times if x > 1e-9]
        f4 = float(max(non_zero) - min(non_zero)) if len(non_zero) > 1 else 0.0

        return np.array([f1, f2, f3, f4], dtype=np.float64)

    def _order_crossover(self, p1, p2):
        L = len(p1)
        if L < 2:
            return p1[:]
        a, b = sorted(self.rng.sample(range(L), 2))
        in_seg = bytearray(L + 2)
        for x in p1[a:b]:
            in_seg[x] = 1
        child = [0] * L
        child[a:b] = p1[a:b]
        fill_pos = b % L
        p2_pos = b % L
        placed = b - a
        while placed < L:
            x = p2[p2_pos]
            if not in_seg[x]:
                child[fill_pos] = x
                fill_pos = (fill_pos + 1) % L
                placed += 1
            p2_pos = (p2_pos + 1) % L
        if not self._is_valid_perm(child):
            child = self._repair_perm(child)
        return child

    def _crossover(self, s1, s2):
        if self.rng.random() > self.params.crossover_rate:
            return self._repair({"perm": s1["perm"][:], "cuts": s1["cuts"][:]})
        child_perm = self._order_crossover(s1["perm"], s2["perm"])
        child_cuts = [
            c1 if self.rng.random() < 0.5 else c2
            for c1, c2 in zip(s1["cuts"], s2["cuts"])
        ]
        return self._repair({"perm": child_perm, "cuts": child_cuts})

    def _mutate(self, sol):
        perm = sol["perm"][:]
        cuts = sol["cuts"][:]
        if self.rng.random() < self.params.mutation_rate and len(perm) >= 2:
            i, j = sorted(self.rng.sample(range(len(perm)), 2))
            perm[i], perm[j] = perm[j], perm[i]
        if self.rng.random() < self.params.mutation_rate and len(perm) >= 3:
            i, j = sorted(self.rng.sample(range(len(perm)), 2))
            perm[i:j+1] = reversed(perm[i:j+1])
        if self.rng.random() < self.params.mutation_rate and cuts:
            idx = self.rng.randrange(len(cuts))
            lo = cuts[idx-1] + 1 if idx > 0 else 1
            hi = cuts[idx+1] - 1 if idx < len(cuts)-1 else self.num_nodes - 1
            if lo <= hi:
                candidates = [c for c in range(lo, hi+1) if c != cuts[idx]]
                if candidates:
                    cuts[idx] = self.rng.choice(candidates)
        return self._repair({"perm": perm, "cuts": cuts})

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns:
            fronts: list of lists, each containing indices of solutions in that front
            rank: array of rank for each solution
        """
        pop_size = len(F)
        domination_count = np.zeros(pop_size, dtype=int)
        dominated_solutions = [[] for _ in range(pop_size)]

        # Calculate domination relationships
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    # i dominates j
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    # j dominates i
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Build fronts
        fronts = []
        rank = np.zeros(pop_size, dtype=int)
        current_front = np.where(domination_count == 0)[0].tolist()

        front_counter = 0
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                rank[i] = front_counter
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            front_counter += 1

        return fronts, rank

    def _crowding_distance(self, F, front_indices):
        """
        Calculate crowding distance for solutions in a front.
        """
        n = len(front_indices)
        if n == 0:
            return np.array([])

        m = F.shape[1]
        distance = np.zeros(n, dtype=np.float64)

        if n <= 2:
            return np.ones(n, dtype=np.float64) * np.inf

        F_front = F[front_indices]
        f_min = np.min(F_front, axis=0)
        f_max = np.max(F_front, axis=0)
        f_range = np.where((f_max - f_min) > 1e-10, (f_max - f_min), 1e-10)
        F_norm = (F_front - f_min) / f_range

        for obj in range(m):
            sorted_idx = np.argsort(F_norm[:, obj])
            distance[sorted_idx[0]] = np.inf
            distance[sorted_idx[-1]] = np.inf

            for i in range(1, n - 1):
                if not np.isinf(distance[sorted_idx[i]]):
                    distance[sorted_idx[i]] += (
                        F_norm[sorted_idx[i + 1], obj] - F_norm[sorted_idx[i - 1], obj]
                    )

        return distance

    def _extract_non_dominated_archive(self, population, F, decimals: int = 12):
        """
        Extract the first Pareto front from a population and deduplicate identical
        objective vectors to keep the archive clean for visualization/logging.
        """
        fronts, _ = self._fast_non_dominated_sort(F)
        if not fronts:
            return [], []

        archive = []
        archive_f_rows = []
        seen = set()
        for idx in fronts[0]:
            key = tuple(np.round(F[idx], decimals=decimals).tolist())
            if key in seen:
                continue
            seen.add(key)
            archive.append({"perm": population[idx]["perm"][:], "cuts": population[idx]["cuts"][:]})
            archive_f_rows.append(F[idx].copy())
        return archive, archive_f_rows

    def _tournament_selection(self, population, F, tournaments=None):
        """
        Binary tournament selection based on rank and crowding distance.
        """
        if tournaments is None:
            tournaments = len(population)

        fronts, rank = self._fast_non_dominated_sort(F)

        crowding_dist = np.zeros(len(population), dtype=np.float64)
        for front in fronts:
            cd = self._crowding_distance(F, front)
            for i, idx in enumerate(front):
                crowding_dist[idx] = cd[i]

        selected = []
        for _ in range(tournaments):
            i = self.rng.randrange(len(population))
            j = self.rng.randrange(len(population))

            if rank[i] < rank[j]:
                selected.append(i)
            elif rank[i] > rank[j]:
                selected.append(j)
            else:
                if crowding_dist[i] >= crowding_dist[j]:
                    selected.append(i)
                else:
                    selected.append(j)

        return selected

    def solve(self):
        """
        NSGA-II main algorithm.
        - Elitist survival on parent + offspring population
        - Archive is taken from the current first non-dominated front
        """
        pop_size = self.params.population_size
        generations = self.params.generations

        population = [self._random_solution() for _ in range(pop_size)]
        F = np.array([self.evaluate(sol) for sol in population], dtype=np.float64)

        archive, archive_f_rows = self._extract_non_dominated_archive(population, F)
        history = []

        for gen in range(generations):
            selected_indices = self._tournament_selection(population, F, tournaments=pop_size)

            offspring = []
            offspring_F = []
            for i in range(0, pop_size, 2):
                parent1_idx = selected_indices[i % len(selected_indices)]
                parent2_idx = selected_indices[(i + 1) % len(selected_indices)]

                child1 = self._mutate(self._crossover(population[parent1_idx], population[parent2_idx]))
                offspring.append(child1)
                offspring_F.append(self.evaluate(child1))

                if i + 1 < pop_size:
                    child2 = self._mutate(self._crossover(population[parent2_idx], population[parent1_idx]))
                    offspring.append(child2)
                    offspring_F.append(self.evaluate(child2))

            offspring_F = np.array(offspring_F, dtype=np.float64)
            combined_population = population + offspring
            combined_F = np.vstack([F, offspring_F])

            fronts, _ = self._fast_non_dominated_sort(combined_F)

            new_population = []
            new_F_rows = []
            for front in fronts:
                if len(new_population) + len(front) <= pop_size:
                    for idx in front:
                        new_population.append(combined_population[idx])
                        new_F_rows.append(combined_F[idx].copy())
                else:
                    remaining = pop_size - len(new_population)
                    if remaining > 0:
                        cd = self._crowding_distance(combined_F, front)
                        sorted_by_cd = np.argsort(-cd)[:remaining]
                        for local_i in sorted_by_cd:
                            idx = front[local_i]
                            new_population.append(combined_population[idx])
                            new_F_rows.append(combined_F[idx].copy())
                    break

            population = new_population
            F = np.array(new_F_rows, dtype=np.float64)
            archive, archive_f_rows = self._extract_non_dominated_archive(population, F)

            history.append({
                "generation": gen,
                "archive_size": len(archive),
                "front0_size": len(archive),
                "best_total_network_energy": float(np.min(F[:, 0])),
                "best_average_aoi": float(np.min(F[:, 1])),
                "best_load_imbalance": float(np.min(F[:, 2])),
                "best_round_time_imbalance": float(np.min(F[:, 3])),
            })

        archive_f_list = [row.copy() for row in archive_f_rows]
        return archive, archive_f_list, history

    def map_solution(self, sol):
        routes = self._decode(sol)
        return [["Depot" if idx == 0 else self.visit_ids[idx] for idx in r] for r in routes]

    def map_solution_indices(self, sol):
        return self._decode(sol)


# Alias for backward compatibility with existing runner names
MOEADParams = NSGAIIParams
MOEADMultiAUV = NSGAIIMultiAUV


# ─────────────────────────────────────────────────────────────────────────────
# PRECOMPUTE ROUND DATA
# ─────────────────────────────────────────────────────────────────────────────

def build_round_precomputed(visit_ids, upload_schedules, eparams) -> RoundPrecomputed:
    n = len(visit_ids)
    dist_arrays = [np.array([], dtype=np.float64)] * n
    upload_arrays = [np.array([], dtype=np.float64)] * n

    for idx in range(1, n):
        cid = visit_ids[idx]
        schedule = upload_schedules.get(cid, [])
        dist_arrays[idx] = np.array([float(item["distance_to_hp"]) for item in schedule], dtype=np.float64)
        upload_arrays[idx] = np.array([float(item["t_upload"]) for item in schedule], dtype=np.float64)

    a_fc_lin = _absorption_linear(eparams.fc_khz)
    N_fc = _total_noise(eparams.fc_khz, eparams.shipping_activity, eparams.wind_speed)

    return RoundPrecomputed(
        dist_arrays=dist_arrays,
        upload_arrays=upload_arrays,
        rate_arrays=[np.array([], dtype=np.float64)] * n,
        a_fc_lin=a_fc_lin,
        N_fc=N_fc,
    )


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def noise_turbulence(fc_khz):
    return 10.0 ** ((17.0 - 30.0 * math.log10(fc_khz)) / 10.0)


def noise_shipping(fc_khz, shipping_activity):
    val_db = (
        40.0 + 20.0 * (shipping_activity - 0.5)
        + 26.0 * math.log10(fc_khz)
        - 60.0 * math.log10(fc_khz + 0.03)
    )
    return 10.0 ** (val_db / 10.0)


def noise_waves(fc_khz, wind_speed):
    val_db = (
        50.0 + 7.5 * math.sqrt(max(wind_speed, 0.0))
        + 20.0 * math.log10(fc_khz)
        - 40.0 * math.log10(fc_khz + 0.4)
    )
    return 10.0 ** (val_db / 10.0)


def noise_thermal(fc_khz):
    return 10.0 ** ((-15.0 + 20.0 * math.log10(fc_khz)) / 10.0)


def path_loss_linear(distance_m, fc_khz, spreading_factor, A0=1.0):
    d_km = max(distance_m / 1000.0, 1e-9)
    a_fc = _absorption_linear(fc_khz)
    return A0 * (d_km ** spreading_factor) * (a_fc ** d_km)


def normalized_snr(distance_m, fc_khz, spreading_factor, shipping_activity, wind_speed, A0=1.0):
    A_df = path_loss_linear(distance_m, fc_khz, spreading_factor, A0=A0)
    N_fc = _total_noise(fc_khz, shipping_activity, wind_speed)
    return 1.0 / max(A_df * N_fc, 1e-30)


def load_nodes_from_json(input_path, initial_energy=100.0, sensor_death_threshold=0.5):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    node_positions, nodes = {}, {}
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list of nodes")
    for node in data:
        nid = node["id"]
        init_e = float(node.get("initial_energy", initial_energy))
        res_e = float(node.get("residual_energy", init_e))
        nodes[nid] = {
            "initial_energy": init_e,
            "residual_energy": res_e,
            "alive": res_e >= sensor_death_threshold,
        }
        node_positions[nid] = (
            float(node["x"]),
            float(node["y"]),
            float(node["z"]),
        )
    return nodes, node_positions


def initialize_sensor_aoi_state(node_data):
    return {
        nid: {
            "last_upload_finish_time": 0.0,
            "next_batch_ready_time": 0.0,
            "last_delivery_time": 0.0,
            "current_aoi": 0.0,
            "delivered_rounds": 0,
            "alive": bool(info.get("alive", True)),
        }
        for nid, info in node_data.items()
    }


def alive_node_ids(node_data, eparams):
    return sorted([
        nid for nid, nd in node_data.items()
        if nd.get("alive", True) and nd["residual_energy"] >= eparams.sensor_death_threshold
    ])


def auv_move_energy_from_routes(routes_idx, time_matrix, eparams):
    total_move_time = sum(
        float(time_matrix[route[i-1], route[i]])
        for route in routes_idx
        for i in range(1, len(route))
    )
    return {"move_time": float(total_move_time), "E_move": float(total_move_time * eparams.P_move)}


def build_sensor_upload_schedule_for_cluster(cluster_node_ids, hp, node_positions, eparams):
    schedule = []
    for nid in cluster_node_ids:
        pos = np.array(node_positions[nid], dtype=np.float64)
        d = float(np.linalg.norm(pos - hp))
        rate, tu = _get_rate_cached(d, eparams)
        schedule.append({
            "sensor_id": nid,
            "distance_to_hp": d,
            "t_upload": tu,
            "R_mn": rate,
        })
    schedule.sort(key=lambda x: (-x["t_upload"], str(x["sensor_id"])))
    current = 0.0
    for item in schedule:
        item["upload_start_at_hp"] = float(current)
        current += item["t_upload"]
        item["upload_finish_at_hp"] = float(current)
    return schedule


def build_all_cluster_upload_schedules(active_clusters, node_positions, eparams):
    return {
        cid: build_sensor_upload_schedule_for_cluster(
            cinfo["nodes"],
            np.array(cinfo["hovering_point"], dtype=np.float64),
            node_positions,
            eparams,
        )
        for cid, cinfo in active_clusters.items()
    }


def auv_hover_and_receive_energy(active_clusters, upload_schedules, eparams):
    total_hover_time = 0.0
    cluster_details = {}
    for cid, cinfo in active_clusters.items():
        schedule = upload_schedules.get(cid, [])
        hover_time = float(sum(item["t_upload"] for item in schedule))
        cluster_details[cid] = {
            "hover_time": hover_time,
            "E_hover": hover_time * eparams.P_hover,
            "E_receive": hover_time * eparams.P_receive,
            "upload_order": [item["sensor_id"] for item in schedule],
        }
        total_hover_time += hover_time
    return {
        "total_hover_time": float(total_hover_time),
        "E_hover_total": float(total_hover_time * eparams.P_hover),
        "E_receive_total": float(total_hover_time * eparams.P_receive),
        "clusters": cluster_details,
    }


def build_fixed_clusters(node_data, node_positions, clustering):
    ids = sorted(node_positions.keys())
    if not ids:
        return {}
    coords = np.array([node_positions[nid] for nid in ids], dtype=np.float64)
    clusters_data = clustering.cluster_with_constraints(coords, ids)
    clusters = {}
    for i, (cluster_coords, cluster_ids) in enumerate(clusters_data):
        hp = clustering.compute_hovering_point(cluster_coords)
        stats = clustering.compute_hovering_point_stats(cluster_coords, hp)
        clusters[i] = {
            "all_nodes": list(cluster_ids),
            "hovering_point": hp.tolist(),
            "hovering_point_avg_distance": stats["avg_distance"],
            "hovering_point_max_distance": stats["max_distance"],
        }
    return clusters


def get_active_clusters(fixed_clusters, node_data, eparams):
    active = {}
    for cid, cinfo in fixed_clusters.items():
        alive = [
            nid for nid in cinfo["all_nodes"]
            if node_data[nid].get("alive", True)
            and node_data[nid]["residual_energy"] >= eparams.sensor_death_threshold
        ]
        if not alive:
            continue
        ch = max(alive, key=lambda nid: node_data[nid]["residual_energy"])
        active[cid] = {
            "nodes": alive,
            "all_nodes": list(cinfo["all_nodes"]),
            "hovering_point": list(cinfo["hovering_point"]),
            "hovering_point_avg_distance": cinfo["hovering_point_avg_distance"],
            "hovering_point_max_distance": cinfo["hovering_point_max_distance"],
            "cluster_head": ch,
        }
    return active


def log_clustering(clusters, node_data, round_idx):
    SEP, W_CUM, W_CH, W_ND, W_EN = "─", 6, 16, 8, 30
    total = sum(len(v["nodes"]) for v in clusters.values())
    print(f"  [Round {round_idx}] ACTIVE CLUSTERS — {len(clusters)} clusters ({total} alive sensors)")
    div = "├" + SEP*(W_CUM+2) + "┼" + SEP*(W_CH+2) + "┼" + SEP*(W_ND+2) + "┼" + SEP*(W_EN+2) + "┤"
    print("┌" + SEP*(W_CUM+2) + "┬" + SEP*(W_CH+2) + "┬" + SEP*(W_ND+2) + "┬" + SEP*(W_EN+2) + "┐")
    print(f"│ {'Cluster':^{W_CUM}} │ {'Head':^{W_CH}} │ {'#Alive':^{W_ND}} │ {'Energy (J)':^{W_EN}} │")
    print(div)
    repr_energies = {v["cluster_head"]: node_data[v["cluster_head"]]["residual_energy"] for v in clusters.values()}
    min_e = min(repr_energies.values()) if repr_energies else float("inf")
    for cid in sorted(clusters.keys()):
        cinfo = clusters[cid]
        ch = cinfo["cluster_head"]
        ch_e = node_data[ch]["residual_energy"]
        flag = "  ← lowest" if abs(ch_e - min_e) < 1e-9 and len(clusters) > 1 else ""
        print(f"│ {cid:^{W_CUM}} │ {str(ch):^{W_CH}} │ {len(cinfo['nodes']):^{W_ND}} │ {ch_e:8.3f}{flag:<{W_EN-8}} │")
    print("└" + SEP*(W_CUM+2) + "┴" + SEP*(W_CH+2) + "┴" + SEP*(W_ND+2) + "┴" + SEP*(W_EN+2) + "┘\n")


# ── THAY ĐỔI 1: thêm 2 tham số optional auv_energy_state và per_auv_energy ──
def log_round_summary(round_idx, alive_before, alive_after, dead_this_round, objectives,
                      auv_energy_state=None, per_auv_energy=None):
    bar = "═" * 62
    f1, f2, f3, f4 = objectives
    print(bar)
    print(f"  [Round {round_idx}] RESULT")
    print(
        f"    Alive sensors : {alive_before} → {alive_after}"
        f"  (dead: {len(dead_this_round)}"
        + (f"  ids: {sorted(dead_this_round)}" if dead_this_round else "") + ")"
    )
    print(f"    Net energy    : {f1:.3f} J")
    print(f"    Avg AoI       : {f2:.3f} s")
    print(f"    Load imbalance: {int(f3)} sensors")
    print(f"    Time imbalance: {f4:.3f} s")

    # ── in năng lượng còn lại từng AUV ──────────────────────────────
    if auv_energy_state:
        print(f"    AUV residual energy after round {round_idx}:")
        for auv_id in sorted(auv_energy_state.keys()):
            info = auv_energy_state[auv_id]
            e_used = per_auv_energy[auv_id]["E_total"] if per_auv_energy and auv_id in per_auv_energy else None
            used_str = f"  used: {e_used:.4f} J" if e_used is not None else ""
            alive_str = "alive" if info["alive"] else "DEAD"
            print(f"      {auv_id}: residual={info['residual_energy']:.4f} J{used_str}  [{alive_str}]")

    print(bar + "\n")


def build_center_coords_from_hp(active_clusters, bs=(200, 200, 400)):
    centers = [tuple(bs)]
    visit_ids = [None]
    for cid in sorted(active_clusters.keys()):
        centers.append(tuple(active_clusters[cid]["hovering_point"]))
        visit_ids.append(cid)
    return np.array(centers, dtype=np.float64), visit_ids


def pick_representative_solution(archive_f):
    arr = np.array([f for f in archive_f])
    f_min, f_max = np.min(arr, axis=0), np.max(arr, axis=0)
    denom = np.maximum(f_max - f_min, 1e-9)
    scores = np.sum((arr - f_min) / denom, axis=1)
    return int(np.argmin(scores))


def compute_cluster_service_times(active_clusters, node_positions, eparams):
    upload_schedules = build_all_cluster_upload_schedules(active_clusters, node_positions, eparams)
    service_times = {
        cid: float(sum(item["t_upload"] for item in sched))
        for cid, sched in upload_schedules.items()
    }
    return service_times, upload_schedules


def build_service_times_vector(visit_ids, service_times_by_cluster):
    return [0.0] + [float(service_times_by_cluster.get(visit_ids[i], 0.0)) for i in range(1, len(visit_ids))]


def build_aoi_uploads_vector(visit_ids, upload_schedules):
    return [[]] + [[float(item["t_upload"]) for item in upload_schedules.get(visit_ids[i], [])]
                   for i in range(1, len(visit_ids))]


def build_sensor_distances_vector(visit_ids, upload_schedules):
    return [[]] + [[float(item["distance_to_hp"]) for item in upload_schedules.get(visit_ids[i], [])]
                   for i in range(1, len(visit_ids))]


def build_sensor_rates_vector(visit_ids, upload_schedules):
    return [[]] + [[float(item.get("R_mn", 0.0)) for item in upload_schedules.get(visit_ids[i], [])]
                   for i in range(1, len(visit_ids))]


def build_sensor_uploads_vector(visit_ids, upload_schedules):
    return [[]] + [[float(item["t_upload"]) for item in upload_schedules.get(visit_ids[i], [])]
                   for i in range(1, len(visit_ids))]


def build_aoi_bases_vector(visit_ids, upload_schedules, sensor_aoi_state):
    result = [[]]
    for i in range(1, len(visit_ids)):
        cid = visit_ids[i]
        schedule = upload_schedules.get(cid, [])
        arr = []
        for item in schedule:
            nid = item["sensor_id"]
            upload_start = float(item.get("upload_start_global", 0.0))
            ready_time = float(sensor_aoi_state.get(nid, {}).get("next_batch_ready_time", 0.0))
            arr.append(max(upload_start - ready_time, 0.0))
        result.append(arr)
    return result


def compute_cluster_arrival_and_finish_times(routes_idx, time_matrix, visit_ids, service_times_by_cluster):
    arrival, finish = {}, {}
    for route in routes_idx:
        t = 0.0
        for i in range(1, len(route) - 1):
            prev_idx, cur_idx = route[i-1], route[i]
            cid = visit_ids[cur_idx]
            t += float(time_matrix[prev_idx, cur_idx])
            arrival[cid] = float(t)
            t += float(service_times_by_cluster.get(cid, 0.0))
            finish[cid] = float(t)
    return arrival, finish


def initialize_auv_energy(num_auvs, eparams):
    return {
        f"AUV_{i+1}": {
            "initial_energy": float(eparams.auv_initial_energy),
            "residual_energy": float(eparams.auv_initial_energy),
            "alive": True,
        }
        for i in range(num_auvs)
    }


def compute_per_auv_energy_from_routes(routes_idx, time_matrix, service_times_vector, eparams):
    per_auv = {}
    for auv_idx, route in enumerate(routes_idx):
        move_time = sum(float(time_matrix[route[i-1], route[i]]) for i in range(1, len(route)))
        hover_time = sum(float(service_times_vector[idx]) for idx in route if idx != 0)
        e_move = float(move_time * eparams.P_move)
        e_hover = float(hover_time * eparams.P_hover)
        per_auv[f"AUV_{auv_idx+1}"] = {
            "move_time": float(move_time),
            "hover_time": float(hover_time),
            "E_move": e_move,
            "E_hover": e_hover,
            "E_receive": 0.0,
            "E_total": float(e_move + e_hover),
        }
    return per_auv


def update_auv_energy(auv_energy_state, per_auv_energy):
    dead = []
    for auv_id, usage in per_auv_energy.items():
        if auv_id not in auv_energy_state or not auv_energy_state[auv_id].get("alive", True):
            continue
        consumption = float(
            usage.get("E_move", 0.0) +
            usage.get("E_hover", 0.0) +
            usage.get("E_receive", 0.0)
        )
        auv_energy_state[auv_id]["residual_energy"] = max(
            0.0,
            auv_energy_state[auv_id]["residual_energy"] - consumption
        )
        if auv_energy_state[auv_id]["residual_energy"] <= 0.0:
            auv_energy_state[auv_id]["alive"] = False
            dead.append(auv_id)
    return auv_energy_state, dead


def enrich_upload_schedules_with_global_times(routes_idx, visit_ids, time_matrix, upload_schedules):
    schedules = {
        cid: {
            "arrival_at_hp": float(info.get("arrival_at_hp", 0.0)),
            "finish_collect_at_hp": float(info.get("finish_collect_at_hp", 0.0)),
            "schedule": [dict(x) for x in info.get("schedule", [])],
            "upload_order": list(info.get("upload_order", [])),
            "cluster_collect_time": float(info.get("cluster_collect_time", 0.0)),
            "cluster_hover_time": float(info.get("cluster_hover_time", 0.0)),
        }
        for cid, info in upload_schedules.items()
    }
    for route in routes_idx:
        t = 0.0
        for i in range(1, len(route) - 1):
            prev_idx, cur_idx = route[i-1], route[i]
            cid = visit_ids[cur_idx]
            t += float(time_matrix[prev_idx, cur_idx])
            info = schedules.get(cid)
            if info is None:
                continue
            info["arrival_at_hp"] = float(t)
            current = t
            for item in info["schedule"]:
                item["upload_start_global"] = float(current)
                current += float(item["t_upload"])
                item["upload_finish_global"] = float(current)
            info["finish_collect_at_hp"] = float(current)
            t = current
    return schedules


def update_next_batch_ready_times(sensor_aoi_state, upload_schedules, eparams):
    for cid, info in upload_schedules.items():
        for item in info.get("schedule", []):
            nid = item["sensor_id"]
            rate, _ = _get_rate_cached(float(item["distance_to_hp"]), eparams)
            upload_finish = float(item.get("upload_finish_global", 0.0))
            generation_time = (eparams.G * eparams.L) / max(rate, 1e-9)
            state = sensor_aoi_state.setdefault(nid, {
                "last_upload_finish_time": 0.0,
                "next_batch_ready_time": 0.0,
                "last_delivery_time": 0.0,
                "current_aoi": 0.0,
                "delivered_rounds": 0,
                "alive": True,
            })
            state["last_upload_finish_time"] = upload_finish
            state["next_batch_ready_time"] = upload_finish + generation_time
    return sensor_aoi_state


def compute_aoi_metrics(routes_idx, visit_ids, time_matrix, upload_schedules, sensor_aoi_state):
    per_sensor_aoi, per_cluster_aoi = {}, {}
    total_aoi, total_sensors = 0.0, 0
    updated_state = {
        nid: {
            k: (float(v) if k not in ("delivered_rounds", "alive") else
                (int(v) if k == "delivered_rounds" else bool(v)))
            for k, v in info.items()
        }
        for nid, info in sensor_aoi_state.items()
    }

    for auv_idx, route in enumerate(routes_idx):
        cluster_route = [n for n in route if n != 0]
        if not cluster_route:
            continue
        t = 0.0
        for i in range(1, len(route) - 1):
            prev_idx, cur_idx = route[i-1], route[i]
            cid = visit_ids[cur_idx]
            t += float(time_matrix[prev_idx, cur_idx])
            schedule = upload_schedules.get(cid, {}).get("schedule", [])
            t += float(sum(item["t_upload"] for item in schedule))
        if len(route) >= 2:
            t += float(time_matrix[route[-2], route[-1]])
        delivery_time = float(t)

        for hp_idx in cluster_route:
            cid = visit_ids[hp_idx]
            schedule = upload_schedules.get(cid, {}).get("schedule", [])
            cluster_sum = 0.0
            cluster_items = []

            for j, item in enumerate(schedule):
                sensor_id = item["sensor_id"]
                ready_time = float(sensor_aoi_state.get(sensor_id, {}).get("next_batch_ready_time", 0.0))
                usg = float(item.get("upload_start_global", 0.0))
                base_age = max(usg - ready_time, 0.0)
                aoi_sensor = max(base_age + (delivery_time - usg), 0.0)

                cluster_items.append({
                    "sensor_id": sensor_id,
                    "upload_rank": j + 1,
                    "batch_ready_time": ready_time,
                    "upload_start_global": usg,
                    "upload_finish_global": float(item.get("upload_finish_global", 0.0)),
                    "delivery_time_at_depot": delivery_time,
                    "base_age_at_upload_start": float(base_age),
                    "t_upload": float(item["t_upload"]),
                    "AoI": float(aoi_sensor),
                })

                per_sensor_aoi[sensor_id] = {
                    "cluster_id": cid,
                    "auv_id": f"AUV_{auv_idx+1}",
                    "upload_rank": j + 1,
                    "batch_ready_time": ready_time,
                    "upload_start_global": usg,
                    "upload_finish_global": float(item.get("upload_finish_global", 0.0)),
                    "delivery_time_at_depot": delivery_time,
                    "base_age_at_upload_start": float(base_age),
                    "t_upload": float(item["t_upload"]),
                    "AoI": float(aoi_sensor),
                }

                updated_state.setdefault(sensor_id, {
                    "last_upload_finish_time": 0.0,
                    "next_batch_ready_time": 0.0,
                    "last_delivery_time": 0.0,
                    "current_aoi": 0.0,
                    "delivered_rounds": 0,
                    "alive": True,
                })
                updated_state[sensor_id]["last_delivery_time"] = delivery_time
                updated_state[sensor_id]["current_aoi"] = float(aoi_sensor)
                updated_state[sensor_id]["delivered_rounds"] += 1

                cluster_sum += aoi_sensor
                total_aoi += aoi_sensor
                total_sensors += 1

            per_cluster_aoi[cid] = {
                "sum_AoI": float(cluster_sum),
                "num_sensors": len(schedule),
                "average_AoI": float(cluster_sum / max(len(schedule), 1)),
                "sensors": cluster_items,
            }

    return {
        "sum_AoI": float(total_aoi),
        "average_AoI": float(total_aoi / max(total_sensors, 1)),
        "num_sensors": int(total_sensors),
        "per_sensor": per_sensor_aoi,
        "per_cluster": per_cluster_aoi,
        "updated_sensor_aoi_state": updated_state,
    }


def update_energy_after_round(
    active_clusters, node_data, node_positions, routes_idx, visit_ids,
    time_matrix, eparams, T_total, service_times_by_cluster, upload_schedules,
):
    round_energy = {
        "sensors": {},
        "auv": {},
        "network": {},
        "upload_schedules": {},
        "cluster_arrival_times": {},
    }
    dead_nodes = []

    cluster_arrival_times, cluster_finish_times = compute_cluster_arrival_and_finish_times(
        routes_idx, time_matrix, visit_ids, service_times_by_cluster
    )

    total_sensor_energy = 0.0

    for cid, cinfo in active_clusters.items():
        hp = np.array(cinfo["hovering_point"], dtype=np.float64)
        schedule = upload_schedules.get(cid, [])
        cluster_sensor_energy = {}
        arrival_at_hp = float(cluster_arrival_times.get(cid, 0.0))

        for rank, item in enumerate(schedule, start=1):
            nid = item["sensor_id"]
            if not node_data[nid].get("alive", True):
                continue
            pos = np.array(node_positions[nid], dtype=np.float64)
            d = float(np.linalg.norm(pos - hp))
            comps = dict(_get_energy_cached(d, T_total, eparams))
            comps["cluster_arrival_time"] = arrival_at_hp
            comps["upload_start_global"] = float(arrival_at_hp + item["upload_start_at_hp"])
            comps["upload_finish_global"] = float(arrival_at_hp + item["upload_finish_at_hp"])
            comps["priority_rank"] = rank
            node_data[nid]["residual_energy"] -= comps["E_sensor_total"]
            cluster_sensor_energy[nid] = comps
            total_sensor_energy += comps["E_sensor_total"]

        round_energy["sensors"][cid] = cluster_sensor_energy
        for item in schedule:
            nid = item["sensor_id"]
            if nid in cluster_sensor_energy:
                item["R_mn"] = float(cluster_sensor_energy[nid]["R_mn"])

        round_energy["upload_schedules"][cid] = {
            "arrival_at_hp": arrival_at_hp,
            "finish_collect_at_hp": float(cluster_finish_times.get(cid, arrival_at_hp)),
            "schedule": schedule,
            "upload_order": [x["sensor_id"] for x in schedule],
            "cluster_collect_time": float(service_times_by_cluster.get(cid, 0.0)),
            "cluster_hover_time": float(service_times_by_cluster.get(cid, 0.0)),
        }

    move_info = auv_move_energy_from_routes(routes_idx, time_matrix, eparams)
    hover_info = auv_hover_and_receive_energy(active_clusters, upload_schedules, eparams)
    E_auv_total = move_info["E_move"] + hover_info["E_hover_total"] + hover_info["E_receive_total"]

    round_energy["cluster_arrival_times"] = cluster_arrival_times
    round_energy["cluster_finish_times"] = cluster_finish_times
    round_energy["auv"] = {
        "move_time": move_info["move_time"],
        "E_move": move_info["E_move"],
        "hover_time": hover_info["total_hover_time"],
        "E_hover": hover_info["E_hover_total"],
        "E_receive": hover_info["E_receive_total"],
        "E_AUV_total": float(E_auv_total),
        "cluster_hover_details": hover_info["clusters"],
    }

    T_collect = float(sum(service_times_by_cluster.values()))
    T_move = float(move_info["move_time"])
    round_energy["network"] = {
        "E_sensors_total": float(total_sensor_energy),
        "E_AUV_total": float(E_auv_total),
        "E_total": float(total_sensor_energy + E_auv_total),
        "T_collect": T_collect,
        "T_move": T_move,
        "T_total": float(T_total),
    }

    for nid, nd in node_data.items():
        if not nd.get("alive", True):
            continue
        if nd["residual_energy"] < eparams.sensor_death_threshold:
            nd["alive"] = False
            dead_nodes.append(nid)

    return round_energy, dead_nodes


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATION PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────

def build_round_visualization_payload(
    round_idx,
    total_nodes,
    alive_before_ids,
    node_data,
    active_clusters,
    visit_ids,
    representative_routes_idx,
    archive_f,
    current_objectives,
    round_energy,
    aoi_metrics,
    auv_energy_state,
    base_station=(200, 200, 400),
):
    alive_after_ids = sorted([nid for nid, nd in node_data.items() if nd.get("alive", True)])
    dead_after_ids = sorted([nid for nid, nd in node_data.items() if not nd.get("alive", True)])

    sensor_residuals = {
        str(nid): float(node_data[nid]["residual_energy"])
        for nid in sorted(node_data.keys(), key=lambda x: str(x))
    }

    sensor_aoi = {
        str(nid): float(info["AoI"])
        for nid, info in aoi_metrics.get("per_sensor", {}).items()
    }

    route_3d = []
    for auv_idx, route in enumerate(representative_routes_idx):
        coords = []
        for idx in route:
            if idx == 0:
                coords.append({
                    "type": "depot",
                    "id": "Depot",
                    "coord": [float(base_station[0]), float(base_station[1]), float(base_station[2])],
                })
            else:
                cid = visit_ids[idx]
                hp = active_clusters[cid]["hovering_point"]
                coords.append({
                    "type": "hp",
                    "id": int(cid) if isinstance(cid, (int, np.integer)) else str(cid),
                    "coord": [float(hp[0]), float(hp[1]), float(hp[2])],
                })
        route_3d.append({
            "auv_id": f"AUV_{auv_idx+1}",
            "path": coords,
        })

    hp_positions = {
        str(cid): [float(x) for x in cinfo["hovering_point"]]
        for cid, cinfo in active_clusters.items()
    }

    cluster_nodes = {
        str(cid): [str(nid) for nid in cinfo["nodes"]]
        for cid, cinfo in active_clusters.items()
    }

    pareto_front = [
        {
            "f1_total_energy": float(f[0]),
            "f2_average_aoi": float(f[1]),
            "f3_load_imbalance": float(f[2]),
            "f4_time_imbalance": float(f[3]),
        }
        for f in archive_f
    ]

    return {
        "round": int(round_idx),
        "survival": {
            "total_nodes": int(total_nodes),
            "alive_before_count": int(len(alive_before_ids)),
            "alive_after_count": int(len(alive_after_ids)),
            "dead_after_count": int(len(dead_after_ids)),
            "alive_before_ids": [str(x) for x in alive_before_ids],
            "alive_after_ids": [str(x) for x in alive_after_ids],
            "dead_after_ids": [str(x) for x in dead_after_ids],
        },
        "energy": {
            "objective_f1_total_energy": float(current_objectives[0]),
            "sensor_total": float(round_energy["network"]["E_sensors_total"]),
            "auv_total": float(round_energy["network"]["E_AUV_total"]),
            "total": float(round_energy["network"]["E_total"]),
            "auv_move": float(round_energy["auv"]["E_move"]),
            "auv_hover": float(round_energy["auv"]["E_hover"]),
            "auv_receive": float(round_energy["auv"]["E_receive"]),
            "t_move": float(round_energy["network"]["T_move"]),
            "t_collect": float(round_energy["network"]["T_collect"]),
            "t_total": float(round_energy["network"]["T_total"]),
            "sensor_residual_after_round": sensor_residuals,
            "auv_residual_after_round": {
                auv_id: float(info["residual_energy"])
                for auv_id, info in auv_energy_state.items()
            },
        },
        "aoi": {
            "objective_f2_average_aoi": float(current_objectives[1]),
            "sum_aoi": float(aoi_metrics["sum_AoI"]),
            "average_aoi": float(aoi_metrics["average_AoI"]),
            "num_sensors": int(aoi_metrics["num_sensors"]),
            "per_sensor_aoi": sensor_aoi,
        },
        "pareto_front": pareto_front,
        "route_3d": {
            "base_station": [float(base_station[0]), float(base_station[1]), float(base_station[2])],
            "hp_positions": hp_positions,
            "cluster_nodes": cluster_nodes,
            "routes": route_3d,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_fixed_cluster_hp_moead(
    input_path,
    output_path=None,
    num_auvs=3,
    space_size=400,
    r_sen=60,
    max_cluster_size=25,
    min_cluster_size=10,
    v_f=1.2,
    v_AUV=3.0,
    generations=200,
    population_size=60,
    neighborhood_size=10,
    service_time=1.0,
    seed=42,
    energy_params=None,
    verbose=True,
):
    energy_params = energy_params or EnergyParams()

    _RATE_CACHE.clear(); _rate_cache_keys.clear()
    _ENERGY_CACHE.clear(); _energy_cache_keys.clear()
    _thorp_absorption_db.cache_clear()
    _absorption_linear.cache_clear()
    _total_noise.cache_clear()

    node_data, node_positions = load_nodes_from_json(
        input_path, sensor_death_threshold=energy_params.sensor_death_threshold
    )
    clustering = Clustering(
        space_size=space_size,
        r_sen=r_sen,
        max_cluster_size=max_cluster_size,
        min_cluster_size=min_cluster_size,
    )
    fixed_clusters = build_fixed_clusters(node_data, node_positions, clustering)
    if len(fixed_clusters) < num_auvs:
        raise ValueError(
            f"Fixed clusters ({len(fixed_clusters)}) < num_auvs ({num_auvs}). "
            "Reduce num_auvs or adjust clustering params."
        )

    all_rounds = []
    dead_nodes_total = set()
    final_status = "completed"
    total_nodes = len(node_data)
    auv_energy_state = initialize_auv_energy(num_auvs, energy_params)
    sensor_aoi_state = initialize_sensor_aoi_state(node_data)
    base_station = (200.0, 200.0, 400.0)

    if verbose:
        print(f"\n{'═'*62}")
        print(f"  START SIMULATION — {os.path.basename(input_path)}")
        print(f"  Nodes: {total_nodes}  AUVs: {num_auvs}")
        print(f"  Stop condition: first AUV dies OR first sensor node dies")
        print(f"  Mode: FIXED CLUSTERS + FIXED HP + SEQUENTIAL UPLOAD")
        print(f"  Optimizations: cache, Numba JIT parallel, NumPy Pareto, list-buffer archive")
        print(f"{'═'*62}\n")

    fixed_clusters_export = {
        str(cid): {
            "all_nodes": list(cinfo["all_nodes"]),
            "hovering_point": list(cinfo["hovering_point"]),
            "hovering_point_avg_distance": float(cinfo["hovering_point_avg_distance"]),
            "hovering_point_max_distance": float(cinfo["hovering_point_max_distance"]),
        }
        for cid, cinfo in fixed_clusters.items()
    }

    node_positions_export = {
        str(nid): [float(x), float(y), float(z)]
        for nid, (x, y, z) in sorted(node_positions.items(), key=lambda kv: str(kv[0]))
    }

    round_idx = 1
    while True:
        alive_before = alive_node_ids(node_data, energy_params)

        if verbose:
            print(f"{'─'*62}")
            print(f"  [Round {round_idx}] Start — alive: {len(alive_before)}/{total_nodes}")
            print(f"{'─'*62}")

        if len(alive_before) < num_auvs:
            final_status = "stopped_not_enough_alive_nodes"
            if verbose:
                print(f"  [Round {round_idx}] STOP: not enough alive nodes ({len(alive_before)} < {num_auvs})\n")
            break

        active_clusters = get_active_clusters(fixed_clusters, node_data, energy_params)
        if len(active_clusters) < num_auvs:
            final_status = "stopped_not_enough_active_clusters"
            if verbose:
                print(f"  [Round {round_idx}] STOP: not enough active clusters ({len(active_clusters)} < {num_auvs})\n")
            break

        if verbose:
            log_clustering(active_clusters, node_data, round_idx)

        center_coords, visit_ids = build_center_coords_from_hp(active_clusters)

        service_times_by_cluster, upload_schedules = compute_cluster_service_times(
            active_clusters, node_positions, energy_params
        )
        service_times_vector = build_service_times_vector(visit_ids, service_times_by_cluster)
        aoi_uploads_vector = build_aoi_uploads_vector(visit_ids, upload_schedules)
        aoi_bases_vector = build_aoi_bases_vector(visit_ids, upload_schedules, sensor_aoi_state)
        sensor_distances_vector = build_sensor_distances_vector(visit_ids, upload_schedules)
        sensor_rates_vector = build_sensor_rates_vector(visit_ids, upload_schedules)
        sensor_uploads_vector = build_sensor_uploads_vector(visit_ids, upload_schedules)

        precomputed = build_round_precomputed(visit_ids, upload_schedules, energy_params)
        clear_round_energy_cache()

        params = MOEADParams(
            num_auvs=num_auvs,
            population_size=population_size,
            neighborhood_size=neighborhood_size,
            generations=generations,
            service_times=service_times_vector,
            aoi_uploads=aoi_uploads_vector,
            aoi_bases=aoi_bases_vector,
            sensor_distances=sensor_distances_vector,
            sensor_rates=sensor_rates_vector,
            sensor_uploads=sensor_uploads_vector,
            P_c=energy_params.P_c,
            P_trans=energy_params.P_trans,
            P_idle=energy_params.P_idle,
            P_hover=energy_params.P_hover,
            P_move=energy_params.P_move,
            G=energy_params.G,
            L=energy_params.L,
            B=energy_params.B,
            v_sound=energy_params.v_sound,
            fc_khz=energy_params.fc_khz,
            PSL_tr_db=energy_params.PSL_tr_db,
            spreading_factor=energy_params.spreading_factor,
            shipping_activity=energy_params.shipping_activity,
            wind_speed=energy_params.wind_speed,
            A0=energy_params.A0,
            v_f=v_f,
            v_AUV=v_AUV,
            random_seed=seed + round_idx,
        )

        solver = MOEADMultiAUV(center_coords, visit_ids, params, precomputed=precomputed)
        archive, archive_f, history = solver.solve()
        best_idx = pick_representative_solution(archive_f)
        representative_routes_named = solver.map_solution(archive[best_idx])
        representative_routes_idx = solver.map_solution_indices(archive[best_idx])

        flat_perm, cuts, count = [], [], 0
        for route in representative_routes_idx[:-1]:
            inner = [idx for idx in route if idx != 0]
            flat_perm.extend(inner)
            count += len(inner)
            cuts.append(count)
        if representative_routes_idx:
            flat_perm.extend([idx for idx in representative_routes_idx[-1] if idx != 0])

        current_sol = {"perm": flat_perm, "cuts": cuts}
        current_objectives = solver.evaluate(current_sol)

        T_total = float(max(
            sum(float(solver.time_matrix[route[i-1], route[i]]) for i in range(1, len(route))) +
            sum(float(service_times_vector[idx]) for idx in route if idx != 0)
            for route in representative_routes_idx
        )) if representative_routes_idx else 0.0

        round_energy, dead_nodes_this_round = update_energy_after_round(
            active_clusters, node_data, node_positions,
            representative_routes_idx, visit_ids,
            solver.time_matrix, energy_params, T_total,
            service_times_by_cluster, upload_schedules,
        )

        detailed_upload_schedules = enrich_upload_schedules_with_global_times(
            representative_routes_idx, visit_ids, solver.time_matrix, round_energy["upload_schedules"]
        )

        aoi_metrics = compute_aoi_metrics(
            representative_routes_idx, visit_ids, solver.time_matrix,
            detailed_upload_schedules, sensor_aoi_state,
        )
        sensor_aoi_state = aoi_metrics["updated_sensor_aoi_state"]
        sensor_aoi_state = update_next_batch_ready_times(sensor_aoi_state, detailed_upload_schedules, energy_params)

        round_energy["upload_schedules"] = detailed_upload_schedules
        round_energy["aoi_metrics"] = aoi_metrics

        per_auv_energy = compute_per_auv_energy_from_routes(
            representative_routes_idx, solver.time_matrix, service_times_vector, energy_params
        )
        auv_energy_state, dead_auvs_this_round = update_auv_energy(auv_energy_state, per_auv_energy)

        round_energy["auv_per_vehicle"] = per_auv_energy
        round_energy["auv_energy_state_after_round"] = {
            auv_id: {
                "initial_energy": float(info["initial_energy"]),
                "residual_energy": float(info["residual_energy"]),
                "alive": bool(info["alive"]),
            }
            for auv_id, info in auv_energy_state.items()
        }
        round_energy["dead_auvs_this_round"] = list(dead_auvs_this_round)

        dead_nodes_total.update(dead_nodes_this_round)

        alive_after_list = [nid for nid, nd in node_data.items() if nd.get("alive", True)]
        alive_after = len(alive_after_list)
        objectives = current_objectives.tolist()

        visualization = build_round_visualization_payload(
            round_idx=round_idx,
            total_nodes=total_nodes,
            alive_before_ids=alive_before,
            node_data=node_data,
            active_clusters=active_clusters,
            visit_ids=visit_ids,
            representative_routes_idx=representative_routes_idx,
            archive_f=archive_f,
            current_objectives=current_objectives,
            round_energy=round_energy,
            aoi_metrics=aoi_metrics,
            auv_energy_state=auv_energy_state,
            base_station=base_station,
        )

        if dead_nodes_this_round:
            final_status = "stopped_sensor_energy_depleted"
        if dead_auvs_this_round:
            final_status = "stopped_auv_energy_depleted"

        # ── THAY ĐỔI 2: truyền auv_energy_state và per_auv_energy vào log ──
        if verbose:
            log_round_summary(
                round_idx, len(alive_before), alive_after,
                dead_nodes_this_round, objectives,
                auv_energy_state=auv_energy_state,
                per_auv_energy=per_auv_energy,
            )

        all_rounds.append({
            "round": round_idx,
            "alive_nodes_before_round": alive_before,
            "num_active_clusters": len(active_clusters),
            "active_cluster_heads": [active_clusters[cid]["cluster_head"] for cid in sorted(active_clusters.keys())],
            "active_clusters": {
                str(cid): {
                    "nodes": list(active_clusters[cid]["nodes"]),
                    "all_nodes": list(active_clusters[cid]["all_nodes"]),
                    "hovering_point": list(active_clusters[cid]["hovering_point"]),
                    "hovering_point_avg_distance": float(active_clusters[cid]["hovering_point_avg_distance"]),
                    "hovering_point_max_distance": float(active_clusters[cid]["hovering_point_max_distance"]),
                    "cluster_head": active_clusters[cid]["cluster_head"],
                }
                for cid in sorted(active_clusters.keys())
            },
            "pareto_size": len(archive),
            "pareto_front": [[float(x) for x in f] for f in archive_f],
            "representative_objectives": objectives,
            "moead_rerun_this_round": True,
            "representative_routes": representative_routes_named,
            "representative_routes_indices": representative_routes_idx,
            "history": history,
            "energy_details": round_energy,
            "sum_AoI": round_energy["aoi_metrics"]["sum_AoI"],
            "average_AoI": round_energy["aoi_metrics"]["average_AoI"],
            "sensor_aoi_state_after_round": round_energy["aoi_metrics"]["updated_sensor_aoi_state"],
            "dead_nodes_this_round": dead_nodes_this_round,
            "dead_auvs_this_round": round_energy["dead_auvs_this_round"],
            "auv_energy_after_round": round_energy["auv_energy_state_after_round"],
            # ── THAY ĐỔI 3: thêm 2 trường mới vào JSON mỗi round ──────────
            "auv_residual_energy_per_round": {
                auv_id: float(info["residual_energy"])
                for auv_id, info in auv_energy_state.items()
            },
            "auv_time_per_round": {
                auv_id: {
                    "move_time": float(per_auv_energy[auv_id]["move_time"]),
                    "hover_time": float(per_auv_energy[auv_id]["hover_time"]),
                    "total_time": float(per_auv_energy[auv_id]["move_time"] + per_auv_energy[auv_id]["hover_time"]),
                }
                for auv_id in per_auv_energy
            },
            "residual_energy_after_round": {
                str(nid): float(node_data[nid]["residual_energy"])
                for nid in sorted(node_data.keys(), key=lambda x: str(x))
            },
            "visualization": visualization,
        })

        if dead_auvs_this_round:
            if verbose:
                print(f"  [Round {round_idx}] STOP: AUV energy depleted — {dead_auvs_this_round}\n")
            break

        if dead_nodes_this_round:
            if verbose:
                print(f"  [Round {round_idx}] STOP: sensor energy depleted — {sorted(dead_nodes_this_round)}\n")
            break

        round_idx += 1

    final_alive = sorted([nid for nid, nd in node_data.items() if nd.get("alive", True)])
    final_dead = sorted([nid for nid, nd in node_data.items() if not nd.get("alive", True)])

    if verbose:
        print(f"{'═'*62}")
        print(f"  END — status: {final_status}")
        print(f"  Rounds completed : {len(all_rounds)}")
        print(f"  Alive nodes      : {len(final_alive)}/{total_nodes}")
        print(f"  Dead nodes       : {len(final_dead)}")
        if final_dead:
            print(f"  Dead ids         : {final_dead}")
        print("  AUV residual energy:")
        for auv_id, info in auv_energy_state.items():
            print(f"    {auv_id}: {info['residual_energy']:.3f} J (alive={info['alive']})")
        print(f"{'═'*62}\n")

    result = {
        "algorithm": "NSGAII_FIXED_CLUSTERS_HP_SEQUENTIAL_UPLOAD_OPTIMIZED_V1_VISUAL",
        "input_file": input_path,
        "status": final_status,
        "num_auvs": num_auvs,
        "base_station": [200.0, 200.0, 400.0],
        "node_positions": node_positions_export,
        "fixed_clusters": fixed_clusters_export,
        "rounds": all_rounds,
        "dead_nodes_total": sorted(dead_nodes_total),
        "final_alive_nodes": final_alive,
        "final_dead_nodes": final_dead,
        "final_residual_energy": {
            str(nid): float(node_data[nid]["residual_energy"])
            for nid in sorted(node_data.keys(), key=lambda x: str(x))
        },
        "final_auv_energy": {
            auv_id: {
                "initial_energy": float(info["initial_energy"]),
                "residual_energy": float(info["residual_energy"]),
                "alive": bool(info["alive"]),
            }
            for auv_id, info in auv_energy_state.items()
        },
        "final_round_average_aoi": float(all_rounds[-1]["average_AoI"]) if all_rounds else None,
        "final_sensor_aoi_state": sensor_aoi_state,
        "energy_parameters": energy_params.__dict__,
        "plotting_ready": True,
    }

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def run_folder_moead(
    input_folder="/kaggle/input/datasets/hoan2107/data-nodes",
    output_main_folder="/kaggle/working/results_history_fixed_hp_seq_upload_nsgaii",
    num_auvs=3,
    space_size=400,
    r_sen=60,
    max_cluster_size=25,
    min_cluster_size=10,
    v_f=1.2,
    v_AUV=3.0,
    generations=200,
    population_size=60,
    neighborhood_size=10,
    service_time=1.0,
    seed=42,
    energy_params=None,
):
    os.makedirs(output_main_folder, exist_ok=True)
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    files = sorted([f for f in os.listdir(input_folder) if f.endswith(".json")])
    if not files:
        raise FileNotFoundError(f"No .json files in {input_folder}")

    energy_params = energy_params or EnergyParams()
    global_results = {}

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        base_name = filename.replace(".json", "")
        file_output_folder = os.path.join(output_main_folder, f"folder_{base_name}")
        os.makedirs(file_output_folder, exist_ok=True)
        output_path = os.path.join(file_output_folder, f"nsgaii_fixed_hp_seq_upload_{base_name}.json")

        result = run_fixed_cluster_hp_moead(
            input_path=input_path,
            output_path=output_path,
            num_auvs=num_auvs,
            space_size=space_size,
            r_sen=r_sen,
            max_cluster_size=max_cluster_size,
            min_cluster_size=min_cluster_size,
            v_f=v_f,
            v_AUV=v_AUV,
            generations=generations,
            population_size=population_size,
            neighborhood_size=neighborhood_size,
            service_time=service_time,
            seed=seed,
            energy_params=energy_params,
            verbose=True,
        )

        file_summary = {
            "algorithm": "NSGAII_FIXED_CLUSTERS_HP_SEQUENTIAL_UPLOAD_OPTIMIZED_V1_VISUAL",
            "input_file": input_path,
            "output_file": output_path,
            "status": result["status"],
            "num_rounds": len(result["rounds"]),
            "final_alive_nodes": result["final_alive_nodes"],
            "final_dead_nodes": result["final_dead_nodes"],
        }
        with open(os.path.join(file_output_folder, f"summary_{base_name}.json"), "w", encoding="utf-8") as f:
            json.dump(file_summary, f, indent=4, ensure_ascii=False)
        global_results[filename] = file_summary

    summary = {
        "algorithms": ["NSGAII_FIXED_CLUSTERS_HP_SEQUENTIAL_UPLOAD_OPTIMIZED_V1_VISUAL"],
        "detailed_results": global_results,
    }
    with open(os.path.join(output_main_folder, "summary_all_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    return summary


def main():
    return run_folder_moead(
        input_folder="/kaggle/input/datasets/hoan2107/data-nodes",
        output_main_folder="/kaggle/working/results_history_fixed_hp_seq_upload_nsgaii",
        num_auvs=3,
    )


if __name__ == "__main__":
    main()
