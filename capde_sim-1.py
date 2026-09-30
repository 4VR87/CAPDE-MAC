from __future__ import annotations

import csv
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# IEEE 802.15.6 UP mapping used in the manuscript:
# background=UP0, periodic=UP1, bursty=UP5, real-time=UP6, emergency=UP7.
CLASS_PRIORITY = {
    "emergency": 1.00,
    "real-time": 0.875,
    "bursty": 0.75,
    "periodic": 0.25,
    "background": 0.125,
}

CW_BY_CLASS = {
    "emergency": 2,
    "real-time": 4,
    "bursty": 5,
    "periodic": 8,
    "background": 16,
}


@dataclass
class Packet:
    packet_id: int
    sensor_id: str
    traffic_class: str
    gen_time: float
    deadline_abs: float
    retries: int = 0
    predicted_completion: Optional[float] = None


@dataclass
class Sensor:
    sensor_id: str
    name: str
    traffic_class: str
    base_rate_pps: float
    deadline_s: float
    reliability_target: float
    base_success: float
    burst_enabled: bool
    burst_multiplier: float
    burst_start_prob: float
    burst_end_prob: float
    node_penalty: float = 0.0  # Pi_node,i; supply 0--0.05 per node in the sensor profile
    queue: deque = field(default_factory=deque)
    burst_on: bool = False
    service_ewma_s: float = 0.10
    retry_prob_ewma: float = 0.05
    opp_ewma: float = 0.50
    ageing_credit: float = 0.0
    energy_mj: float = 0.0
    attempts: int = 0
    collisions: int = 0
    delivered: int = 0
    expired: int = 0
    generated: int = 0
    prediction_errors: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_emergency_times: deque = field(default_factory=lambda: deque(maxlen=128))


@dataclass
class SimulationConfig:
    duration_s: float = 180.0
    frame_s: float = 0.10
    opportunities_per_frame: int = 5
    beta_service_ewma: float = 0.875  # beta; service EWMA, effective memory = 8 frames
    beta_retry_ewma: float = 0.75   # gamma_r; retry/failure EWMA, effective memory = 4 frames
    beta_opp_ewma: float = 0.75     # opportunity EWMA smoothing
    qmax_norm: int = 12
    token_capacity: float = 5.0
    token_replenish: float = 0.25
    ageing_delta: float = 1.0
    ageing_max: float = 40.0
    recovery_period_frames: int = 4
    prediction_error_mad_threshold_s: float = 0.020
    prediction_weight_reduction_factor: float = 0.50
    emergency_envelope_pps: float = 4.0
    # Eq. (6) objective weights: mean delay, P95 delay, deadline misses, E_bit, collisions, success reward.
    objective_w1: float = 0.10
    objective_w2: float = 0.25
    objective_w3: float = 0.35
    objective_w4: float = 0.10
    objective_w5: float = 0.05
    objective_w6: float = 0.15
    # Equation 9 operationalization. alpha_link is SUBTRACTED, matching the prose that poor links are penalized.
    alpha_class: float = 0.15
    alpha_prediction: float = 0.05
    alpha_queue: float = 0.10
    alpha_link: float = 0.10
    alpha_age: float = 0.55
    alpha_cost: float = 0.05
    # Energy model (illustrative, consistent with Table 5's 0.082 mJ/attempt input)
    e_tx_attempt_mj: float = 0.082
    e_success_aux_mj: float = 0.010
    e_cca_mj: float = 0.004
    e_collision_aux_mj: float = 0.002
    e_security_mj: float = 0.001
    # Posture/link model
    posture_bad_start_prob: float = 0.010
    posture_bad_end_prob: float = 0.100
    posture_bad_multiplier: float = 0.80  # 1 - Pi_post, Pi_post = 0.20
    min_success_prob: float = 0.65
    max_success_prob: float = 0.975
    kappa_posture_link_penalty: float = 0.20
    p_r_max: float = 0.95
    retry_limit: int = 3  # retransmissions; up to 4 total attempts
    node_penalty_max: float = 0.05  # Pi_node,i range; per-node values belong in the sensor profile
    load_success_penalty_per_unit: float = 0.125  # Pi_load in multiplicative Eq. (A9)


@dataclass
class RunMetrics:
    protocol: str
    load: float
    seed: int
    generated: int
    delivered: int
    missed: int
    mean_delay_ms: float
    p95_delay_ms: float
    p99_delay_ms: float
    emergency_mean_delay_ms: float
    deadline_miss_ratio: float
    packet_delivery_ratio: float
    collision_ratio: float
    energy_per_delivered_mj: float
    attempts: int
    collision_attempts: int
    integrity_flags: int
    token_uses: int

    def as_dict(self):
        return self.__dict__.copy()


def load_sensor_profiles(csv_path: str | Path) -> List[Sensor]:
    sensors: List[Sensor] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sensors.append(
                Sensor(
                    sensor_id=row["sensor_id"],
                    name=row["name"],
                    traffic_class=row["traffic_class"],
                    base_rate_pps=float(row["base_rate_pps"]),
                    deadline_s=float(row["deadline_s"]),
                    reliability_target=float(row["reliability_target"]),
                    base_success=float(row["base_success"]),
                    burst_enabled=row["burst_enabled"].strip().lower() in {"1", "true", "yes"},
                    burst_multiplier=float(row["burst_multiplier"]),
                    burst_start_prob=float(row["burst_start_prob"]),
                    burst_end_prob=float(row["burst_end_prob"]),
                    node_penalty=float(row.get("node_penalty", 0.0)),
                )
            )
    return sensors


def load_config(json_path: str | Path) -> SimulationConfig:
    with open(json_path, encoding="utf-8") as f:
        values = json.load(f)
    return SimulationConfig(**values)


class CAPDESimulator:
    """
    Discrete-time / discrete-event hybrid reference simulator reconstructed from the manuscript.

    Important: This is a transparent research reference implementation, not an IEEE conformance tool.
    Where the manuscript did not disclose exact numerical parameters, values are externalized in config files.
    """

    def __init__(
        self,
        sensors: List[Sensor],
        config: SimulationConfig,
        protocol: str,
        load: float,
        seed: int,
        collect_packet_trace: bool = False,
    ):
        allowed = {"capde", "no_prediction", "no_preemption", "static", "priority_csma", "adt_mac", "mdp_hymac"}
        if protocol not in allowed:
            raise ValueError(f"Unknown protocol {protocol}; expected one of {sorted(allowed)}")
        self.sensors = sensors
        self.cfg = config
        self.protocol = protocol
        self.load = float(load)
        self.seed = int(seed)
        # Independent random streams keep traffic/posture realizations identical across
        # protocols for the same seed while separating them from protocol-specific access
        # and transmission decisions.
        ss = np.random.SeedSequence(self.seed)
        a_ss, p_ss, ch_ss, mac_ss = ss.spawn(4)
        self.arrival_rng = np.random.default_rng(a_ss)
        self.posture_rng = np.random.default_rng(p_ss)
        self.channel_rng = np.random.default_rng(ch_ss)
        self.mac_rng = np.random.default_rng(mac_ss)
        self.packet_counter = 0
        self.posture_bad = False
        self.token_count = config.token_capacity
        self.static_rr_index = 0
        self.recovery_rr_index = 0
        self.adt_rr_index = 0
        self.mdp_rr_index = 0
        self.delays: List[float] = []
        self.emergency_delays: List[float] = []
        self.integrity_flags = 0
        self.token_uses = 0
        self.collect_packet_trace = collect_packet_trace
        self.packet_trace: List[dict] = []

    @property
    def slot_s(self) -> float:
        return self.cfg.frame_s / self.cfg.opportunities_per_frame

    def _update_posture(self):
        if self.posture_bad:
            if self.posture_rng.random() < self.cfg.posture_bad_end_prob:
                self.posture_bad = False
        else:
            if self.posture_rng.random() < self.cfg.posture_bad_start_prob:
                self.posture_bad = True

    def _success_probability(self, sensor: Sensor) -> float:
        # Eq. (A9): baseline success multiplied by posture, load, and node penalties.
        p = sensor.base_success
        if self.posture_bad:
            p *= self.cfg.posture_bad_multiplier  # 1 - Pi_post = 0.80
        u = max(0.0, self.load - 1.0) / 0.8
        p *= max(0.0, 1.0 - self.cfg.load_success_penalty_per_unit * u)  # 1 - Pi_load*u
        p *= max(0.0, 1.0 - sensor.node_penalty)  # 1 - Pi_node,i
        return float(np.clip(p, self.cfg.min_success_prob, self.cfg.max_success_prob))

    def _generate_arrivals(self, interval_start: float, interval_end: float):
        dt = interval_end - interval_start
        for s in self.sensors:
            if s.burst_enabled:
                if s.burst_on:
                    if self.arrival_rng.random() < s.burst_end_prob:
                        s.burst_on = False
                else:
                    if self.arrival_rng.random() < s.burst_start_prob:
                        s.burst_on = True
            multiplier = s.burst_multiplier if s.burst_on else 1.0
            lam = max(0.0, s.base_rate_pps * self.load * multiplier * dt)
            n = int(self.arrival_rng.poisson(lam))
            if n <= 0:
                continue
            times = np.sort(self.arrival_rng.uniform(interval_start, interval_end, size=n))
            for gt in times:
                self.packet_counter += 1
                p = Packet(
                    packet_id=self.packet_counter,
                    sensor_id=s.sensor_id,
                    traffic_class=s.traffic_class,
                    gen_time=float(gt),
                    deadline_abs=float(gt + s.deadline_s),
                )
                s.queue.append(p)
                s.generated += 1
                if s.traffic_class == "emergency":
                    s.recent_emergency_times.append(float(gt))
                if self.collect_packet_trace:
                    self.packet_trace.append({
                        "packet_id": p.packet_id,
                        "sensor_id": p.sensor_id,
                        "traffic_class": p.traffic_class,
                        "gen_time_s": p.gen_time,
                        "deadline_abs_s": p.deadline_abs,
                        "status": "generated",
                        "completion_time_s": "",
                        "delay_ms": "",
                        "retries": 0,
                    })

    def _expire_packets(self, now: float):
        for s in self.sensors:
            while s.queue and s.queue[0].deadline_abs <= now + 1e-12:
                p = s.queue.popleft()
                s.expired += 1
                if self.collect_packet_trace:
                    self._trace_update(p.packet_id, "expired", now, None, p.retries)

    def _trace_update(self, packet_id: int, status: str, completion: float, delay: Optional[float], retries: int):
        # Trace is only enabled for one diagnostic run, so O(n) update is acceptable and keeps dependencies minimal.
        for row in reversed(self.packet_trace):
            if row["packet_id"] == packet_id:
                row["status"] = status
                row["completion_time_s"] = completion
                row["delay_ms"] = "" if delay is None else delay * 1000.0
                row["retries"] = retries
                return

    def _attempt(self, sensor: Sensor, packet: Packet, service_time: float, collision: bool = False) -> bool:
        sensor.attempts += 1
        sensor.energy_mj += self.cfg.e_tx_attempt_mj + self.cfg.e_security_mj
        if collision:
            sensor.collisions += 1
            sensor.energy_mj += self.cfg.e_collision_aux_mj
            packet.retries += 1
            if packet.retries > self.cfg.retry_limit:
                sensor.queue.popleft()
                sensor.expired += 1
            return False

        success = self.channel_rng.random() < self._success_probability(sensor)
        sensor.retry_prob_ewma = (
            self.cfg.beta_retry_ewma * sensor.retry_prob_ewma
            + (1.0 - self.cfg.beta_retry_ewma) * (0.0 if success else 1.0)
        )
        if not success:
            packet.retries += 1
            if packet.retries > self.cfg.retry_limit:
                sensor.queue.popleft()
                sensor.expired += 1
            return False

        sensor.energy_mj += self.cfg.e_success_aux_mj
        sensor.queue.popleft()
        sensor.delivered += 1
        delay = max(0.0, service_time - packet.gen_time)
        self.delays.append(delay)
        if sensor.traffic_class == "emergency":
            self.emergency_delays.append(delay)

        # EWMA observed service interval: generation-to-service delay is the observable effective interval.
        sensor.service_ewma_s = (
            self.cfg.beta_service_ewma * sensor.service_ewma_s
            + (1.0 - self.cfg.beta_service_ewma) * max(self.slot_s, delay)
        )
        if packet.predicted_completion is not None:
            sensor.prediction_errors.append(delay - packet.predicted_completion)
        if self.collect_packet_trace:
            self._trace_update(packet.packet_id, "delivered", service_time, delay, packet.retries)
        return True

    def _prediction_weight(self) -> float:
        if self.protocol == "no_prediction":
            return 0.0
        errs = []
        for s in self.sensors:
            errs.extend(s.prediction_errors)
        if not errs:
            return self.cfg.alpha_prediction
        med = float(np.median(errs))
        mad = float(np.median(np.abs(np.asarray(errs) - med)))
        if mad <= 1e-12:
            return self.cfg.alpha_prediction
        return self.cfg.alpha_prediction * min(
            1.0, self.cfg.prediction_error_mad_threshold_s / mad
        )

    def _completion_prediction(self, s: Sensor) -> float:
        opp = max(s.opp_ewma, 0.10)
        retry = float(np.clip(s.retry_prob_ewma, 0.0, self.cfg.p_r_max))
        # Manuscript Eq. (3): queue/service estimate divided by opportunity rate plus retry term.
        return ((len(s.queue) + 1.0) * s.service_ewma_s / opp) + self.slot_s * retry / max(1e-6, 1.0 - retry)

    def _score(self, s: Sensor, now: float, prediction_weight: Optional[float] = None) -> float:
        if not s.queue:
            return -math.inf
        p = s.queue[0]
        slack = max(1e-6, p.deadline_abs - now)
        dhat = self._completion_prediction(s)
        p.predicted_completion = dhat
        risk = min(2.0, dhat / slack)
        qterm = min(1.0, len(s.queue) / max(1.0, self.cfg.qmax_norm))
        link_penalty = 1.0 - self._success_probability(s)
        age_term = min(1.0, s.ageing_credit / max(self.cfg.ageing_max, 1e-9))
        total_energy = sum(x.energy_mj for x in self.sensors) + 1e-9
        cost = min(1.0, s.energy_mj / total_energy * len(self.sensors)) if total_energy > 1e-8 else 0.0
        return (
            self.cfg.alpha_class * CLASS_PRIORITY[s.traffic_class]
            + (self._prediction_weight() if prediction_weight is None else prediction_weight) * risk
            + self.cfg.alpha_queue * qterm
            - self.cfg.alpha_link * link_penalty
            + self.cfg.alpha_age * age_term
            - self.cfg.alpha_cost * cost
        )

    def _eligible_emergency(self, now: float) -> List[Sensor]:
        out = []
        for s in self.sensors:
            if s.traffic_class != "emergency" or not s.queue:
                continue
            while s.recent_emergency_times and s.recent_emergency_times[0] < now - 1.0:
                s.recent_emergency_times.popleft()
            if len(s.recent_emergency_times) <= self.cfg.emergency_envelope_pps:
                out.append(s)
            else:
                self.integrity_flags += 1
        return out

    def _service_capde_frame(self, frame_start: float, frame_index: int):
        C = self.cfg.opportunities_per_frame
        scheduled: List[Sensor] = []
        used_ids = set()
        tokens_used_this_frame = 0

        # Guarded emergency override. No-preemption ablation disables token bonus entirely.
        if self.protocol != "no_preemption" and self.token_count >= 1.0:
            emerg = sorted(self._eligible_emergency(frame_start), key=lambda s: s.queue[0].deadline_abs)
            for s in emerg:
                if len(scheduled) >= C or self.token_count < 1.0:
                    break
                scheduled.append(s)
                used_ids.add(s.sensor_id)
                self.token_count -= 1.0
                tokens_used_this_frame += 1
                self.token_uses += 1

        # Deterministic recovery reserve every H frames while background backlog exists.
        if (
            len(scheduled) < C
            and self.cfg.recovery_period_frames > 0
            and frame_index % self.cfg.recovery_period_frames == 0
        ):
            bg = [s for s in self.sensors if s.traffic_class == "background" and s.queue and s.sensor_id not in used_ids]
            if bg:
                bg = sorted(bg, key=lambda s: s.sensor_id)
                chosen = bg[self.recovery_rr_index % len(bg)]
                self.recovery_rr_index += 1
                scheduled.append(chosen)
                used_ids.add(chosen.sensor_id)

        # Rank remaining head-of-line packets.
        candidates = [s for s in self.sensors if s.queue and s.sensor_id not in used_ids]
        prediction_weight = self._prediction_weight()
        candidates.sort(key=lambda s: self._score(s, frame_start, prediction_weight), reverse=True)
        for s in candidates:
            if len(scheduled) >= C:
                break
            scheduled.append(s)
            used_ids.add(s.sensor_id)

        # Equation (10) operationalization: replenish only in frames with no token consumption.
        if tokens_used_this_frame == 0:
            self.token_count = min(self.cfg.token_capacity, self.token_count + self.cfg.token_replenish)

        served_ids = set()
        for slot, s in enumerate(scheduled):
            if not s.queue:
                continue
            service_time = frame_start + (slot + 1) * self.slot_s
            self._expire_packets(service_time)
            if not s.queue:
                continue
            self._attempt(s, s.queue[0], service_time, collision=False)
            served_ids.add(s.sensor_id)

        # Update opportunity EWMA and ageing.
        for s in self.sensors:
            got = 1.0 if s.sensor_id in used_ids else 0.0
            s.opp_ewma = self.cfg.beta_opp_ewma * s.opp_ewma + (1.0 - self.cfg.beta_opp_ewma) * got
            if s.queue and s.sensor_id not in served_ids:
                s.ageing_credit = min(self.cfg.ageing_max, s.ageing_credit + self.cfg.ageing_delta)
            else:
                s.ageing_credit = 0.0

    def _service_static_frame(self, frame_start: float):
        C = self.cfg.opportunities_per_frame
        # One fixed opportunity reserved for emergency traffic; it idles if no emergency packet exists.
        emergency = [s for s in self.sensors if s.traffic_class == "emergency" and s.queue]
        if emergency:
            s = min(emergency, key=lambda x: x.queue[0].deadline_abs)
            service_time = frame_start + self.slot_s
            self._expire_packets(service_time)
            if s.queue:
                self._attempt(s, s.queue[0], service_time)

        non_em = [s for s in self.sensors if s.traffic_class != "emergency"]
        if not non_em:
            return
        for j in range(1, C):
            idx = (self.static_rr_index + (j - 1)) % len(non_em)
            s = non_em[idx]
            service_time = frame_start + (j + 1) * self.slot_s
            self._expire_packets(service_time)
            if s.queue:
                self._attempt(s, s.queue[0], service_time)
        self.static_rr_index = (self.static_rr_index + (C - 1)) % len(non_em)

    def _service_csma_frame(self, frame_start: float):
        for slot in range(self.cfg.opportunities_per_frame):
            service_time = frame_start + (slot + 1) * self.slot_s
            self._expire_packets(service_time)
            active = [s for s in self.sensors if s.queue]
            if not active:
                continue
            draws: Dict[int, List[Sensor]] = {}
            for s in active:
                cw = CW_BY_CLASS[s.traffic_class]
                b = int(self.mac_rng.integers(0, cw))
                draws.setdefault(b, []).append(s)
                s.energy_mj += self.cfg.e_cca_mj
            winning_backoff = min(draws)
            contenders = draws[winning_backoff]
            if len(contenders) > 1:
                for s in contenders:
                    if s.queue:
                        self._attempt(s, s.queue[0], service_time, collision=True)
            else:
                s = contenders[0]
                self._attempt(s, s.queue[0], service_time, collision=False)

    def _adt_priority_factor(self, s: Sensor) -> float:
        # Mechanism-level mapping of ADT-MAC's UP7 emergency class and three
        # periodic priority levels to the five traffic classes in this study.
        return {
            "emergency": 4.0,
            "real-time": 3.0,
            "bursty": 2.6,
            "periodic": 2.0,
            "background": 1.0,
        }[s.traffic_class]

    def _service_adt_frame(self, frame_start: float):
        """
        Common-scenario mechanism mapping of Hassan et al. ADT-MAC (IEEE Access, 2024).

        Preserved mechanisms: priority-based TDMA/SAP service; proportional dynamic slot
        allocation according to traffic contribution; emergency pre-emption; and repeated
        polled service when emergency queues contain More Data. RAP association overhead is
        omitted after initialization, as in the other steady-state protocol comparisons.

        The original ADT-MAC frame contains 32 x 10-ms slots, whereas the present common
        experiment deliberately holds capacity at five opportunities per 100-ms frame.
        Therefore the original S_min guarantee cannot be applied literally for 12 nodes;
        the proportional extra-slot rule is projected onto the common five-opportunity budget.
        """
        C = self.cfg.opportunities_per_frame
        scheduled: List[Sensor] = []

        # UP7-style pre-emptive emergency transmission followed by polling when More Data=1.
        # Multiple slots may be assigned to the same emergency sensor within a frame.
        while len(scheduled) < C:
            emerg = [s for s in self.sensors if s.traffic_class == "emergency" and s.queue]
            if not emerg:
                break
            # Earliest absolute deadline first; break ties by larger backlog.
            s = min(emerg, key=lambda x: (x.queue[0].deadline_abs, -len(x.queue), x.sensor_id))
            scheduled.append(s)
            # Do not pre-pop queues here; repeated assignment emulates PAP polling.
            # Limit to the current observed queue size so a single node does not receive
            # more slots than the More-Data backlog available at frame start.
            count_for_s = sum(1 for x in scheduled if x.sensor_id == s.sensor_id)
            if count_for_s >= len(s.queue):
                # Temporarily mark exhausted-for-allocation by excluding in next iteration.
                # Rebuild candidates explicitly below if all emergencies are already covered.
                covered = {x.sensor_id: sum(1 for y in scheduled if y.sensor_id == x.sensor_id) for x in emerg}
                avail = [x for x in emerg if covered.get(x.sensor_id, 0) < len(x.queue)]
                if not avail:
                    break

        remaining = C - len(scheduled)
        if remaining > 0:
            active = [s for s in self.sensors if s.queue and s.traffic_class != "emergency"]
            if active:
                # ADT-MAC Eq. (1): E_i proportional to node traffic contribution. Priority
                # is applied before rate, reflecting the paper's sort by priority and rate.
                weights = []
                for s in active:
                    traffic = max(1e-9, s.base_rate_pps * self.load)
                    if s.burst_enabled and s.burst_on:
                        traffic *= s.burst_multiplier
                    weights.append(self._adt_priority_factor(s) * traffic)
                total = sum(weights)
                quotas = [remaining * w / total for w in weights]
                slots = [int(math.floor(q)) for q in quotas]
                left = remaining - sum(slots)
                # Largest-remainder allocation, with priority/rate tie-break.
                order = sorted(
                    range(len(active)),
                    key=lambda j: (quotas[j] - slots[j], self._adt_priority_factor(active[j]), active[j].base_rate_pps, -j),
                    reverse=True,
                )
                for j in order[:left]:
                    slots[j] += 1
                # If small capacity yields zero allocations to all but a few nodes, rotate
                # equal-priority ties across frames without overriding priority ordering.
                allocation=[]
                for j,s in enumerate(active):
                    allocation.extend([s]*slots[j])
                allocation.sort(key=lambda x: (self._adt_priority_factor(x), x.base_rate_pps, len(x.queue)), reverse=True)
                scheduled.extend(allocation[:remaining])

        for slot, s in enumerate(scheduled[:C]):
            service_time = frame_start + (slot + 1) * self.slot_s
            self._expire_packets(service_time)
            if s.queue:
                self._attempt(s, s.queue[0], service_time, collision=False)

    def _mdp_policy_score(self, s: Sensor) -> float:
        """One-step policy score used to project MDP-HYMAC onto the common service budget.

        The score uses only mechanisms explicitly present in Olatinwo et al. (2024): event
        priority, channel status, buffer occupancy, heterogeneous data rate, and energy cost.
        It intentionally does not use CAPDE-MAC's deadline-risk predictor or ageing credit.
        """
        # Critical/less-critical event mapping. Emergency is critical; real-time/bursty are
        # less-critical; periodic/background are normal.
        event_priority = {
            "emergency": 1.0,
            "real-time": 0.75,
            "bursty": 0.65,
            "periodic": 0.45,
            "background": 0.25,
        }[s.traffic_class]
        channel_free = self._success_probability(s)
        buffer_pressure = min(1.0, len(s.queue) / max(1.0, self.cfg.qmax_norm))
        max_rate = max(x.base_rate_pps for x in self.sensors)
        rate_term = s.base_rate_pps / max_rate
        total_energy = sum(x.energy_mj for x in self.sensors) + 1e-9
        energy_cost = min(1.0, s.energy_mj / total_energy * len(self.sensors)) if total_energy > 1e-8 else 0.0
        # One-step reward proxy for the dynamic-programming objective: favor critical events,
        # usable channels, non-overflowing queues, and high-rate heterogeneous devices while
        # penalizing accumulated energy cost.
        return 1.50*event_priority + 0.80*channel_free + 0.75*buffer_pressure + 0.45*rate_term - 0.20*energy_cost

    def _service_mdp_hymac_frame(self, frame_start: float):
        """
        Mechanism-level common-scenario mapping of MDP-HYMAC (IEEE Sensors Journal, 2024).

        Preserved mechanisms: event-priority queue order, channel-state awareness, buffer-state
        awareness, heterogeneous data-rate based slot assignment, low-contention scheduled
        service, and retransmission by retaining failed packets. The paper's native multiple
        parallel channels are NOT converted into extra capacity here; all protocols receive the
        same five transmission opportunities per 100-ms frame for a controlled comparison.
        """
        C = self.cfg.opportunities_per_frame
        scheduled: List[Sensor] = []
        # Re-evaluate the state after each slot, analogous to state/policy updates. Allow
        # repeated slots for high-rate/high-priority devices as in the published slot scheme.
        virtual_counts = {s.sensor_id: 0 for s in self.sensors}
        for _ in range(C):
            active=[s for s in self.sensors if len(s.queue) > virtual_counts[s.sensor_id]]
            if not active:
                break
            # Penalize already assigned slots so the allocation follows heterogeneous demand
            # rather than assigning the whole frame to a single node.
            def score(s):
                target = max(0.25, s.base_rate_pps * self.load)
                fairness_penalty = 0.22 * virtual_counts[s.sensor_id] / target
                return self._mdp_policy_score(s) - fairness_penalty
            chosen=max(active, key=lambda s: (score(s), -s.queue[0].deadline_abs, s.sensor_id))
            scheduled.append(chosen)
            virtual_counts[chosen.sensor_id]+=1

        for slot,s in enumerate(scheduled):
            service_time=frame_start+(slot+1)*self.slot_s
            self._expire_packets(service_time)
            if not s.queue:
                continue
            # MDP-HYMAC's CAP performs CCA/back-off before its TDMA transmission phase.
            # Charge a small CCA overhead once per selected service opportunity while keeping
            # the common opportunity count fixed.
            s.energy_mj += 0.25*self.cfg.e_cca_mj
            self._attempt(s, s.queue[0], service_time, collision=False)

    def run(self) -> Tuple[RunMetrics, List[dict]]:
        nframes = int(round(self.cfg.duration_s / self.cfg.frame_s))
        # Arrivals for each frame are generated over the immediately preceding frame interval.
        for f in range(nframes):
            frame_start = f * self.cfg.frame_s
            if f > 0:
                self._generate_arrivals(frame_start - self.cfg.frame_s, frame_start)
            self._update_posture()
            self._expire_packets(frame_start)
            if self.protocol in {"capde", "no_prediction", "no_preemption"}:
                self._service_capde_frame(frame_start, f)
            elif self.protocol == "static":
                self._service_static_frame(frame_start)
            elif self.protocol == "priority_csma":
                self._service_csma_frame(frame_start)
            elif self.protocol == "adt_mac":
                self._service_adt_frame(frame_start)
            elif self.protocol == "mdp_hymac":
                self._service_mdp_hymac_frame(frame_start)
            else:
                raise RuntimeError(f"Unhandled protocol {self.protocol}")

        # Generate arrivals from the final interval and count all residual packets as misses/end-of-run queued traffic.
        self._generate_arrivals(self.cfg.duration_s - self.cfg.frame_s, self.cfg.duration_s)
        self._expire_packets(self.cfg.duration_s)
        for s in self.sensors:
            while s.queue:
                p = s.queue.popleft()
                s.expired += 1
                if self.collect_packet_trace:
                    self._trace_update(p.packet_id, "end_of_run_miss", self.cfg.duration_s, None, p.retries)

        generated = sum(s.generated for s in self.sensors)
        delivered = sum(s.delivered for s in self.sensors)
        missed = sum(s.expired for s in self.sensors)
        attempts = sum(s.attempts for s in self.sensors)
        collision_attempts = sum(s.collisions for s in self.sensors)
        energy = sum(s.energy_mj for s in self.sensors)

        # Defensive identity check for the model's accounting semantics.
        if generated != delivered + missed:
            raise RuntimeError(f"Packet accounting mismatch: generated={generated}, delivered={delivered}, missed={missed}")

        metrics = RunMetrics(
            protocol=self.protocol,
            load=self.load,
            seed=self.seed,
            generated=generated,
            delivered=delivered,
            missed=missed,
            mean_delay_ms=float(np.mean(self.delays) * 1000.0) if self.delays else math.nan,
            p95_delay_ms=float(np.percentile(self.delays, 95) * 1000.0) if self.delays else math.nan,
            p99_delay_ms=float(np.percentile(self.delays, 99) * 1000.0) if self.delays else math.nan,
            emergency_mean_delay_ms=float(np.mean(self.emergency_delays) * 1000.0) if self.emergency_delays else math.nan,
            deadline_miss_ratio=(missed / generated) if generated else 0.0,
            packet_delivery_ratio=(delivered / generated) if generated else 0.0,
            collision_ratio=(collision_attempts / attempts) if attempts else 0.0,
            energy_per_delivered_mj=(energy / delivered) if delivered else math.nan,
            attempts=attempts,
            collision_attempts=collision_attempts,
            integrity_flags=self.integrity_flags,
            token_uses=self.token_uses,
        )
        return metrics, self.packet_trace


def run_one(sensor_csv: str | Path, config_json: str | Path, protocol: str, load: float, seed: int, trace=False):
    sensors = load_sensor_profiles(sensor_csv)
    cfg = load_config(config_json)
    return CAPDESimulator(sensors, cfg, protocol, load, seed, collect_packet_trace=trace).run()