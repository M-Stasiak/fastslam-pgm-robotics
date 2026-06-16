from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np

from .simulation import Observation, move_unicycle
from .utils import gaussian_logpdf, normalize_angle


@dataclass
class Particle:
    pose: np.ndarray
    weight: float
    landmark_means: np.ndarray
    landmark_covariances: np.ndarray
    observed: np.ndarray

    def clone(self) -> "Particle":
        return Particle(
            pose=self.pose.copy(),
            weight=float(self.weight),
            landmark_means=self.landmark_means.copy(),
            landmark_covariances=self.landmark_covariances.copy(),
            observed=self.observed.copy(),
        )


@dataclass
class FastSLAMConfig:
    number_of_particles: int = 90
    dt: float = 0.20
    particle_velocity_noise_std: float = 0.08
    particle_yaw_rate_noise_std: float = np.deg2rad(2.0)
    range_noise_std: float = 0.12
    bearing_noise_std: float = np.deg2rad(1.8)
    resampling_ratio: float = 0.60
    seed: int = 2026


class FastSLAM:
    """FastSLAM 1.0 with known landmark correspondences.

    The robot path is represented by particles. Every particle owns a separate
    bank of two-dimensional EKFs, one for each landmark.
    """

    def __init__(self, number_of_landmarks: int, config: FastSLAMConfig):
        self.config = config
        self.number_of_landmarks = number_of_landmarks
        self.rng = np.random.default_rng(config.seed)
        uniform_weight = 1.0 / config.number_of_particles

        self.particles = [
            Particle(
                pose=np.zeros(3, dtype=float),
                weight=uniform_weight,
                landmark_means=np.zeros((number_of_landmarks, 2), dtype=float),
                landmark_covariances=np.zeros((number_of_landmarks, 2, 2), dtype=float),
                observed=np.zeros(number_of_landmarks, dtype=bool),
            )
            for _ in range(config.number_of_particles)
        ]

        self.measurement_covariance = np.diag(
            [config.range_noise_std**2, config.bearing_noise_std**2]
        )

    def predict(self, odometry: np.ndarray) -> None:
        for particle in self.particles:
            sampled_control = odometry + np.array(
                [
                    self.rng.normal(0.0, self.config.particle_velocity_noise_std),
                    self.rng.normal(0.0, self.config.particle_yaw_rate_noise_std),
                ]
            )
            particle.pose = move_unicycle(
                particle.pose, sampled_control, self.config.dt
            )

    def _initialize_landmark(self, particle: Particle, observation: Observation) -> None:
        x, y, yaw = particle.pose
        angle = normalize_angle(yaw + observation.bearing)
        distance = observation.distance

        particle.landmark_means[observation.landmark_id] = np.array(
            [x + distance * np.cos(angle), y + distance * np.sin(angle)]
        )

        inverse_jacobian = np.array(
            [
                [np.cos(angle), -distance * np.sin(angle)],
                [np.sin(angle), distance * np.cos(angle)],
            ]
        )
        covariance = (
            inverse_jacobian
            @ self.measurement_covariance
            @ inverse_jacobian.T
        )
        particle.landmark_covariances[observation.landmark_id] = covariance + np.eye(2) * 1e-9
        particle.observed[observation.landmark_id] = True

    def _update_landmark(self, particle: Particle, observation: Observation) -> float:
        landmark_id = observation.landmark_id
        mean = particle.landmark_means[landmark_id]
        covariance = particle.landmark_covariances[landmark_id]
        x, y, yaw = particle.pose

        dx = mean[0] - x
        dy = mean[1] - y
        squared_distance = max(dx * dx + dy * dy, 1e-12)
        predicted_distance = np.sqrt(squared_distance)
        predicted_measurement = np.array([predicted_distance, normalize_angle(np.arctan2(dy, dx) - yaw)])

        measurement_jacobian = np.array(
            [
                [dx / predicted_distance, dy / predicted_distance],
                [-dy / squared_distance, dx / squared_distance],
            ]
        )

        innovation_covariance = measurement_jacobian @ covariance @ measurement_jacobian.T + self.measurement_covariance
        kalman_gain = (covariance @ measurement_jacobian.T @ np.linalg.pinv(innovation_covariance))

        innovation = np.array(
            [
                observation.distance - predicted_measurement[0],
                normalize_angle(observation.bearing - predicted_measurement[1]),
            ]
        )

        particle.landmark_means[landmark_id] = mean + kalman_gain @ innovation
        identity = np.eye(2)

        residual = identity - kalman_gain @ measurement_jacobian
        particle.landmark_covariances[landmark_id] = residual @ covariance @ residual.T + kalman_gain @ self.measurement_covariance @ kalman_gain.T

        return gaussian_logpdf(innovation, innovation_covariance)

    def update(self, observations: list[Observation]) -> None:
        if not observations: return

        log_weights = np.empty(len(self.particles), dtype=float)
        for index, particle in enumerate(self.particles):
            log_weight = np.log(max(particle.weight, 1e-300))
            for observation in observations:
                if not particle.observed[observation.landmark_id]:
                    self._initialize_landmark(particle, observation)
                else:
                    log_weight += self._update_landmark(particle, observation)
            log_weights[index] = log_weight

        maximum_log_weight = float(np.max(log_weights))
        unnormalized = np.exp(log_weights - maximum_log_weight)
        total = float(np.sum(unnormalized))
        if not np.isfinite(total) or total <= 0.0:
            normalized = np.full(len(self.particles), 1.0 / len(self.particles))
        else:
            normalized = unnormalized / total

        for particle, weight in zip(self.particles, normalized):
            particle.weight = float(weight)

    def estimate_pose(self) -> np.ndarray:
        weights = np.array([particle.weight for particle in self.particles])
        poses = np.array([particle.pose for particle in self.particles])
        x = float(np.sum(weights * poses[:, 0]))
        y = float(np.sum(weights * poses[:, 1]))
        sin_yaw = float(np.sum(weights * np.sin(poses[:, 2])))
        cos_yaw = float(np.sum(weights * np.cos(poses[:, 2])))
        yaw = float(np.arctan2(sin_yaw, cos_yaw))
        return np.array([x, y, yaw])

    def estimate_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weighted map mean, covariance approximation and visibility mask."""
        weights = np.array([particle.weight for particle in self.particles])
        means = np.full((self.number_of_landmarks, 2), np.nan)
        covariances = np.full((self.number_of_landmarks, 2, 2), np.nan)
        observed_mask = np.zeros(self.number_of_landmarks, dtype=bool)

        for landmark_id in range(self.number_of_landmarks):
            valid_indices = [
                i for i, particle in enumerate(self.particles)
                if particle.observed[landmark_id]
            ]
            if not valid_indices: continue

            local_weights = weights[valid_indices]
            weight_sum = float(np.sum(local_weights))
            if weight_sum <= 0.0:
                local_weights = np.full(len(valid_indices), 1.0 / len(valid_indices))
            else:
                local_weights = local_weights / weight_sum

            local_means = np.array(
                [self.particles[i].landmark_means[landmark_id] for i in valid_indices]
            )
            weighted_mean = np.sum(local_weights[:, None] * local_means, axis=0)
            means[landmark_id] = weighted_mean

            total_covariance = np.zeros((2, 2), dtype=float)
            for local_weight, particle_index, local_mean in zip(
                local_weights, valid_indices, local_means
            ):
                delta = (local_mean - weighted_mean).reshape(2, 1)
                total_covariance += local_weight * (
                    self.particles[particle_index].landmark_covariances[landmark_id]
                    + delta @ delta.T
                )
            covariances[landmark_id] = total_covariance
            observed_mask[landmark_id] = True

        return means, covariances, observed_mask

    def effective_sample_size(self) -> float:
        weights = np.array([particle.weight for particle in self.particles])
        return float(1.0 / np.sum(np.square(weights)))

    def resample_if_needed(self) -> bool:
        particle_count = len(self.particles)
        threshold = self.config.resampling_ratio * particle_count
        if self.effective_sample_size() >= threshold:
            return False

        weights = np.array([particle.weight for particle in self.particles])
        cumulative = np.cumsum(weights)
        cumulative[-1] = 1.0

        start = self.rng.uniform(0.0, 1.0 / particle_count)
        positions = start + np.arange(particle_count) / particle_count
        indices = np.searchsorted(cumulative, positions)

        new_particles = [self.particles[index].clone() for index in indices]
        uniform_weight = 1.0 / particle_count
        for particle in new_particles:
            particle.weight = uniform_weight
        self.particles = new_particles
        return True

    def particle_poses(self) -> np.ndarray:
        return np.array([particle.pose for particle in self.particles])

    def step(self, odometry: np.ndarray, observations: list[Observation]) -> dict[str, np.ndarray | float | bool]:
        self.predict(odometry)
        self.update(observations)

        pose_estimate = self.estimate_pose()
        map_means, map_covariances, observed_mask = self.estimate_map()
        particle_poses_before_resampling = self.particle_poses().copy()
        effective_sample_size = self.effective_sample_size()
        resampled = self.resample_if_needed()

        return {
            "pose": pose_estimate,
            "map_means": map_means,
            "map_covariances": map_covariances,
            "observed_mask": observed_mask,
            "particle_poses": particle_poses_before_resampling,
            "effective_sample_size": effective_sample_size,
            "resampled": resampled,
        }
