from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .utils import normalize_angle


@dataclass(frozen=True)
class Observation:
    landmark_id: int
    distance: float
    bearing: float


@dataclass
class SimulationConfig:
    dt: float = 0.20
    steps: int = 170
    sensor_range: float = 7.5
    range_noise_std: float = 0.12
    bearing_noise_std: float = np.deg2rad(1.8)
    true_velocity_noise_std: float = 0.025
    true_yaw_rate_noise_std: float = np.deg2rad(0.8)
    odom_velocity_noise_std: float = 0.05
    odom_yaw_rate_noise_std: float = np.deg2rad(1.4)
    seed: int = 14


ROUTE_LABELS: dict[str, str] = {
    "trasa1": "Trasa 1",
    "trasa2": "Trasa 2",
}


def available_routes() -> tuple[str, ...]:
    return tuple(ROUTE_LABELS)


def route_label(route_name: str) -> str:
    try:
        return ROUTE_LABELS[route_name]
    except KeyError as exc:
        raise ValueError(f"Nieznana trasa: {route_name}. Dostępne: {', '.join(available_routes())}") from exc


def make_landmarks() -> np.ndarray:
    return np.array(
        [
            [1.5, 4.5], [3.0, -3.8], [5.0, 2.0], [6.5, -5.0],
            [8.0, 5.5], [9.5, -1.5], [11.5, 3.5], [12.5, -5.5],
            [14.0, 0.5], [15.5, 5.8], [17.0, -3.2], [18.5, 2.8],
            [20.0, -5.0], [21.5, 4.8], [23.0, 0.0], [24.5, -3.8],
            [26.0, 4.0], [28.0, -0.8], [29.0, 5.0], [30.5, -4.5],
        ],
        dtype=float,
    )


def control_command(step: int, dt: float, route_name: str = "trasa1", total_steps: int = 170) -> np.ndarray:
    """Return controls for robot trajectories."""
    time = step * dt

    if route_name == "trasa1":
        velocity = 0.90 + 0.08 * np.sin(0.35 * time)
        yaw_rate = 0.16 * np.cos(0.45 * time)
        return np.array([velocity, yaw_rate], dtype=float)

    if route_name == "trasa2":
        duration = max(total_steps * dt, dt)
        cycles = 3.0
        angular_frequency = 2.0 * np.pi * cycles / duration
        heading_amplitude = 0.75
        velocity = 0.88 + 0.05 * np.cos(0.25 * time)
        yaw_rate = heading_amplitude * angular_frequency * np.cos(
            angular_frequency * time
        )
        return np.array([velocity, yaw_rate], dtype=float)

    route_label(route_name)
    raise AssertionError("unreachable")


def move_unicycle(pose: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    """Apply the planar unicycle motion model."""
    x, y, yaw = pose
    velocity, yaw_rate = control

    if abs(yaw_rate) < 1e-8:
        x += velocity * dt * np.cos(yaw)
        y += velocity * dt * np.sin(yaw)
    else:
        x += (velocity / yaw_rate) * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
        y += (velocity / yaw_rate) * (-np.cos(yaw + yaw_rate * dt) + np.cos(yaw))
    yaw = normalize_angle(yaw + yaw_rate * dt)
    return np.array([x, y, yaw], dtype=float)


def simulate_true_motion(pose: np.ndarray, command: np.ndarray, config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    noisy_control = command + np.array(
        [
            rng.normal(0.0, config.true_velocity_noise_std),
            rng.normal(0.0, config.true_yaw_rate_noise_std),
        ]
    )
    return move_unicycle(pose, noisy_control, config.dt)


def measure_odometry(command: np.ndarray, config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    return command + np.array(
        [
            rng.normal(0.0, config.odom_velocity_noise_std),
            rng.normal(0.0, config.odom_yaw_rate_noise_std),
        ]
    )


def sense_landmarks(pose: np.ndarray, landmarks: np.ndarray, config: SimulationConfig, rng: np.random.Generator) -> list[Observation]:
    observations: list[Observation] = []
    x, y, yaw = pose
    for landmark_id, (landmark_x, landmark_y) in enumerate(landmarks):
        dx = landmark_x - x
        dy = landmark_y - y
        distance = float(np.hypot(dx, dy))
        if distance > config.sensor_range: continue

        bearing = normalize_angle(np.arctan2(dy, dx) - yaw)
        measured_distance = max(0.02, distance + rng.normal(0.0, config.range_noise_std))
        measured_bearing = normalize_angle(bearing + rng.normal(0.0, config.bearing_noise_std))
        observations.append(Observation(landmark_id, measured_distance, measured_bearing))
        
    return observations
