from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from src.fastslam import FastSLAM, FastSLAMConfig
from src.simulation import SimulationConfig, available_routes, control_command, make_landmarks, measure_odometry, route_label, sense_landmarks, simulate_true_motion
from src.visualization import LiveSimulationView, save_animation, save_error_plot, save_trajectory_plot


def compute_map_rmse(
    estimated_means: np.ndarray,
    observed_mask: np.ndarray,
    true_landmarks: np.ndarray,
) -> float:
    if not np.any(observed_mask):
        return float("nan")
    errors = estimated_means[observed_mask] - true_landmarks[observed_mask]
    return float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))


def run_project(
    graph_directory: Path,
    output_directory: Path,
    route_name: str = "trasa1",
    create_gif: bool = True,
    show_live: bool = True,
    live_delay: float = 0.03,
) -> dict[str, float | int | str]:
    output_directory.mkdir(parents=True, exist_ok=True)

    simulation_config = SimulationConfig()
    fastslam_config = FastSLAMConfig(
        dt=simulation_config.dt,
        range_noise_std=simulation_config.range_noise_std,
        bearing_noise_std=simulation_config.bearing_noise_std,
    )

    landmarks = make_landmarks()
    filter_ = FastSLAM(len(landmarks), fastslam_config)
    rng = np.random.default_rng(simulation_config.seed)
    selected_route_label = route_label(route_name)

    live_view = None
    if show_live:
        try:
            live_view = LiveSimulationView(
                landmarks, selected_route_label, live_delay
            )
        except Exception as exc:
            print(
                "Nie udało się otworzyć okna animacji na żywo. "
                f"Symulacja będzie kontynuowana bez niego: {exc}"
            )

    true_pose = np.zeros(3, dtype=float)
    true_path: list[np.ndarray] = []
    estimated_path: list[np.ndarray] = []
    particle_history: list[np.ndarray] = []
    map_mean_history: list[np.ndarray] = []
    map_covariance_history: list[np.ndarray] = []
    map_observed_history: list[np.ndarray] = []
    observation_id_history: list[list[int]] = []
    position_errors: list[float] = []
    map_errors: list[float] = []
    effective_sample_sizes: list[float] = []
    resampling_count = 0

    try:
        for step in range(simulation_config.steps):
            command = control_command(
                step,
                simulation_config.dt,
                route_name=route_name,
                total_steps=simulation_config.steps,
            )
            true_pose = simulate_true_motion(true_pose, command, simulation_config, rng)
            odometry = measure_odometry(command, simulation_config, rng)
            observations = sense_landmarks(true_pose, landmarks, simulation_config, rng)

            result = filter_.step(odometry, observations)
            estimated_pose = np.asarray(result["pose"])
            map_means = np.asarray(result["map_means"])
            map_covariances = np.asarray(result["map_covariances"])
            observed_mask = np.asarray(result["observed_mask"], dtype=bool)
            particle_poses = np.asarray(result["particle_poses"]).copy()
            observation_ids = [obs.landmark_id for obs in observations]
            position_error = float(np.linalg.norm(true_pose[:2] - estimated_pose[:2]))

            true_path.append(true_pose.copy())
            estimated_path.append(estimated_pose.copy())
            particle_history.append(particle_poses)
            map_mean_history.append(map_means.copy())
            map_covariance_history.append(map_covariances.copy())
            map_observed_history.append(observed_mask.copy())
            observation_id_history.append(observation_ids)
            position_errors.append(position_error)
            map_errors.append(compute_map_rmse(map_means, observed_mask, landmarks))
            effective_sample_sizes.append(float(result["effective_sample_size"]))
            resampling_count += int(bool(result["resampled"]))

            if live_view is not None:
                live_view.update(
                    step=step,
                    total_steps=simulation_config.steps,
                    true_path=np.asarray(true_path),
                    estimated_path=np.asarray(estimated_path),
                    particle_poses=particle_poses,
                    map_means=map_means,
                    map_covariances=map_covariances,
                    observed_mask=observed_mask,
                    observation_ids=observation_ids,
                    robot_position_error=position_error,
                )
    finally:
        if live_view is not None:
            live_view.close()

    true_path_array = np.array(true_path)
    estimated_path_array = np.array(estimated_path)
    particle_history_array = np.array(particle_history)
    map_mean_history_array = np.array(map_mean_history)
    map_covariance_history_array = np.array(map_covariance_history)
    map_observed_history_array = np.array(map_observed_history)
    position_errors_array = np.array(position_errors)
    map_errors_array = np.array(map_errors)
    effective_sample_sizes_array = np.array(effective_sample_sizes)

    print("Symulacja zakończona. Zapisywanie wykresów...")
    save_trajectory_plot(
        output_directory / "trajectory_and_map.png",
        landmarks,
        true_path_array,
        estimated_path_array,
        map_mean_history_array[-1],
        map_covariance_history_array[-1],
        map_observed_history_array[-1],
        selected_route_label,
    )
    save_error_plot(
        output_directory / "errors.png",
        position_errors_array,
        map_errors_array,
        effective_sample_sizes_array,
        fastslam_config.number_of_particles,
        selected_route_label,
    )

    if create_gif:
        print("Generowanie pliku GIF...")
        save_animation(
            output_directory / "fastslam_animation.gif",
            landmarks,
            true_path_array,
            estimated_path_array,
            particle_history_array,
            map_mean_history_array,
            map_covariance_history_array,
            map_observed_history_array,
            observation_id_history,
            position_errors_array,
            selected_route_label,
        )

    finite_map_errors = map_errors_array[np.isfinite(map_errors_array)]
    metrics: dict[str, float | int | str] = {
        "route": route_name,
        "route_label": selected_route_label,
        "steps": int(simulation_config.steps),
        "particles": int(fastslam_config.number_of_particles),
        "landmarks": int(len(landmarks)),
        "mean_robot_position_error_m": float(np.mean(position_errors_array)),
        "final_robot_position_error_m": float(position_errors_array[-1]),
        "mean_map_rmse_m": float(np.mean(finite_map_errors)),
        "final_map_rmse_m": float(finite_map_errors[-1]),
        "mapped_landmarks": int(np.sum(map_observed_history_array[-1])),
        "resampling_count": int(resampling_count),
    }
    (output_directory / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="FastSLAM PGM Robotics")
    parser.add_argument(
        "--route",
        choices=available_routes(),
        default="trasa1",
        help="Select one of the prepared robot trajectories",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF creation for a faster test run",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Do not show the animation window while the simulation is running",
    )
    parser.add_argument(
        "--live-delay",
        type=float,
        default=0.03,
        help="Pause between live frames in seconds (default: 0.03)",
    )
    args = parser.parse_args()

    graph_directory = Path("outputs")
    output_directory = Path("outputs") / args.route
    metrics = run_project(
        graph_directory,
        output_directory,
        route_name=args.route,
        create_gif=not args.no_gif,
        show_live=not args.no_live,
        live_delay=args.live_delay,
    )
    print("FastSLAM finished successfully.")
    print(f"Output directory: {output_directory.resolve()}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
