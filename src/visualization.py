from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle


def _equal_axis_limits(landmarks: np.ndarray, true_path: np.ndarray | None = None) -> tuple[float, float, float, float]:
    all_x = landmarks[:, 0]
    all_y = landmarks[:, 1]
    if true_path is not None and len(true_path):
        all_x = np.concatenate([all_x, true_path[:, 0]])
        all_y = np.concatenate([all_y, true_path[:, 1]])
    margin = 2.0
    return (
        float(all_x.min() - margin),
        float(all_x.max() + margin),
        float(all_y.min() - margin),
        float(all_y.max() + margin),
    )


def _ellipse_parameters(covariance: np.ndarray, confidence_scale: float = 2.4477) -> tuple[float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    width, height = 2.0 * confidence_scale * np.sqrt(eigenvalues)
    return float(width), float(height), angle


def covariance_ellipse(covariance: np.ndarray, center: np.ndarray, confidence_scale: float = 2.4477) -> Ellipse:
    width, height, angle = _ellipse_parameters(covariance, confidence_scale)
    return Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=1.0,
        edgecolor="tab:red",
        alpha=0.55,
    )


def _build_simulation_artists(landmarks: np.ndarray, title: str, limits: tuple[float, float, float, float]):
    x_min, x_max, y_min, y_max = limits
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)

    ax.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        marker="*",
        s=115,
        color="tab:blue",
        edgecolors="navy",
        linewidths=0.6,
        zorder=5,
        label="Prawdziwe landmarki",
    )
    true_line, = ax.plot(
        [], [], linestyle="--", linewidth=1.8, color="black", label="Prawdziwa trasa"
    )
    estimated_line, = ax.plot(
        [], [], linewidth=2.2, color="tab:orange", label="Trasa oszacowana"
    )
    particles_scatter = ax.scatter(
        [], [], s=1, alpha=0.5, color="tab:cyan", label="Cząstki", zorder=10
    )
    estimated_landmarks_scatter = ax.scatter(
        [],
        [],
        marker="o",
        facecolors="none",
        edgecolors="tab:red",
        linewidths=2.2,
        s=95,
        zorder=7,
        label="Landmarki oszacowane",
    )
    true_robot, = ax.plot(
        [],
        [],
        marker="s",
        color="black",
        markersize=7,
        linestyle="None",
        zorder=8,
        label="Robot prawdziwy",
    )
    estimated_robot, = ax.plot(
        [],
        [],
        marker="^",
        color="tab:orange",
        markeredgecolor="black",
        markersize=9,
        linestyle="None",
        zorder=9,
        label="Robot oszacowany",
    )
    true_heading, = ax.plot([], [], linewidth=2.0, color="black", zorder=8)
    estimated_heading, = ax.plot(
        [], [], linewidth=2.0, color="tab:orange", zorder=8
    )
    sensor_lines = LineCollection(
        [], linewidths=0.9, alpha=0.38, colors="tab:green", zorder=1
    )
    ax.add_collection(sensor_lines)
    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "0.7"},
        zorder=10,
    )
    covariance_ellipses: list[Ellipse] = []
    for _ in range(len(landmarks)):
        ellipse = Ellipse(
            (0.0, 0.0),
            width=0.0,
            height=0.0,
            fill=False,
            edgecolor="tab:red",
            linewidth=1.0,
            alpha=0.35,
            visible=False,
            zorder=6,
        )
        ax.add_patch(ellipse)
        covariance_ellipses.append(ellipse)

    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return {
        "fig": fig,
        "ax": ax,
        "true_line": true_line,
        "estimated_line": estimated_line,
        "particles_scatter": particles_scatter,
        "estimated_landmarks_scatter": estimated_landmarks_scatter,
        "true_robot": true_robot,
        "estimated_robot": estimated_robot,
        "true_heading": true_heading,
        "estimated_heading": estimated_heading,
        "sensor_lines": sensor_lines,
        "status_text": status_text,
        "covariance_ellipses": covariance_ellipses,
    }


def _update_covariance_ellipses(ellipses: list[Ellipse], means: np.ndarray, covariances: np.ndarray, observed_mask: np.ndarray) -> None:
    for landmark_id, ellipse in enumerate(ellipses):
        if (observed_mask[landmark_id] and np.all(np.isfinite(means[landmark_id])) and np.all(np.isfinite(covariances[landmark_id]))):
            width, height, angle = _ellipse_parameters(covariances[landmark_id])
            ellipse.center = tuple(means[landmark_id])
            ellipse.width = width
            ellipse.height = height
            ellipse.angle = angle
            ellipse.set_visible(True)
        else:
            ellipse.set_visible(False)


class LiveSimulationView:

    def __init__(self, landmarks: np.ndarray, route_name: str, delay_seconds: float = 0.03) -> None:
        self.landmarks = landmarks
        self.delay_seconds = max(float(delay_seconds), 0.001)
        self.active = True
        plt.ion()
        self.artists = _build_simulation_artists(landmarks, f"FastSLAM — {route_name}", _equal_axis_limits(landmarks))
        self.artists["fig"].canvas.draw_idle()
        plt.show(block=False)
        plt.pause(0.05)

    def update(
        self,
        step: int,
        total_steps: int,
        true_path: np.ndarray,
        estimated_path: np.ndarray,
        particle_poses: np.ndarray,
        map_means: np.ndarray,
        map_covariances: np.ndarray,
        observed_mask: np.ndarray,
        observation_ids: list[int],
        robot_position_error: float,
    ) -> None:
        fig = self.artists["fig"]
        if not self.active or not plt.fignum_exists(fig.number):
            self.active = False
            return

        self.artists["true_line"].set_data(true_path[:, 0], true_path[:, 1])
        self.artists["estimated_line"].set_data(estimated_path[:, 0], estimated_path[:, 1])
        self.artists["particles_scatter"].set_offsets(particle_poses[:, :2])

        if np.any(observed_mask):
            self.artists["estimated_landmarks_scatter"].set_offsets(map_means[observed_mask])
        else:
            self.artists["estimated_landmarks_scatter"].set_offsets(np.empty((0, 2)))
        _update_covariance_ellipses(
            self.artists["covariance_ellipses"],
            map_means,
            map_covariances,
            observed_mask,
        )

        true_pose = true_path[-1]
        estimated_pose = estimated_path[-1]
        self.artists["true_robot"].set_data([true_pose[0]], [true_pose[1]])
        self.artists["estimated_robot"].set_data([estimated_pose[0]], [estimated_pose[1]])

        heading_length = 0.8
        self.artists["true_heading"].set_data(
            [true_pose[0], true_pose[0] + heading_length * np.cos(true_pose[2])],
            [true_pose[1], true_pose[1] + heading_length * np.sin(true_pose[2])],
        )
        self.artists["estimated_heading"].set_data(
            [
                estimated_pose[0],
                estimated_pose[0] + heading_length * np.cos(estimated_pose[2]),
            ],
            [
                estimated_pose[1],
                estimated_pose[1] + heading_length * np.sin(estimated_pose[2]),
            ],
        )

        segments = [
            [(true_pose[0], true_pose[1]), tuple(self.landmarks[landmark_id])]
            for landmark_id in observation_ids
        ]
        self.artists["sensor_lines"].set_segments(segments)
        mapped_count = int(np.sum(observed_mask))
        self.artists["status_text"].set_text(
            f"krok: {step + 1:3d}/{total_steps}\n"
            f"błąd pozycji: {robot_position_error:.2f} m\n"
            f"landmarki w mapie: {mapped_count}/{len(self.landmarks)}"
        )

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(self.delay_seconds)

    def close(self, pause_seconds: float = 0.6) -> None:
        fig = self.artists["fig"]
        if self.active and plt.fignum_exists(fig.number):
            current = self.artists["status_text"].get_text()
            self.artists["status_text"].set_text(current + "\nSYMULACJA ZAKOŃCZONA")
            fig.canvas.draw_idle()
            plt.pause(max(pause_seconds, 0.001))
            plt.close(fig)
        plt.ioff()
        self.active = False


def save_trajectory_plot(
    output_path: Path,
    landmarks: np.ndarray,
    true_path: np.ndarray,
    estimated_path: np.ndarray,
    final_map_means: np.ndarray,
    final_map_covariances: np.ndarray,
    final_observed_mask: np.ndarray,
    route_name: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        landmarks[:, 0],
        landmarks[:, 1],
        marker="*",
        s=115,
        color="tab:blue",
        label="Prawdziwe landmarki",
    )
    ax.plot(
        true_path[:, 0],
        true_path[:, 1],
        linestyle="--",
        linewidth=2.0,
        color="black",
        label="Prawdziwa trasa",
    )
    ax.plot(
        estimated_path[:, 0],
        estimated_path[:, 1],
        linewidth=2.2,
        color="tab:orange",
        label="Trasa oszacowana",
    )

    visible = np.where(final_observed_mask)[0]
    if visible.size:
        ax.scatter(
            final_map_means[visible, 0],
            final_map_means[visible, 1],
            marker="o",
            facecolors="none",
            edgecolors="tab:red",
            linewidths=2.2,
            s=95,
            label="Landmarki oszacowane",
        )
        for landmark_id in visible:
            covariance = final_map_covariances[landmark_id]
            if np.all(np.isfinite(covariance)):
                ax.add_patch(
                    covariance_ellipse(covariance, final_map_means[landmark_id])
                )

    suffix = f" — {route_name}" if route_name else ""
    ax.set_title(f"FastSLAM {suffix}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_error_plot(
    output_path: Path,
    robot_position_errors: np.ndarray,
    map_rmse: np.ndarray,
    effective_sample_sizes: np.ndarray,
    number_of_particles: int,
    route_name: str = "",
) -> None:
    steps = np.arange(len(robot_position_errors))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, robot_position_errors, label="Błąd pozycji robota [m]")
    ax.plot(steps, map_rmse, label="RMSE landmarków [m]")
    suffix = f" — {route_name}" if route_name else ""
    ax.set_title(f"Błędy estymacji FastSLAM{suffix}")
    ax.set_xlabel("Krok symulacji")
    ax.set_ylabel("Błąd [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # secondary = ax.twinx()
    # secondary.plot(steps, effective_sample_sizes, linestyle=":", label="N_eff")
    # secondary.axhline(number_of_particles, linestyle="--", linewidth=0.8)
    # secondary.set_ylabel("Efektywna liczba cząstek")
    # secondary.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_animation(
    output_path: Path,
    landmarks: np.ndarray,
    true_path: np.ndarray,
    estimated_path: np.ndarray,
    particle_history: np.ndarray,
    map_mean_history: np.ndarray,
    map_covariance_history: np.ndarray,
    map_observed_history: np.ndarray,
    observation_id_history: list[list[int]],
    robot_position_errors: np.ndarray,
    route_name: str = "",
    frame_stride: int = 2,
) -> None:
    frame_indices = np.arange(0, len(true_path), frame_stride)
    if frame_indices[-1] != len(true_path) - 1:
        frame_indices = np.append(frame_indices, len(true_path) - 1)

    title = "FastSLAM"
    if route_name:
        title += f" — {route_name}"
    artists = _build_simulation_artists(landmarks, title, _equal_axis_limits(landmarks, true_path))
    fig = artists["fig"]

    def update(frame_number: int):
        index = int(frame_indices[frame_number])
        artists["true_line"].set_data(true_path[: index + 1, 0], true_path[: index + 1, 1])
        artists["estimated_line"].set_data(estimated_path[: index + 1, 0], estimated_path[: index + 1, 1])
        artists["particles_scatter"].set_offsets(particle_history[index, :, :2])

        visible = map_observed_history[index]
        if np.any(visible):
            artists["estimated_landmarks_scatter"].set_offsets(map_mean_history[index, visible])
        else:
            artists["estimated_landmarks_scatter"].set_offsets(np.empty((0, 2)))
        _update_covariance_ellipses(
            artists["covariance_ellipses"],
            map_mean_history[index],
            map_covariance_history[index],
            visible,
        )

        true_pose = true_path[index]
        estimated_pose = estimated_path[index]
        artists["true_robot"].set_data([true_pose[0]], [true_pose[1]])
        artists["estimated_robot"].set_data([estimated_pose[0]], [estimated_pose[1]])

        heading_length = 0.8
        artists["true_heading"].set_data(
            [true_pose[0], true_pose[0] + heading_length * np.cos(true_pose[2])],
            [true_pose[1], true_pose[1] + heading_length * np.sin(true_pose[2])],
        )
        artists["estimated_heading"].set_data(
            [
                estimated_pose[0],
                estimated_pose[0] + heading_length * np.cos(estimated_pose[2]),
            ],
            [
                estimated_pose[1],
                estimated_pose[1] + heading_length * np.sin(estimated_pose[2]),
            ],
        )

        segments = [
            [(true_pose[0], true_pose[1]), tuple(landmarks[landmark_id])]
            for landmark_id in observation_id_history[index]
        ]
        artists["sensor_lines"].set_segments(segments)

        mapped_count = int(np.sum(visible))
        artists["status_text"].set_text(
            f"krok: {index + 1:3d}/{len(true_path)}\n"
            f"błąd pozycji: {robot_position_errors[index]:.2f} m\n"
            f"landmarki w mapie: {mapped_count}/{len(landmarks)}"
        )
        return tuple(
            [
                artists["true_line"],
                artists["estimated_line"],
                artists["particles_scatter"],
                artists["estimated_landmarks_scatter"],
                artists["true_robot"],
                artists["estimated_robot"],
                artists["true_heading"],
                artists["estimated_heading"],
                artists["sensor_lines"],
                artists["status_text"],
                *artists["covariance_ellipses"],
            ]
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=100,
        blit=False,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=10), dpi=110)
    plt.close(fig)
