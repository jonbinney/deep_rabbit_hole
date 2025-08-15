"""
Video renderer for Quoridor games using image sequences and ffmpeg.

Creates video files showing the progression of each game move by move.
Uses matplotlib to generate individual frames, then compiles them into videos.
"""

import copy
import shutil
import subprocess
import tempfile
from pathlib import Path

from agents import Agent
from arena_utils import GameResult
from mpl_visualizer import visualize_board

from renderers import Renderer


class VideoRenderer(Renderer):
    """
    Renderer that creates video files showing game progression move by move.

    Generates individual PNG frames for each move, then compiles them into video
    using ffmpeg (if available) or saves as individual images.
    """

    def __init__(
        self,
        fps: float = 2.0,
        format: str = "mp4",
        output_dir: str = "videos",
        figsize: tuple = (10, 10),
        keep_frames: bool = False,
    ):
        """
        Initialize video renderer.

        Args:
            fps: Frames per second for the video
            format: Output format ('mp4', 'gif') - requires ffmpeg
            output_dir: Directory to save video files
            figsize: Figure size for each frame
            keep_frames: Whether to keep individual frame images
        """
        super().__init__()
        self.fps = fps
        self.format = format.lower()
        self.output_dir = Path(output_dir)
        self.keep_frames = keep_frames

        # Game state storage
        self.game_states = []
        self.game_info = None
        self.move_descriptions = []
        self.temp_dir = None
        self.agent_action_logs = []

        # Ensure figsize has even dimensions for ffmpeg compatibility
        self.figsize = (
            int(figsize[0]) if int(figsize[0]) % 2 == 0 else int(figsize[0]) + 1,
            int(figsize[1]) if int(figsize[1]) % 2 == 0 else int(figsize[1]) + 1,
        )

        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)

        # Check if ffmpeg is available
        self.has_ffmpeg = shutil.which("ffmpeg") is not None

    def start_game(self, env, agent1: Agent, agent2: Agent):
        """Start of game - initialize and store initial state."""
        # Clear previous game data
        self.game_states = []
        self.move_descriptions = []
        self.agent_action_logs = []

        # Store game info for filename and agent references
        self.game_info = {"agent1": agent1.name(), "agent2": agent2.name()}
        self.agents = {"player_0": agent1, "player_1": agent2}

        # Store initial game state
        initial_state = self._extract_game_state(env)
        self.game_states.append(initial_state)
        self.move_descriptions.append("Initial position")

        agent1.action_log.set_enabled(True)
        agent2.action_log.set_enabled(True)

    def after_action(self, env, step, agent_id, action):
        """After each action - store the current game state."""
        current_state = self._extract_game_state(env)
        self.game_states.append(current_state)

        # Create move description
        if self.game_info:
            agent_name = self.game_info["agent1"] if agent_id == "player_0" else self.game_info["agent2"]
            move_desc = f"Move {step + 1}: {agent_name} - {action}"
        else:
            move_desc = f"Move {step + 1}: {agent_id} - {action}"
        self.move_descriptions.append(move_desc)

        # Store a copy of the agent's action log for this frame
        if hasattr(self, "agents") and agent_id in self.agents:
            agent = self.agents[agent_id]
            if hasattr(agent, "action_log"):
                # Create a copy of the action log to preserve its state at this moment
                import copy

                action_log_copy = copy.deepcopy(agent.action_log)
                self.agent_action_logs.append(action_log_copy)
            else:
                self.agent_action_logs.append(None)
        else:
            self.agent_action_logs.append(None)

    def end_game(self, env, result: GameResult):
        """End of game - create frames and compile video."""
        if not self.game_states:
            print("No game states recorded, skipping video creation")
            return

        try:
            # Generate filename
            video_filename = self._generate_filename(result)
            video_path = self.output_dir / video_filename

            # Create temporary directory for frames
            self.temp_dir = Path(tempfile.mkdtemp(prefix=f"quoridor_video_{result.game_id}_"))

            # Generate all frames
            frame_paths = self._generate_frames()

            # Create video
            if self.has_ffmpeg and len(frame_paths) > 1:
                self._create_video_ffmpeg(frame_paths, video_path)
            else:
                # Fallback: just keep the frame images
                final_dir = self.output_dir / f"frames_{result.game_id}"
                if self.temp_dir.exists():
                    if final_dir.exists():
                        shutil.rmtree(final_dir)
                    shutil.move(str(self.temp_dir), str(final_dir))
                print(f"✓ Frame images saved: {final_dir}")
                print("  (Install ffmpeg to create video files)")
                # Reset temp_dir so cleanup doesn't try to process it again
                self.temp_dir = None

            # Clean up temporary files
            if self.temp_dir and self.temp_dir.exists():
                if self.keep_frames:
                    # Move temp directory to permanent location in output directory
                    permanent_dir = self.output_dir / f"frames_{result.game_id}"
                    if permanent_dir.exists():
                        shutil.rmtree(permanent_dir)
                    shutil.move(str(self.temp_dir), str(permanent_dir))
                    print(f"  Frame images kept in: {permanent_dir}")
                else:
                    # Delete temporary directory
                    shutil.rmtree(self.temp_dir)

        except Exception as e:
            print(f"✗ Error creating video: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Ensure temp directory is always cleaned up in case of errors
            if self.temp_dir and self.temp_dir.exists() and not self.keep_frames:
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as cleanup_error:
                    print(f"Warning: Could not clean up temp directory: {cleanup_error}")

    def _extract_game_state(self, env):
        """Extract the current game state from the environment."""
        return copy.deepcopy(env.game)

    def _generate_frames(self):
        """Generate PNG frames for each game state."""
        frame_paths = []

        for i, (game_state, move_desc, action_log) in enumerate(
            zip(self.game_states, self.move_descriptions, self.agent_action_logs)
        ):
            frame_filename = f"frame_{i:04d}.png"
            if self.temp_dir:
                frame_path = self.temp_dir / frame_filename
            else:
                frame_path = self.output_dir / frame_filename

            # Generate the frame using visualize_board, passing the action log if available
            fig = visualize_board(
                game_state,
                show=False,
                save_path=str(frame_path),
                figsize=self.figsize,
                title=move_desc,
                action_log=action_log,
            )
            # Explicitly close the figure to free memory
            import matplotlib.pyplot as plt

            plt.close(fig)

            frame_paths.append(frame_path)

        return frame_paths

    def _create_video_ffmpeg(self, frame_paths, output_path):
        """Create video using ffmpeg."""
        if not frame_paths:
            return

        # Build ffmpeg command
        if self.temp_dir:
            input_pattern = str(self.temp_dir / "frame_%04d.png")
        else:
            input_pattern = str(self.output_dir / "frame_%04d.png")

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-framerate",
            str(self.fps),
            "-i",
            input_pattern,
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure even dimensions
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]

        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    def _generate_filename(self, result: GameResult) -> str:
        """Generate filename for the video."""
        if self.game_info:
            agent1 = self._clean_filename(self.game_info["agent1"])
            agent2 = self._clean_filename(self.game_info["agent2"])
        else:
            agent1 = "unknown"
            agent2 = "unknown"
        game_id = result.game_id or "unknown"

        return f"game_{game_id}_{agent1}_vs_{agent2}.{self.format}"

    def _clean_filename(self, name: str) -> str:
        """Clean name for use in filename."""
        # Replace problematic characters
        clean = name.replace(" ", "_").replace(":", "").replace("/", "_")
        # Remove other problematic characters
        clean = "".join(c for c in clean if c.isalnum() or c in "_-")
        return clean[:20]  # Limit length

    def end_arena(self, game, results: list[GameResult]):
        print(f"Videos of the games saved to {self.output_dir.absolute()}")
