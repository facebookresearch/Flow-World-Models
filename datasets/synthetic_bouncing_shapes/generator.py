import torch
from einops import rearrange


class SyntheticBouncingShapesVideoGenerator:
    """Generator for bouncing shapes videos with explicit two-index access:
    - First index: initial conditions
    - Second index: trajectory variant for those initial conditions
    """

    def __init__(self, config):
        self.config = config
        self.frames = config.frames
        self.image_size = config.image_size
        self.square_size = config.square_size
        self.disk_radius = config.disk_radius
        self._shapes = ("square", "circle")
        self._pos_keys = tuple(f"{shape}_position" for shape in self._shapes)
        self._vel_keys = tuple(f"{shape}_speed" for shape in self._shapes)
        self._state_keys = self._pos_keys + self._vel_keys
        self.max_speed = float(config.max_speed)
        self.velocity_bins = int(config.velocity_bins)
        self.train_ratio = float(config.train_ratio)
        self.dataset_seed = int(config.dataset_seed)
        self.split = config.split.lower()
        (
            self._all_velocity_states,
            self._train_velocity_states,
            self._eval_velocity_states,
        ) = self._initialize_velocity_hypercube()
        if self.split == "train":
            self._velocity_subset = self._train_velocity_states
        else:
            self._velocity_subset = self._eval_velocity_states

        sz = float(self.image_size - self.square_size)
        half_square = float(self.square_size / 2)
        dr = float(self.disk_radius)
        ch = float(self.image_size - self.disk_radius)

        self._bounds = {
            "square": torch.tensor(
                [[half_square, half_square], [sz + half_square, sz + half_square]],
                dtype=torch.float32,
            ),
            "circle": torch.tensor([[dr, dr], [ch, ch]], dtype=torch.float32),
        }

        self._background = torch.full(
            (3, self.image_size, self.image_size),
            20,
            dtype=torch.uint8,
        )
        self._square_color = torch.tensor((200, 50, 50), dtype=torch.uint8).view(
            3, 1, 1
        )
        self._circle_color = torch.tensor((50, 180, 250), dtype=torch.uint8).view(
            3, 1, 1
        )

        # Cache for trajectory trees per initial condition.
        # During export we only need one IC at a time, so callers should
        # release entries after finishing an IC to avoid unbounded growth.
        self._trajectory_cache = {}
        # Per-IC random generators to produce deterministic but non-repeating positions
        self._ic_generators = {}

    def _initialize_velocity_hypercube(self):
        axis = torch.linspace(-self.max_speed, self.max_speed, self.velocity_bins)
        hypercube = torch.cartesian_prod(axis, axis, axis, axis).view(
            -1, len(self._vel_keys), 2
        )
        generator = torch.Generator().manual_seed(self.dataset_seed)
        perm = torch.randperm(hypercube.shape[0], generator=generator)
        split = int(perm.numel() * self.train_ratio)
        return hypercube, hypercube[perm[:split]], hypercube[perm[split:]]

    def get_num_initial_conditions(self):
        """Returns the number of different initial conditions available."""
        return self._velocity_subset.shape[0]

    def get_num_trajectories(self, ic_index):
        """Returns the number of trajectory variants for a given initial condition."""
        tree_data = self._get_or_build_tree(ic_index)
        return len(tree_data["all_trajectories"])

    def get_trajectory_info(self, ic_index):
        """Returns metadata about trajectories for a given initial condition."""
        tree_data = self._get_or_build_tree(ic_index)
        return {
            "num_trajectories": len(tree_data["all_trajectories"]),
            "initial_state": tree_data["initial_state"],
            "frames": self.frames,
        }

    def _get_or_build_tree(self, ic_index):
        """Get cached tree or build it if not cached."""
        if ic_index not in self._trajectory_cache:
            initial_state = self._get_initial_conditions(ic_index)
            tree_data = self._build_trajectory_tree(ic_index, initial_state)
            self._trajectory_cache[ic_index] = tree_data
        return self._trajectory_cache[ic_index]

    def release_initial_condition(self, ic_index):
        """Drop cached data for an IC once all its trajectories are exported."""
        self._trajectory_cache.pop(ic_index, None)
        self._ic_generators.pop(ic_index, None)

    def generate(self, ic_index, traj_index):
        """Generate a video for specific initial condition and trajectory indices.

        Args:
            ic_index: Index of initial conditions
            traj_index: Index of trajectory variant

        Returns:
            Dictionary containing video and target heatmaps
        """
        tree_data = self._get_or_build_tree(ic_index)

        if traj_index >= len(tree_data["all_trajectories"]):
            raise IndexError(
                f"Trajectory index {traj_index} out of range. "
                f"IC {ic_index} has {len(tree_data['all_trajectories'])} trajectories."
            )

        # Get bounce decisions for this trajectory
        bounce_decisions = tree_data["all_trajectories"][traj_index]

        # Get first half (deterministic)
        first_half = tree_data["first_half"]

        # Get midpoint state
        midpoint_state = {
            key: first_half[key][self.frames // 2 - 1].clone()
            for key in self._state_keys
        }

        # Compute second half with specific decisions
        second_half = self._compute_trajectory_with_decisions(
            midpoint_state, self.frames // 2, self.frames, bounce_decisions
        )

        # Combine trajectories
        full_trajectory = {}
        for key in first_half.keys():
            full_trajectory[key] = torch.cat([first_half[key], second_half[key]], dim=0)

        # Generate targets
        targets = self._generate_targets_from_trajectory(full_trajectory)

        # Render video
        video = self._render_video_from_trajectory(full_trajectory)

        return {
            "video": video,
            "ic_index": ic_index,
            "traj_index": traj_index,
            "bounce_decisions": bounce_decisions,
            **targets,
        }

    def generate_batch(self, ic_traj_pairs):
        """Generate multiple videos from list of (ic_index, traj_index) pairs.

        Args:
            ic_traj_pairs: List of tuples (ic_index, traj_index)

        Returns:
            List of dictionaries containing video and targets
        """
        return [self.generate(ic_idx, traj_idx) for ic_idx, traj_idx in ic_traj_pairs]

    def _get_initial_conditions(self, ic_index):
        """Get initial conditions for a given index.

        Extend this method to support multiple initial conditions.
        """
        if ic_index < 0 or ic_index >= self._velocity_subset.shape[0]:
            raise IndexError(
                f"ic_index {ic_index} out of range for velocity subset size "
                f"{self._velocity_subset.shape[0]}"
            )
        if ic_index not in self._ic_generators:
            seed = self.dataset_seed + int(ic_index)
            self._ic_generators[ic_index] = torch.Generator().manual_seed(seed)
        generator = self._ic_generators[ic_index]
        velocities = self._velocity_subset[ic_index].to(dtype=torch.float32)
        square_low, square_high = self._bounds["square"]
        circle_low, circle_high = self._bounds["circle"]
        square_position = square_low + (square_high - square_low) * torch.rand(
            2, generator=generator, dtype=torch.float32
        )
        circle_position = circle_low + (circle_high - circle_low) * torch.rand(
            2, generator=generator, dtype=torch.float32
        )

        return {
            "square_position": square_position,
            "circle_position": circle_position,
            "square_speed": velocities[0],
            "circle_speed": velocities[1],
        }

    def _build_trajectory_tree(self, ic_index, initial_state):
        """Build a binary tree of all possible trajectories from initial conditions."""
        first_half_trajectory, first_half_decisions = (
            self._compute_trajectory_with_random_decisions(
                initial_state, 0, self.frames // 2, ic_index
            )
        )

        # Get state at the midpoint
        midpoint_state = {
            key: first_half_trajectory[key][self.frames // 2 - 1].clone()
            for key in self._state_keys
        }

        # Build tree for second half
        tree_root = self._explore_trajectory_tree(
            midpoint_state, self.frames // 2, self.frames, bounce_decisions=[]
        )

        all_trajectories = self._flatten_tree(tree_root)

        return {
            "initial_state": initial_state,
            "first_half": first_half_trajectory,
            "first_half_decisions": first_half_decisions,
            "second_half_tree": tree_root,
            "all_trajectories": all_trajectories,
        }

    def _explore_trajectory_tree(self, state, start_frame, end_frame, bounce_decisions):
        """Recursively explore all possible trajectories from a given state."""
        if start_frame >= end_frame:
            return {
                "is_leaf": True,
                "bounce_decisions": bounce_decisions.copy(),
                "trajectory": None,
            }

        # Take one step forward
        next_positions = {
            pos_key: state[pos_key] + state[vel_key]
            for pos_key, vel_key in zip(self._pos_keys, self._vel_keys)
        }

        # Check which shapes bounce
        bounce_info = self._check_bounce_occurrence(next_positions)
        bouncing_shapes = [shape for shape, bounced in bounce_info.items() if bounced]

        if not bouncing_shapes:
            new_state = self._apply_no_bounce(state, next_positions)
            return self._explore_trajectory_tree(
                new_state, start_frame + 1, end_frame, bounce_decisions
            )

        # Create all combinations for bouncing shapes
        num_bouncing = len(bouncing_shapes)
        tree_node = {
            "is_leaf": False,
            "bounce_decisions": bounce_decisions.copy(),
            "bouncing_shapes": bouncing_shapes,
            "branches": [],
        }

        # Create 2^num_bouncing branches
        for i in range(2**num_bouncing):
            decisions_this_branch = []
            for j in range(num_bouncing):
                decision = (i >> j) & 1 == 1
                decisions_this_branch.append((bouncing_shapes[j], decision))

            new_state = self._apply_bounce_decisions(
                state, next_positions, decisions_this_branch
            )

            branch = self._explore_trajectory_tree(
                new_state,
                start_frame + 1,
                end_frame,
                bounce_decisions + decisions_this_branch,
            )
            tree_node["branches"].append(branch)

        return tree_node

    def _apply_bounce_decisions(self, state, next_positions, frame_decisions):
        """Apply bounce decisions for the current frame."""
        new_state = {key: value.clone() for key, value in state.items()}
        decision_dict = {shape: decision for shape, decision in frame_decisions}

        for shape, pos_key, vel_key in zip(
            self._shapes, self._pos_keys, self._vel_keys
        ):
            pos = next_positions[pos_key].clone()
            vel = state[vel_key].clone()

            if shape in decision_dict:
                if decision_dict[shape]:
                    # Reflect
                    low, high = self._bounds[shape]
                    below = pos < low
                    above = pos > high
                    pos = torch.where(below, 2.0 * low - pos, pos)
                    vel = torch.where(below, -vel, vel)
                    above = pos > high
                    pos = torch.where(above, 2.0 * high - pos, pos)
                    vel = torch.where(above, -vel, vel)
                else:
                    # Reverse
                    vel = -vel
                    pos = state[pos_key] + vel

            new_state[pos_key] = pos
            new_state[vel_key] = vel

        return new_state

    def _check_bounce_occurrence(self, next_positions):
        """Check which shapes will bounce at next positions."""
        bounce_info = {}
        for shape, pos_key in zip(self._shapes, self._pos_keys):
            low, high = self._bounds[shape]
            pos = next_positions[pos_key]
            bounce_info[shape] = (pos < low).any() or (pos > high).any()
        return bounce_info

    def _apply_no_bounce(self, state, next_positions):
        """Apply movement without bounce."""
        new_state = {key: value.clone() for key, value in state.items()}
        new_state.update(next_positions)
        return new_state

    def _apply_reflect_bounce(self, state, next_positions):
        """Apply bounce with reflection for shapes that are bouncing."""
        updated_positions = {}
        updated_velocities = {}

        for shape, pos_key, vel_key in zip(
            self._shapes, self._pos_keys, self._vel_keys
        ):
            low, high = self._bounds[shape]
            pos = next_positions[pos_key].clone()
            vel = state[vel_key].clone()

            below = pos < low
            above = pos > high
            is_bouncing = (below | above).any()

            if is_bouncing:
                pos = torch.where(below, 2.0 * low - pos, pos)
                vel = torch.where(below, -vel, vel)
                above = pos > high
                pos = torch.where(above, 2.0 * high - pos, pos)
                vel = torch.where(above, -vel, vel)

            updated_positions[pos_key] = pos
            updated_velocities[vel_key] = vel

        new_state = {key: value.clone() for key, value in state.items()}
        new_state.update(updated_positions)
        new_state.update(updated_velocities)
        return new_state

    def _flatten_tree(self, tree_root):
        """Extract all unique trajectories from the tree."""
        trajectories = []

        def traverse(node):
            if node["is_leaf"]:
                trajectories.append(node["bounce_decisions"])
            else:
                for branch in node["branches"]:
                    traverse(branch)

        traverse(tree_root)
        return trajectories

    def _compute_trajectory_with_random_decisions(
        self, initial_state, start_frame, end_frame, ic_index
    ):
        trajectory = {
            key: torch.empty((end_frame - start_frame, 2), dtype=torch.float32)
            for key in self._state_keys
        }
        rounded_pos_keys = [key + "_rounded" for key in self._pos_keys]
        for key in rounded_pos_keys:
            trajectory[key] = torch.empty(
                (end_frame - start_frame, 2), dtype=torch.long
            )

        state = {key: value.clone() for key, value in initial_state.items()}
        generator = torch.Generator().manual_seed(self.dataset_seed + int(ic_index))
        random_decisions = []

        for i in range(end_frame - start_frame):
            next_positions = {
                pos_key: state[pos_key] + state[vel_key]
                for pos_key, vel_key in zip(self._pos_keys, self._vel_keys)
            }

            bounce_info = self._check_bounce_occurrence(next_positions)
            bouncing_shapes = [
                shape for shape, bounced in bounce_info.items() if bounced
            ]

            if bouncing_shapes:
                frame_decisions = []
                for shape in bouncing_shapes:
                    decision = torch.rand((), generator=generator).item() >= 0.5
                    frame_decisions.append((shape, decision))
                state = self._apply_bounce_decisions(
                    state, next_positions, frame_decisions
                )
                random_decisions.extend(frame_decisions)
            else:
                state = self._apply_no_bounce(state, next_positions)

            for key in self._state_keys:
                trajectory[key][i] = state[key]
            for pos_key in self._pos_keys:
                trajectory[pos_key + "_rounded"][i] = state[pos_key].round().long()

        return trajectory, random_decisions

    def _compute_deterministic_trajectory(self, initial_state, start_frame, end_frame):
        """Compute deterministic trajectory with regular bounces."""
        trajectory = {
            key: torch.empty((end_frame - start_frame, 2), dtype=torch.float32)
            for key in self._state_keys
        }
        rounded_pos_keys = [key + "_rounded" for key in self._pos_keys]
        for key in rounded_pos_keys:
            trajectory[key] = torch.empty(
                (end_frame - start_frame, 2), dtype=torch.long
            )

        state = {key: value.clone() for key, value in initial_state.items()}

        for i in range(end_frame - start_frame):
            next_positions = {
                pos_key: state[pos_key] + state[vel_key]
                for pos_key, vel_key in zip(self._pos_keys, self._vel_keys)
            }
            state = self._apply_reflect_bounce(state, next_positions)

            for key in self._state_keys:
                trajectory[key][i] = state[key]
            for pos_key in self._pos_keys:
                trajectory[pos_key + "_rounded"][i] = state[pos_key].round().long()

        return trajectory

    def _compute_trajectory_with_decisions(
        self, initial_state, start_frame, end_frame, bounce_decisions
    ):
        """Compute trajectory following specific bounce decisions."""
        trajectory = {
            key: torch.empty((end_frame - start_frame, 2), dtype=torch.float32)
            for key in self._state_keys
        }
        rounded_pos_keys = [key + "_rounded" for key in self._pos_keys]
        for key in rounded_pos_keys:
            trajectory[key] = torch.empty(
                (end_frame - start_frame, 2), dtype=torch.long
            )

        state = {key: value.clone() for key, value in initial_state.items()}
        decision_idx = 0

        for i in range(end_frame - start_frame):
            next_positions = {
                pos_key: state[pos_key] + state[vel_key]
                for pos_key, vel_key in zip(self._pos_keys, self._vel_keys)
            }

            bounce_info = self._check_bounce_occurrence(next_positions)
            bouncing_shapes = [
                shape for shape, bounced in bounce_info.items() if bounced
            ]

            for shape, pos_key, vel_key in zip(
                self._shapes, self._pos_keys, self._vel_keys
            ):
                pos = next_positions[pos_key].clone()
                vel = state[vel_key].clone()

                if shape in bouncing_shapes:
                    decision = None
                    for dec_shape, dec_value in bounce_decisions[
                        decision_idx : decision_idx + len(bouncing_shapes)
                    ]:
                        if dec_shape == shape:
                            decision = dec_value
                            break

                    if decision is not None:
                        if decision:
                            # Reflect
                            low, high = self._bounds[shape]
                            below = pos < low
                            above = pos > high
                            pos = torch.where(below, 2.0 * low - pos, pos)
                            vel = torch.where(below, -vel, vel)
                            above = pos > high
                            pos = torch.where(above, 2.0 * high - pos, pos)
                            vel = torch.where(above, -vel, vel)
                        else:
                            # Reverse
                            vel = -vel
                            pos = state[pos_key] + vel

                state[pos_key] = pos
                state[vel_key] = vel

            decision_idx += len(bouncing_shapes)

            for key in self._state_keys:
                trajectory[key][i] = state[key]
            for pos_key in self._pos_keys:
                trajectory[pos_key + "_rounded"][i] = state[pos_key].round().long()

        return trajectory

    def _generate_targets_from_trajectory(self, trajectory):
        square_centers = trajectory["square_position_rounded"]
        circle_centers = trajectory["circle_position_rounded"]
        return {
            "target_square_heatmap_pixel": self._build_pixel_heatmap(square_centers),
            "target_circle_heatmap_pixel": self._build_pixel_heatmap(circle_centers),
            "target_square_heatmap_patch": self._build_patch_heatmap(square_centers),
            "target_circle_heatmap_patch": self._build_patch_heatmap(circle_centers),
        }

    def _build_pixel_heatmap(self, centers):
        heatmap = torch.zeros(
            (len(centers), self.image_size, self.image_size, 1), dtype=torch.float32
        )
        idx = torch.arange(len(centers))
        heatmap[idx, centers[:, 1], centers[:, 0], 0] = 1.0
        return heatmap

    def _build_patch_heatmap(self, centers):
        patch_size = 16
        patch_h = self.image_size // patch_size
        patch_w = self.image_size // patch_size
        heatmap = torch.zeros((len(centers), patch_h, patch_w, 1), dtype=torch.float32)
        idx = torch.arange(len(centers))
        px = torch.clamp(centers[:, 0] // patch_size, max=patch_w - 1)
        py = torch.clamp(centers[:, 1] // patch_size, max=patch_h - 1)
        heatmap[idx, py, px, 0] = 1.0
        return heatmap

    def _render_video_from_trajectory(self, trajectory):
        num_frames = len(trajectory["square_position_rounded"])
        square_pos = trajectory["square_position_rounded"]
        circle_pos = trajectory["circle_position_rounded"]

        video = self._background.unsqueeze(0).expand(num_frames, -1, -1, -1).clone()

        xx = torch.arange(self.image_size, dtype=torch.int32).view(1, 1, -1)
        yy = torch.arange(self.image_size, dtype=torch.int32).view(1, -1, 1)

        cx, cy = square_pos[:, 0].view(-1, 1, 1), square_pos[:, 1].view(-1, 1, 1)
        half_sq = self.square_size // 2
        sq_mask = (
            (xx >= cx - half_sq)
            & (xx < cx + half_sq)
            & (yy >= cy - half_sq)
            & (yy < cy + half_sq)
        )
        rearrange(video, "t c h w -> c t h w")[:, sq_mask] = (
            self._square_color.flatten()[:, None]
        )

        cxi, cyi = circle_pos[:, 0].view(-1, 1, 1), circle_pos[:, 1].view(-1, 1, 1)
        circ_mask = ((xx - cxi) ** 2 + (yy - cyi) ** 2) <= self.disk_radius**2
        rearrange(video, "t c h w -> c t h w")[:, circ_mask] = (
            self._circle_color.flatten()[:, None]
        )

        return video
