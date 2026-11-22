import os
import RocketSim as rsim
from rlbot.flat import BallAnchor, ControllerState, GamePacket
from rlbot.managers import Hivemind

import sys
from pathlib import Path


def model_path(filename: str) -> str:
    """Resolve a model file from a PyInstaller bundle or from source."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return str(base / filename)


def mesh_path() -> str:
    """Resolve the collision mesh directory from a PyInstaller bundle or from source."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent
    return str(base / "collision_meshes")


COLLISION_MESHES = mesh_path()
os.environ["RS_COLLISION_MESHES"] = COLLISION_MESHES
try:
    rsim.init(COLLISION_MESHES)
except RuntimeError as error:
    if str(error) != "Already inited":
        raise


try:
    import torch
except ImportError:
    sys.path.insert(0, "../../torch-archive")
    import torch


from rlgym_compat import GameState
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

import math
from typing import Any
import numpy as np
import random
from torch import nn
from torch.distributions import Categorical
from rlgym_compat import V1GameState


class MaskableCategorical(Categorical):
    def __init__(self, logits, validate_args, ZERO):
        super().__init__(None, logits, validate_args)
        self.mask = None
        self.zero = ZERO

    def entropy(self) -> torch.Tensor:
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.mask, p_log_p, self.zero)
        return -p_log_p.sum(-1)


class CategoricalDistribution:
    """
     Categorical distribution for discrete actions.
     :param action_dim: Number of discrete actions
     """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution = None
        self.mask = None
        self.zero = None

    def proba_distribution(self, action_logits: torch.Tensor):
        if self.zero is None:
            self.zero = torch.zeros(1, device=action_logits.device)
        self.distribution = MaskableCategorical(logits=action_logits, validate_args=False, ZERO=self.zero)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        self.distribution.mask = self.mask
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

    def apply_mask(self, mask):
        self.mask = mask


class SeerNetworkV4(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = nn.LeakyReLU(inplace=True)

        self.OBS_SIZE = 144

        self.ENCODER_INTERMEDIATE_SIZE = 1024

        self.encoder = nn.Sequential(
            nn.Linear(self.OBS_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
        )

        self.value_network = nn.Sequential(
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, 1),
        )

        self.policy_network = nn.Sequential(
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, 90),
        )

        self.distribution = CategoricalDistribution(90)
        self.HUGE_NEG = None
        self.ones = None
        self.jump_mask = None
        self.boost_mask = None
        self.pitch_roll_mask = None
        self.handbrake_mask = None

    def encode(self, obs):


        if obs.shape[-1] == 144:
            return self.encoder(obs)

    def forward(self, obs, deterministic):

        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

        # Rollout
        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        mask = self.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        log_prob = self.distribution.log_prob(actions)
        return actions, value, log_prob,

    def predict_value(self, obs):
        # Rollout
        x = self.encode(obs)
        value = self.value_network(x)
        return value

    def predict_actions(self, obs, deterministic):
        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

            # Rollout
        x = self.encode(obs)

        policy_logits = self.policy_network(x)
        mask = self.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs, actions, mask):

        if self.HUGE_NEG is None:
            self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32).to(obs.device)

        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        log_prob = self.distribution.log_prob(actions)

        entropy = self.distribution.entropy()

        return value, log_prob, entropy

    def create_mask(self, obs):

        has_boost = obs[:, 13] > 0.0
        on_ground = obs[:, 14] == 1
        has_flip = obs[:, 15] == 1
        in_air = torch.logical_not(on_ground)

        if self.ones is None:
            self.ones = torch.ones(1, dtype=torch.bool, device=obs.device, requires_grad=False)

        if self.jump_mask is None:
            self.jump_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                           1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.,
                                           1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                           1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.], dtype=torch.bool, device=obs.device, requires_grad=False)

        if self.boost_mask is None:
            self.boost_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.,
                                            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 10., 1., 0., 1., 0.,
                                            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                            1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.], dtype=torch.bool, device=obs.device, requires_grad=False)

        if self.pitch_roll_mask is None:
            self.pitch_roll_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.bool, device=obs.device, requires_grad=False)

        if self.handbrake_mask is None:
            self.handbrake_mask = torch.tensor([1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.bool, device=obs.device, requires_grad=False)

        with torch.no_grad():

            mask_0 = torch.where(has_boost.unsqueeze(1), self.ones, self.boost_mask)
            mask_1 = torch.where(has_flip.unsqueeze(1), self.ones, self.jump_mask)
            mask_2 = torch.where(in_air.unsqueeze(1), self.ones, self.pitch_roll_mask)
            mask_3 = torch.where(on_ground.unsqueeze(1), self.ones, self.handbrake_mask)

            res = (mask_0 * mask_1) * (mask_2 * mask_3)

        return res


class SeerGameCondition:  # Mimics a Rocket League game
    def __init__(self, tick_skip=8, overtime_prob=0.1, no_touch_timeout=512):
        super().__init__()
        self.tick_skip = tick_skip
        self.timer = 0
        self.overtime = False
        self.done = True
        self.initial_state = None
        self.score = 0
        self.overtime_prob = overtime_prob
        self.last_touch = 0
        # self.no_touch_timeout = NoTouchTimeoutCondition(no_touch_timeout)
        self.scored = 0.0

    def reset(self, initial_state):
        self.initial_state = initial_state
        self.done = False
        # self.no_touch_timeout.reset(initial_state)
        self.scored = 0.0

        if random.random() < self.overtime_prob:
            self.overtime = True
            self.timer = 0
            self.score = 0
        else:
            self.overtime = False
            self.timer = random.uniform(10, 300)
            self.score = random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    def is_terminal(self, current_state) -> bool:
        reset = False

        self.scored = ((current_state.blue_score - self.initial_state.blue_score)
                       - (current_state.orange_score - self.initial_state.orange_score))

        self.score += self.scored

        if self.scored != 0:  # Goal scored
            reset = True

        if self.overtime:
            if self.scored != 0:  # Overtime
                self.done = True

        else:
            if self.timer <= 0 and current_state.ball.position[2] <= 110:
                reset = True

                if self.score != 0:
                    self.done = True

        self.timer -= self.tick_skip / 120
        self.timer = max(0, self.timer)

        # if self.no_touch_timeout.is_terminal(current_state):
        #     reset = True

        return reset or self.done


def get_encoded_actionV4(action: np.ndarray) -> np.ndarray:
    # throttle, steer, pitch, yaw, roll, jump, boost, handbrake

    action_encoding = np.zeros(15, dtype=np.float32)

    acc = 0
    throttle_index = action[0] + 1 + acc
    acc += 3
    steer_yaw_index = action[1] + 1 + acc
    acc += 3
    pitch_index = action[2] + 1 + acc
    acc += 3
    roll_index = action[4] + 1 + acc

    action_encoding[int(throttle_index)] = 1.0
    action_encoding[int(steer_yaw_index)] = 1.0
    action_encoding[int(pitch_index)] = 1.0
    action_encoding[int(roll_index)] = 1.0

    action_encoding[12] = action[5]
    action_encoding[13] = action[6]
    action_encoding[14] = action[7]

    return action_encoding


def encode_all_playersV4(player, state, inverted, demo_timers, ball):
    player_encoding = _encode_playerV4(player, inverted, demo_timers.get(player.car_id), ball)

    same_team = []
    opponent_team = []

    for p in state.players:

        if p == player:
            continue

        if p.team_num == player.team_num:
            same_team.append(p)
        else:
            opponent_team.append(p)


    same_team.sort(key=lambda x: x.car_id)
    opponent_team.sort(key=lambda x: x.car_id)

    encodings = [player_encoding]
    for p in same_team + opponent_team:
        encodings.append(_encode_playerV4(p, inverted, demo_timers.get(p.car_id), ball))

    return encodings


def _encode_rotationV4(rot):
    tol = 1e-4
    if rot > math.pi:
        rot -= 2 * math.pi
    return rot

def _encode_playerV4(player, inverted: bool, demo_timer: float, ball):
    if inverted:
        player_car = player.inverted_car_data
    else:
        player_car = player.car_data

    vel_norm = np.linalg.norm([player_car.linear_velocity[0],
                               player_car.linear_velocity[1],
                               player_car.linear_velocity[2]])

    ball_diff_x = ball.position[0] - player_car.position[0]
    ball_diff_y = ball.position[1] - player_car.position[1]
    ball_diff_z = ball.position[2] - player_car.position[2]
    ball_diff_norm = np.linalg.norm([ball_diff_x, ball_diff_y, ball_diff_z])

    pitch = _encode_rotationV4(player_car.pitch())
    yaw = _encode_rotationV4(player_car.yaw())
    roll = _encode_rotationV4(player_car.roll())

    array = np.array([
        player_car.position[0] * (1.0 / 4096.0),
        player_car.position[1] * (1.0 / 5120.0),
        player_car.position[2] * (1.0 / 2048.0),
        pitch * (1.0 / math.pi),
        yaw * (1.0 / math.pi),
        roll * (1.0 / math.pi),
        player_car.linear_velocity[0] * (1.0 / 2300.0),
        player_car.linear_velocity[1] * (1.0 / 2300.0),
        player_car.linear_velocity[2] * (1.0 / 2300.0),
        player_car.angular_velocity[0] * (1.0 / 5.5),
        player_car.angular_velocity[1] * (1.0 / 5.5),
        player_car.angular_velocity[2] * (1.0 / 5.5),
        demo_timer * (1 / 3.0),
        player.boost_amount,
        player.on_ground,
        player.has_flip,
        demo_timer > 0,
        vel_norm * (1.0 / 6000.0),
        vel_norm > 2200,
        ball_diff_x * (1.0 / (4096.0 * 2.0)),
        ball_diff_y * (1.0 / (5120.0 * 2.0)),
        ball_diff_z * (1.0 / 2048.0),
        ball_diff_norm * (1.0 / 13272.55),
        player.has_jump
    ], dtype=np.float32)


    return array


def _encode_ballV4(ball):
    state = np.array([
        ball.position[0] * (1.0 / 4096.0),
        ball.position[1] * (1.0 / 5120.0),
        ball.position[2] * (1.0 / 2048.0),
        ball.linear_velocity[0] * (1.0 / 6000.0),
        ball.linear_velocity[1] * (1.0 / 6000.0),
        ball.linear_velocity[2] * (1.0 / 6000.0),
        ball.angular_velocity[0] * (1.0 / 6.0),
        ball.angular_velocity[1] * (1.0 / 6.0),
        ball.angular_velocity[2] * (1.0 / 6.0),
        np.linalg.norm([ball.linear_velocity[0], ball.linear_velocity[1], ball.linear_velocity[2]]) * (1.0 / 6000.0),
    ], dtype=np.float32)


    return state


pads_scalerV4 = np.array([
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
], dtype=np.float32)


def encode_boost_padsV4(boost_pads_timers):
    pads_active = boost_pads_timers == 0.0

    timers = boost_pads_timers * pads_scalerV4

    return [pads_active, timers]


def encode_game_stateV4(condition, inverted):
    dif, timer, overtime = condition.score, condition.timer, condition.overtime

    if inverted:
        dif *= -1

    game_state = [dif / 10, timer / 300, overtime]

    return game_state


class SeerObsV4:
    def __init__(self, team_size, condition, default_tick_skip=8.0, physics_ticks_per_second=120.0):
        super().__init__()
        self.team_size = team_size
        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}
        self.time_diff_tick = default_tick_skip / physics_ticks_per_second
        self.condition = condition

    def reset(self, initial_state):

        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}
        for p in initial_state.players:
            self.demo_timers.update({p.car_id: 0})

    def pre_step(self, state):
        self.update_boostpads(state.boost_pads)
        self.update_demo_timers(state)

    def build_obs(self, player, state, previous_action: np.ndarray) -> Any:

        if player.team_num == 1:
            inverted = True
            ball = state.inverted_ball
            pads = self.boost_pads_timers[::-1]

        else:
            inverted = False
            ball = state.ball
            pads = self.boost_pads_timers

        ball_data = _encode_ballV4(ball)
        player_encodings = encode_all_playersV4(player, state, inverted, self.demo_timers, ball)
        prev_action_enc = get_encoded_actionV4(previous_action)
        pads_encoding = encode_boost_padsV4(pads)
        game_state = encode_game_stateV4(self.condition, inverted)
        obs = np.concatenate([*player_encodings, ball_data, prev_action_enc, *pads_encoding, game_state], dtype=np.float32)
        return obs

    def update_boostpads(self, pads):

        mask = pads == 1.0
        not_mask = np.logical_not(mask)

        self.boost_pads_timers[mask] = 0.0
        self.boost_pads_timers[not_mask] += self.time_diff_tick

    def update_demo_timers(self, state):

        for p in state.players:

            if p.is_demoed:
                self.demo_timers.update({p.car_id: self.demo_timers.get(p.car_id) + self.time_diff_tick})
            else:
                self.demo_timers.update({p.car_id: 0})


class SeerActionV4:
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, 0])
        actions = np.array(actions)
        return actions

    # def get_action_space(self) -> gym.spaces.Space:
    #     return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state) -> np.ndarray:
        return self._lookup_table[actions]


class AgentV4:
    def __init__(self, filename):
        self.filename = filename
        print("{} Loading...".format(self.filename))

        torch.set_num_threads(2)
        self.policy = SeerNetworkV4()

        state_dict = torch.load(self.filename, map_location=torch.device('cpu'), weights_only=True)
        consume_prefix_in_state_dict_if_present(state_dict, "module.")
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        print("Ready: {}".format(self.filename))

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state)
            action = self.policy.predict_actions(state, True)
        return action.numpy()


class SeerMaskCreatorV6:

    def __init__(self, device):
        self.jump_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                       1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.,
                                       1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                       1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.], dtype=torch.bool, device=device, requires_grad=False)

        self.pitch_roll_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.bool, device=device, requires_grad=False)

        self.boost_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.,
                                        1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                        1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                        1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.], dtype=torch.bool, device=device, requires_grad=False)

        self.handbrake_mask = torch.tensor([1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.bool, device=device, requires_grad=False)

        self.ones = torch.ones(1, dtype=torch.bool, device=device, requires_grad=False)

    def create_mask(self, obs):
        has_boost = obs[:, 13] > 0.0
        on_ground = obs[:, 14] == 1
        has_flip = obs[:, 15] == 1
        in_air = torch.logical_not(on_ground)

        with torch.no_grad():
            mask_0 = torch.where(has_boost.unsqueeze(1), self.ones, self.boost_mask)
            mask_1 = torch.where(has_flip.unsqueeze(1), self.ones, self.jump_mask)
            mask_2 = torch.where(in_air.unsqueeze(1), self.ones, self.pitch_roll_mask)
            mask_3 = torch.where(on_ground.unsqueeze(1), self.ones, self.handbrake_mask)

            res = (mask_0 * mask_1) * (mask_2 * mask_3)

        return res


class ResBlockV6(nn.Module):
    def __init__(self, size, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer0 = nn.Linear(size, size)
        self.layer1 = nn.Linear(size, size)
        self.activation = activation

    def forward(self, x):
        out = self.layer0(x)
        out = self.activation(out)
        out = self.layer1(out)
        out += x
        out = self.activation(out)
        return out


class SeerNetworkV6(nn.Module):
    def __init__(self, device):
        super(SeerNetworkV6, self).__init__()
        self.activation = nn.LeakyReLU(inplace=True)

        self.OBS_SIZE = 267

        self.ENCODER_INTERMEDIATE_SIZE = 768

        self.mask_creator = SeerMaskCreatorV6(device)

        self.encoder = nn.Sequential(
            nn.Linear(self.OBS_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            ResBlockV6(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlockV6(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlockV6(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
        ).to(device)

        self.value_network = nn.Sequential(
            ResBlockV6(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlockV6(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, 1),
        ).to(device)

        self.policy_network = nn.Sequential(
            ResBlockV6(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlockV6(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlockV6(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, 90),
        ).to(device)

        self.distribution = CategoricalDistribution(90)

        self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32, device=device)

    def encode(self, obs):

        if obs.shape[-1] == 267:
            return self.encoder(obs)

    def forward(self, obs, deterministic):
        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        mask = self.mask_creator.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        log_prob = self.distribution.log_prob(actions)
        return actions, value, log_prob,

    def predict_value(self, obs):
        x = self.encode(obs)
        value = self.value_network(x)
        return value

    def predict_actions(self, obs, deterministic):
        x = self.encode(obs)

        policy_logits = self.policy_network(x)
        mask = self.mask_creator.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs, actions, mask):
        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        log_prob = self.distribution.log_prob(actions)

        entropy = self.distribution.entropy()

        return value, log_prob, entropy


def get_encoded_actionV6(action: np.ndarray) -> np.ndarray:
    # throttle, steer, pitch, yaw, roll, jump, boost, handbrake

    action_encoding = np.zeros(15, dtype=np.float32)

    acc = 0
    throttle_index = action[0] + 1 + acc
    acc += 3
    steer_yaw_index = action[1] + 1 + acc
    acc += 3
    pitch_index = action[2] + 1 + acc
    acc += 3
    roll_index = action[4] + 1 + acc

    action_encoding[int(throttle_index)] = 1.0
    action_encoding[int(steer_yaw_index)] = 1.0
    action_encoding[int(pitch_index)] = 1.0
    action_encoding[int(roll_index)] = 1.0

    action_encoding[12] = action[5]
    action_encoding[13] = action[6]
    action_encoding[14] = action[7]

    return action_encoding


def encode_all_playersV6(player, state, inverted, demo_timers, ball):
    player_encoding = _encode_playerV6(player, inverted, demo_timers.get(player.car_id), ball, player)

    same_team = []
    opponent_team = []

    for p in state.players:

        if p == player:
            continue

        if p.team_num == player.team_num:
            same_team.append(p)
        else:
            opponent_team.append(p)


    same_team_encoding = []
    opponent_team_encoding = []

    for p in same_team:
        same_team_encoding.append(_encode_playerV6(p, inverted, demo_timers.get(p.car_id), ball, player))

    for p in opponent_team:
        opponent_team_encoding.append(_encode_playerV6(p, inverted, demo_timers.get(p.car_id), ball, player))

    while len(same_team_encoding) < 2:
        same_team_encoding.append(np.zeros(28, dtype=np.float32))

    while len(opponent_team_encoding) < 3:
        opponent_team_encoding.append(np.zeros(28, dtype=np.float32))

    encodings = [player_encoding] + same_team_encoding + opponent_team_encoding

    return encodings


def _encode_rotationV6(rot):
    tol = 1e-4
    if rot > math.pi:
        rot -= 2 * math.pi
    return rot


def _encode_playerV6(player, inverted: bool, demo_timer: float, ball, main_player):
    if inverted:
        player_car = player.inverted_car_data
        main_player_car = main_player.inverted_car_data
    else:
        player_car = player.car_data
        main_player_car = main_player.car_data

    vel_norm = np.linalg.norm([player_car.linear_velocity[0],
                               player_car.linear_velocity[1],
                               player_car.linear_velocity[2]])

    ball_diff_x = ball.position[0] - player_car.position[0]
    ball_diff_y = ball.position[1] - player_car.position[1]
    ball_diff_z = ball.position[2] - player_car.position[2]
    ball_diff_norm = np.linalg.norm([ball_diff_x, ball_diff_y, ball_diff_z])

    main_player_diff_x = main_player_car.position[0] - player_car.position[0]
    main_player_diff_y = main_player_car.position[1] - player_car.position[1]
    main_player_diff_z = main_player_car.position[2] - player_car.position[2]
    main_player_diff_norm = np.linalg.norm([main_player_diff_x, main_player_diff_y, main_player_diff_z])

    pitch = _encode_rotationV6(player_car.pitch())
    yaw = _encode_rotationV6(player_car.yaw())
    roll = _encode_rotationV6(player_car.roll())

    array = np.array([
        player_car.position[0] * (1.0 / 4096.0),
        player_car.position[1] * (1.0 / 5120.0),
        player_car.position[2] * (1.0 / 2048.0),
        pitch * (1.0 / math.pi),
        yaw * (1.0 / math.pi),
        roll * (1.0 / math.pi),
        player_car.linear_velocity[0] * (1.0 / 2300.0),
        player_car.linear_velocity[1] * (1.0 / 2300.0),
        player_car.linear_velocity[2] * (1.0 / 2300.0),
        player_car.angular_velocity[0] * (1.0 / 5.5),
        player_car.angular_velocity[1] * (1.0 / 5.5),
        player_car.angular_velocity[2] * (1.0 / 5.5),
        demo_timer * (1 / 3.0),
        player.boost_amount,
        player.on_ground,
        player.has_flip,
        demo_timer > 0,
        vel_norm * (1.0 / 2300.0),
        vel_norm > 2200,
        ball_diff_x * (1.0 / (4096.0 * 2.0)),
        ball_diff_y * (1.0 / (5120.0 * 2.0)),
        ball_diff_z * (1.0 / 2048.0),
        ball_diff_norm * (1.0 / 13272.55),
        player.has_jump,
        main_player_diff_x * (1.0 / (4096.0 * 2.0)),
        main_player_diff_y * (1.0 / (5120.0 * 2.0)),
        main_player_diff_z * (1.0 / 2048.0),
        main_player_diff_norm * (1.0 / 13272.55)
    ], dtype=np.float32)


    return array


def _encode_ballV6(ball):
    state = np.array([
        ball.position[0] * (1.0 / 4096.0),
        ball.position[1] * (1.0 / 5120.0),
        ball.position[2] * (1.0 / 2048.0),
        ball.linear_velocity[0] * (1.0 / 6000.0),
        ball.linear_velocity[1] * (1.0 / 6000.0),
        ball.linear_velocity[2] * (1.0 / 6000.0),
        ball.angular_velocity[0] * (1.0 / 6.0),
        ball.angular_velocity[1] * (1.0 / 6.0),
        ball.angular_velocity[2] * (1.0 / 6.0),
        np.linalg.norm([ball.linear_velocity[0], ball.linear_velocity[1], ball.linear_velocity[2]]) * (1.0 / 6000.0),
    ], dtype=np.float32)


    return state


pads_scalerV6 = np.array([
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
], dtype=np.float32)


def encode_boost_padsV6(boost_pads_timers):
    pads_active = boost_pads_timers == 0.0

    timers = boost_pads_timers * pads_scalerV6

    return [pads_active, timers]


def encode_game_stateV6(state, condition, inverted):
    dif, timer, overtime = condition.score, condition.timer, condition.overtime

    is_ones = 0
    is_twos = 0
    is_threes = 0

    if len(state.players) == 2:
        is_ones = 1
    elif len(state.players) == 4:
        is_twos = 1
    elif len(state.players) == 6:
        is_threes = 1
    else:
        raise NotImplementedError(len(state.players))

    if inverted:
        dif *= -1

    game_state = [is_ones, is_twos, is_threes, dif / 10, timer / 300, overtime]

    return game_state


class SeerObsV6:
    def __init__(self, condition, default_tick_skip=8.0, physics_ticks_per_second=120.0):
        super().__init__()
        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}
        self.time_diff_tick = default_tick_skip / physics_ticks_per_second
        self.condition = condition

    def reset(self, initial_state):

        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}
        for p in initial_state.players:
            self.demo_timers.update({p.car_id: 0})

    def pre_step(self, state):
        self.update_boostpads(state.boost_pads)
        self.update_demo_timers(state)

    def build_obs(self, player, state, previous_action: np.ndarray) -> Any:

        if player.team_num == 1:
            inverted = True
            ball = state.inverted_ball
            pads = self.boost_pads_timers[::-1]

        else:
            inverted = False
            ball = state.ball
            pads = self.boost_pads_timers

        ball_data = _encode_ballV6(ball)
        player_encodings = encode_all_playersV6(player, state, inverted, self.demo_timers, ball)
        prev_action_enc = get_encoded_actionV6(previous_action)
        pads_encoding = encode_boost_padsV6(pads)
        game_state = encode_game_stateV6(state, self.condition, inverted)
        obs = np.concatenate([*player_encodings, ball_data, prev_action_enc, *pads_encoding, game_state], dtype=np.float32)
        return obs

    def update_boostpads(self, pads):

        mask = pads == 1.0
        not_mask = np.logical_not(mask)

        self.boost_pads_timers[mask] = 0.0
        self.boost_pads_timers[not_mask] += self.time_diff_tick

    def update_demo_timers(self, state):

        for p in state.players:

            if p.is_demoed:
                self.demo_timers.update({p.car_id: self.demo_timers.get(p.car_id) + self.time_diff_tick})
            else:
                self.demo_timers.update({p.car_id: 0})


class SeerActionV6:
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, 0])
        actions = np.array(actions)
        return actions

    # def get_action_space(self) -> gym.spaces.Space:
    #     return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state) -> np.ndarray:
        return self._lookup_table[actions]


class AgentV6:
    def __init__(self, filename):
        self.filename = filename
        print("{} Loading...".format(self.filename))

        torch.set_num_threads(2)
        self.policy = SeerNetworkV6("cpu")

        state_dict = torch.load(self.filename, map_location=torch.device('cpu'), weights_only=True)
        consume_prefix_in_state_dict_if_present(state_dict, "module.")
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        print("Ready: {}".format(self.filename))

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state)
            action = self.policy.predict_actions(state, True)
        return action.numpy()


class SeerMaskCreatorV7:

    def __init__(self, device):
        self.jump_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                       1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.,
                                       1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                       1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.], dtype=torch.bool, device=device, requires_grad=False)

        self.pitch_roll_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.bool, device=device, requires_grad=False)

        self.boost_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.,
                                        1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                        1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                        1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.], dtype=torch.bool, device=device, requires_grad=False)

        self.handbrake_mask = torch.tensor([1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.bool, device=device, requires_grad=False)

        self.ones = torch.ones(1, dtype=torch.bool,
                               device=device, requires_grad=False)

    def create_mask(self, obs):
        has_boost = obs[:, 13] > 0.0
        on_ground = obs[:, 14] == 1
        has_flip = obs[:, 15] == 1
        in_air = torch.logical_not(on_ground)

        with torch.no_grad():
            mask_0 = torch.where(has_boost.unsqueeze(1),
                                 self.ones, self.boost_mask)
            mask_1 = torch.where(has_flip.unsqueeze(1),
                                 self.ones, self.jump_mask)
            mask_2 = torch.where(in_air.unsqueeze(
                1), self.ones, self.pitch_roll_mask)
            mask_3 = torch.where(on_ground.unsqueeze(
                1), self.ones, self.handbrake_mask)

            res = (mask_0 * mask_1) * (mask_2 * mask_3)

        return res


class ResBlockV7(nn.Module):
    def __init__(self, size, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer0 = nn.Linear(size, size)
        self.layer1 = nn.Linear(size, size)
        self.activation = activation

    def forward(self, x):
        out = self.layer0(x)
        out = self.activation(out)
        out = self.layer1(out)
        out += x
        out = self.activation(out)
        return out


class SeerNetworkV7(nn.Module):
    def __init__(self, device):
        super(SeerNetworkV7, self).__init__()
        self.activation = nn.LeakyReLU(inplace=True)

        self.OBS_SIZE = 215

        self.ENCODER_INTERMEDIATE_SIZE = 1024

        self.mask_creator = SeerMaskCreatorV7(device)

        self.encoder = nn.Sequential(
            nn.Linear(self.OBS_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            ResBlockV7(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlockV7(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
        ).to(device)

        self.value_network = nn.Sequential(
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE,
                      self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            ResBlockV7(self.ENCODER_INTERMEDIATE_SIZE // 2, self.activation),
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, 1),
        ).to(device)

        self.policy_network = nn.Sequential(
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE,
                      self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            ResBlockV7(self.ENCODER_INTERMEDIATE_SIZE // 2, self.activation),
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, 90),
        ).to(device)

        self.distribution = CategoricalDistribution(90)

        self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32, device=device)

    def encode(self, obs):

        if obs.shape[-1] == 215:
            return self.encoder(obs)

    def forward(self, obs, deterministic):
        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        mask = self.mask_creator.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        log_prob = self.distribution.log_prob(actions)
        return actions, value, log_prob,

    def predict_value(self, obs):
        x = self.encode(obs)
        value = self.value_network(x)
        return value

    def predict_actions(self, obs, deterministic):
        x = self.encode(obs)

        policy_logits = self.policy_network(x)
        mask = self.mask_creator.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs, actions, mask):
        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        log_prob = self.distribution.log_prob(actions)

        entropy = self.distribution.entropy()

        return value, log_prob, entropy


def get_encoded_actionV7(action: np.ndarray) -> np.ndarray:
    # throttle, steer, pitch, yaw, roll, jump, boost, handbrake

    action_encoding = np.zeros(15, dtype=np.float32)

    acc = 0
    throttle_index = action[0] + 1 + acc
    acc += 3
    steer_yaw_index = action[1] + 1 + acc
    acc += 3
    pitch_index = action[2] + 1 + acc
    acc += 3
    roll_index = action[4] + 1 + acc

    action_encoding[int(throttle_index)] = 1.0
    action_encoding[int(steer_yaw_index)] = 1.0
    action_encoding[int(pitch_index)] = 1.0
    action_encoding[int(roll_index)] = 1.0

    action_encoding[12] = action[5]
    action_encoding[13] = action[6]
    action_encoding[14] = action[7]

    return action_encoding


def _encode_rotationV7(rot):
    tol = 1e-4
    if rot > math.pi:
        rot -= 2 * math.pi
    return rot


def encode_all_playersV7(player, state, inverted, demo_timers, ball, time_since_jumped):
    player_encoding = _encode_playerV7(player, inverted, demo_timers.get(
        player.car_id), ball, player, time_since_jumped.get(player.car_id))

    same_team = []
    opponent_team = []

    for p in state.players:

        if p == player:
            continue

        if p.team_num == player.team_num:
            same_team.append(p)
        else:
            opponent_team.append(p)

    same_team = sorted(same_team, key=lambda x: np.linalg.norm(
        x.car_data.position - player.car_data.position))
    opponent_team = sorted(opponent_team, key=lambda x: np.linalg.norm(
        x.car_data.position - player.car_data.position))

    same_team_encoding = []
    opponent_team_encoding = []

    for p in same_team[:1]:
        same_team_encoding.append(_encode_playerV7(p, inverted, demo_timers.get(
            p.car_id), ball, player, time_since_jumped.get(p.car_id)))

    for p in opponent_team[:2]:
        opponent_team_encoding.append(_encode_playerV7(p, inverted, demo_timers.get(
            p.car_id), ball, player, time_since_jumped.get(p.car_id)))

    # while len(same_team_encoding) < 2:
    #     same_team_encoding.append(np.zeros(28, dtype=np.float32))

    # while len(opponent_team_encoding) < 3:
    #     opponent_team_encoding.append(np.zeros(28, dtype=np.float32))

    encodings = [player_encoding] + same_team_encoding + opponent_team_encoding

    return encodings


def _encode_playerV7(player, inverted: bool, demo_timer: float, ball, main_player, time_since_ground: float):
    if inverted:
        player_car = player.inverted_car_data
        main_player_car = main_player.inverted_car_data
    else:
        player_car = player.car_data
        main_player_car = main_player.car_data

    vel_norm = np.linalg.norm(np.array([player_car.linear_velocity[0],
                                        player_car.linear_velocity[1],
                                        player_car.linear_velocity[2]]))

    ball_diff_x = ball.position[0] - player_car.position[0]
    ball_diff_y = ball.position[1] - player_car.position[1]
    ball_diff_z = ball.position[2] - player_car.position[2]
    ball_diff_norm = np.linalg.norm(
        (np.array([ball_diff_x, ball_diff_y, ball_diff_z])))

    main_player_diff_x = main_player_car.position[0] - player_car.position[0]
    main_player_diff_y = main_player_car.position[1] - player_car.position[1]
    main_player_diff_z = main_player_car.position[2] - player_car.position[2]
    main_player_diff_norm = np.linalg.norm(
        np.array([main_player_diff_x, main_player_diff_y, main_player_diff_z]))

    pitch = _encode_rotationV7(player_car.pitch())
    yaw = _encode_rotationV7(player_car.yaw())
    roll = _encode_rotationV7(player_car.roll())

    array = np.array([
        player_car.position[0] * (1.0 / 4096.0),
        player_car.position[1] * (1.0 / 5120.0),
        player_car.position[2] * (1.0 / 2048.0),
        pitch * (1.0 / math.pi),
        yaw * (1.0 / math.pi),
        roll * (1.0 / math.pi),
        player_car.linear_velocity[0] * (1.0 / 2300.0),
        player_car.linear_velocity[1] * (1.0 / 2300.0),
        player_car.linear_velocity[2] * (1.0 / 2300.0),
        player_car.angular_velocity[0] * (1.0 / 5.5),
        player_car.angular_velocity[1] * (1.0 / 5.5),
        player_car.angular_velocity[2] * (1.0 / 5.5),
        demo_timer * (1 / 3.0),
        player.boost_amount,
        player.on_ground,
        player.has_flip,
        demo_timer > 0,
        vel_norm * (1.0 / 2300.0),
        vel_norm > 2200,
        ball_diff_x * (1.0 / (4096.0 * 2.0)),
        ball_diff_y * (1.0 / (5120.0 * 2.0)),
        ball_diff_z * (1.0 / 2048.0),
        ball_diff_norm * (1.0 / 13272.55),
        player.has_jump,
        main_player_diff_x * (1.0 / (4096.0 * 2.0)),
        main_player_diff_y * (1.0 / (5120.0 * 2.0)),
        main_player_diff_z * (1.0 / 2048.0),
        main_player_diff_norm * (1.0 / 13272.55),
        time_since_ground * (1.0 / 5.0)
    ], dtype=np.float32)


    return array


def _encode_ballV7(ball):
    state = np.array([
        ball.position[0] * (1.0 / 4096.0),
        ball.position[1] * (1.0 / 5120.0),
        ball.position[2] * (1.0 / 2048.0),
        ball.linear_velocity[0] * (1.0 / 6000.0),
        ball.linear_velocity[1] * (1.0 / 6000.0),
        ball.linear_velocity[2] * (1.0 / 6000.0),
        ball.angular_velocity[0] * (1.0 / 6.0),
        ball.angular_velocity[1] * (1.0 / 6.0),
        ball.angular_velocity[2] * (1.0 / 6.0),
        np.linalg.norm(np.array(
            [ball.linear_velocity[0], ball.linear_velocity[1], ball.linear_velocity[2]])) * (1.0 / 6000.0),
    ], dtype=np.float32)


    return state


pads_scalerV7 = np.array([
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
], dtype=np.float32)


def encode_boost_padsV7(boost_pads_timers):
    pads_active = boost_pads_timers == 0.0

    timers = boost_pads_timers * pads_scalerV7

    return [pads_active, timers]


def encode_game_stateV7(state, condition, inverted):
    dif, timer, overtime = condition.score, condition.timer, condition.overtime

    is_ones = 0
    is_twos = 0
    is_threes = 0

    if len(state.players) == 2:
        is_ones = 1
    elif len(state.players) == 4:
        is_twos = 1
    elif len(state.players) == 6:
        is_threes = 1
    else:
        raise NotImplementedError(len(state.players))

    if inverted:
        dif *= -1

    game_state = [is_ones, is_twos, is_threes, np.clip(dif / 10, -0.5, 0.5), timer / 300, overtime]
    return game_state


class SeerObsV7:
    def __init__(self, condition, default_tick_skip=8.0, physics_ticks_per_second=120.0):
        super().__init__()
        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}
        self.time_diff_tick = default_tick_skip / physics_ticks_per_second
        self.condition = condition
        self.time_since_jumped = {}

    def reset(self, initial_state: GameState):

        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}
        self.time_since_jumped = {}
        for p in initial_state.players:
            self.demo_timers.update({p.car_id: 0})
            self.time_since_jumped.update({p.car_id: 0})

    def pre_step(self, state: GameState):
        self.update_boostpads(state.boost_pads)
        self.update_demo_timers(state)
        self.update_time_since_jumped(state)

    def build_obs(self, player, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == 1:
            inverted = True
            ball = state.inverted_ball
            pads = self.boost_pads_timers[::-1]

        else:
            inverted = False
            ball = state.ball
            pads = self.boost_pads_timers

        ball_data = _encode_ballV7(ball)
        player_encodings = encode_all_playersV7(
            player, state, inverted, self.demo_timers, ball, self.time_since_jumped)
        prev_action_enc = get_encoded_actionV7(previous_action)
        pads_encoding = encode_boost_padsV7(pads)
        game_state = encode_game_stateV7(state, self.condition, inverted)
        obs = np.concatenate([*player_encodings, ball_data, prev_action_enc,
                             *pads_encoding, game_state], dtype=np.float32)
        return obs

    def update_boostpads(self, pads):

        mask = pads == 1.0
        not_mask = np.logical_not(mask)

        self.boost_pads_timers[mask] = 0.0
        self.boost_pads_timers[not_mask] += self.time_diff_tick

    def update_demo_timers(self, state: GameState):

        for p in state.players:

            if p.is_demoed:
                self.demo_timers.update(
                    {p.car_id: self.demo_timers.get(p.car_id) + self.time_diff_tick})
            else:
                self.demo_timers.update({p.car_id: 0})

    def update_time_since_jumped(self, state: GameState):
        for p in state.players:
            if p.on_ground:
                self.time_since_jumped.update({p.car_id: 0})
            else:
                if not p.car_id in self.time_since_jumped:
                    self.time_since_jumped.update({p.car_id: 0})
                self.time_since_jumped.update(
                    {p.car_id: min(5.0, self.time_since_jumped.get(p.car_id) + self.time_diff_tick)})


class SeerActionV7:
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append(
                            [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            actions.append(
                                [boost, yaw, pitch, yaw, roll, jump, boost, 0])
        actions = np.array(actions)
        return actions

    # def get_action_space(self) -> gym.spaces.Space:
    #     return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        return self._lookup_table[actions]


class AgentV7:
    def __init__(self, filename):
        self.filename = filename
        print("{} Loading...".format(self.filename))

        torch.set_num_threads(2)
        self.policy = SeerNetworkV7("cpu")

        state_dict = torch.load(self.filename, map_location=torch.device('cpu'), weights_only=True)
        consume_prefix_in_state_dict_if_present(state_dict, "module.")
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        print("Ready: {}".format(self.filename))

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state)
            action = self.policy.predict_actions(state, True)
        return action.numpy()


def _vector_to_numpy(vector):
    return np.asarray([vector.x, vector.y, vector.z])


class SeerMaskCreatorV8:

    def __init__(self, device):
        super(SeerMaskCreatorV8, self).__init__()
        self.jump_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                       1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.,
                                       1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                       1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1.], dtype=torch.bool, device=device, requires_grad=False)

        self.pitch_roll_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.bool, device=device, requires_grad=False)

        self.boost_mask = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0.,
                                        1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                        1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                        1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.], dtype=torch.bool, device=device, requires_grad=False)

        self.handbrake_mask = torch.tensor([1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
                                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], dtype=torch.bool, device=device, requires_grad=False)

        self.demo = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.bool, device=device, requires_grad=False)

        self.ones = torch.ones(1, dtype=torch.bool, device=device, requires_grad=False)

    def create_mask(self, obs):
        has_boost = obs[:, 13] > 0.0
        on_ground = obs[:, 14] == 1
        has_flip = obs[:, 15] == 1
        is_alive = obs[:, 16] == 0
        in_air = torch.logical_not(on_ground)

        with torch.no_grad():
            mask_0 = torch.where(has_boost.unsqueeze(1), self.ones, self.boost_mask)
            mask_1 = torch.where(has_flip.unsqueeze(1), self.ones, self.jump_mask)
            mask_2 = torch.where(in_air.unsqueeze(1), self.ones, self.pitch_roll_mask)
            mask_3 = torch.where(on_ground.unsqueeze(1), self.ones, self.handbrake_mask)
            mask_4 = torch.where(is_alive.unsqueeze(1), self.ones, self.demo)

            res = ((mask_0 * mask_1) * (mask_2 * mask_3)) * mask_4

        return res


class ResBlockV8(nn.Module):
    def __init__(self, size, activation, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer0 = nn.Linear(size, size)
        self.layer1 = nn.Linear(size, size)
        self.activation = activation

    def forward(self, x):
        out = self.layer0(x)
        out = self.activation(out)
        out = self.layer1(out)
        out += x
        out = self.activation(out)
        return out


class SeerNetworkV8(nn.Module):
    def __init__(self, device):
        super(SeerNetworkV8, self).__init__()
        self.activation = nn.LeakyReLU(inplace=True)

        self.OBS_SIZE = 306

        self.ENCODER_INTERMEDIATE_SIZE = 1024

        self.mask_creator = SeerMaskCreatorV8(device)

        self.encoder = nn.Sequential(
            nn.Linear(self.OBS_SIZE, self.ENCODER_INTERMEDIATE_SIZE),
            self.activation,
            ResBlockV8(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlockV8(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
        ).to(device)

        self.value_network = nn.Sequential(
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            ResBlockV8(self.ENCODER_INTERMEDIATE_SIZE // 2, self.activation),
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, 1),
        ).to(device)

        self.policy_network = nn.Sequential(
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, self.ENCODER_INTERMEDIATE_SIZE // 2),
            self.activation,
            ResBlockV8(self.ENCODER_INTERMEDIATE_SIZE // 2, self.activation),
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE // 2, 90),
        ).to(device)

        self.distribution = CategoricalDistribution(90)

        self.HUGE_NEG = torch.tensor(-1e8, dtype=torch.float32, device=device)

    def encode(self, obs):

        if obs.shape[-1] == 306:
            return self.encoder(obs)

    def forward(self, obs, deterministic):
        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        mask = self.mask_creator.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        log_prob = self.distribution.log_prob(actions)
        return actions, value, log_prob,

    def predict_value(self, obs):
        x = self.encode(obs)
        value = self.value_network(x)
        return value

    def predict_actions(self, obs, deterministic):
        x = self.encode(obs)

        policy_logits = self.policy_network(x)
        mask = self.mask_creator.create_mask(obs)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        self.distribution.apply_mask(mask)

        actions = self.distribution.get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs, actions, mask):
        x = self.encode(obs)

        value = self.value_network(x)
        policy_logits = self.policy_network(x)
        policy_logits = torch.where(mask, policy_logits, self.HUGE_NEG)
        self.distribution.proba_distribution(policy_logits)
        log_prob = self.distribution.log_prob(actions)

        entropy = self.distribution.entropy()

        return value, log_prob, entropy


class SeerGameConditionV8:  # Mimics a Rocket League game
    def __init__(self, tick_skip=4, overtime_prob=0.1, no_touch_timeout=512):
        super().__init__()
        self.tick_skip = tick_skip
        self.timer = 0
        self.overtime = False
        self.done = True
        self.initial_state = None
        self.score = 0
        self.overtime_prob = overtime_prob
        self.last_touch = 0
        # self.no_touch_timeout = NoTouchTimeoutCondition(no_touch_timeout)
        self.scored = 0.0
        self.ball_ground_flag = False

    def reset(self, initial_state: GameState):
        self.initial_state = initial_state
        self.done = False
        # self.no_touch_timeout.reset(initial_state)
        self.scored = 0.0

        if random.random() < self.overtime_prob:
            self.overtime = True
            self.timer = 0
            self.score = 0
        else:
            self.overtime = False
            self.timer = random.uniform(5, 300)
            self.score = random.choices([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], weights=[1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], k=1)[0]

    def is_terminal(self, current_state: GameState) -> bool:

        # Setter reset if ball on ground
        if self.ball_ground_flag and current_state.ball.position[2] <= 110:
            return True

        reset = False

        self.scored = ((current_state.blue_score - self.initial_state.blue_score)
                       - (current_state.orange_score - self.initial_state.orange_score))

        self.score += self.scored

        if self.scored != 0:  # Goal scored
            reset = True

        if self.overtime:
            if self.scored != 0:  # Overtime
                self.done = True

        else:
            if self.timer <= 0 and current_state.ball.position[2] <= 110:
                reset = True

                if self.score != 0:
                    self.done = True

        self.timer -= self.tick_skip / 120
        self.timer = max(0, self.timer)

        # if self.no_touch_timeout.is_terminal(current_state):
        #     reset = True

        return reset or self.done


def get_encoded_actionV8(action: np.ndarray) -> np.ndarray:
    # throttle, steer, pitch, yaw, roll, jump, boost, handbrake

    action_encoding = np.zeros(15, dtype=np.float32)

    acc = 0
    throttle_index = action[0] + 1 + acc
    acc += 3
    steer_yaw_index = action[1] + 1 + acc
    acc += 3
    pitch_index = action[2] + 1 + acc
    acc += 3
    roll_index = action[4] + 1 + acc

    action_encoding[int(throttle_index)] = 1.0
    action_encoding[int(steer_yaw_index)] = 1.0
    action_encoding[int(pitch_index)] = 1.0
    action_encoding[int(roll_index)] = 1.0

    action_encoding[12] = action[5]
    action_encoding[13] = action[6]
    action_encoding[14] = action[7]

    return action_encoding


def _encode_rotationV8(rot):
    tol = 1e-4
    if rot > math.pi:
        rot -= 2 * math.pi
    return rot


def encode_all_playersV8(player, state, inverted, demo_timers, ball, time_since_jumped):
    player_encoding = _encode_playerV8(player, inverted, demo_timers.get(player.car_id), ball, player, time_since_jumped.get(player.car_id))

    same_team = []
    opponent_team = []

    for p in state.players:

        if p == player:
            continue

        if p.team_num == player.team_num:
            same_team.append(p)
        else:
            opponent_team.append(p)

    same_team = sorted(same_team, key=lambda x: np.linalg.norm(x.car_data.position - player.car_data.position))
    opponent_team = sorted(opponent_team, key=lambda x: np.linalg.norm(x.car_data.position - player.car_data.position))

    same_team_encoding = []
    opponent_team_encoding = []

    for p in same_team:
        same_team_encoding.append(_encode_playerV8(p, inverted, demo_timers.get(p.car_id), ball, player, time_since_jumped.get(p.car_id)))

    for p in opponent_team:
        opponent_team_encoding.append(_encode_playerV8(p, inverted, demo_timers.get(p.car_id), ball, player, time_since_jumped.get(p.car_id)))

    # while len(same_team_encoding) < 2:
    #     same_team_encoding.append(np.zeros(28, dtype=np.float32))

    # while len(opponent_team_encoding) < 3:
    #     opponent_team_encoding.append(np.zeros(28, dtype=np.float32))

    encodings = [player_encoding] + same_team_encoding + opponent_team_encoding

    return encodings


def _encode_playerV8(player, inverted: bool, demo_timer: float, ball, main_player, time_since_ground: float):
    if inverted:
        player_car = player.inverted_car_data
        main_player_car = main_player.inverted_car_data
    else:
        player_car = player.car_data
        main_player_car = main_player.car_data

    vel_norm = np.linalg.norm(player_car.linear_velocity)
    ball_dif = ball.position - player_car.position
    ball_diff_norm = np.linalg.norm(ball_dif)

    main_player_diff = main_player_car.position - player_car.position
    main_player_diff_norm = np.linalg.norm(main_player_diff)

    pitch = _encode_rotationV8(player_car.pitch())
    yaw = _encode_rotationV8(player_car.yaw())
    roll = _encode_rotationV8(player_car.roll())

    if not player.is_demoed:

        array = np.array([
            player_car.position[0] * (1.0 / 4096.0),
            player_car.position[1] * (1.0 / 5120.0),
            player_car.position[2] * (1.0 / 2048.0),
            pitch * (1.0 / math.pi),
            yaw * (1.0 / math.pi),
            roll * (1.0 / math.pi),
            player_car.linear_velocity[0] * (1.0 / 2300.0),
            player_car.linear_velocity[1] * (1.0 / 2300.0),
            player_car.linear_velocity[2] * (1.0 / 2300.0),
            player_car.angular_velocity[0] * (1.0 / 5.5),
            player_car.angular_velocity[1] * (1.0 / 5.5),
            player_car.angular_velocity[2] * (1.0 / 5.5),
            0,  # demo timer
            player.boost_amount,
            player.on_ground,
            player.has_flip,
            player.is_demoed,
            vel_norm * (1.0 / 2300.0),
            vel_norm > 2200,
            ball_dif[0] * (1.0 / (4096.0 * 2.0)),
            ball_dif[1] * (1.0 / (5120.0 * 2.0)),
            ball_dif[2] * (1.0 / 2048.0),
            ball_diff_norm * (1.0 / 13272.55),
            player.has_jump,
            main_player_diff[0] * (1.0 / (4096.0 * 2.0)),
            main_player_diff[1] * (1.0 / (5120.0 * 2.0)),
            main_player_diff[2] * (1.0 / 2048.0),
            main_player_diff_norm * (1.0 / 13272.55),
            time_since_ground * (1.0 / 5.0)
        ], dtype=np.float32)

    else:
        array = np.array([
            0,  # position[0]
            0,  # position[1]
            0,  # position[2]
            0,  # pitch
            0,  # yaw
            0,  # roll
            0,  # linear_velocity[0]
            0,  # linear_velocity[1]
            0,  # linear_velocity[2]
            0,  # angular_velocity[0]
            0,  # angular_velocity[1]
            0,  # angular_velocity[2]
            max(1.0 - demo_timer * (1 / 3.0), 0.0),
            0,  # player.boost_amount
            0,  # on_ground
            0,  # has_fip
            player.is_demoed,
            0,  # vel_norm
            0,  # vel_norm > 0
            0,  # ball_diff_x
            0,  # ball_diff_y
            0,  # ball_diff_z
            0,  # ball_diff_norm
            0,  # has_jump
            0,  # main_player_diff_x
            0,  # main_player_diff_y
            0,  # main_player_diff_z
            0,  # main_player_diff_norm
            0  # time_since_ground
        ], dtype=np.float32)


    return array


def _encode_ballV8(ball):
    state = np.array([
        ball.position[0] * (1.0 / 4096.0),
        ball.position[1] * (1.0 / 5120.0),
        ball.position[2] * (1.0 / 2048.0),
        ball.linear_velocity[0] * (1.0 / 6000.0),
        ball.linear_velocity[1] * (1.0 / 6000.0),
        ball.linear_velocity[2] * (1.0 / 6000.0),
        ball.angular_velocity[0] * (1.0 / 6.0),
        ball.angular_velocity[1] * (1.0 / 6.0),
        ball.angular_velocity[2] * (1.0 / 6.0),
        np.linalg.norm(ball.linear_velocity) * (1.0 / 6000.0),
    ], dtype=np.float32)


    return state


pads_scaler_V8 = np.array([
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 10.0,
    1.0 / 10.0,
    1.0 / 4.0,
    1.0 / 4.0,
    1.0 / 4.0,
], dtype=np.float32)


def encode_boost_padsV8(boost_pads_timers):
    pads_active = boost_pads_timers == 0.0

    timers = boost_pads_timers * pads_scaler_V8

    return [pads_active, timers]


def encode_game_stateV8(state, condition, inverted):
    dif, timer, overtime = condition.score, condition.timer, condition.overtime

    # is_ones = 0
    # is_twos = 0
    # is_threes = 0

    # if len(state.players) == 2:
    #     is_ones = 1
    # elif len(state.players) == 4:
    #     is_twos = 1
    # elif len(state.players) == 6:
    #     is_threes = 1
    # else:
    #     raise NotImplementedError(len(state.players))

    if inverted:
        dif *= -1

    # game_state = [is_ones, is_twos, is_threes, np.clip(dif / 10,-0.5, 0.5), timer / 300, overtime]
    game_state = [np.clip(dif / 10, -0.5, 0.5), timer / 300, overtime]

    return game_state


def make_ball_predicitionV8(ball, arena, times):
    result = np.empty([len(times) * 6], dtype=np.float32)

    ball_state = arena.ball.get_state()
    ball_state.pos = rsim.Vec(ball.position[0], ball.position[1], ball.position[2])
    ball_state.vel = rsim.Vec(ball.linear_velocity[0], ball.linear_velocity[1], ball.linear_velocity[2])
    ball_state.ang_vel = rsim.Vec(ball.angular_velocity[0], ball.angular_velocity[1], ball.angular_velocity[2])
    arena.ball.set_state(ball_state)

    for i, t in enumerate(times):

        if i == 0:
            ticks = t * 120
        else:
            ticks = (t - times[i - 1]) * 120

        arena.step(int(ticks))

        ball_state2 = arena.ball.get_state()

        result[i * 6 + 0] = ball_state2.pos[0] * (1.0 / 4096.0)
        result[i * 6 + 1] = ball_state2.pos[1] * (1.0 / 5120.0)
        result[i * 6 + 2] = ball_state2.pos[2] * (1.0 / 2048.0)
        result[i * 6 + 3] = ball_state2.vel[0] * (1.0 / 6000.0)
        result[i * 6 + 4] = ball_state2.vel[1] * (1.0 / 6000.0)
        result[i * 6 + 5] = ball_state2.vel[2] * (1.0 / 6000.0)


    return result


class SeerObsV8:
    def __init__(self, condition, default_tick_skip=8.0, physics_ticks_per_second=120.0):


        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}
        self.time_diff_tick = default_tick_skip / physics_ticks_per_second
        self.condition = condition
        self.time_since_jumped = {}
        self.arena = rsim.Arena(rsim.GameMode.SOCCAR)
        self.times = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        self.ball_prediction = None
        self.ball_prediction_inv = None
        self.ball_pred_inverter = np.array([-1, -1, 1, -1, -1, 1] * len(self.times), dtype=np.float32)
        self.pads_encoding = None
        self.pads_encoding_inv = None
        self.game_state = None
        self.game_state_inv = None
        self.ball_data = None
        self.ball_data_inv = None

    def reset(self, initial_state: GameState):

        self.boost_pads_timers = np.zeros(34, dtype=np.float32)
        self.demo_timers = {}
        self.time_since_jumped = {}
        for p in initial_state.players:
            self.demo_timers.update({p.car_id: 0})
            self.time_since_jumped.update({p.car_id: 0})

    def pre_step(self, state: GameState):
        self.update_boostpads(state.boost_pads)
        self.update_demo_timers(state)
        self.update_time_since_jumped(state)

        # Ball Predicition
        self.ball_prediction = make_ball_predicitionV8(state.ball, self.arena, self.times)
        self.ball_prediction_inv = self.ball_prediction * self.ball_pred_inverter

        # Pads
        self.pads_encoding = encode_boost_padsV8(self.boost_pads_timers)
        self.pads_encoding_inv = encode_boost_padsV8(self.boost_pads_timers[::-1])

        # Game state
        self.game_state = encode_game_stateV8(state, self.condition, False)
        self.game_state_inv = encode_game_stateV8(state, self.condition, True)

        # Ball
        self.ball_data = _encode_ballV8(state.ball)
        self.ball_data_inv = _encode_ballV8(state.inverted_ball)

        # has_flip fix
        for player in state.players:
            if player.has_flip and not player.has_jump and self.time_since_jumped[player.car_id] > 1.45:
                player.has_flip = False

    def build_obs(self, player, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == 1:
            inverted = True
            ball_data = self.ball_data_inv
            pads_encoding = self.pads_encoding_inv
            ball_prediction = self.ball_prediction_inv
            game_state = self.game_state_inv
            ball = state.inverted_ball
        else:
            inverted = False
            ball_data = self.ball_data
            pads_encoding = self.pads_encoding
            ball_prediction = self.ball_prediction
            game_state = self.game_state
            ball = state.ball

        player_encodings = encode_all_playersV8(player, state, inverted, self.demo_timers, ball, self.time_since_jumped)
        prev_action_enc = get_encoded_actionV8(previous_action)
        obs = np.concatenate([*player_encodings, ball_data, prev_action_enc, *pads_encoding, game_state, ball_prediction], dtype=np.float32)
        return obs

    def update_boostpads(self, pads):

        mask = pads == 1.0
        not_mask = np.logical_not(mask)

        self.boost_pads_timers[mask] = 0.0
        self.boost_pads_timers[not_mask] += self.time_diff_tick

    def update_demo_timers(self, state: GameState):

        for p in state.players:

            if p.is_demoed:
                self.demo_timers.update({p.car_id: self.demo_timers.get(p.car_id) + self.time_diff_tick})
            else:
                self.demo_timers.update({p.car_id: 0})

    def update_time_since_jumped(self, state: GameState):
        for p in state.players:
            if p.on_ground:
                self.time_since_jumped.update({p.car_id: 0})
            else:
                if p.car_id not in self.time_since_jumped:
                    self.time_since_jumped.update({p.car_id: 0})
                self.time_since_jumped.update({p.car_id: min(5.0, self.time_since_jumped.get(p.car_id) + self.time_diff_tick)})


class SeerActionV8:
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            # Note to self: This handbranke is not reflected in the handbrake mask
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    # def get_action_space(self) -> gym.spaces.Space:
    #     return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        return self._lookup_table[actions]


class AgentV8:
    def __init__(self, filename):
        self.filename = filename
        print("{} Loading...".format(self.filename))

        torch.set_num_threads(2)
        self.policy = SeerNetworkV8("cpu")

        state_dict = torch.load(self.filename, map_location=torch.device('cpu'), weights_only=True)
        consume_prefix_in_state_dict_if_present(state_dict, "module.")
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

        print("Ready: {}".format(self.filename))

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state)
            action = self.policy.predict_actions(state, True)
        return action.numpy()


class MyBot(Hivemind):
    def __init__(self, default_agent_id: str | None = None):
        super().__init__(default_agent_id)

        self.condition = SeerGameCondition()

        self.obs_builderV7 = [SeerObsV7(self.condition), SeerObsV7(self.condition)]
        self.act_parserV7 = SeerActionV7()
        self.agentV7 = AgentV7(model_path("43200.p"))

        self.obs_builderV8 = [SeerObsV8(self.condition), SeerObsV8(self.condition), SeerObsV8(self.condition), ]
        self.act_parserV8 = SeerActionV8()
        self.agentV8 = AgentV8(model_path("11100.p"))

        self.obs_builderV4 = [SeerObsV4(1, self.condition), SeerObsV4(1, self.condition)]
        self.act_parserV4 = SeerActionV4()
        self.agentV4 = AgentV4(model_path("36000.p"))

        self.tick_skip = 8

        self.game_state = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        self.action = None
        self.controls = None
        self.obs_builder = None
        print("Creating")

    def initialize(self):
        print("Initializing")
        # Set up information about the boost pads now that the game is active and the info is available

        self.game_state = V1GameState(self.field_info, tick_skip=self.tick_skip)
        self.actions = np.zeros((len(self.indices), 8))
        self.controls = [ControllerState() for _ in self.indices]

    def get_outputs(self, packet: GamePacket) -> dict[int, ControllerState]:

        cur_time = packet.match_info.seconds_elapsed
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = round(delta * 120)
        self.ticks += ticks_elapsed
        self.game_state.update(packet)

        if self.update_action:

            self.update_action = False
            
            obs_array = []

            for i, index in enumerate(self.indices):

                player = self.game_state.players[index]

                if packet.match_info.is_unlimited_time:
                    self.condition.overtime = False
                    self.condition.timer = 250
                    self.condition.score = 0
                else:
                    self.condition.overtime = packet.match_info.is_overtime
                    self.condition.timer = max(packet.match_info.game_time_remaining, 0)
                    self.condition.score = packet.teams[0].score - packet.teams[1].score

                if len(self.game_state.players) == 2:
                    self.obs_builderV4[i].pre_step(self.game_state)
                    obs = self.obs_builderV4[i].build_obs(player, self.game_state, self.actions[i])
                    obs_array.append(obs)

                elif len(self.game_state.players) == 4:
                    self.obs_builderV7[i].pre_step(self.game_state)
                    obs = self.obs_builderV7[i].build_obs(player, self.game_state, self.actions[i])
                    obs_array.append(obs)
                elif len(self.game_state.players) == 6:
                    self.obs_builderV8[i].pre_step(self.game_state)
                    obs = self.obs_builderV8[i].build_obs(player, self.game_state, self.actions[i])
                    obs_array.append(obs)
                else:
                    pass
                
                
            obs_array = np.array(obs_array, dtype=np.float32) 
            
            if len(self.game_state.players) == 2:
                actions = self.agentV4.act(obs_array)
                self.actions = self.act_parserV4.parse_actions(actions, self.game_state)
            elif len(self.game_state.players) == 4:
                actions = self.agentV7.act(obs_array)
                self.actions = self.act_parserV7.parse_actions(actions, self.game_state)
            elif len(self.game_state.players) == 6:
                actions = self.agentV8.act(obs_array)
                self.actions = self.act_parserV8.parse_actions(actions, self.game_state)
            else:
                pass
        
        if self.ticks >= self.tick_skip - 1:
            self.update_controls(self.actions)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_action = True

        return {index: self.controls[i] for i, index in enumerate(self.indices)}

    def update_controls(self, actions):
        for i, action in enumerate(actions):
            self.controls[i].throttle = action[0]
            self.controls[i].steer = action[1]
            self.controls[i].pitch = action[2]
            self.controls[i].yaw = 0 if action[5] > 0 else action[3]
            self.controls[i].roll = action[4]
            self.controls[i].jump = action[5] > 0
            self.controls[i].boost = action[6] > 0
            self.controls[i].handbrake = action[7] > 0


if __name__ == "__main__":
    MyBot("ultrawidegc/lebot").run()
