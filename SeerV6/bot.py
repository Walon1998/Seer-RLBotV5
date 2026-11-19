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


try:
    import torch
except ImportError:
    sys.path.insert(0, "../../torch-archive")
    import torch


import os
from torch.distributions import Categorical
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch import nn
import random
import numpy as np
import math
from rlgym_compat import V1GameState
from typing import Any


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

    def proba_distribution(self, action_logits: torch.Tensor) -> "MultiCategoricalDistribution":
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


class ResBlock(nn.Module):
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
            ResBlock(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlock(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlock(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
        ).to(device)

        self.value_network = nn.Sequential(
            ResBlock(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlock(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            nn.Linear(self.ENCODER_INTERMEDIATE_SIZE, 1),
        ).to(device)

        self.policy_network = nn.Sequential(
            ResBlock(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlock(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
            ResBlock(self.ENCODER_INTERMEDIATE_SIZE, self.activation),
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


class SeerGameConditionV6:  # Mimics a Rocket League game
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
        self.ball_ground_flag = False

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
            self.timer = random.uniform(0, 300)
            self.score = random.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])

    def is_terminal(self, current_state) -> bool:

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
        # reset = True

        return reset or self.done


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


def _encode_rotation(rot):
    tol = 1e-4
    if rot > math.pi:
        rot -= 2 * math.pi
    return rot


def encode_all_players(player, state, inverted, demo_timers, ball):
    player_encoding = _encode_player(player, inverted, demo_timers.get(player.car_id), ball, player)

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
        same_team_encoding.append(_encode_player(p, inverted, demo_timers.get(p.car_id), ball, player))

    for p in opponent_team:
        opponent_team_encoding.append(_encode_player(p, inverted, demo_timers.get(p.car_id), ball, player))

    while len(same_team_encoding) < 2:
        same_team_encoding.append(np.zeros(28, dtype=np.float32))

    while len(opponent_team_encoding) < 3:
        opponent_team_encoding.append(np.zeros(28, dtype=np.float32))

    encodings = [player_encoding] + same_team_encoding + opponent_team_encoding

    return encodings


def _encode_player(player, inverted: bool, demo_timer: float, ball, main_player):
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

    pitch = _encode_rotation(player_car.pitch())
    yaw = _encode_rotation(player_car.yaw())
    roll = _encode_rotation(player_car.roll())

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


def _encode_ball(ball):
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


pads_scaler = np.array([
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


def encode_boost_pads(boost_pads_timers):
    pads_active = boost_pads_timers == 0.0

    timers = boost_pads_timers * pads_scaler

    return [pads_active, timers]


def encode_game_state(state, condition, inverted):
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

        ball_data = _encode_ball(ball)
        player_encodings = encode_all_players(player, state, inverted, self.demo_timers, ball)
        prev_action_enc = get_encoded_actionV6(previous_action)
        pads_encoding = encode_boost_pads(pads)
        game_state = encode_game_state(state, self.condition, inverted)
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


class Agent:
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


class MyBot(Hivemind):
    def __init__(self, default_agent_id: str | None = None):
        super().__init__(default_agent_id)

        self.condition = SeerGameConditionV6()
        self.act_parser = SeerActionV6()
        self.agent = Agent(model_path("32400.p"))
        self.tick_skip = 8

        self.game_state: V1GameState = None
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
        self.obs_builder = [SeerObsV6(self.condition) for _ in self.indices]
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

                self.condition.overtime = packet.match_info.is_overtime
                self.condition.timer = max(packet.match_info.game_time_remaining, 0)
                self.condition.score = packet.teams[0].score - packet.teams[1].score

                self.obs_builder[i].pre_step(self.game_state)
                obs = self.obs_builder[i].build_obs(
                    player, self.game_state, self.actions[i])
                obs_array.append(obs)

            obs_array = np.array(obs_array, dtype=np.float32)

            self.actions = self.act_parser.parse_actions(
                self.agent.act(obs_array), self.game_state)  # Dim is (N, 8)

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
    MyBot("ultrawidegc/seerv6_32400").run()
