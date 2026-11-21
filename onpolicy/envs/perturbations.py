"""
Distribution-shift suite for the AAMAS rebuttal (experiments E2, E2b, E10).

Every shift is expressed as a vectorized-env wrapper that can also carry ``actor_hooks``:
small transforms the R_Actor applies *inside* its forward pass. The repair runner reads the
hooks off whichever env is active (``runner.eval_envs`` while measuring, ``runner.envs``
while repairing) and installs them on the actor, so baseline / clean-retention measurements
on the unwrapped env automatically run with no hooks.

Shift kinds
-----------
obs (default)   Partner-perception shift applied at the env level, as in the paper.
                ``mirror_scope`` in {all, partner, partner_full} reproduces MirrorObsVecEnv
                exactly. If ``perturb_angle`` (Task 8) is set, every relative-partner vector is
                rotated by that angle instead -- the same transform as
                mirror_wrapper.PartnerRotationObsVecEnv, routed through here so the decoupled
                cells work with rotations too (severity sweep, E10).
                With ``decouple``:
                  - 'msg_only'   : the actor trunk receives the *clean* observation (the env
                                   transform is undone inside the actor), so ONLY the message
                                   head sees the shifted partner positions.
                  - 'actor_only' : the message head receives the clean observation, so ONLY
                                   the actor trunk sees the shift; messages stay semantically
                                   intact.
                  - 'none'       : both see the shift (== the paper's partner_full).
comm_perm       Communication-only shift. Observations are untouched. Each sender's softmax
                weights over the V shared tokens are permuted by a fixed derangement before
                embedding (Eq. 1), so receivers get a scrambled mixture. The channel is a
                continuous mixture, so this is the analogue of a token permutation.
dynamics        Non-communication shift. Observations and messages are untouched; agent
                acceleration is scaled (and optionally damping changed) via a separately built
                env, so reward drops because control changed, not because communication broke.

Observation layout of simple_spread (dim_p = 2):
    [ self_vel(2), self_pos(2), landmark_pos(2*L), other_pos(2*(N-1)), comm(2*(N-1)) ]
"""

import numpy as np


# ----------------------------------------------------------------------------- pair specs
def spatial_pairs(num_agents, num_landmarks, dim_p=2, scope="partner_full"):
    """Return (pairs, M): list of (ix, iy) observation index pairs and the 2x2 matrix applied
    to each pair as [x', y']^T = M [x, y]^T."""
    if dim_p != 2:
        raise ValueError("pair transforms assume dim_p == 2")
    self_block = 2 * dim_p
    other_start = self_block + num_landmarks * dim_p
    other_end = other_start + (num_agents - 1) * dim_p
    partner = [(i, i + 1) for i in range(other_start, other_end, dim_p)]
    allsp = [(i, i + 1) for i in range(0, other_end, dim_p)]
    reflect_x = np.array([[-1.0, 0.0], [0.0, 1.0]])
    if scope == "all":
        return allsp, reflect_x
    if scope == "partner":
        return partner, reflect_x
    if scope == "partner_full":
        return partner, -np.eye(2)
    raise ValueError("Unknown scope: {}".format(scope))


def rotation(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def apply_pairs_np(obs, pairs, M):
    obs = np.array(obs, copy=True)
    if not pairs:
        return obs
    ix = np.array([p[0] for p in pairs])
    iy = np.array([p[1] for p in pairs])
    x, y = obs[..., ix].copy(), obs[..., iy].copy()
    obs[..., ix] = M[0, 0] * x + M[0, 1] * y
    obs[..., iy] = M[1, 0] * x + M[1, 1] * y
    return obs


def derangement(n, seed):
    """Fixed permutation of range(n) with no fixed points (every token is remapped)."""
    rng = np.random.RandomState(seed)
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


# ----------------------------------------------------------------------------- wrapper
class ShiftedVecEnv:
    """Vec-env wrapper: applies an observation pair-transform (possibly identity) and carries
    ``actor_hooks`` for the runner to install on the actor. Forwards everything else."""

    def __init__(self, venv, pairs=None, M=None, actor_hooks=None, label="shift"):
        self.venv = venv
        self.pairs = list(pairs) if pairs else []
        self.M = np.asarray(M, dtype=np.float64) if M is not None else np.eye(2)
        self.actor_hooks = actor_hooks or {}
        self.label = label

    def _t(self, obs):
        return apply_pairs_np(obs, self.pairs, self.M) if self.pairs else obs

    def reset(self, *args, **kwargs):
        return self._t(self.venv.reset(*args, **kwargs))

    def step(self, *args, **kwargs):
        obs, rews, dones, infos = self.venv.step(*args, **kwargs)
        return self._t(obs), rews, dones, infos

    def __getattr__(self, name):
        return getattr(self.venv, name)


def build_shift(venv, all_args, shifted_venv=None):
    """Wrap ``venv`` according to ``all_args.shift``. For 'dynamics' the caller passes
    ``shifted_venv`` (an env built with the altered physics); it is wrapped instead."""
    kind = getattr(all_args, "shift", "obs")
    if kind == "obs":
        scope = all_args.mirror_scope
        angle = getattr(all_args, "perturb_angle", None)
        if angle is not None:
            pairs, _ = spatial_pairs(all_args.num_agents, all_args.num_landmarks, scope="partner_full")
            M = rotation(angle)
            scope = "partner_rot{}".format(angle)
        else:
            pairs, M = spatial_pairs(all_args.num_agents, all_args.num_landmarks, scope=scope)
        Minv = np.linalg.inv(M)
        decouple = getattr(all_args, "decouple", "none")
        hooks = {}
        if decouple == "msg_only":
            hooks["trunk_pairs"] = (pairs, Minv)   # undo shift before the trunk
        elif decouple == "actor_only":
            hooks["msg_pairs"] = (pairs, Minv)     # undo shift before the message head
        elif decouple != "none":
            raise ValueError("Unknown decouple mode: {}".format(decouple))
        return ShiftedVecEnv(venv, pairs, M, hooks, label="obs:{}:{}".format(scope, decouple))
    if kind == "comm_perm":
        perm = derangement(5, all_args.comm_perm_seed)
        return ShiftedVecEnv(venv, None, None, {"token_perm": perm.tolist()},
                             label="comm_perm:{}".format(perm.tolist()))
    if kind == "dynamics":
        if shifted_venv is None:
            raise ValueError("dynamics shift needs an env built with the altered physics")
        return ShiftedVecEnv(shifted_venv, None, None, {}, label="dynamics:accel_x{}:damping{}".format(
            all_args.dyn_accel_scale, all_args.dyn_damping))
    raise ValueError("Unknown shift: {}".format(kind))


def describe_shift(all_args):
    kind = getattr(all_args, "shift", "obs")
    if kind == "obs":
        angle = getattr(all_args, "perturb_angle", None)
        scope = "partner rotation {} deg".format(angle) if angle is not None else all_args.mirror_scope
        return "obs shift scope={} decouple={}".format(scope, all_args.decouple)
    if kind == "comm_perm":
        return "comm-only token permutation {} (seed {})".format(
            derangement(5, all_args.comm_perm_seed).tolist(), all_args.comm_perm_seed)
    return "dynamics shift accel x{} damping={}".format(all_args.dyn_accel_scale, all_args.dyn_damping)
