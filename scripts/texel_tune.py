#!/usr/bin/env python3
"""Texel-tune the HCE eval weights from a feature dump.

Input: the file produced by `chess_uci`'s `tunedump` command, one line per
quiet position:

    <label> <phase> <eval_true> <white feature block> <black feature block>

where each base feature block is:
    mat_q mat_n mat_b mat_r mat_p isolated doubled
    mob_n mob_b mob_r mob_q rook_open rook_semi
    passed_mg passed_eg king_mg king_eg hanging queen_mg queen_eg
    pawn_pushes pawn_threat_minor pawn_threat_major
    connected_pawns phalanx_pawns backward_pawns knight_outposts
    bishop_pair rook_behind_passer
    minor_threat_pawn minor_threat_minor minor_threat_major
    rook_threat_minor safe_push_threat_minor safe_push_threat_major
    residual_mg residual_eg
    pst[K,Q,B,N,R,P][64] flattened
    optional mobility-shape counts appended by newer tunedump binaries

The engine's per-side eval is  total = (mg*phase + eg*(24-phase)) / 24  with an
integer truncation per side. Dropping that truncation makes white_total -
black_total a *linear* function of the tunable weights, so we fit them with
gradient descent through the Texel sigmoid against the game result.

Steps: (1) verify the exact integer reconstruction matches eval_true, (2) fit
the sigmoid scale K, (3) gradient-descent the weights on a train split while
watching a validation split, (4) print old vs new integer weights.
"""
import argparse
import sys

import numpy as np

# Number of scalar (material + positional) features and per-side layout.
N_BASE_SCALAR = 21
N_CURRENT_SCALAR = 35
N_V2_SCALAR = 12
N_V3_SCALAR = 12
N_EXTRA_SCALAR = (
    (N_CURRENT_SCALAR - N_BASE_SCALAR) + N_V2_SCALAR + N_V3_SCALAR
)
N_SCALAR = N_BASE_SCALAR + N_EXTRA_SCALAR
SIDE_OLD = 37
PST_PIECES = 6
PST_SQUARES = 64
N_PST = PST_PIECES * PST_SQUARES
N_PARAMS = N_SCALAR + 2 * N_PST  # 827
N_MOBILITY_SHAPE = 8

# Current engine PST tables (from src/core/engine/hce_eval.c).
K_PAWN_PST = np.array([
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
], dtype=np.float64)

K_KNIGHT_PST = np.array([
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
], dtype=np.float64)

K_BISHOP_PST = np.array([
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
], dtype=np.float64)

K_ROOK_PST = np.array([
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
], dtype=np.float64)

K_QUEEN_PST = np.array([
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
], dtype=np.float64)

K_KING_MID_PST = np.array([
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
], dtype=np.float64)

K_KING_END_PST = np.array([
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
], dtype=np.float64)

# Piece enum order: KING=0, QUEEN=1, BISHOP=2, KNIGHT=3, ROOK=4, PAWN=5.
_PST_TABLES_MG = [
    np.zeros(64, dtype=np.float64),   # KING (zero in the pre-king-PST baseline)
    K_QUEEN_PST,
    K_BISHOP_PST,
    K_KNIGHT_PST,
    K_ROOK_PST,
    K_PAWN_PST,
]
# The engine uses mg = table, eg = table/2 (trunc toward zero) for non-king pieces.
_PST_TABLES_EG = [
    np.sign(t) * (np.abs(t) // 2) for t in _PST_TABLES_MG
]
_PST_DEFAULT_MG = np.concatenate(_PST_TABLES_MG)
_PST_DEFAULT_EG = np.concatenate(_PST_TABLES_EG)

PARAM_NAMES = (
    ["mat_q", "mat_n", "mat_b", "mat_r", "mat_p",
     "iso_mg", "iso_eg", "dbl_mg", "dbl_eg",
     "mob_n_mg", "mob_n_eg", "mob_b_mg", "mob_b_eg",
     "mob_r_mg", "mob_r_eg", "mob_q_mg", "mob_q_eg",
     "rook_open_mg", "rook_open_eg", "rook_semi_mg", "rook_semi_eg",
     "passed_mg_scale", "passed_eg_scale",
     "king_mg_scale", "king_eg_scale",
     "hanging_mg_scale", "hanging_eg_scale",
     "queen_mg_scale", "queen_eg_scale",
     "pawn_push_mg", "pawn_push_eg",
     "pawn_threat_minor_mg", "pawn_threat_minor_eg",
     "pawn_threat_major_mg", "pawn_threat_major_eg",
     "connected_pawn_mg", "connected_pawn_eg",
     "phalanx_pawn_mg", "phalanx_pawn_eg",
     "backward_pawn_mg", "backward_pawn_eg",
     "knight_outpost_mg", "knight_outpost_eg",
     "bishop_pair_mg", "bishop_pair_eg",
     "rook_behind_passer_mg", "rook_behind_passer_eg",
     "minor_threat_pawn_mg", "minor_threat_pawn_eg",
     "minor_threat_minor_mg", "minor_threat_minor_eg",
     "minor_threat_major_mg", "minor_threat_major_eg",
     "rook_threat_minor_mg", "rook_threat_minor_eg",
     "safe_push_threat_minor_mg", "safe_push_threat_minor_eg",
     "safe_push_threat_major_mg", "safe_push_threat_major_eg"]
    + [f"pst{p}_{s}_mg" for p in range(PST_PIECES) for s in range(PST_SQUARES)]
    + [f"pst{p}_{s}_eg" for p in range(PST_PIECES) for s in range(PST_SQUARES)]
)

# Current engine scalar defaults (post 44dc6b5 texel tune).
_SCALAR_DEFAULTS = np.array([
    1235, 409, 466, 537, 100,   # material q n b r p (knight=409, bishop=466)
    -13, -16, -17, -15,         # isolated, doubled (mg, eg)
    8, 4, 9, 4, 9, 6, 9, 2,     # mobility n,b,r,q (mg, eg)
    19, 12, 11, 6,              # rook open, semi (mg, eg)
    100, 100,                    # passed-pawn mg/eg percentage scales
    -100, -100,                  # king-danger mg/eg percentage scales
    -100, -100,                  # hanging-piece mg/eg percentage scales
    -100, -100,                  # queen-trap mg/eg percentage scales
    0, 0,                        # pawn push mg/eg
    0, 0,                        # pawn threat vs minor mg/eg
    0, 0,                        # pawn threat vs major mg/eg
    0, 0,                        # connected pawn mg/eg
    0, 0,                        # phalanx pawn mg/eg
    0, 0,                        # backward pawn mg/eg
    0, 0,                        # knight outpost mg/eg
    0, 0,                        # bishop pair mg/eg
    0, 0,                        # rook behind passer mg/eg
    0, 0,                        # minor threat vs weak pawn mg/eg
    0, 0,                        # minor threat vs weak minor mg/eg
    0, 0,                        # minor threat vs weak major mg/eg
    0, 0,                        # rook threat vs weak minor mg/eg
    0, 0,                        # safe pawn-push threat vs minor mg/eg
    0, 0,                        # safe pawn-push threat vs major mg/eg
], dtype=np.float64)

DEFAULTS = np.concatenate([
    _SCALAR_DEFAULTS,
    _PST_DEFAULT_MG,
    _PST_DEFAULT_EG,
])

# Column indices within the first 15 old scalar features.
F_MATQ, F_MATN, F_MATB, F_MATR, F_MATP = 0, 1, 2, 3, 4
F_ISO, F_DBL = 5, 6
F_MN, F_MB, F_MR, F_MQ = 7, 8, 9, 10
F_ROPEN, F_RSEMI = 11, 12
F_PASSMG, F_PASSEG = 13, 14
F_KINGMG, F_KINGEG = 15, 16
F_HANGING = 17
F_QUEENMG, F_QUEENEG = 18, 19
F_PAWN_PUSH, F_PAWN_THREAT_MINOR, F_PAWN_THREAT_MAJOR = 20, 21, 22
F_CONNECTED, F_PHALANX, F_BACKWARD = 23, 24, 25
F_KNIGHT_OUTPOST, F_BISHOP_PAIR, F_ROOK_BEHIND_PASSER = 26, 27, 28
F_MINOR_THREAT_PAWN = 29
F_MINOR_THREAT_MINOR = 30
F_MINOR_THREAT_MAJOR = 31
F_ROOK_THREAT_MINOR = 32
F_SAFE_PUSH_THREAT_MINOR = 33
F_SAFE_PUSH_THREAT_MAJOR = 34
F_RESMG, F_RESEG = 35, 36


def trunc_div24(a):
    """C integer division by 24 truncating toward zero (vectorized)."""
    q = np.abs(a) // 24
    return np.where(a < 0, -q, q).astype(np.int64)


def trunc_div100(a):
    """C integer division by 100 truncating toward zero (vectorized)."""
    q = np.abs(a) // 100
    return np.where(a < 0, -q, q).astype(np.int64)


def build(feats_path):
    raw = np.atleast_2d(np.loadtxt(feats_path, dtype=np.float32))
    base_side = SIDE_OLD + N_PST
    base_columns = 3 + 2 * base_side
    extended_side = base_side + N_MOBILITY_SHAPE
    extended_columns = 3 + 2 * extended_side
    if raw.shape[1] not in (base_columns, extended_columns):
        raise ValueError(
            f"feature dump has {raw.shape[1]} columns, expected "
            f"{base_columns} or {extended_columns}; regenerate it with "
            "a compatible tunedump")
    label = raw[:, 0].astype(np.float64)
    phase = raw[:, 1].astype(np.int64)
    eval_true = raw[:, 2].astype(np.int64)
    side_feats = extended_side if raw.shape[1] == extended_columns else base_side
    w = raw[:, 3:3 + base_side]
    b_start = 3 + side_feats
    b = raw[:, b_start:b_start + base_side]
    return label, phase, eval_true, w, b


def split_side(side, dtype=np.int64):
    """Return (old_scalar, pst) from a side feature vector."""
    old = side[:, :SIDE_OLD].astype(dtype, copy=False)
    pst = side[:, SIDE_OLD:].astype(dtype, copy=False).reshape(
        -1, PST_PIECES, PST_SQUARES)
    return old, pst


def side_mg_eg_int(side, theta):
    """Return exact pre-blend middlegame/endgame totals for each side row."""
    old, pst = split_side(side)
    scalar = theta[:N_SCALAR]
    wmg = theta[N_SCALAR:N_SCALAR + N_PST].reshape(PST_PIECES, PST_SQUARES)
    weg = theta[N_SCALAR + N_PST:].reshape(PST_PIECES, PST_SQUARES)

    mat = (old[:, F_MATQ] * scalar[0] + old[:, F_MATN] * scalar[1] +
           old[:, F_MATB] * scalar[2] + old[:, F_MATR] * scalar[3] +
           old[:, F_MATP] * scalar[4]).astype(np.int64)
    ps_mg = np.sum(pst * wmg, axis=(1, 2)).astype(np.int64)
    ps_eg = np.sum(pst * weg, axis=(1, 2)).astype(np.int64)

    mg = (mat + ps_mg +
          old[:, F_ISO] * scalar[5] + old[:, F_DBL] * scalar[7] +
          old[:, F_MN] * scalar[9] + old[:, F_MB] * scalar[11] +
          old[:, F_MR] * scalar[13] + old[:, F_MQ] * scalar[15] +
          old[:, F_ROPEN] * scalar[17] + old[:, F_RSEMI] * scalar[19] +
          trunc_div100(old[:, F_PASSMG] * scalar[21]) +
          trunc_div100(old[:, F_KINGMG] * scalar[23]) +
          trunc_div100(old[:, F_HANGING] * scalar[25]) +
          trunc_div100(old[:, F_QUEENMG] * scalar[27]) +
          old[:, F_PAWN_PUSH] * scalar[29] +
          old[:, F_PAWN_THREAT_MINOR] * scalar[31] +
          old[:, F_PAWN_THREAT_MAJOR] * scalar[33] +
          old[:, F_CONNECTED] * scalar[35] +
          old[:, F_PHALANX] * scalar[37] +
          old[:, F_BACKWARD] * scalar[39] +
          old[:, F_KNIGHT_OUTPOST] * scalar[41] +
          old[:, F_BISHOP_PAIR] * scalar[43] +
          old[:, F_ROOK_BEHIND_PASSER] * scalar[45] +
          old[:, F_MINOR_THREAT_PAWN] * scalar[47] +
          old[:, F_MINOR_THREAT_MINOR] * scalar[49] +
          old[:, F_MINOR_THREAT_MAJOR] * scalar[51] +
          old[:, F_ROOK_THREAT_MINOR] * scalar[53] +
          old[:, F_SAFE_PUSH_THREAT_MINOR] * scalar[55] +
          old[:, F_SAFE_PUSH_THREAT_MAJOR] * scalar[57] +
          old[:, F_RESMG]).astype(np.int64)
    eg = (mat + ps_eg +
          old[:, F_ISO] * scalar[6] + old[:, F_DBL] * scalar[8] +
          old[:, F_MN] * scalar[10] + old[:, F_MB] * scalar[12] +
          old[:, F_MR] * scalar[14] + old[:, F_MQ] * scalar[16] +
          old[:, F_ROPEN] * scalar[18] + old[:, F_RSEMI] * scalar[20] +
          trunc_div100(old[:, F_PASSEG] * scalar[22]) +
          trunc_div100(old[:, F_KINGEG] * scalar[24]) +
          trunc_div100(old[:, F_HANGING] * scalar[26]) +
          trunc_div100(old[:, F_QUEENEG] * scalar[28]) +
          old[:, F_PAWN_PUSH] * scalar[30] +
          old[:, F_PAWN_THREAT_MINOR] * scalar[32] +
          old[:, F_PAWN_THREAT_MAJOR] * scalar[34] +
          old[:, F_CONNECTED] * scalar[36] +
          old[:, F_PHALANX] * scalar[38] +
          old[:, F_BACKWARD] * scalar[40] +
          old[:, F_KNIGHT_OUTPOST] * scalar[42] +
          old[:, F_BISHOP_PAIR] * scalar[44] +
          old[:, F_ROOK_BEHIND_PASSER] * scalar[46] +
          old[:, F_MINOR_THREAT_PAWN] * scalar[48] +
          old[:, F_MINOR_THREAT_MINOR] * scalar[50] +
          old[:, F_MINOR_THREAT_MAJOR] * scalar[52] +
          old[:, F_ROOK_THREAT_MINOR] * scalar[54] +
          old[:, F_SAFE_PUSH_THREAT_MINOR] * scalar[56] +
          old[:, F_SAFE_PUSH_THREAT_MAJOR] * scalar[58] +
          old[:, F_RESEG]).astype(np.int64)
    return mg, eg


def side_totals_int(side, phase, theta):
    """Exact integer per-side total using the current weights."""
    mg, eg = side_mg_eg_int(side, theta)
    return trunc_div24(mg * phase + eg * (24 - phase))


def design_matrix(phase, w, b):
    """X (N x N_PARAMS) and c (N,) so eval_white_float ~= X @ theta + c."""
    n = w.shape[0]
    ph = phase.astype(np.float32)
    mgw = ph / 24.0
    egw = (24.0 - ph) / 24.0
    w_old, w_pst = split_side(w, np.float32)
    b_old, b_pst = split_side(b, np.float32)
    d_old = (w_old - b_old).astype(np.float32)
    d_pst = (w_pst - b_pst).astype(np.float32).reshape(n, N_PST)

    X = np.zeros((n, N_PARAMS), dtype=np.float32)
    # Material: phase-independent.
    X[:, 0] = d_old[:, F_MATQ]
    X[:, 1] = d_old[:, F_MATN]
    X[:, 2] = d_old[:, F_MATB]
    X[:, 3] = d_old[:, F_MATR]
    X[:, 4] = d_old[:, F_MATP]
    # mg/eg scalar pairs.
    X[:, 5] = d_old[:, F_ISO] * mgw
    X[:, 6] = d_old[:, F_ISO] * egw
    X[:, 7] = d_old[:, F_DBL] * mgw
    X[:, 8] = d_old[:, F_DBL] * egw
    X[:, 9] = d_old[:, F_MN] * mgw
    X[:, 10] = d_old[:, F_MN] * egw
    X[:, 11] = d_old[:, F_MB] * mgw
    X[:, 12] = d_old[:, F_MB] * egw
    X[:, 13] = d_old[:, F_MR] * mgw
    X[:, 14] = d_old[:, F_MR] * egw
    X[:, 15] = d_old[:, F_MQ] * mgw
    X[:, 16] = d_old[:, F_MQ] * egw
    X[:, 17] = d_old[:, F_ROPEN] * mgw
    X[:, 18] = d_old[:, F_ROPEN] * egw
    X[:, 19] = d_old[:, F_RSEMI] * mgw
    X[:, 20] = d_old[:, F_RSEMI] * egw
    X[:, 21] = d_old[:, F_PASSMG] * mgw / 100.0
    X[:, 22] = d_old[:, F_PASSEG] * egw / 100.0
    X[:, 23] = d_old[:, F_KINGMG] * mgw / 100.0
    X[:, 24] = d_old[:, F_KINGEG] * egw / 100.0
    X[:, 25] = d_old[:, F_HANGING] * mgw / 100.0
    X[:, 26] = d_old[:, F_HANGING] * egw / 100.0
    X[:, 27] = d_old[:, F_QUEENMG] * mgw / 100.0
    X[:, 28] = d_old[:, F_QUEENEG] * egw / 100.0
    X[:, 29] = d_old[:, F_PAWN_PUSH] * mgw
    X[:, 30] = d_old[:, F_PAWN_PUSH] * egw
    X[:, 31] = d_old[:, F_PAWN_THREAT_MINOR] * mgw
    X[:, 32] = d_old[:, F_PAWN_THREAT_MINOR] * egw
    X[:, 33] = d_old[:, F_PAWN_THREAT_MAJOR] * mgw
    X[:, 34] = d_old[:, F_PAWN_THREAT_MAJOR] * egw
    X[:, 35] = d_old[:, F_CONNECTED] * mgw
    X[:, 36] = d_old[:, F_CONNECTED] * egw
    X[:, 37] = d_old[:, F_PHALANX] * mgw
    X[:, 38] = d_old[:, F_PHALANX] * egw
    X[:, 39] = d_old[:, F_BACKWARD] * mgw
    X[:, 40] = d_old[:, F_BACKWARD] * egw
    X[:, 41] = d_old[:, F_KNIGHT_OUTPOST] * mgw
    X[:, 42] = d_old[:, F_KNIGHT_OUTPOST] * egw
    X[:, 43] = d_old[:, F_BISHOP_PAIR] * mgw
    X[:, 44] = d_old[:, F_BISHOP_PAIR] * egw
    X[:, 45] = d_old[:, F_ROOK_BEHIND_PASSER] * mgw
    X[:, 46] = d_old[:, F_ROOK_BEHIND_PASSER] * egw
    X[:, 47] = d_old[:, F_MINOR_THREAT_PAWN] * mgw
    X[:, 48] = d_old[:, F_MINOR_THREAT_PAWN] * egw
    X[:, 49] = d_old[:, F_MINOR_THREAT_MINOR] * mgw
    X[:, 50] = d_old[:, F_MINOR_THREAT_MINOR] * egw
    X[:, 51] = d_old[:, F_MINOR_THREAT_MAJOR] * mgw
    X[:, 52] = d_old[:, F_MINOR_THREAT_MAJOR] * egw
    X[:, 53] = d_old[:, F_ROOK_THREAT_MINOR] * mgw
    X[:, 54] = d_old[:, F_ROOK_THREAT_MINOR] * egw
    X[:, 55] = d_old[:, F_SAFE_PUSH_THREAT_MINOR] * mgw
    X[:, 56] = d_old[:, F_SAFE_PUSH_THREAT_MINOR] * egw
    X[:, 57] = d_old[:, F_SAFE_PUSH_THREAT_MAJOR] * mgw
    X[:, 58] = d_old[:, F_SAFE_PUSH_THREAT_MAJOR] * egw
    # PST mg/eg.
    X[:, N_SCALAR:N_SCALAR + N_PST] = d_pst * mgw[:, None]
    X[:, N_SCALAR + N_PST:] = d_pst * egw[:, None]

    c = d_old[:, F_RESMG] * mgw + d_old[:, F_RESEG] * egw
    return X, c


def grouped_split(groups, val_frac, seed):
    """Return train/validation rows while keeping every game indivisible."""
    groups = np.asarray(groups)
    if groups.ndim != 1:
        raise ValueError("groups must be one-dimensional")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("grouped validation needs at least two games")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    nval_groups = min(len(unique) - 1,
                      max(1, int(round(len(unique) * val_frac))))
    is_val = np.isin(groups, unique[:nval_groups])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def load_groups(path, expected_rows):
    with open(path, "r", encoding="utf-8") as fp:
        groups = np.array([line.strip() for line in fp if line.strip()])
    if len(groups) != expected_rows:
        raise ValueError(
            f"group sidecar has {len(groups)} rows, expected {expected_rows}")
    return groups


def verify_reconstruction(w, b, phase, eval_true, defaults, batch_size=8192):
    """Verify the integer feature contract without huge int64 copies."""
    mismatches = []
    for start in range(0, len(phase), batch_size):
        stop = min(start + batch_size, len(phase))
        wt = side_totals_int(w[start:stop], phase[start:stop], defaults)
        bt = side_totals_int(b[start:stop], phase[start:stop], defaults)
        eval_white = wt - bt
        bad = np.flatnonzero(
            np.abs(eval_true[start:stop] - 12) != np.abs(eval_white))
        if len(bad):
            room = 5 - len(mismatches)
            mismatches.extend((bad[:room] + start).tolist())
        if len(mismatches) >= 5:
            break
    return mismatches


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def load_tuned_defaults(path):
    with open(path, "r", encoding="utf-8") as fp:
        lines = [line.strip() for line in fp if line.strip().startswith("TUNED ")]
    if not lines:
        raise SystemExit(f"no TUNED line found in {path}")
    values = np.array([int(value) for value in lines[-1].split()[1:]], dtype=np.float64)
    old_params = N_BASE_SCALAR + 2 * N_PST
    current_params = N_CURRENT_SCALAR + 2 * N_PST
    v2_scalar = N_CURRENT_SCALAR + N_V2_SCALAR
    v2_params = v2_scalar + 2 * N_PST
    if len(values) == old_params:
        values = np.concatenate([
            values[:N_BASE_SCALAR],
            _SCALAR_DEFAULTS[N_BASE_SCALAR:],
            values[N_BASE_SCALAR:],
        ])
    elif len(values) == current_params:
        values = np.concatenate([
            values[:N_CURRENT_SCALAR],
            np.zeros(N_V2_SCALAR + N_V3_SCALAR, dtype=np.float64),
            values[N_CURRENT_SCALAR:],
        ])
    elif len(values) == v2_params:
        values = np.concatenate([
            values[:v2_scalar],
            np.zeros(N_V3_SCALAR, dtype=np.float64),
            values[v2_scalar:],
        ])
    if len(values) != N_PARAMS:
        raise SystemExit(f"expected {N_PARAMS} values in {path}, got {len(values)}")
    return values


def loss_for(K, evals, y):
    return np.mean((y - sigmoid(K * evals)) ** 2)


def fit_K(evals, y):
    best_K, best_L = 0.004, 1e9
    for K in np.linspace(0.0005, 0.02, 200):
        L = loss_for(K, evals, y)
        if L < best_L:
            best_L, best_K = L, K
    return best_K, best_L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups",
                    help="One game ID per feature row for leakage-free validation.")
    ap.add_argument("--l2", type=float, default=3.0,
                    help="L2 pull toward defaults (relative), tames overfit.")
    ap.add_argument("--anchor-pawn", action="store_true", default=True,
                    help="Freeze pawn=100 so the eval stays in centipawns.")
    ap.add_argument("--no-anchor-pawn", dest="anchor_pawn", action="store_false")
    ap.add_argument("--freeze-material", action="store_true",
                    help="Hold all 5 material values fixed; tune only the "
                         "positional terms + PST.")
    ap.add_argument("--only-extra-scalars", action="store_true",
                    help="Tune only passed-pawn, king-danger, hanging, and "
                         "queen-trap scales; freeze established scalars/PSTs.")
    ap.add_argument("--only-new-features", action="store_true",
                    help="Tune only pawn activity/threat weights.")
    ap.add_argument("--only-v2-features", action="store_true",
                    help="Tune only the new pawn/minor/rook structure weights.")
    ap.add_argument("--only-v3-features", action="store_true",
                    help="Tune only contextual weak-piece and safe-push threats.")
    ap.add_argument("--retune-established", action="store_true",
                    help="Freeze material and all experimental scalar terms; "
                         "retune the established 16 positional scalars and PSTs.")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="Use random mini-batches of this many training rows; "
                         "zero keeps full-batch optimization.")
    ap.add_argument("--out-c", help="Optional path to write tuned PST/material C snippet.")
    ap.add_argument("--initial-tuned-file",
                    help="Use the last TUNED line in this file as the exact current defaults.")
    args = ap.parse_args()

    defaults = (load_tuned_defaults(args.initial_tuned_file)
                if args.initial_tuned_file else DEFAULTS.copy())

    label, phase, eval_true, w, b = build(args.feats)
    n = len(label)
    print(f"positions: {n}", file=sys.stderr)

    # (1) Exact integer verification against the engine's own eval.
    bad = verify_reconstruction(w, b, phase, eval_true, defaults)
    print(f"verify: |eval_true-12| != |recon| on "
          f"{'at least ' if bad else ''}{len(bad)}/{n} rows", file=sys.stderr)
    if bad:
        for i in bad:
            wt = side_totals_int(w[i:i + 1], phase[i:i + 1], defaults)
            bt = side_totals_int(b[i:i + 1], phase[i:i + 1], defaults)
            eval_white = int(wt[0] - bt[0])
            print(f"  row {i}: true={eval_true[i]} recon_white={eval_white}",
                  file=sys.stderr)
        print("ABORT: feature reconstruction is not exact.", file=sys.stderr)
        return 1
    print("verify: OK (linear features reproduce engine eval exactly)",
          file=sys.stderr)

    # Linear design for float tuning.
    X, c = design_matrix(phase, w, b)
    y = label
    if args.groups:
        groups = load_groups(args.groups, n)
        tr, val = grouped_split(groups, args.val_frac, args.seed)
        print(f"split: {len(np.unique(groups[tr]))} train games, "
              f"{len(np.unique(groups[val]))} validation games",
              file=sys.stderr)
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(n)
        nval = max(1, int(n * args.val_frac))
        val, tr = idx[:nval], idx[nval:]
        print("warning: row-random split; pass --groups for game-held-out validation",
              file=sys.stderr)

    theta = defaults.copy()
    evals_tr = X[tr] @ theta + c[tr]
    K, L0 = fit_K(evals_tr, y[tr])
    print(f"fit K={K:.5f}  baseline train loss={L0:.6f}  "
          f"val loss={loss_for(K, X[val] @ theta + c[val], y[val]):.6f}",
          file=sys.stderr)

    # (3) Gradient descent (Adam) on weights; refit K periodically.
    # Relative L2 pull toward defaults keeps low-count terms sane.
    reg_scale = np.maximum(np.abs(defaults), 20.0)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Xtr, ctr, ytr = X[tr], c[tr], y[tr]
    ntr = len(tr)
    selection_count = sum((
        args.only_v2_features,
        args.only_v3_features,
        args.only_new_features,
        args.only_extra_scalars,
        args.freeze_material,
        args.retune_established,
    ))
    if selection_count > 1:
        raise SystemExit("choose at most one parameter-selection mode")
    if args.batch_size < 0:
        raise SystemExit("--batch-size must be non-negative")

    if args.only_v3_features:
        active = np.arange(
            N_CURRENT_SCALAR + N_V2_SCALAR,
            N_SCALAR,
        )
    elif args.only_v2_features:
        active = np.arange(
            N_CURRENT_SCALAR,
            N_CURRENT_SCALAR + N_V2_SCALAR,
        )
    elif args.only_new_features:
        active = np.arange(N_BASE_SCALAR + 8, N_CURRENT_SCALAR)
    elif args.only_extra_scalars:
        active = np.arange(N_BASE_SCALAR, N_SCALAR)
    elif args.retune_established:
        active = np.concatenate((
            np.arange(5, N_BASE_SCALAR),
            np.arange(N_SCALAR, N_PARAMS),
        ))
    elif args.freeze_material:
        active = np.arange(5, N_PARAMS)
    else:
        active = np.arange(N_PARAMS)
    Xactive = Xtr[:, active]
    fixed_tr = Xtr @ defaults + ctr - Xactive @ defaults[active]
    m = np.zeros(len(active))
    v = np.zeros(len(active))
    best_theta = theta.copy()
    best_iteration = 0
    best_val = loss_for(K, X[val] @ theta + c[val], y[val])
    batch_rng = np.random.default_rng(args.seed ^ 0x5EED5EED)
    for it in range(1, args.iters + 1):
        if 0 < args.batch_size < ntr:
            batch = batch_rng.integers(0, ntr, size=args.batch_size)
            batch_x = Xactive[batch]
            batch_fixed = fixed_tr[batch]
            batch_y = ytr[batch]
        else:
            batch_x = Xactive
            batch_fixed = fixed_tr
            batch_y = ytr
        evals = batch_fixed + batch_x @ theta[active]
        p = sigmoid(K * evals)
        g = (batch_x.T @
             (2.0 * (p - batch_y) * p * (1.0 - p) * K)) / len(batch_y)
        g += (args.l2 * 1e-3 * (theta[active] - defaults[active]) /
              (reg_scale[active] ** 2))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** it)
        vhat = v / (1 - b2 ** it)
        theta[active] -= args.lr * mhat / (np.sqrt(vhat) + eps)
        if args.anchor_pawn and 4 in active:
            theta[4] = 100.0
        if it % 500 == 0:
            K, _ = fit_K(Xtr @ theta + ctr, ytr)
            Ltr = loss_for(K, Xtr @ theta + ctr, ytr)
            Lval = loss_for(K, X[val] @ theta + c[val], y[val])
            if Lval < best_val:
                best_val = Lval
                best_theta = theta.copy()
                best_iteration = it
            print(f"  it {it:5d}  K={K:.5f}  train={Ltr:.6f}  val={Lval:.6f}",
                  file=sys.stderr)

    theta = best_theta
    print(f"selected iteration {best_iteration} with val loss={best_val:.6f}",
          file=sys.stderr)
    rounded = np.round(theta).astype(np.int64)
    print("\n# tuned scalar weights (round to int):", file=sys.stderr)
    for name, d0, t in zip(PARAM_NAMES[:N_SCALAR], defaults[:N_SCALAR], rounded[:N_SCALAR]):
        print(f"  {name:14s} {int(round(d0)):5d} -> {int(t):5d}",
              file=sys.stderr)

    # Machine-readable line for porting.
    print("TUNED " + " ".join(str(int(t)) for t in rounded))

    if args.out_c:
        write_c_snippet(args.out_c, rounded)
    return 0


def write_c_snippet(path, rounded):
    """Write a C snippet with tuned hce_piece_value and PST tables."""
    mat = rounded[:5]
    scalar = rounded[:N_SCALAR]
    pst_mg = rounded[N_SCALAR:N_SCALAR + N_PST].reshape(PST_PIECES, PST_SQUARES)
    pst_eg = rounded[N_SCALAR + N_PST:].reshape(PST_PIECES, PST_SQUARES)
    with open(path, "w", encoding="utf-8") as fp:
        fp.write("// Tuned HCE tables (machine-generated).\n")
        fp.write("const int hce_piece_value[PIECE_TYPE_COUNT] = {\n")
        fp.write("    0,\n")
        piece_values = [
            (mat[0], "QUEEN"),
            (mat[2], "BISHOP"),
            (mat[1], "KNIGHT"),
            (mat[3], "ROOK"),
            (mat[4], "PAWN"),
        ]
        for value, name in piece_values:
            fp.write(f"    {value:4d},  // {name}\n")
        fp.write("};\n\n")
        _write_table(fp, "k_king_mid_pst", pst_mg[0])
        _write_table(fp, "k_king_end_pst", pst_eg[0])
        _write_table(fp, "k_queen_pst", pst_mg[1])
        _write_table(fp, "k_bishop_pst", pst_mg[2])
        _write_table(fp, "k_knight_pst", pst_mg[3])
        _write_table(fp, "k_rook_pst", pst_mg[4])
        _write_table(fp, "k_pawn_pst", pst_mg[5])
        fp.write("\n// Suggested scalar weights if tuning them separately:\n")
        for name, val in zip(PARAM_NAMES[:N_SCALAR], scalar):
            fp.write(f"// {name} = {val}\n")


def _write_table(fp, name, arr):
    fp.write(f"static const int {name}[64] = {{\n")
    for r in range(8):
        row = ", ".join(f"{int(arr[r * 8 + c]):4d}" for c in range(8))
        fp.write("    " + row + ",\n")
    fp.write("};\n")


if __name__ == "__main__":
    raise SystemExit(main())
