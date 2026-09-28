#!/usr/bin/env python3
"""Label FENs with a Stockfish win probability for tunedump / Texel tuning.

Writes "<fen>;<p>" lines, where p is white's expected score
1 / (1 + 10^(-cp_white / 250)) from a fixed-depth Stockfish search
(scores clipped to +-2500, mates as +-2500). One Stockfish process with one
thread; run it under `nice` on a shared machine.

    sf_winprob_labels.py --fens quiet.fen --out labels.txt --depth 8
    printf 'tunedump labels.txt feats.txt\\nquit\\n' | chess_uci
"""
import argparse
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--stockfish", default="stockfish")
    args = ap.parse_args()

    sf = subprocess.Popen([args.stockfish], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, bufsize=1)

    def send(cmd):
        sf.stdin.write(cmd + "\n")
        sf.stdin.flush()

    send("uci")
    send("setoption name Threads value 1")
    send("setoption name Hash value 32")
    send("isready")
    while sf.stdout.readline().strip() != "readyok":
        pass
    with open(args.fens) as fin, open(args.out, "w") as out:
        for raw in fin:
            fen = raw.split(";", 1)[0].strip()
            if not fen:
                continue
            send("ucinewgame")
            send("position fen " + fen)
            send(f"go depth {args.depth}")
            score = 0
            while True:
                line = sf.stdout.readline()
                if line.startswith("info") and " score " in line:
                    t = line.split()
                    i = t.index("score")
                    if t[i + 1] == "cp":
                        score = int(t[i + 2])
                    elif t[i + 1] == "mate":
                        score = 2500 if int(t[i + 2]) > 0 else -2500
                if line.startswith("bestmove"):
                    break
            score = max(-2500, min(2500, score))
            white = score if fen.split()[1] == "w" else -score
            out.write(f"{fen};{1.0 / (1.0 + 10 ** (-white / 250.0)):.4f}\n")
    send("quit")


if __name__ == "__main__":
    main()
