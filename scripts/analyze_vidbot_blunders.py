#!/usr/bin/env python3
"""Find the largest engine-move evaluation drops in recent VidBot games."""

import argparse
import concurrent.futures
import json

import chess
import chess.engine
import chess.pgn


def load_losses(path, username, count):
    games = []
    with open(path, "r", encoding="utf-8") as source:
        while len(games) < count:
            game = chess.pgn.read_game(source)
            if game is None:
                break
            white = game.headers.get("White", "").lower() == username.lower()
            black = game.headers.get("Black", "").lower() == username.lower()
            result = game.headers.get("Result", "")
            lost = (white and result == "0-1") or (black and result == "1-0")
            if lost:
                games.append(game)
    return games


def analyse_game(game, username, engine_path, depth, hash_mb):
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    engine.configure({"Threads": 1, "Hash": hash_mb})
    bot_color = chess.WHITE if game.headers.get("White", "").lower() == username.lower() else chess.BLACK
    board = game.board()
    rows = []
    try:
        for ply, move in enumerate(game.mainline_moves(), start=1):
            if board.turn != bot_color:
                board.push(move)
                continue
            before_fen = board.fen()
            before = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = before.get("pv", [None])[0]
            candidates = [move]
            if best_move is not None and best_move != move:
                candidates.append(best_move)
            # Score both root moves together. Separate searches can discover
            # different mate horizons even at the same nominal depth, which
            # previously assigned huge losses to moves compared with themselves.
            comparison = engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                root_moves=candidates,
                multipv=len(candidates),
            )
            if isinstance(comparison, dict):
                comparison = [comparison]
            scores = {
                info["pv"][0]: info["score"].pov(bot_color).score(mate_score=30000)
                for info in comparison
                if info.get("pv")
            }
            played_score = scores[move]
            best_score = max(scores.values())
            rows.append({
                "game_id": game.headers.get("GameId", ""),
                "time_control": game.headers.get("TimeControl", ""),
                "opponent": game.headers.get("Black" if bot_color else "White", ""),
                "ply": ply,
                "fen": before_fen,
                "played": move.uci(),
                "best": best_move.uci() if best_move is not None else "",
                "best_cp": best_score,
                "played_cp": played_score,
                "loss_cp": max(0, best_score - played_score),
            })
            board.push(move)
    finally:
        engine.quit()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--username", default="VidBot")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--depth", type=int, default=11)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--hash-mb", type=int, default=32)
    parser.add_argument("--engine", default="/opt/homebrew/bin/stockfish")
    args = parser.parse_args()

    games = load_losses(args.pgn, args.username, args.games)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(analyse_game, game, args.username, args.engine,
                        args.depth, args.hash_mb)
            for game in games
        ]
        for future in concurrent.futures.as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: row["loss_cp"], reverse=True)
    with open(args.out, "w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"analysed {len(games)} losses and {len(rows)} VidBot moves: {args.out}")
    for row in rows[:20]:
        print(f"{row['loss_cp']:5d} cp  {row['game_id']} ply {row['ply']:3d}  "
              f"{row['played']} -> {row['best']}  {row['fen']}")


if __name__ == "__main__":
    main()
