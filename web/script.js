"use strict";

(() => {
  const WHITE = "w";
  const BLACK = "b";

  const FILES = "abcdefgh";
  const START_WHITE_BACK = ["R", "N", "B", "Q", "K", "B", "N", "R"];
  const START_BLACK_BACK = ["r", "n", "b", "q", "k", "b", "n", "r"];

  const PROMOTION_TYPES = ["q", "r", "b", "n"];

  const KNIGHT_STEPS = [
    [1, 2],
    [2, 1],
    [2, -1],
    [1, -2],
    [-1, -2],
    [-2, -1],
    [-2, 1],
    [-1, 2],
  ];

  const KING_STEPS = [
    [1, 1],
    [1, 0],
    [1, -1],
    [0, 1],
    [0, -1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
  ];

  const BISHOP_DIRS = [
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1],
  ];

  const ROOK_DIRS = [
    [1, 0],
    [-1, 0],
    [0, 1],
    [0, -1],
  ];

  const PIECE_TO_SPRITE_COLUMN = {
    k: 0,
    q: 1,
    b: 2,
    n: 3,
    r: 4,
    p: 5,
  };

  const BOARD_COLORS = {
    bg0: "#000000",
    bg1: "#000000",
    frame: "#3e3e3e",
    light: "#d2b896",
    dark: "#8f6f55",
    selected: "rgba(201, 150, 0, 0.82)",
    lastMove: "rgba(179, 138, 57, 0.5)",
    legalDot: "rgba(27, 27, 27, 0.62)",
    captureRing: "rgba(196, 43, 43, 0.82)",
    check: "rgba(219, 47, 47, 0.88)",
    promoBackdrop: "rgba(0, 0, 0, 0.58)",
    promoCard: "#232c3d",
    promoStroke: "#5e6f8f",
    promoBtn: "#415170",
    promoBtnHover: "#536489",
    coord: "rgba(241, 246, 255, 0.75)",
    dragShadow: "rgba(0, 0, 0, 0.28)",
  };

  const defaultBridge = {
    async ensureUserSession() {
      return null;
    },
    async onMoveCommitted(_payload) {},
    async requestAIMove(_payload) {
      return null;
    },
    async requestRemoteMove(_payload) {
      return null;
    },
  };

  function opposite(color) {
    return color === WHITE ? BLACK : WHITE;
  }

  function clamp(value, min, max) {
    if (value < min) {
      return min;
    }
    if (value > max) {
      return max;
    }
    return value;
  }

  function inBounds(file, rank) {
    return file >= 0 && file < 8 && rank >= 0 && rank < 8;
  }

  function makeSquare(file, rank) {
    return rank * 8 + file;
  }

  function fileOf(square) {
    return square & 7;
  }

  function rankOf(square) {
    return square >> 3;
  }

  function squareName(square) {
    if (square < 0 || square >= 64) {
      return "-";
    }
    return `${FILES[fileOf(square)]}${rankOf(square) + 1}`;
  }

  function pieceColor(piece) {
    if (!piece) {
      return null;
    }
    return piece === piece.toUpperCase() ? WHITE : BLACK;
  }

  function pieceType(piece) {
    if (!piece) {
      return null;
    }
    return piece.toLowerCase();
  }

  function makePiece(color, type) {
    return color === WHITE ? type.toUpperCase() : type;
  }

  function colorName(color) {
    return color === WHITE ? "White" : "Black";
  }

  function formatClock(ms) {
    const totalSec = Math.max(0, Math.ceil(ms / 1000));
    const minutes = Math.floor(totalSec / 60);
    const seconds = totalSec % 60;
    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
  }

  function sameMove(a, b) {
    const promoA = a.promotion || null;
    const promoB = b.promotion || null;
    return a.from === b.from && a.to === b.to && promoA === promoB;
  }

  function squareParity(square) {
    return (fileOf(square) + rankOf(square)) & 1;
  }

  function roundRectPath(ctx, x, y, w, h, r) {
    const radius = Math.min(r, w * 0.5, h * 0.5);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
  }

  class ChessEngine {
    constructor() {
      this.reset();
    }

    createInitialBoard() {
      const board = new Array(64).fill(null);
      for (let file = 0; file < 8; file += 1) {
        board[makeSquare(file, 0)] = START_WHITE_BACK[file];
        board[makeSquare(file, 1)] = "P";
        board[makeSquare(file, 6)] = "p";
        board[makeSquare(file, 7)] = START_BLACK_BACK[file];
      }
      return board;
    }

    createInitialState() {
      return {
        board: this.createInitialBoard(),
        sideToMove: WHITE,
        castling: { K: true, Q: true, k: true, q: true },
        enPassant: -1,
        halfmoveClock: 0,
        fullmoveNumber: 1,
        result: { code: "ongoing", winner: null },
      };
    }

    reset() {
      this.state = this.createInitialState();
      this.history = [];
      this.repetition = new Map();
      this.bumpRepetition(this.positionKey(this.state), 1);
    }

    cloneState(state = this.state) {
      return {
        board: state.board.slice(),
        sideToMove: state.sideToMove,
        castling: {
          K: state.castling.K,
          Q: state.castling.Q,
          k: state.castling.k,
          q: state.castling.q,
        },
        enPassant: state.enPassant,
        halfmoveClock: state.halfmoveClock,
        fullmoveNumber: state.fullmoveNumber,
        result: {
          code: state.result.code,
          winner: state.result.winner,
        },
      };
    }

    cloneMove(move) {
      return {
        from: move.from,
        to: move.to,
        piece: move.piece,
        captured: move.captured,
        promotion: move.promotion || null,
        flags: {
          capture: Boolean(move.flags.capture),
          enPassant: Boolean(move.flags.enPassant),
          doublePush: Boolean(move.flags.doublePush),
          castleKing: Boolean(move.flags.castleKing),
          castleQueen: Boolean(move.flags.castleQueen),
        },
      };
    }

    castlingToString(castling) {
      let rights = "";
      if (castling.K) {
        rights += "K";
      }
      if (castling.Q) {
        rights += "Q";
      }
      if (castling.k) {
        rights += "k";
      }
      if (castling.q) {
        rights += "q";
      }
      return rights || "-";
    }

    positionKey(state = this.state) {
      const boardPart = state.board.map((piece) => piece || ".").join("");
      const castlingPart = this.castlingToString(state.castling);
      const epPart = state.enPassant >= 0 ? squareName(state.enPassant) : "-";
      return `${boardPart}|${state.sideToMove}|${castlingPart}|${epPart}`;
    }

    bumpRepetition(key, delta) {
      const current = this.repetition.get(key) || 0;
      const next = current + delta;
      if (next <= 0) {
        this.repetition.delete(key);
      } else {
        this.repetition.set(key, next);
      }
    }

    findKingSquare(state, color) {
      const kingPiece = color === WHITE ? "K" : "k";
      for (let sq = 0; sq < 64; sq += 1) {
        if (state.board[sq] === kingPiece) {
          return sq;
        }
      }
      return -1;
    }

    isSquareAttacked(state, square, byColor) {
      const file = fileOf(square);
      const rank = rankOf(square);

      if (byColor === WHITE) {
        const fromRight = square - 7;
        const fromLeft = square - 9;
        if (file < 7 && fromRight >= 0 && state.board[fromRight] === "P") {
          return true;
        }
        if (file > 0 && fromLeft >= 0 && state.board[fromLeft] === "P") {
          return true;
        }
      } else {
        const fromRight = square + 7;
        const fromLeft = square + 9;
        if (file > 0 && fromRight < 64 && state.board[fromRight] === "p") {
          return true;
        }
        if (file < 7 && fromLeft < 64 && state.board[fromLeft] === "p") {
          return true;
        }
      }

      for (const [df, dr] of KNIGHT_STEPS) {
        const nf = file + df;
        const nr = rank + dr;
        if (!inBounds(nf, nr)) {
          continue;
        }
        const piece = state.board[makeSquare(nf, nr)];
        if (piece && pieceColor(piece) === byColor && pieceType(piece) === "n") {
          return true;
        }
      }

      for (const [df, dr] of KING_STEPS) {
        const nf = file + df;
        const nr = rank + dr;
        if (!inBounds(nf, nr)) {
          continue;
        }
        const piece = state.board[makeSquare(nf, nr)];
        if (piece && pieceColor(piece) === byColor && pieceType(piece) === "k") {
          return true;
        }
      }

      for (const [df, dr] of BISHOP_DIRS) {
        let nf = file + df;
        let nr = rank + dr;
        while (inBounds(nf, nr)) {
          const piece = state.board[makeSquare(nf, nr)];
          if (piece) {
            if (pieceColor(piece) === byColor) {
              const type = pieceType(piece);
              if (type === "b" || type === "q") {
                return true;
              }
            }
            break;
          }
          nf += df;
          nr += dr;
        }
      }

      for (const [df, dr] of ROOK_DIRS) {
        let nf = file + df;
        let nr = rank + dr;
        while (inBounds(nf, nr)) {
          const piece = state.board[makeSquare(nf, nr)];
          if (piece) {
            if (pieceColor(piece) === byColor) {
              const type = pieceType(piece);
              if (type === "r" || type === "q") {
                return true;
              }
            }
            break;
          }
          nf += df;
          nr += dr;
        }
      }

      return false;
    }

    isKingInCheck(state, color) {
      const kingSquare = this.findKingSquare(state, color);
      if (kingSquare < 0) {
        return false;
      }
      return this.isSquareAttacked(state, kingSquare, opposite(color));
    }

    pushMove(moves, state, from, to, flags = {}, promotion = null) {
      const piece = state.board[from];
      if (!piece) {
        return;
      }

      const captureSquare = flags.enPassant
        ? (pieceColor(piece) === WHITE ? to - 8 : to + 8)
        : to;

      const captured =
        captureSquare >= 0 && captureSquare < 64 ? state.board[captureSquare] : null;

      moves.push({
        from,
        to,
        piece,
        captured,
        promotion,
        flags: {
          capture: Boolean(captured),
          enPassant: Boolean(flags.enPassant),
          doublePush: Boolean(flags.doublePush),
          castleKing: Boolean(flags.castleKing),
          castleQueen: Boolean(flags.castleQueen),
        },
      });
    }

    generatePawnMoves(state, color, from, moves) {
      const file = fileOf(from);
      const rank = rankOf(from);
      const oneStep = color === WHITE ? from + 8 : from - 8;
      const twoStep = color === WHITE ? from + 16 : from - 16;
      const startRank = color === WHITE ? 1 : 6;
      const promoRank = color === WHITE ? 6 : 1;

      if (oneStep >= 0 && oneStep < 64 && !state.board[oneStep]) {
        if (rank === promoRank) {
          for (const promo of PROMOTION_TYPES) {
            this.pushMove(moves, state, from, oneStep, {}, promo);
          }
        } else {
          this.pushMove(moves, state, from, oneStep);
          if (rank === startRank && !state.board[twoStep]) {
            this.pushMove(moves, state, from, twoStep, { doublePush: true });
          }
        }
      }

      if (color === WHITE) {
        if (file > 0) {
          const to = from + 7;
          if (to >= 0 && to < 64) {
            const target = state.board[to];
            if (target && pieceColor(target) === BLACK) {
              if (rank === promoRank) {
                for (const promo of PROMOTION_TYPES) {
                  this.pushMove(moves, state, from, to, {}, promo);
                }
              } else {
                this.pushMove(moves, state, from, to);
              }
            } else if (to === state.enPassant) {
              this.pushMove(moves, state, from, to, { enPassant: true });
            }
          }
        }

        if (file < 7) {
          const to = from + 9;
          if (to >= 0 && to < 64) {
            const target = state.board[to];
            if (target && pieceColor(target) === BLACK) {
              if (rank === promoRank) {
                for (const promo of PROMOTION_TYPES) {
                  this.pushMove(moves, state, from, to, {}, promo);
                }
              } else {
                this.pushMove(moves, state, from, to);
              }
            } else if (to === state.enPassant) {
              this.pushMove(moves, state, from, to, { enPassant: true });
            }
          }
        }
      } else {
        if (file > 0) {
          const to = from - 9;
          if (to >= 0 && to < 64) {
            const target = state.board[to];
            if (target && pieceColor(target) === WHITE) {
              if (rank === promoRank) {
                for (const promo of PROMOTION_TYPES) {
                  this.pushMove(moves, state, from, to, {}, promo);
                }
              } else {
                this.pushMove(moves, state, from, to);
              }
            } else if (to === state.enPassant) {
              this.pushMove(moves, state, from, to, { enPassant: true });
            }
          }
        }

        if (file < 7) {
          const to = from - 7;
          if (to >= 0 && to < 64) {
            const target = state.board[to];
            if (target && pieceColor(target) === WHITE) {
              if (rank === promoRank) {
                for (const promo of PROMOTION_TYPES) {
                  this.pushMove(moves, state, from, to, {}, promo);
                }
              } else {
                this.pushMove(moves, state, from, to);
              }
            } else if (to === state.enPassant) {
              this.pushMove(moves, state, from, to, { enPassant: true });
            }
          }
        }
      }
    }

    generateKnightMoves(state, color, from, moves) {
      const file = fileOf(from);
      const rank = rankOf(from);
      for (const [df, dr] of KNIGHT_STEPS) {
        const nf = file + df;
        const nr = rank + dr;
        if (!inBounds(nf, nr)) {
          continue;
        }

        const to = makeSquare(nf, nr);
        const target = state.board[to];
        if (!target || pieceColor(target) !== color) {
          this.pushMove(moves, state, from, to);
        }
      }
    }

    generateSlidingMoves(state, color, from, moves, directions) {
      const file = fileOf(from);
      const rank = rankOf(from);
      for (const [df, dr] of directions) {
        let nf = file + df;
        let nr = rank + dr;
        while (inBounds(nf, nr)) {
          const to = makeSquare(nf, nr);
          const target = state.board[to];
          if (!target) {
            this.pushMove(moves, state, from, to);
          } else {
            if (pieceColor(target) !== color) {
              this.pushMove(moves, state, from, to);
            }
            break;
          }
          nf += df;
          nr += dr;
        }
      }
    }

    generateKingMoves(state, color, from, moves) {
      const file = fileOf(from);
      const rank = rankOf(from);

      for (const [df, dr] of KING_STEPS) {
        const nf = file + df;
        const nr = rank + dr;
        if (!inBounds(nf, nr)) {
          continue;
        }
        const to = makeSquare(nf, nr);
        const target = state.board[to];
        if (!target || pieceColor(target) !== color) {
          this.pushMove(moves, state, from, to);
        }
      }

      if (this.isKingInCheck(state, color)) {
        return;
      }

      if (color === WHITE && from === 4) {
        if (
          state.castling.K &&
          state.board[5] === null &&
          state.board[6] === null &&
          state.board[7] === "R" &&
          !this.isSquareAttacked(state, 5, BLACK) &&
          !this.isSquareAttacked(state, 6, BLACK)
        ) {
          this.pushMove(moves, state, from, 6, { castleKing: true });
        }

        if (
          state.castling.Q &&
          state.board[3] === null &&
          state.board[2] === null &&
          state.board[1] === null &&
          state.board[0] === "R" &&
          !this.isSquareAttacked(state, 3, BLACK) &&
          !this.isSquareAttacked(state, 2, BLACK)
        ) {
          this.pushMove(moves, state, from, 2, { castleQueen: true });
        }
      }

      if (color === BLACK && from === 60) {
        if (
          state.castling.k &&
          state.board[61] === null &&
          state.board[62] === null &&
          state.board[63] === "r" &&
          !this.isSquareAttacked(state, 61, WHITE) &&
          !this.isSquareAttacked(state, 62, WHITE)
        ) {
          this.pushMove(moves, state, from, 62, { castleKing: true });
        }

        if (
          state.castling.q &&
          state.board[59] === null &&
          state.board[58] === null &&
          state.board[57] === null &&
          state.board[56] === "r" &&
          !this.isSquareAttacked(state, 59, WHITE) &&
          !this.isSquareAttacked(state, 58, WHITE)
        ) {
          this.pushMove(moves, state, from, 58, { castleQueen: true });
        }
      }
    }

    generatePseudoMoves(state = this.state, color = state.sideToMove) {
      const moves = [];

      for (let from = 0; from < 64; from += 1) {
        const piece = state.board[from];
        if (!piece || pieceColor(piece) !== color) {
          continue;
        }

        const type = pieceType(piece);
        if (type === "p") {
          this.generatePawnMoves(state, color, from, moves);
          continue;
        }
        if (type === "n") {
          this.generateKnightMoves(state, color, from, moves);
          continue;
        }
        if (type === "b") {
          this.generateSlidingMoves(state, color, from, moves, BISHOP_DIRS);
          continue;
        }
        if (type === "r") {
          this.generateSlidingMoves(state, color, from, moves, ROOK_DIRS);
          continue;
        }
        if (type === "q") {
          this.generateSlidingMoves(state, color, from, moves, BISHOP_DIRS);
          this.generateSlidingMoves(state, color, from, moves, ROOK_DIRS);
          continue;
        }
        if (type === "k") {
          this.generateKingMoves(state, color, from, moves);
        }
      }

      return moves;
    }

    updateCastlingRights(state, side, from, movingType, capturedPiece, capturedSquare) {
      if (movingType === "k") {
        if (side === WHITE) {
          state.castling.K = false;
          state.castling.Q = false;
        } else {
          state.castling.k = false;
          state.castling.q = false;
        }
      }

      if (movingType === "r") {
        if (from === 0) {
          state.castling.Q = false;
        } else if (from === 7) {
          state.castling.K = false;
        } else if (from === 56) {
          state.castling.q = false;
        } else if (from === 63) {
          state.castling.k = false;
        }
      }

      if (capturedPiece && pieceType(capturedPiece) === "r") {
        if (capturedSquare === 0) {
          state.castling.Q = false;
        } else if (capturedSquare === 7) {
          state.castling.K = false;
        } else if (capturedSquare === 56) {
          state.castling.q = false;
        } else if (capturedSquare === 63) {
          state.castling.k = false;
        }
      }
    }

    applyMoveToState(state, move) {
      const side = state.sideToMove;
      const movingPiece = state.board[move.from];
      if (!movingPiece) {
        return false;
      }

      const movingType = pieceType(movingPiece);

      const captureSquare = move.flags.enPassant
        ? (side === WHITE ? move.to - 8 : move.to + 8)
        : move.to;
      const capturedPiece =
        captureSquare >= 0 && captureSquare < 64 ? state.board[captureSquare] : null;

      state.board[move.from] = null;

      if (move.flags.enPassant && captureSquare >= 0 && captureSquare < 64) {
        state.board[captureSquare] = null;
      }

      let placedPiece = movingPiece;
      if (move.promotion) {
        placedPiece = makePiece(side, move.promotion);
      }
      state.board[move.to] = placedPiece;

      if (move.flags.castleKing) {
        if (side === WHITE) {
          state.board[7] = null;
          state.board[5] = "R";
        } else {
          state.board[63] = null;
          state.board[61] = "r";
        }
      } else if (move.flags.castleQueen) {
        if (side === WHITE) {
          state.board[0] = null;
          state.board[3] = "R";
        } else {
          state.board[56] = null;
          state.board[59] = "r";
        }
      }

      this.updateCastlingRights(
        state,
        side,
        move.from,
        movingType,
        capturedPiece,
        captureSquare,
      );

      state.enPassant = -1;
      if (movingType === "p" && Math.abs(move.to - move.from) === 16) {
        state.enPassant = side === WHITE ? move.from + 8 : move.from - 8;
      }

      if (movingType === "p" || capturedPiece) {
        state.halfmoveClock = 0;
      } else {
        state.halfmoveClock += 1;
      }

      if (side === BLACK) {
        state.fullmoveNumber += 1;
      }

      state.sideToMove = opposite(side);
      state.result = { code: "ongoing", winner: null };
      return true;
    }

    generateLegalMoves(state = this.state) {
      const side = state.sideToMove;
      const pseudo = this.generatePseudoMoves(state, side);
      const legal = [];

      for (const move of pseudo) {
        const next = this.cloneState(state);
        this.applyMoveToState(next, move);
        if (!this.isKingInCheck(next, side)) {
          legal.push(move);
        }
      }

      return legal;
    }

    hasInsufficientMaterial(state = this.state) {
      let whiteKnights = 0;
      let blackKnights = 0;
      const whiteBishops = [];
      const blackBishops = [];

      for (let sq = 0; sq < 64; sq += 1) {
        const piece = state.board[sq];
        if (!piece) {
          continue;
        }

        const type = pieceType(piece);
        const color = pieceColor(piece);

        if (type === "k") {
          continue;
        }

        if (type === "p" || type === "r" || type === "q") {
          return false;
        }

        if (type === "n") {
          if (color === WHITE) {
            whiteKnights += 1;
          } else {
            blackKnights += 1;
          }
          continue;
        }

        if (type === "b") {
          if (color === WHITE) {
            whiteBishops.push(squareParity(sq));
          } else {
            blackBishops.push(squareParity(sq));
          }
        }
      }

      const whiteMinor = whiteKnights + whiteBishops.length;
      const blackMinor = blackKnights + blackBishops.length;

      if (whiteMinor === 0 && blackMinor === 0) {
        return true;
      }

      if (whiteMinor === 1 && blackMinor === 0) {
        return true;
      }

      if (blackMinor === 1 && whiteMinor === 0) {
        return true;
      }

      if (whiteKnights === 2 && whiteBishops.length === 0 && blackMinor === 0) {
        return true;
      }

      if (blackKnights === 2 && blackBishops.length === 0 && whiteMinor === 0) {
        return true;
      }

      if (
        whiteKnights === 0 &&
        blackKnights === 0 &&
        whiteBishops.length === 1 &&
        blackBishops.length === 1 &&
        whiteBishops[0] === blackBishops[0]
      ) {
        return true;
      }

      return false;
    }

    evaluateResult() {
      const side = this.state.sideToMove;
      const legalMoves = this.generateLegalMoves(this.state);

      if (legalMoves.length === 0) {
        if (this.isKingInCheck(this.state, side)) {
          return { code: "checkmate", winner: opposite(side) };
        }
        return { code: "stalemate", winner: null };
      }

      if (this.state.halfmoveClock >= 100) {
        return { code: "fifty_move", winner: null };
      }

      if ((this.repetition.get(this.positionKey(this.state)) || 0) >= 3) {
        return { code: "threefold", winner: null };
      }

      if (this.hasInsufficientMaterial(this.state)) {
        return { code: "insufficient", winner: null };
      }

      return { code: "ongoing", winner: null };
    }

    formatMove(move, resultAfterMove) {
      let text = "";

      if (move.flags.castleKing) {
        text = "O-O";
      } else if (move.flags.castleQueen) {
        text = "O-O-O";
      } else {
        const type = pieceType(move.piece);
        const pieceLabel = type === "p" ? "" : type.toUpperCase();
        const separator = move.flags.capture || move.flags.enPassant ? "x" : "-";
        text = `${pieceLabel}${squareName(move.from)}${separator}${squareName(move.to)}`;
        if (move.promotion) {
          text += `=${move.promotion.toUpperCase()}`;
        }
        if (move.flags.enPassant) {
          text += " e.p.";
        }
      }

      if (resultAfterMove.code === "checkmate") {
        text += "#";
      } else if (this.isKingInCheck(this.state, this.state.sideToMove)) {
        text += "+";
      }

      return text;
    }

    makeMove(inputMove) {
      const legalMoves = this.generateLegalMoves(this.state);
      const chosen = legalMoves.find((move) => sameMove(move, inputMove));
      if (!chosen) {
        return null;
      }

      const snapshot = this.cloneState(this.state);
      const keyBefore = this.positionKey(this.state);

      this.applyMoveToState(this.state, chosen);

      const keyAfter = this.positionKey(this.state);
      this.bumpRepetition(keyAfter, 1);

      const result = this.evaluateResult();
      this.state.result = result;

      const notation = this.formatMove(chosen, result);

      const record = {
        move: this.cloneMove(chosen),
        notation,
        fen: this.toFEN(this.state),
        keyBefore,
        keyAfter,
        snapshot,
      };

      this.history.push(record);
      return record;
    }

    undoMove() {
      if (this.history.length === 0) {
        return null;
      }

      const currentKey = this.positionKey(this.state);
      this.bumpRepetition(currentKey, -1);

      const entry = this.history.pop();
      this.state = this.cloneState(entry.snapshot);
      return entry;
    }

    toFEN(state = this.state) {
      const ranks = [];

      for (let rank = 7; rank >= 0; rank -= 1) {
        let empty = 0;
        let line = "";

        for (let file = 0; file < 8; file += 1) {
          const piece = state.board[makeSquare(file, rank)];
          if (!piece) {
            empty += 1;
            continue;
          }
          if (empty > 0) {
            line += String(empty);
            empty = 0;
          }
          line += piece;
        }

        if (empty > 0) {
          line += String(empty);
        }

        ranks.push(line);
      }

      const castling = this.castlingToString(state.castling);
      const ep = state.enPassant >= 0 ? squareName(state.enPassant) : "-";
      return `${ranks.join("/")} ${state.sideToMove} ${castling} ${ep} ${state.halfmoveClock} ${state.fullmoveNumber}`;
    }
  }

  class ChessUI {
    constructor() {
      this.engine = new ChessEngine();

      this.bridge = {
        ...defaultBridge,
        ...(window.ChessFeatureBridge || {}),
      };
      window.ChessFeatureBridge = this.bridge;

      this.canvas = document.getElementById("board-canvas");
      this.ctx = this.canvas.getContext("2d");
      this.turnEl = document.getElementById("turn-indicator");
      this.statusEl = document.getElementById("status-indicator");
      this.fenEl = document.getElementById("fen-indicator");
      this.clockWhiteEl = document.getElementById("clock-white");
      this.clockBlackEl = document.getElementById("clock-black");
      this.moveListEl = document.getElementById("move-list");

      this.btnNew = document.getElementById("btn-new");
      this.btnUndo = document.getElementById("btn-undo");
      this.btnFlip = document.getElementById("btn-flip");
      this.btnCopyFen = document.getElementById("btn-copy-fen");

      this.sprite = new Image();
      this.spriteReady = false;
      this.sprite.onload = () => {
        this.spriteReady = true;
      };
      this.sprite.src = "chesspieces.png";

      this.boardFlipped = false;
      this.selectedSquare = -1;
      this.legalMoves = [];
      this.promotionHitboxes = [];
      this.pendingPromotion = null;

      this.pointer = { x: -1, y: -1 };
      this.isPointerDown = false;
      this.pressSquare = -1;
      this.pressPoint = { x: 0, y: 0 };
      this.dragCandidateFrom = -1;

      this.drag = {
        active: false,
        from: -1,
        piece: null,
        x: 0,
        y: 0,
      };

      this.clockInitialMs = 10 * 60 * 1000;
      this.clockIncrementMs = 0;
      this.clock = {
        remaining: {
          w: this.clockInitialMs,
          b: this.clockInitialMs,
        },
        started: false,
        running: false,
        lastTickMs: performance.now(),
      };
      this.clockHistory = [];

      this.dpr = 1;
      this.layout = {
        width: 0,
        height: 0,
        board: {
          x: 0,
          y: 0,
          size: 0,
          cell: 0,
        },
      };

      this.syncLegalMoves();
      this.bindEvents();
      this.resizeCanvas();
      this.refreshSidebar();
      this.startRenderLoop();

      Promise.resolve(this.bridge.ensureUserSession()).catch(() => {});
    }

    bindEvents() {
      this.btnNew.addEventListener("click", () => this.newGame());
      this.btnUndo.addEventListener("click", () => this.undoMove());
      this.btnFlip.addEventListener("click", () => this.flipBoard());
      this.btnCopyFen.addEventListener("click", () => this.copyFEN());

      this.canvas.addEventListener("pointerdown", (event) => this.onPointerDown(event));
      this.canvas.addEventListener("pointermove", (event) => this.onPointerMove(event));
      this.canvas.addEventListener("pointerup", (event) => this.onPointerUp(event));
      this.canvas.addEventListener("pointercancel", (event) => this.onPointerCancel(event));

      window.addEventListener("resize", () => this.resizeCanvas());
      window.addEventListener("keydown", (event) => this.onKeyDown(event));
    }

    resetClock() {
      this.clock.remaining.w = this.clockInitialMs;
      this.clock.remaining.b = this.clockInitialMs;
      this.clock.started = false;
      this.clock.running = false;
      this.clock.lastTickMs = performance.now();
      this.clockHistory = [];
      this.updateClockLabels();
    }

    captureClockSnapshot() {
      return {
        remainingW: this.clock.remaining.w,
        remainingB: this.clock.remaining.b,
        started: this.clock.started,
        running: this.clock.running,
      };
    }

    restoreClockSnapshot(snapshot) {
      if (!snapshot) {
        return;
      }
      this.clock.remaining.w = snapshot.remainingW;
      this.clock.remaining.b = snapshot.remainingB;
      this.clock.started = snapshot.started;
      this.clock.running = snapshot.running;
      this.clock.lastTickMs = performance.now();
      this.updateClockLabels();
    }

    tickClock(now = performance.now()) {
      if (!this.clock.running || !this.clock.started) {
        return false;
      }

      if (this.engine.state.result.code !== "ongoing") {
        this.clock.running = false;
        return false;
      }

      const delta = now - this.clock.lastTickMs;
      if (delta <= 0) {
        return false;
      }

      const side = this.engine.state.sideToMove;
      const key = side === WHITE ? "w" : "b";
      this.clock.remaining[key] -= delta;
      this.clock.lastTickMs = now;

      if (this.clock.remaining[key] <= 0) {
        this.clock.remaining[key] = 0;
        this.clock.running = false;
        if (this.engine.state.result.code === "ongoing") {
          this.engine.state.result = {
            code: "timeout",
            winner: opposite(side),
          };
          this.syncLegalMoves();
          this.refreshSidebar();
        }
      }

      return true;
    }

    updateClockLabels() {
      this.clockWhiteEl.textContent = formatClock(this.clock.remaining.w);
      this.clockBlackEl.textContent = formatClock(this.clock.remaining.b);

      if (this.clock.running && this.engine.state.result.code === "ongoing") {
        const active = this.engine.state.sideToMove;
        this.clockWhiteEl.style.color = active === WHITE ? "#ffffff" : "#cad7f0";
        this.clockBlackEl.style.color = active === BLACK ? "#ffffff" : "#cad7f0";
      } else {
        this.clockWhiteEl.style.color = "#cad7f0";
        this.clockBlackEl.style.color = "#cad7f0";
      }
    }

    onKeyDown(event) {
      if (event.key === "f" || event.key === "F") {
        this.flipBoard();
        return;
      }

      if (event.key === "u" || event.key === "U") {
        this.undoMove();
        return;
      }

      if (event.key === "Escape" && this.pendingPromotion) {
        this.pendingPromotion = null;
        this.promotionHitboxes = [];
      }
    }

    newGame() {
      this.engine.reset();
      this.selectedSquare = -1;
      this.pendingPromotion = null;
      this.promotionHitboxes = [];
      this.drag.active = false;
      this.resetClock();
      this.syncLegalMoves();
      this.refreshSidebar();
    }

    undoMove() {
      this.pendingPromotion = null;
      this.promotionHitboxes = [];
      this.drag.active = false;

      if (!this.engine.undoMove()) {
        return;
      }

      this.selectedSquare = -1;
      if (this.clockHistory.length > 0) {
        this.restoreClockSnapshot(this.clockHistory.pop());
      } else {
        this.resetClock();
      }
      this.syncLegalMoves();
      this.refreshSidebar();
    }

    flipBoard() {
      this.boardFlipped = !this.boardFlipped;
    }

    async copyFEN() {
      const fen = this.engine.toFEN();
      try {
        await navigator.clipboard.writeText(fen);
      } catch (_err) {
        window.prompt("Copy FEN:", fen);
      }
    }

    syncLegalMoves() {
      this.legalMoves = this.engine.generateLegalMoves();
    }

    refreshSidebar() {
      const state = this.engine.state;
      if (state.result.code !== "ongoing") {
        this.clock.running = false;
      }
      const sideLabel = colorName(state.sideToMove);
      this.turnEl.textContent = sideLabel;

      let statusText = "In progress";
      if (state.result.code !== "ongoing") {
        statusText = this.resultToText(state.result);
      } else if (this.engine.isKingInCheck(state, state.sideToMove)) {
        statusText = `${sideLabel} is in check`;
      }
      this.statusEl.textContent = statusText;

      this.fenEl.textContent = this.engine.toFEN();

      this.btnUndo.disabled = this.engine.history.length === 0;
      this.updateClockLabels();
      this.renderMoveList();
    }

    renderMoveList() {
      this.moveListEl.innerHTML = "";
      const records = this.engine.history;
      if (records.length === 0) {
        const row = document.createElement("div");
        row.className = "move-row";
        row.innerHTML = "<span>-</span><span>No moves yet</span><span></span>";
        this.moveListEl.appendChild(row);
        return;
      }

      for (let i = 0; i < records.length; i += 2) {
        const moveNumber = Math.floor(i / 2) + 1;
        const whiteMove = records[i] ? records[i].notation : "";
        const blackMove = records[i + 1] ? records[i + 1].notation : "";

        const row = document.createElement("div");
        row.className = "move-row";

        const n = document.createElement("span");
        n.textContent = `${moveNumber}.`;
        const w = document.createElement("span");
        w.textContent = whiteMove;
        const b = document.createElement("span");
        b.textContent = blackMove;

        row.appendChild(n);
        row.appendChild(w);
        row.appendChild(b);
        this.moveListEl.appendChild(row);
      }

      this.moveListEl.scrollTop = this.moveListEl.scrollHeight;
    }

    resultToText(result) {
      if (result.code === "checkmate") {
        return `Checkmate. ${colorName(result.winner)} wins`;
      }
      if (result.code === "timeout") {
        return `Time out. ${colorName(result.winner)} wins`;
      }
      if (result.code === "stalemate") {
        return "Draw by stalemate";
      }
      if (result.code === "fifty_move") {
        return "Draw by 50-move rule";
      }
      if (result.code === "threefold") {
        return "Draw by repetition";
      }
      if (result.code === "insufficient") {
        return "Draw by insufficient material";
      }
      return "In progress";
    }

    getCanvasPoint(event) {
      const rect = this.canvas.getBoundingClientRect();
      return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
    }

    onPointerDown(event) {
      if (event.button !== 0) {
        return;
      }

      const point = this.getCanvasPoint(event);
      this.pointer = point;

      if (this.pendingPromotion) {
        this.handlePromotionClick(point.x, point.y);
        event.preventDefault();
        return;
      }

      this.isPointerDown = true;
      this.pressSquare = this.screenToSquare(point.x, point.y);
      this.pressPoint = point;
      this.dragCandidateFrom = -1;

      if (this.pressSquare >= 0 && this.engine.state.result.code === "ongoing") {
        const piece = this.engine.state.board[this.pressSquare];
        if (piece && pieceColor(piece) === this.engine.state.sideToMove) {
          this.dragCandidateFrom = this.pressSquare;
        }
      }

      try {
        this.canvas.setPointerCapture(event.pointerId);
      } catch (_err) {}

      event.preventDefault();
    }

    onPointerMove(event) {
      const point = this.getCanvasPoint(event);
      this.pointer = point;

      if (!this.isPointerDown) {
        return;
      }

      if (this.drag.active) {
        this.drag.x = point.x;
        this.drag.y = point.y;
        event.preventDefault();
        return;
      }

      if (this.dragCandidateFrom >= 0) {
        const dx = point.x - this.pressPoint.x;
        const dy = point.y - this.pressPoint.y;
        if (dx * dx + dy * dy >= 16) {
          this.startDrag(this.dragCandidateFrom, point);
        }
      }

      event.preventDefault();
    }

    onPointerUp(event) {
      if (event.button !== 0) {
        return;
      }

      if (!this.isPointerDown && !this.drag.active) {
        return;
      }

      const point = this.getCanvasPoint(event);
      this.pointer = point;

      if (this.pendingPromotion) {
        this.handlePromotionClick(point.x, point.y);
        this.resetPointerState(event);
        event.preventDefault();
        return;
      }

      const releaseSquare = this.screenToSquare(point.x, point.y);

      if (this.drag.active) {
        this.finishDrag(releaseSquare);
      } else {
        this.handleClickRelease(releaseSquare);
      }

      this.resetPointerState(event);
      event.preventDefault();
    }

    onPointerCancel(event) {
      this.drag.active = false;
      this.canvas.classList.remove("dragging");
      this.resetPointerState(event);
    }

    resetPointerState(event) {
      this.isPointerDown = false;
      this.pressSquare = -1;
      this.dragCandidateFrom = -1;
      this.drag.active = false;
      this.canvas.classList.remove("dragging");

      try {
        if (event && this.canvas.hasPointerCapture(event.pointerId)) {
          this.canvas.releasePointerCapture(event.pointerId);
        }
      } catch (_err) {}
    }

    startDrag(fromSquare, point) {
      const piece = this.engine.state.board[fromSquare];
      if (!piece) {
        return;
      }

      this.drag.active = true;
      this.drag.from = fromSquare;
      this.drag.piece = piece;
      this.drag.x = point.x;
      this.drag.y = point.y;
      this.selectedSquare = fromSquare;
      this.canvas.classList.add("dragging");
    }

    finishDrag(releaseSquare) {
      const from = this.drag.from;
      const moved = releaseSquare >= 0 && this.tryMove(from, releaseSquare);

      if (!moved) {
        if (releaseSquare === from) {
          this.selectedSquare = from;
        } else if (releaseSquare >= 0) {
          const piece = this.engine.state.board[releaseSquare];
          if (piece && pieceColor(piece) === this.engine.state.sideToMove) {
            this.selectedSquare = releaseSquare;
          } else {
            this.selectedSquare = -1;
          }
        } else {
          this.selectedSquare = -1;
        }
      }
    }

    handleClickRelease(releaseSquare) {
      if (releaseSquare < 0) {
        this.selectedSquare = -1;
        return;
      }

      if (this.engine.state.result.code !== "ongoing") {
        this.selectedSquare = -1;
        return;
      }

      if (this.selectedSquare >= 0 && this.selectedSquare !== releaseSquare) {
        if (this.tryMove(this.selectedSquare, releaseSquare)) {
          return;
        }
      }

      const piece = this.engine.state.board[releaseSquare];
      if (piece && pieceColor(piece) === this.engine.state.sideToMove) {
        this.selectedSquare = this.selectedSquare === releaseSquare ? -1 : releaseSquare;
      } else {
        this.selectedSquare = -1;
      }
    }

    movesForPair(from, to) {
      return this.legalMoves.filter((move) => move.from === from && move.to === to);
    }

    tryMove(from, to, promotion = null) {
      if (from < 0 || to < 0) {
        return false;
      }

      const candidates = this.movesForPair(from, to);
      if (candidates.length === 0) {
        return false;
      }

      let chosen = null;
      if (promotion) {
        chosen = candidates.find((move) => move.promotion === promotion);
        if (!chosen) {
          return false;
        }
      } else {
        const promoMoves = candidates.filter((move) => Boolean(move.promotion));
        if (promoMoves.length > 0) {
          this.pendingPromotion = {
            from,
            to,
            color: this.engine.state.sideToMove,
            moves: promoMoves,
          };
          this.promotionHitboxes = [];
          this.selectedSquare = from;
          return true;
        }

        chosen = candidates[0];
      }

      return this.executeMove(chosen);
    }

    handlePromotionClick(x, y) {
      for (const hit of this.promotionHitboxes) {
        if (x >= hit.x && x <= hit.x + hit.w && y >= hit.y && y <= hit.y + hit.h) {
          const pending = this.pendingPromotion;
          if (!pending) {
            return;
          }
          this.pendingPromotion = null;
          this.promotionHitboxes = [];
          this.tryMove(pending.from, pending.to, hit.promotion);
          return;
        }
      }
    }

    executeMove(move) {
      const now = performance.now();
      this.tickClock(now);
      if (this.engine.state.result.code !== "ongoing") {
        return false;
      }
      const movingSide = this.engine.state.sideToMove;
      const clockSnapshot = this.captureClockSnapshot();

      const record = this.engine.makeMove(move);
      if (!record) {
        return false;
      }

      this.clockHistory.push(clockSnapshot);
      if (!this.clock.started) {
        this.clock.started = true;
      }
      if (this.clockIncrementMs > 0) {
        const key = movingSide === WHITE ? "w" : "b";
        this.clock.remaining[key] += this.clockIncrementMs;
      }
      this.clock.running = this.engine.state.result.code === "ongoing";
      this.clock.lastTickMs = now;

      this.selectedSquare = -1;
      this.pendingPromotion = null;
      this.promotionHitboxes = [];

      this.syncLegalMoves();
      this.refreshSidebar();

      this.notifyMove(record).catch(() => {});
      return true;
    }

    async notifyMove(record) {
      const payload = {
        move: {
          from: squareName(record.move.from),
          to: squareName(record.move.to),
          promotion: record.move.promotion || null,
        },
        notation: record.notation,
        fen: record.fen,
        sideToMove: this.engine.state.sideToMove,
        history: this.engine.history.map((entry) => entry.notation),
      };
      await this.bridge.onMoveCommitted(payload);
    }

    resizeCanvas() {
      const rect = this.canvas.getBoundingClientRect();
      const width = Math.max(1, Math.round(rect.width));
      const height = Math.max(1, Math.round(rect.height));

      this.dpr = window.devicePixelRatio || 1;
      const targetW = Math.round(width * this.dpr);
      const targetH = Math.round(height * this.dpr);

      if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
        this.canvas.width = targetW;
        this.canvas.height = targetH;
      }

      this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      this.layout.width = width;
      this.layout.height = height;

      const pad = clamp(Math.round(Math.min(width, height) * 0.04), 10, 28);
      const size = Math.max(80, Math.min(width, height) - pad * 2);

      this.layout.board = {
        x: (width - size) * 0.5,
        y: (height - size) * 0.5,
        size,
        cell: size / 8,
      };
    }

    squareToRect(square) {
      const board = this.layout.board;
      const file = fileOf(square);
      const rank = rankOf(square);

      const col = this.boardFlipped ? 7 - file : file;
      const row = this.boardFlipped ? rank : 7 - rank;

      return {
        x: board.x + col * board.cell,
        y: board.y + row * board.cell,
        w: board.cell,
        h: board.cell,
      };
    }

    screenToSquare(x, y) {
      const board = this.layout.board;
      if (
        x < board.x ||
        y < board.y ||
        x > board.x + board.size ||
        y > board.y + board.size
      ) {
        return -1;
      }

      const col = Math.floor((x - board.x) / board.cell);
      const row = Math.floor((y - board.y) / board.cell);
      if (col < 0 || col > 7 || row < 0 || row > 7) {
        return -1;
      }

      const file = this.boardFlipped ? 7 - col : col;
      const rank = this.boardFlipped ? row : 7 - row;

      return makeSquare(file, rank);
    }

    drawPieceInRect(piece, rect, scale = 0.9, alpha = 1) {
      const ctx = this.ctx;
      const padW = (rect.w * (1 - scale)) * 0.5;
      const padH = (rect.h * (1 - scale)) * 0.5;
      const dx = rect.x + padW;
      const dy = rect.y + padH;
      const dw = rect.w - padW * 2;
      const dh = rect.h - padH * 2;

      ctx.save();
      ctx.globalAlpha = alpha;

      if (this.spriteReady) {
        const type = pieceType(piece);
        const color = pieceColor(piece);
        const col = PIECE_TO_SPRITE_COLUMN[type];
        const row = color === WHITE ? 0 : 1;
        const sw = this.sprite.width / 6;
        const sh = this.sprite.height / 2;
        const sx = col * sw;
        const sy = row * sh;
        ctx.drawImage(this.sprite, sx, sy, sw, sh, dx, dy, dw, dh);
      } else {
        const label = piece;
        ctx.fillStyle = pieceColor(piece) === WHITE ? "#f8f8f8" : "#111";
        ctx.font = `${Math.floor(rect.h * 0.58)}px "Avenir Next", sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, rect.x + rect.w * 0.5, rect.y + rect.h * 0.54);
      }

      ctx.restore();
    }

    drawCoordinates() {
      const ctx = this.ctx;
      const board = this.layout.board;
      const cell = board.cell;

      ctx.save();
      ctx.fillStyle = BOARD_COLORS.coord;
      ctx.font = `${Math.floor(cell * 0.17)}px Menlo, Consolas, monospace`;

      for (let col = 0; col < 8; col += 1) {
        const fileLabel = this.boardFlipped ? FILES[7 - col] : FILES[col];
        const x = board.x + col * cell + cell * 0.83;
        const y = board.y + board.size - 6;
        ctx.fillText(fileLabel, x, y);
      }

      for (let row = 0; row < 8; row += 1) {
        const rankLabel = this.boardFlipped ? String(row + 1) : String(8 - row);
        const x = board.x + 5;
        const y = board.y + row * cell + cell * 0.2;
        ctx.fillText(rankLabel, x, y);
      }

      ctx.restore();
    }

    drawMoveHints() {
      if (this.selectedSquare < 0 || this.engine.state.result.code !== "ongoing") {
        return;
      }

      const ctx = this.ctx;
      const moves = this.legalMoves.filter((move) => move.from === this.selectedSquare);
      if (moves.length === 0) {
        return;
      }

      const byTarget = new Map();
      for (const move of moves) {
        const prev = byTarget.get(move.to) || false;
        byTarget.set(move.to, prev || move.flags.capture || move.flags.enPassant);
      }

      for (const [target, isCapture] of byTarget.entries()) {
        const rect = this.squareToRect(target);
        const cx = rect.x + rect.w * 0.5;
        const cy = rect.y + rect.h * 0.5;

        if (isCapture) {
          ctx.strokeStyle = BOARD_COLORS.captureRing;
          ctx.lineWidth = Math.max(2, rect.w * 0.055);
          ctx.beginPath();
          ctx.arc(cx, cy, rect.w * 0.34, 0, Math.PI * 2);
          ctx.stroke();
        } else {
          ctx.fillStyle = BOARD_COLORS.legalDot;
          ctx.beginPath();
          ctx.arc(cx, cy, rect.w * 0.12, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    drawBoardSquares() {
      const ctx = this.ctx;
      const board = this.layout.board;

      const bg = ctx.createLinearGradient(0, 0, this.layout.width, this.layout.height);
      bg.addColorStop(0, BOARD_COLORS.bg0);
      bg.addColorStop(1, BOARD_COLORS.bg1);
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, this.layout.width, this.layout.height);

      const framePad = Math.max(10, board.size * 0.02);
      ctx.fillStyle = BOARD_COLORS.frame;
      ctx.fillRect(
        board.x - framePad,
        board.y - framePad,
        board.size + framePad * 2,
        board.size + framePad * 2,
      );

      for (let sq = 0; sq < 64; sq += 1) {
        const rect = this.squareToRect(sq);
        const light = ((fileOf(sq) + rankOf(sq)) & 1) === 0;
        ctx.fillStyle = light ? BOARD_COLORS.light : BOARD_COLORS.dark;
        ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
      }

      const last = this.engine.history[this.engine.history.length - 1];
      if (last) {
        const fromRect = this.squareToRect(last.move.from);
        const toRect = this.squareToRect(last.move.to);
        ctx.fillStyle = BOARD_COLORS.lastMove;
        ctx.fillRect(fromRect.x, fromRect.y, fromRect.w, fromRect.h);
        ctx.fillRect(toRect.x, toRect.y, toRect.w, toRect.h);
      }

      if (this.selectedSquare >= 0) {
        const selectedRect = this.squareToRect(this.selectedSquare);
        ctx.fillStyle = BOARD_COLORS.selected;
        ctx.fillRect(selectedRect.x, selectedRect.y, selectedRect.w, selectedRect.h);
      }

      this.drawMoveHints();
      this.drawCheckHighlights();
      this.drawCoordinates();
    }

    drawCheckHighlights() {
      const ctx = this.ctx;
      const state = this.engine.state;

      for (const color of [WHITE, BLACK]) {
        if (!this.engine.isKingInCheck(state, color)) {
          continue;
        }
        const kingSquare = this.engine.findKingSquare(state, color);
        if (kingSquare < 0) {
          continue;
        }
        const rect = this.squareToRect(kingSquare);
        ctx.strokeStyle = BOARD_COLORS.check;
        ctx.lineWidth = Math.max(2, rect.w * 0.06);
        ctx.strokeRect(rect.x + 1, rect.y + 1, rect.w - 2, rect.h - 2);
      }
    }

    drawPieces() {
      for (let sq = 0; sq < 64; sq += 1) {
        if (this.drag.active && sq === this.drag.from) {
          continue;
        }
        const piece = this.engine.state.board[sq];
        if (!piece) {
          continue;
        }
        this.drawPieceInRect(piece, this.squareToRect(sq), 0.9);
      }

      if (this.drag.active && this.drag.piece) {
        const cell = this.layout.board.cell;
        const size = cell * 0.96;
        const rect = {
          x: this.drag.x - size * 0.5,
          y: this.drag.y - size * 0.5,
          w: size,
          h: size,
        };

        const ctx = this.ctx;
        ctx.save();
        ctx.fillStyle = BOARD_COLORS.dragShadow;
        ctx.beginPath();
        ctx.ellipse(
          rect.x + rect.w * 0.5,
          rect.y + rect.h * 0.88,
          rect.w * 0.28,
          rect.h * 0.09,
          0,
          0,
          Math.PI * 2,
        );
        ctx.fill();
        ctx.restore();

        this.drawPieceInRect(this.drag.piece, rect, 1);
      }
    }

    drawPromotionModal() {
      if (!this.pendingPromotion) {
        this.promotionHitboxes = [];
        return;
      }

      const ctx = this.ctx;
      const board = this.layout.board;

      this.promotionHitboxes = [];

      ctx.fillStyle = BOARD_COLORS.promoBackdrop;
      ctx.fillRect(0, 0, this.layout.width, this.layout.height);

      const modalW = Math.min(board.size * 0.9, 430);
      const modalH = Math.min(board.size * 0.44, 220);
      const modalX = board.x + (board.size - modalW) * 0.5;
      const modalY = board.y + (board.size - modalH) * 0.5;

      roundRectPath(ctx, modalX, modalY, modalW, modalH, 16);
      ctx.fillStyle = BOARD_COLORS.promoCard;
      ctx.fill();
      ctx.strokeStyle = BOARD_COLORS.promoStroke;
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = "#edf2ff";
      ctx.font = `${Math.floor(modalH * 0.15)}px "Avenir Next", sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText("Choose Promotion", modalX + modalW * 0.5, modalY + modalH * 0.2);

      const buttonSize = Math.min(88, (modalW - 70) / 4);
      const gap = (modalW - buttonSize * 4) / 5;
      const startX = modalX + gap;
      const y = modalY + modalH * 0.36;

      for (let i = 0; i < PROMOTION_TYPES.length; i += 1) {
        const promo = PROMOTION_TYPES[i];
        const x = startX + i * (buttonSize + gap);
        const hover =
          this.pointer.x >= x &&
          this.pointer.x <= x + buttonSize &&
          this.pointer.y >= y &&
          this.pointer.y <= y + buttonSize;

        roundRectPath(ctx, x, y, buttonSize, buttonSize, 10);
        ctx.fillStyle = hover ? BOARD_COLORS.promoBtnHover : BOARD_COLORS.promoBtn;
        ctx.fill();
        ctx.strokeStyle = "rgba(211, 223, 250, 0.68)";
        ctx.lineWidth = 2;
        ctx.stroke();

        const piece = makePiece(this.pendingPromotion.color, promo);
        this.drawPieceInRect(piece, { x, y, w: buttonSize, h: buttonSize }, 0.88);

        this.promotionHitboxes.push({
          promotion: promo,
          x,
          y,
          w: buttonSize,
          h: buttonSize,
        });
      }

      ctx.textAlign = "center";
      ctx.fillStyle = "#bccdeb";
      ctx.font = `${Math.floor(modalH * 0.11)}px "Avenir Next", sans-serif`;
      ctx.fillText("Tap the piece you want", modalX + modalW * 0.5, modalY + modalH * 0.9);
    }

    renderFrame(now) {
      if (this.tickClock(now)) {
        this.updateClockLabels();
      }
      this.drawBoardSquares();
      this.drawPieces();
      this.drawPromotionModal();
    }

    startRenderLoop() {
      const tick = (now) => {
        this.renderFrame(now);
        window.requestAnimationFrame(tick);
      };
      window.requestAnimationFrame(tick);
    }
  }

  window.ChessEngine = ChessEngine;
  window.ChessUI = ChessUI;

  window.addEventListener("DOMContentLoaded", () => {
    const app = new ChessUI();
    window.chessUI = app;
    window.chessEngine = app.engine;
  });
})();
