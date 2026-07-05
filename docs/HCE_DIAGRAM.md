# Current HCE Map

This is the current handcrafted engine path. It covers the Classic backend and
also the search shell used by the NN backend.

```mermaid
flowchart TD
    A["chess_ai_pick_move / UCI go"] --> B["hce_pick_move"]
    B --> C["run_search"]
    C --> D{"Opening book usable?"}
    D -- yes --> E["Return weighted book move"]
    D -- no --> F["Iterative deepening"]
    F --> G["Aspiration window"]
    G --> H["search_root"]
    H --> I["Move ordering: TT, previous root, captures, killers, history"]
    I --> J["negamax"]
    J --> K{"Terminal / draw / max ply?"}
    K -- yes --> L["Mate/draw/static score"]
    K -- no --> M["TT probe"]
    M --> N["Static eval"]
    N --> O["Reverse futility pruning"]
    O --> P["Null move pruning"]
    P --> Q["Generate legal moves"]
    Q --> R["Order moves"]
    R --> S["PVS search"]
    S --> T["Extensions: check, promotion, recapture"]
    S --> U["LMR for late quiets"]
    U --> V["TT store + killer/history updates"]
    J --> W["quiescence at depth 0"]
    W --> X["Stand pat"]
    X --> Y["Captures/promotions/check evasions"]
    Y --> Z["SEE and delta pruning"]
```

## Eval Flow

```mermaid
flowchart TD
    A["engine_eval_cp_stm"] --> B{"Backend"}
    B -- classic --> C["hce_eval_cp_stm"]
    B -- experimental --> D["hce_experimental_eval_cp_stm"]
    B -- nn --> E["nn_eval_cp_stm / accumulator frame"]
    C --> F["phase_value"]
    F --> G["eval_side white"]
    F --> H["eval_side black"]
    G --> I["white terms"]
    H --> J["black terms"]
    I --> K["material"]
    I --> L["piece-square tables"]
    I --> M["pawn structure"]
    I --> N["passed pawns"]
    I --> O["outposts"]
    I --> P["bishop quality / pair"]
    I --> Q["rook files"]
    I --> R["mobility"]
    I --> S["king safety"]
    I --> T["hanging pieces"]
    I --> U["queen trap"]
    K --> V["MG/EG blend"]
    L --> V
    M --> V
    N --> V
    O --> V
    P --> V
    Q --> V
    R --> V
    S --> V
    T --> V
    U --> V
    V --> W["white - black"]
    W --> X["side-to-move sign"]
    X --> Y["+12 tempo"]
```

## Current Tuning Hotspots

- Main-search capture ordering is MVV/LVA. Prior SEE capture-ordering variants
  were measured and rejected, so do not revive them without a new isolated
  hypothesis.
- Promotions receive a tactical bonus but are not forced above all ordinary
  captures. A promotion-first ordering test was measured and rejected.
- The eval is rich but not tuned from data. Most values are hand-balanced and
  should be treated as hypotheses.
- The current source data experiments showed early Lichess eval slices are
  opening-heavy, so any tuning/eval learning needs phase-balanced position sets.
- Search and eval both contain king-safety logic. That is good, but it means
  bad king-safety weights can amplify through extensions and move ordering.
- The Classic and NN backend share the same search shell, so search improvements
  should help both unless they depend on HCE-specific eval confidence.
