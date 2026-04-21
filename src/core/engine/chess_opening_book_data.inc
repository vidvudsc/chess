    {"", {"e2e4", "d2d4", "g1f3", "c2c4", "f2f4", "e2e4", "d2d4"}, 7},

    // First replies
    {"e2e4", {"e7e5", "c7c5", "e7e6", "c7c6", "d7d6", "g8f6", "b8c6", "e7e5"}, 8},
    {"d2d4", {"d7d5", "g8f6", "e7e6", "g7g6", "c7c6", "f7f5", "d7d6", "d7d5"}, 8},
    {"c2c4", {"e7e5", "g8f6", "e7e6", "c7c5", "d7d5", "e7e5"}, 6},
    {"g1f3", {"d7d5", "g8f6", "d7d5", "g7g6", "e7e6", "c7c5"}, 6},
    {"f2f4", {"d7d5", "g8f6", "g7g6", "e7e6", "d7d5"}, 5},

    // Open games / King's pawn game: 1.e4 e5
    {"e2e4 e7e5", {"g1f3", "f1c4", "b1c3", "f2f4", "d2d4", "g1f3"}, 6},
    {"e2e4 e7e5 g1f3", {"b8c6", "g8f6", "d7d6", "b8c6"}, 4},
    {"e2e4 e7e5 g1f3 b8c6", {"f1b5", "f1c4", "d2d4", "b1c3", "f1b5"}, 5},

    // Ruy Lopez
    {"e2e4 e7e5 g1f3 b8c6 f1b5", {"a7a6", "g8f6", "a7a6"}, 3},
    {"e2e4 e7e5 g1f3 b8c6 f1b5 a7a6", {"b5a4", "b5c4"}, 2},
    {"e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4", {"g8f6", "f8e7"}, 2},

    // Italian / Giuoco Piano / Two Knights
    {"e2e4 e7e5 g1f3 b8c6 f1c4", {"f8c5", "g8f6", "f8c5"}, 3},
    {"e2e4 e7e5 g1f3 b8c6 f1c4 f8c5", {"c2c3", "d2d3", "b1c3", "c2c3"}, 4},
    {"e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3", {"g8f6", "d7d6"}, 2},
    {"e2e4 e7e5 g1f3 b8c6 f1c4 g8f6", {"d2d3", "b1c3", "g2g3"}, 3}, // Two Knights quiet systems
    {"e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 b1c3", {"f8c5", "f8e7"}, 2},

    // Evans Gambit
    {"e2e4 e7e5 g1f3 b8c6 f1c4 f8c5", {"b2b4", "c2c3", "d2d3", "b2b4"}, 4},
    {"e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4", {"c5b4", "c5e7"}, 2},
    {"e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 b2b4 c5b4", {"c2c3"}, 1},

    // Four Knights
    {"e2e4 e7e5 g1f3 b8c6 b1c3", {"g8f6", "f8c5", "g8f6"}, 3},
    {"e2e4 e7e5 g1f3 b8c6 b1c3 g8f6", {"f1b5", "d2d4", "f1c4"}, 3},
    {"e2e4 e7e5 g1f3 b8c6 b1c3 g8f6 f1b5", {"f8b4"}, 1}, // Double Spanish

    // Scotch
    {"e2e4 e7e5 g1f3 b8c6", {"d2d4", "f1b5", "f1c4", "d2d4"}, 4},
    {"e2e4 e7e5 g1f3 b8c6 d2d4", {"e5d4"}, 1},
    {"e2e4 e7e5 g1f3 b8c6 d2d4 e5d4", {"f3d4"}, 1},
    {"e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4", {"f8c5", "g8f6"}, 2},

    // Petrov
    {"e2e4 e7e5 g1f3 g8f6", {"f3e5", "b1c3", "d2d4"}, 3},
    {"e2e4 e7e5 g1f3 g8f6 f3e5", {"d7d6"}, 1},

    // Philidor
    {"e2e4 e7e5 g1f3 d7d6", {"d2d4", "f1c4", "c2c3"}, 3},
    {"e2e4 e7e5 g1f3 d7d6 d2d4", {"e5d4"}, 1},

    // King's Gambit
    {"e2e4 e7e5", {"f2f4", "g1f3", "f1c4", "f2f4"}, 4},
    {"e2e4 e7e5 f2f4", {"e5f4", "d7d5", "e5f4"}, 3},
    {"e2e4 e7e5 f2f4 e5f4", {"g1f3"}, 1},
    {"e2e4 e7e5 f2f4 d7d5", {"e4d5"}, 1}, // Falkbeer countergambit style

    // Vienna
    {"e2e4 e7e5", {"b1c3", "g1f3", "f1c4", "b1c3"}, 4},
    {"e2e4 e7e5 b1c3", {"g8f6", "b8c6", "g8f6"}, 3},
    {"e2e4 e7e5 b1c3 g8f6", {"g2g3", "f2f4", "g1f3"}, 3},

    // Sicilian Defense
    {"e2e4 c7c5", {"g1f3", "b1c3", "c2c3", "g1f3"}, 4},
    {"e2e4 c7c5 g1f3", {"d7d6", "e7e6", "b8c6", "g8f6"}, 4},
    {"e2e4 c7c5 g1f3 d7d6", {"d2d4", "b1c3", "d2d4"}, 3},
    {"e2e4 c7c5 g1f3 d7d6 d2d4", {"c5d4"}, 1},
    {"e2e4 c7c5 g1f3 d7d6 d2d4 c5d4", {"f3d4"}, 1},
    {"e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4", {"g8f6", "b8c6", "a7a6", "e7e6"}, 4},
    {"e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6", {"b1c3"}, 1},
    {"e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3", {"a7a6"}, 1}, // Najdorf

    // French Defense
    {"e2e4 e7e6", {"d2d4", "g1f3", "d2d4"}, 3},
    {"e2e4 e7e6 d2d4", {"d7d5"}, 1},
    {"e2e4 e7e6 d2d4 d7d5", {"b1c3", "b1d2", "e4e5", "b1c3"}, 4},
    {"e2e4 e7e6 d2d4 d7d5 b1c3", {"f8b4", "g8f6", "d5e4"}, 3}, // Winawer / classical split
    {"e2e4 e7e6 d2d4 d7d5 b1d2", {"g8f6", "c7c5"}, 2},

    // Caro-Kann Defense
    {"e2e4 c7c6", {"d2d4", "b1c3", "d2d4"}, 3},
    {"e2e4 c7c6 d2d4", {"d7d5"}, 1},
    {"e2e4 c7c6 d2d4 d7d5", {"b1c3", "b1d2", "e4e5", "e4d5", "b1c3"}, 5},
    {"e2e4 c7c6 d2d4 d7d5 e4e5", {"c8f5", "c8g4"}, 2}, // Advance
    {"e2e4 c7c6 d2d4 d7d5 e4d5", {"c6d5"}, 1},

    // Alekhine / Nimzowitsch / Pirc / Modern / Scandinavian
    {"e2e4 g8f6", {"e4e5", "b1c3", "d2d4", "e4e5"}, 4}, // Alekhine
    {"e2e4 g8f6 e4e5", {"f6d5"}, 1},
    {"e2e4 g8f6 e4e5 f6d5", {"d2d4", "c2c4", "g1f3"}, 3},

    {"e2e4 b8c6", {"d2d4", "g1f3", "d2d4"}, 3}, // Nimzowitsch
    {"e2e4 b8c6 d2d4", {"d7d5", "e7e5"}, 2},

    {"e2e4 d7d6", {"d2d4", "g1f3", "d2d4"}, 3}, // Pirc
    {"e2e4 d7d6 d2d4", {"g8f6", "g7g6"}, 2},
    {"e2e4 d7d6 d2d4 g8f6", {"b1c3", "g1f3"}, 2},
    {"e2e4 d7d6 d2d4 g8f6 b1c3", {"g7g6"}, 1},
    {"e2e4 d7d6 d2d4 g8f6 b1c3 g7g6", {"g1f3", "f1e2"}, 2},

    {"d2d4 g7g6", {"e2e4", "c2c4", "g1f3"}, 3}, // Modern defense setups
    {"d2d4 g7g6 e2e4", {"d7d6", "f8g7"}, 2},
    {"d2d4 g7g6 e2e4 d7d6", {"b1c3", "g1f3"}, 2},

    {"e2e4 d7d5", {"e4d5"}, 1}, // Scandinavian
    {"e2e4 d7d5 e4d5", {"d8d5"}, 1},
    {"e2e4 d7d5 e4d5 d8d5", {"b1c3"}, 1},
    {"e2e4 d7d5 e4d5 d8d5 b1c3", {"d5a5", "d5d6", "d5d8"}, 3},

    // Queen's pawn game / Queen's gambit complex
    {"d2d4 d7d5", {"c2c4", "g1f3", "c1f4", "c2c4"}, 4},
    {"d2d4 d7d5 c2c4", {"e7e6", "c7c6", "d5c4", "e7e6"}, 4}, // QGD / Slav / QGA
    {"d2d4 d7d5 c2c4 e7e6", {"b1c3", "g1f3", "b1c3"}, 3}, // QGD
    {"d2d4 d7d5 c2c4 e7e6 g1f3", {"g8f6", "c7c6"}, 2},     // Semi-Slav style
    {"d2d4 d7d5 c2c4 c7c6", {"b1c3", "g1f3", "b1c3"}, 3}, // Slav
    {"d2d4 d7d5 c2c4 d5c4", {"g1f3", "e2e3", "a2a4"}, 3}, // QGA
    {"d2d4 d7d5 c2c4 d5c4 g1f3", {"g8f6", "a7a6"}, 2},

    // London / Colle / Torre
    {"d2d4 g8f6", {"g1f3", "c2c4", "c1f4", "c2c4"}, 4},
    {"d2d4 g8f6 g1f3", {"d7d5", "e7e6", "g7g6"}, 3},
    {"d2d4 g8f6 g1f3 d7d5", {"c1f4", "e2e3", "c1g5"}, 3}, // London / Colle / Torre
    {"d2d4 g8f6 g1f3 d7d5 c1f4", {"e7e6", "c7c5"}, 2},    // London
    {"d2d4 g8f6 g1f3 d7d5 e2e3", {"e7e6", "c7c5"}, 2},    // Colle
    {"d2d4 g8f6 g1f3 e7e6", {"c1g5", "c1f4", "e2e3"}, 3}, // Torre

    // Dutch / Wade
    {"d2d4 f7f5", {"g2g3", "c2c4", "g1f3"}, 3}, // Dutch defense
    {"d2d4 d7d6", {"g1f3", "e2e4", "c2c4"}, 3},
    {"d2d4 d7d6 g1f3", {"c8g4", "g8f6", "c7c6"}, 3}, // Wade
    {"d2d4 d7d6 g1f3 c8g4", {"c2c4", "e2e4"}, 2},

    // Indian complex (Nimzo / Bogo / QID / KID / Gruenfeld / Benoni / Benko / Budapest)
    {"d2d4 g8f6 c2c4", {"e7e6", "g7g6", "c7c5", "e7e6"}, 4},
    {"d2d4 g8f6 c2c4 e7e6", {"b1c3", "g1f3", "b1c3"}, 3},
    {"d2d4 g8f6 c2c4 e7e6 b1c3", {"f8b4", "d7d5", "f8b4"}, 3}, // Nimzo-Indian
    {"d2d4 g8f6 c2c4 e7e6 g1f3", {"b7b6", "f8b4", "d7d5", "c7c5"}, 4}, // QID / Bogo / QGD
    {"d2d4 g8f6 c2c4 e7e6 g1f3 f8b4", {"c1d2"}, 1}, // Bogo-Indian check line idea
    {"d2d4 g8f6 c2c4 e7e6 g1f3 b7b6", {"g2g3", "b1c3"}, 2},    // Queen's Indian

    {"d2d4 g8f6 c2c4 g7g6", {"b1c3", "g1f3", "b1c3"}, 3},      // KID / Gruenfeld split
    {"d2d4 g8f6 c2c4 g7g6 b1c3", {"f8g7", "d7d5", "f8g7"}, 3},
    {"d2d4 g8f6 c2c4 g7g6 b1c3 d7d5", {"c4d5", "g1f3"}, 2},    // Gruenfeld themes
    {"d2d4 g8f6 c2c4 g7g6 b1c3 f8g7", {"e2e4", "g1f3"}, 2},    // KID classical setups

    {"d2d4 g8f6 c2c4 c7c5", {"d4d5", "g1f3"}, 2},              // Benoni
    {"d2d4 g8f6 c2c4 c7c5 d4d5", {"e7e6", "b7b5"}, 2},         // Modern Benoni / Benko
    {"d2d4 g8f6 c2c4 c7c5 d4d5 b7b5", {"c4b5", "g1f3"}, 2},    // Benko gambit
    {"d2d4 g8f6 c2c4 c7c5 d4d5 b7b5 c4b5", {"a7a6"}, 1},

    {"d2d4 g8f6 c2c4 e7e5", {"d4e5", "g1f3"}, 2},              // Budapest gambit
    {"d2d4 g8f6 c2c4 e7e5 d4e5", {"f6g4", "f6e4"}, 2},
    {"d2d4 g8f6 c2c4 e7e5 d4e5 f6g4", {"c1f4", "g1f3"}, 2},

    // Catalan
    {"d2d4 g8f6 c2c4 e7e6", {"g2g3", "g1f3", "b1c3"}, 3},
    {"d2d4 g8f6 c2c4 e7e6 g2g3", {"d7d5", "f8b4"}, 2},
    {"d2d4 g8f6 c2c4 e7e6 g2g3 d7d5", {"f1g2", "g1f3"}, 2},

    // English opening
    {"c2c4 e7e5", {"b1c3", "g2g3", "g1f3", "b1c3"}, 4},
    {"c2c4 g8f6", {"b1c3", "g2g3", "g1f3", "b1c3"}, 4},
    {"c2c4 e7e6", {"b1c3", "g2g3", "g1f3"}, 3},
    {"c2c4 c7c5", {"b1c3", "g2g3", "g1f3"}, 3},

    // Reti / King's Indian Attack
    {"g1f3 d7d5", {"c2c4", "g2g3", "d2d4", "c2c4"}, 4},        // Reti themes
    {"g1f3 g8f6", {"c2c4", "g2g3", "d2d4", "g2g3"}, 4},
    {"g1f3 d7d5 g2g3", {"g8f6", "c7c6", "c8g4"}, 3},           // KIA / Barcza style
    {"g1f3 d7d5 g2g3 g8f6", {"f1g2", "e1g1"}, 2},
    {"g1f3 d7d5 c2c4", {"e7e6", "d5c4", "c7c6"}, 3},

    // Bird opening
    {"f2f4 d7d5", {"g1f3", "e2e3", "g2g3"}, 3},
    {"f2f4 d7d5 g1f3", {"g8f6", "e7e6"}, 2},
    {"f2f4 g8f6", {"g1f3", "e2e3", "g2g3"}, 3},
