CC ?= clang
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O3
HCE_SEARCH_OPT ?= -O2
DEPFLAGS ?= -MMD -MP
LDFLAGS ?=
THREAD_FLAGS ?= -pthread
LDLIBS ?= -lm

RAYLIB_CFLAGS := $(shell pkg-config --cflags raylib)
RAYLIB_LIBS := $(shell pkg-config --libs raylib)
UNAME_S := $(shell uname -s)

SRC_DIR := src
SRC_CORE_DIR := $(SRC_DIR)/core
SRC_ENGINE_DIR := $(SRC_CORE_DIR)/engine
SRC_BOT_DIR := $(SRC_CORE_DIR)/bot
SRC_APP_DIR := $(SRC_DIR)/app
TEST_DIR := tests
BUILD_DIR := build
BIN_DIR := bin

INCLUDES := -I$(SRC_ENGINE_DIR) -I$(SRC_APP_DIR)

CORE_SRCS := \
	$(SRC_ENGINE_DIR)/chess_state.c \
	$(SRC_ENGINE_DIR)/chess_hash.c \
	$(SRC_ENGINE_DIR)/chess_rules.c \
	$(SRC_ENGINE_DIR)/chess_io.c \
	$(SRC_ENGINE_DIR)/chess_opening_book.c \
	$(SRC_ENGINE_DIR)/chess_ai.c \
	$(SRC_ENGINE_DIR)/nn_eval.c \
	$(SRC_ENGINE_DIR)/hce_eval.c \
	$(SRC_ENGINE_DIR)/hce_search.c

APP_SRCS := \
	$(SRC_APP_DIR)/main.c \
	$(SRC_APP_DIR)/game_log.c \
	$(SRC_APP_DIR)/ui.c \
	$(SRC_APP_DIR)/ai_test_lab_cli.c \
	$(SRC_APP_DIR)/chess_ai_worker.c \
	$(SRC_APP_DIR)/stockfish_eval_worker.c \
	$(SRC_APP_DIR)/ai_test_runner.c \
	$(SRC_APP_DIR)/assets.c

UCI_SRC := $(SRC_ENGINE_DIR)/uci_main.c
AI_TEST_LAB_CLI_SRC := $(SRC_APP_DIR)/ai_test_lab_cli.c
AI_TEST_RUNNER_SRC := $(SRC_APP_DIR)/ai_test_runner.c
AI_TEST_LAB_MAIN_SRC := $(SRC_APP_DIR)/ai_test_lab_main.c

CORE_OBJS := $(patsubst $(SRC_ENGINE_DIR)/%.c,$(BUILD_DIR)/engine/%.o,$(CORE_SRCS))
APP_OBJS := $(patsubst $(SRC_APP_DIR)/%.c,$(BUILD_DIR)/app/%.o,$(APP_SRCS))
UCI_OBJ := $(BUILD_DIR)/engine/uci_main.o
AI_TEST_LAB_CLI_OBJ := $(patsubst $(SRC_APP_DIR)/%.c,$(BUILD_DIR)/app/%.o,$(AI_TEST_LAB_CLI_SRC))
AI_TEST_RUNNER_OBJ := $(patsubst $(SRC_APP_DIR)/%.c,$(BUILD_DIR)/app/%.o,$(AI_TEST_RUNNER_SRC))
AI_TEST_LAB_MAIN_OBJ := $(patsubst $(SRC_APP_DIR)/%.c,$(BUILD_DIR)/app/%.o,$(AI_TEST_LAB_MAIN_SRC))
TEST_OBJS := \
	$(BUILD_DIR)/tests/test_rules.o \
	$(BUILD_DIR)/tests/test_clock.o \
	$(BUILD_DIR)/tests/test_perft_suite.o \
	$(BUILD_DIR)/tests/test_ai.o \
	$(BUILD_DIR)/tests/test_tactical_regressions.o \
	$(BUILD_DIR)/tests/perft_main.o \
	$(BUILD_DIR)/tests/bench_engine.o
DEPS := $(CORE_OBJS:.o=.d) $(APP_OBJS:.o=.d) $(UCI_OBJ:.o=.d) $(TEST_OBJS:.o=.d)

ICON_SRC := images/chess.png
ICNS_PATH := images/chess.icns
APP_DIR := dist/Chess.app
CONTENTS_DIR := $(APP_DIR)/Contents
MACOS_DIR := $(CONTENTS_DIR)/MacOS
RES_DIR := $(CONTENTS_DIR)/Resources

.PHONY: all run run-bin run-app test perft bench uci ai_test_lab snapshot_engine fetch_positions build_testlab_positions hce_testlab umbrel_bundle deploy_umbrel clean icns bundle

all: $(BIN_DIR)/chess

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR)/engine:
	mkdir -p $(BUILD_DIR)/engine

$(BUILD_DIR)/app:
	mkdir -p $(BUILD_DIR)/app

$(BUILD_DIR)/tests:
	mkdir -p $(BUILD_DIR)/tests

$(BUILD_DIR)/engine/%.o: $(SRC_ENGINE_DIR)/%.c | $(BUILD_DIR)/engine
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/engine/hce_search.o: $(SRC_ENGINE_DIR)/hce_search.c | $(BUILD_DIR)/engine
	$(CC) $(filter-out -O%,$(CFLAGS)) $(HCE_SEARCH_OPT) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/app/%.o: $(SRC_APP_DIR)/%.c | $(BUILD_DIR)/app
	$(CC) $(CFLAGS) $(DEPFLAGS) $(THREAD_FLAGS) $(RAYLIB_CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)/tests
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/chess: $(CORE_OBJS) $(APP_OBJS) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) $(CORE_OBJS) $(APP_OBJS) -o $@ $(RAYLIB_LIBS) $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/chess_uci: $(CORE_OBJS) $(UCI_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) $(CORE_OBJS) $(UCI_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/ai_test_lab: $(CORE_OBJS) $(AI_TEST_RUNNER_OBJ) $(AI_TEST_LAB_CLI_OBJ) $(AI_TEST_LAB_MAIN_OBJ) | $(BIN_DIR)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) $(CORE_OBJS) $(AI_TEST_RUNNER_OBJ) $(AI_TEST_LAB_CLI_OBJ) $(AI_TEST_LAB_MAIN_OBJ) -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/test_rules: $(CORE_OBJS) $(BUILD_DIR)/tests/test_rules.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJS) $(BUILD_DIR)/tests/test_rules.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/test_clock: $(CORE_OBJS) $(BUILD_DIR)/tests/test_clock.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJS) $(BUILD_DIR)/tests/test_clock.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/test_perft_suite: $(CORE_OBJS) $(BUILD_DIR)/tests/test_perft_suite.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJS) $(BUILD_DIR)/tests/test_perft_suite.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/test_ai: $(CORE_OBJS) $(BUILD_DIR)/tests/test_ai.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJS) $(BUILD_DIR)/tests/test_ai.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/test_tactical_regressions: $(CORE_OBJS) $(BUILD_DIR)/tests/test_tactical_regressions.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJS) $(BUILD_DIR)/tests/test_tactical_regressions.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/perft: $(CORE_OBJS) $(BUILD_DIR)/tests/perft_main.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJS) $(BUILD_DIR)/tests/perft_main.o -o $@ $(LDFLAGS) $(LDLIBS)

$(BIN_DIR)/bench_engine: $(CORE_OBJS) $(BUILD_DIR)/tests/bench_engine.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CORE_OBJS) $(BUILD_DIR)/tests/bench_engine.o -o $@ $(LDFLAGS) $(LDLIBS)

run:
	$(MAKE) run-bin

run-bin: $(BIN_DIR)/chess
	./$(BIN_DIR)/chess

run-app: bundle
ifeq ($(UNAME_S),Darwin)
	open "$(APP_DIR)"
else
	@echo "run-app is macOS-only; falling back to run-bin."
	$(MAKE) run-bin
endif

test: $(BIN_DIR)/test_rules $(BIN_DIR)/test_clock $(BIN_DIR)/test_perft_suite $(BIN_DIR)/test_ai $(BIN_DIR)/test_tactical_regressions
	./$(BIN_DIR)/test_rules
	./$(BIN_DIR)/test_clock
	./$(BIN_DIR)/test_perft_suite
	./$(BIN_DIR)/test_ai
	./$(BIN_DIR)/test_tactical_regressions

perft: $(BIN_DIR)/perft
	./$(BIN_DIR)/perft

bench: $(BIN_DIR)/bench_engine
	./$(BIN_DIR)/bench_engine

uci: $(BIN_DIR)/chess_uci
	./$(BIN_DIR)/chess_uci

ai_test_lab: $(BIN_DIR)/ai_test_lab
	./$(BIN_DIR)/ai_test_lab $(ARGS)

snapshot_engine: $(BIN_DIR)/chess_uci
	@set -eu; \
	NAME="$${NAME:-snapshot_`date +%Y%m%d_%H%M%S`}"; \
	DEST_DIR="current/engine_snapshots/$$NAME"; \
	mkdir -p "$$DEST_DIR"; \
	cp "$(BIN_DIR)/chess_uci" "$$DEST_DIR/chess_uci"; \
	chmod +x "$$DEST_DIR/chess_uci"; \
	printf 'Saved engine snapshot to %s\n' "$$DEST_DIR/chess_uci"

fetch_positions:
	python3 scripts/build_testlab_positions.py \
		--pgn data/pgn/lichess_db_standard_rated_2016-02.pgn.zst \
		--out data/positions/lichess_equal_positions.fen \
		--count 500

build_testlab_positions:
	python3 scripts/build_testlab_positions.py \
		--pgn data/pgn/lichess_db_standard_rated_2016-02.pgn.zst \
		--out data/positions/lichess_equal_positions.fen \
		--count 500 \
		--start-game-index 120000 \
		--max-games-scan 300000 \
		--min-avg-elo 2350 \
		--min-each-elo 2200 \
		--min-ply 12 \
		--max-ply 60 \
		--sample-every 4 \
		--eval-window-cp 70 \
		--material-window-cp 260 \
		--movetime-ms 18

hce_testlab:
	python3 $(SRC_BOT_DIR)/test_lab.py $(ARGS)

umbrel_bundle:
	./scripts/package_umbrel_hce.sh

deploy_umbrel:
	./scripts/deploy_umbrel_hce.sh $(ARGS)

icns: $(ICON_SRC)
	@python3 -c "from PIL import Image; im=Image.open('$(ICON_SRC)').convert('RGBA'); im.save('$(ICNS_PATH)', format='ICNS', sizes=[(16,16),(32,32),(64,64),(128,128),(256,256),(512,512),(1024,1024)]); print('Created $(ICNS_PATH)')"

bundle: $(BIN_DIR)/chess icns
	@set -eu; \
	rm -rf "$(APP_DIR)"; \
	mkdir -p "$(MACOS_DIR)" "$(RES_DIR)"; \
	cp "$(BIN_DIR)/chess" "$(MACOS_DIR)/Chess"; \
	chmod +x "$(MACOS_DIR)/Chess"; \
	ICON_FILE="chess"; \
	if [ -f "$(ICNS_PATH)" ]; then \
	  cp "$(ICNS_PATH)" "$(RES_DIR)/chess.icns"; \
	else \
	  cp "$(ICON_SRC)" "$(RES_DIR)/chess.png"; \
	  ICON_FILE="chess.png"; \
	fi; \
	{ \
	  echo '<?xml version="1.0" encoding="UTF-8"?>'; \
	  echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'; \
	  echo '<plist version="1.0">'; \
	  echo '<dict>'; \
	  echo '  <key>CFBundleName</key>'; \
	  echo '  <string>Chess</string>'; \
	  echo '  <key>CFBundleDisplayName</key>'; \
	  echo '  <string>Chess</string>'; \
	  echo '  <key>CFBundleIdentifier</key>'; \
	  echo '  <string>com.local.chess</string>'; \
	  echo '  <key>CFBundleVersion</key>'; \
	  echo '  <string>1.0</string>'; \
	  echo '  <key>CFBundleShortVersionString</key>'; \
	  echo '  <string>1.0</string>'; \
	  echo '  <key>CFBundleExecutable</key>'; \
	  echo '  <string>Chess</string>'; \
	  echo '  <key>CFBundlePackageType</key>'; \
	  echo '  <string>APPL</string>'; \
	  echo '  <key>LSMinimumSystemVersion</key>'; \
	  echo '  <string>12.0</string>'; \
	  echo '  <key>NSHighResolutionCapable</key>'; \
	  echo '  <true/>'; \
	  echo '  <key>CFBundleIconFile</key>'; \
	  echo "  <string>$$ICON_FILE</string>"; \
	  echo '</dict>'; \
	  echo '</plist>'; \
	} > "$(CONTENTS_DIR)/Info.plist"
	@echo "Created $(APP_DIR)"

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)/chess $(BIN_DIR)/chess_uci $(BIN_DIR)/ai_test_lab $(BIN_DIR)/test_rules $(BIN_DIR)/test_clock $(BIN_DIR)/test_perft_suite $(BIN_DIR)/test_ai $(BIN_DIR)/test_tactical_regressions $(BIN_DIR)/perft $(BIN_DIR)/bench_engine dist/Chess.app dist/umbrel_hce $(ICNS_PATH) src/core/engine/build src/core/engine/chess_uci src/core/bot/__pycache__

-include $(DEPS)
