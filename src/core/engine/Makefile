CC ?= cc
CFLAGS ?= -std=c11 -O3 -Wall -Wextra -Wpedantic
HCE_SEARCH_OPT ?= -O2
DEPFLAGS ?= -MMD -MP
LDFLAGS ?=
THREAD_FLAGS ?= -pthread
LDLIBS ?= -lm

SRC_DIR := $(if $(wildcard src/chess_ai.c),src,.)
BUILD_DIR := build
ENGINE := chess_uci
INCLUDES := -I$(SRC_DIR)

SRCS := \
	$(SRC_DIR)/chess_state.c \
	$(SRC_DIR)/chess_hash.c \
	$(SRC_DIR)/chess_rules.c \
	$(SRC_DIR)/chess_io.c \
	$(SRC_DIR)/chess_opening_book.c \
	$(SRC_DIR)/chess_ai.c \
	$(SRC_DIR)/nn_eval.c \
	$(SRC_DIR)/hce_eval.c \
	$(SRC_DIR)/hce_search.c \
	$(SRC_DIR)/uci_main.c

OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean

all: $(ENGINE)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/hce_search.o: $(SRC_DIR)/hce_search.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(filter-out -O%,$(CFLAGS)) $(HCE_SEARCH_OPT) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(ENGINE): $(OBJS)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR) $(ENGINE)

-include $(DEPS)
