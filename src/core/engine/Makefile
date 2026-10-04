CC ?= cc
CFLAGS ?= -std=c11 -O3 -Wall -Wextra -Wpedantic
# w64devkit/mingw hides POSIX pieces (clock_gettime) under strict c11.
ifeq ($(OS),Windows_NT)
CFLAGS := $(subst -std=c11,-std=gnu11,$(CFLAGS))
endif
HCE_SEARCH_OPT ?= -O2
DEPFLAGS ?= -MMD -MP
LDFLAGS ?=
THREAD_FLAGS ?= -pthread
LDLIBS ?= -lm

SRC_DIR := $(if $(wildcard src/chess_ai.c),src,.)
BUILD_DIR := build
ENGINE := chess_uci
INCLUDES := -I$(SRC_DIR) -I$(SRC_DIR)/fathom

SRCS := \
	$(SRC_DIR)/chess_state.c \
	$(SRC_DIR)/chess_hash.c \
	$(SRC_DIR)/chess_rules.c \
	$(SRC_DIR)/chess_io.c \
	$(SRC_DIR)/chess_opening_book.c \
	$(SRC_DIR)/chess_ai.c \
	$(SRC_DIR)/nn_eval.c \
	$(SRC_DIR)/nn_search.c \
	$(SRC_DIR)/nn_search_support.c \
	$(SRC_DIR)/hce_eval.c \
	$(SRC_DIR)/hce_search.c \
	$(SRC_DIR)/hce_tb.c \
	$(SRC_DIR)/fathom/tbprobe.c \
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

# Vendored Fathom (MIT): upstream code, built without our pedantic warnings.
$(BUILD_DIR)/fathom/tbprobe.o: $(SRC_DIR)/fathom/tbprobe.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(filter-out -Wall -Wextra -Wpedantic -std=c11,$(CFLAGS)) -std=gnu11 -w $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/hce_search.o: $(SRC_DIR)/hce_search.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(filter-out -O%,$(CFLAGS)) $(HCE_SEARCH_OPT) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/nn_search.o: $(SRC_DIR)/nn_search.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(filter-out -O%,$(CFLAGS)) $(HCE_SEARCH_OPT) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

# Vendored Fathom (MIT): upstream code, built without our pedantic warnings.
$(BUILD_DIR)/fathom/tbprobe.o: $(SRC_DIR)/fathom/tbprobe.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(filter-out -Wall -Wextra -Wpedantic -std=c11,$(CFLAGS)) -std=gnu11 -w $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(ENGINE): $(OBJS)
	$(CC) $(CFLAGS) $(THREAD_FLAGS) $(OBJS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf $(BUILD_DIR) $(ENGINE)

-include $(DEPS)
