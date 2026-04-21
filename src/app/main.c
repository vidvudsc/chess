#include <stdbool.h>
#include <string.h>

#include "ai_test_lab_cli.h"
#include "ui.h"

static bool should_run_ai_test_cli(int argc, char **argv) {
    if (argc < 2 || argv == NULL || argv[1] == NULL) {
        return false;
    }
    return strcmp(argv[1], "--ai-test-lab") == 0 || strcmp(argv[1], "ai-test-lab") == 0;
}

int main(int argc, char **argv) {
    if (should_run_ai_test_cli(argc, argv)) {
        return chess_run_ai_test_cli(argc, argv);
    }
    return chess_run_ui();
}
