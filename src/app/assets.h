#ifndef ASSETS_H
#define ASSETS_H

#include <raylib.h>

typedef struct ChessAssets {
    Texture2D pieces_texture;
    Rectangle piece_src[2][6];
    Image icon_image;
    bool loaded;
} ChessAssets;

bool assets_load(ChessAssets *assets);
void assets_set_window_icon(ChessAssets *assets);
void assets_unload(ChessAssets *assets);

#endif
