#include "assets.h"

#include <string.h>

bool assets_load(ChessAssets *assets) {
    if (assets == NULL) {
        return false;
    }

    memset(assets, 0, sizeof(*assets));

    assets->pieces_texture = LoadTexture("images/chesspieces.png");
    if (assets->pieces_texture.id == 0) {
        return false;
    }

    int sprite_w = assets->pieces_texture.width / 6;
    int sprite_h = assets->pieces_texture.height / 2;

    for (int color = 0; color < 2; ++color) {
        for (int piece = 0; piece < 6; ++piece) {
            assets->piece_src[color][piece] = (Rectangle){
                .x = (float)(piece * sprite_w),
                .y = (float)(color * sprite_h),
                .width = (float)sprite_w,
                .height = (float)sprite_h,
            };
        }
    }

    SetTextureFilter(assets->pieces_texture, TEXTURE_FILTER_BILINEAR);

    assets->icon_image = LoadImage("images/chess.png");
    assets->loaded = true;
    return true;
}

void assets_set_window_icon(ChessAssets *assets) {
    if (assets == NULL) {
        return;
    }
#if defined(__APPLE__)
    // GLFW/raylib regular windows do not support custom icons on macOS.
    (void)assets;
    return;
#else
    if (assets->icon_image.data != NULL) {
        SetWindowIcon(assets->icon_image);
    }
#endif
}

void assets_unload(ChessAssets *assets) {
    if (assets == NULL || !assets->loaded) {
        return;
    }

    if (assets->pieces_texture.id != 0) {
        UnloadTexture(assets->pieces_texture);
    }
    if (assets->icon_image.data != NULL) {
        UnloadImage(assets->icon_image);
    }

    memset(assets, 0, sizeof(*assets));
}
