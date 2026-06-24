#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <inttypes.h>
#include <switch.h>
#include <time.h>
#include <ft2build.h>
#include FT_FREETYPE_H

#define FB_WIDTH  1280
#define FB_HEIGHT 720

#define MAX_LINES 8
#define MAX_LEN 256
#define MAX_VERSES 20
#define MAX_AUDIO_SIZE (2 * 1024 * 1024)

#define COLOR_BLACK   RGBA8_MAXALPHA(0, 0, 0)
#define COLOR_CYAN    RGBA8_MAXALPHA(0, 210, 255)
#define COLOR_YELLOW  RGBA8_MAXALPHA(255, 210, 0)
#define COLOR_WHITE   RGBA8_MAXALPHA(255, 255, 255)
#define COLOR_GREEN   RGBA8_MAXALPHA(50, 220, 80)
#define COLOR_RED     RGBA8_MAXALPHA(255, 70, 70)
#define COLOR_BLUE    RGBA8_MAXALPHA(100, 150, 255)
#define COLOR_MAGENTA RGBA8_MAXALPHA(255, 0, 255)

typedef enum {
    SCENE_MENU,
    SCENE_PRACTICE
} Scene;

typedef struct {
    char quarter[256];
    char verse[MAX_LINES][MAX_LEN];
    int currentVerse;
    bool answerShown;
    int verseOrder[MAX_VERSES];
    int verseIndex;
    
    Scene currentScene;
    bool isVsMode;
    int playerTurn; 
    int scores[2];
    
    AudioDriver drv;
    AudioDriverWaveBuf wavebuf;
    void* mempool_ptr;
    int mempool_id;
    bool initedDriver;
    bool initedAudren;
} AppState;

static u32 framebuf_width = 0;
static u64 LanguageCode;

uint32_t sanitize_unicode(uint32_t c) {
    switch(c) {
        case 0x0101: return 'a'; case 0x0100: return 'A'; 
        case 0x012B: return 'i'; case 0x012A: return 'I'; 
        case 0x016B: return 'u'; case 0x016A: return 'U'; 
        case 0x1E5B: return 'r'; case 0x1E5A: return 'R'; 
        case 0x1E5D: return 'r'; case 0x1E5C: return 'R'; 
        case 0x1E37: return 'l'; case 0x1E36: return 'L'; 
        case 0x1E39: return 'l'; case 0x1E38: return 'L'; 
        case 0x1E43: return 'm'; case 0x1E42: return 'M'; 
        case 0x1E25: return 'h'; case 0x1E24: return 'H'; 
        case 0x1E45: return 'n'; case 0x1E44: return 'N'; 
        case 0x00F1: return 'n'; case 0x00D1: return 'N'; 
        case 0x1E6D: return 't'; case 0x1E6C: return 'T'; 
        case 0x1E0D: return 'd'; case 0x1E0C: return 'D'; 
        case 0x1E47: return 'n'; case 0x1E46: return 'N'; 
        case 0x015B: return 's'; case 0x015A: return 'S'; 
        case 0x1E63: return 's'; case 0x1E62: return 'S'; 
        default: return c;
    }
}

u32 get_smooth_loop_color(u32 frame) {
    u32 phase = (frame / 2) % 360; 
    u8 r = 0, g = 0, b = 0;

    if (phase < 60) {
        r = 255; g = (phase * 255) / 60; b = 0;             
    } else if (phase < 120) {
        r = 255 - ((phase - 60) * 255) / 60; g = 255; b = 0; 
    } else if (phase < 180) {
        r = 0; g = 255; b = ((phase - 120) * 255) / 60;     
    } else if (phase < 240) {
        r = 0; g = 255 - ((phase - 180) * 255) / 60; b = 255; 
    } else if (phase < 300) {
        r = ((phase - 240) * 255) / 60; g = 0; b = 255;     
    } else {
        r = 255; g = 0; b = 255 - ((phase - 300) * 255) / 60; 
    }

    return RGBA8_MAXALPHA(r, g, b);
}

void draw_glyph(FT_Bitmap* bitmap, u32* framebuf, u32 x, u32 y, u32 color) {
    if (bitmap->pixel_mode != FT_PIXEL_MODE_GRAY) return;
    u8* imageptr = bitmap->buffer;
    for (u32 tmpy = 0; tmpy < bitmap->rows; tmpy++) {
        for (u32 tmpx = 0; tmpx < bitmap->width; tmpx++) {
            u32 framex = x + tmpx;
            u32 framey = y + tmpy;
            if (framex >= FB_WIDTH || framey >= FB_HEIGHT) continue;

            u8 alpha = imageptr[tmpx];
            if (alpha > 0) framebuf[framey * framebuf_width + framex] = color;
        }
        imageptr += bitmap->pitch;
    }
}

void draw_text(FT_Face face, u32* framebuf, u32 x, u32 y, const char* str, u32 color) {
    u32 tmpx = x;
    FT_UInt glyph_index;
    FT_GlyphSlot slot = face->glyph;
    u32 i = 0;
    u32 str_size = strlen(str);
    uint32_t tmpchar;
    ssize_t unitcount = 0;

    for (i = 0; i < str_size; ) {
        unitcount = decode_utf8(&tmpchar, (const uint8_t*)&str[i]);
        if (unitcount <= 0) break;
        i += unitcount;

        if (tmpchar == '\n') {
            tmpx = x;
            y += face->size->metrics.height / 64;
            continue;
        }

        glyph_index = FT_Get_Char_Index(face, tmpchar);
        
        if (glyph_index == 0) {
            uint32_t fallback_char = sanitize_unicode(tmpchar);
            glyph_index = FT_Get_Char_Index(face, fallback_char);
        }

        if (FT_Load_Glyph(face, glyph_index, FT_LOAD_DEFAULT) == 0) {
            FT_Render_Glyph(face->glyph, FT_RENDER_MODE_NORMAL);
        } else continue;

        draw_glyph(&slot->bitmap, framebuf, tmpx + slot->bitmap_left, y - slot->bitmap_top, color);
        tmpx += slot->advance.x >> 6;
        y += slot->advance.y >> 6;
    }
}

void draw_text_shadow(FT_Face face, u32* framebuf, u32 x, u32 y, const char* str, u32 color) {
    draw_text(face, framebuf, x + 2, y + 2, str, COLOR_BLACK); 
    draw_text(face, framebuf, x, y, str, color);              
}

bool loadPcm(AppState* state, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    size_t pcm_size = ftell(f);
    rewind(f);

    if (pcm_size > MAX_AUDIO_SIZE) { fclose(f); return false; }
    fread(state->mempool_ptr, 1, pcm_size, f);
    fclose(f);

    int16_t* pcm_data = (int16_t*)state->mempool_ptr;
    for (size_t i = 0; i < pcm_size / 2; i++) {
        int32_t sample = pcm_data[i] * 4; 
        if (sample > 32767) sample = 32767;    
        if (sample < -32768) sample = -32768;  
        pcm_data[i] = (int16_t)sample;
    }

    armDCacheFlush(state->mempool_ptr, pcm_size);
    state->wavebuf.data_raw = state->mempool_ptr;
    state->wavebuf.size = pcm_size;
    state->wavebuf.start_sample_offset = 0;
    state->wavebuf.end_sample_offset = pcm_size / 2;
    state->wavebuf.state = AudioDriverWaveBufState_Free;
    return true;
}

void playLoadedAudio(AppState* state) {
    audrvVoiceStop(&state->drv, 0);
    state->wavebuf.state = AudioDriverWaveBufState_Free;
    audrvVoiceAddWaveBuf(&state->drv, 0, &state->wavebuf);
    audrvVoiceStart(&state->drv, 0);
}

void loadQuarter(AppState* state, int verseNum) {
    char path[256];
    snprintf(path, sizeof(path), "sdmc:/switch/Anukrama/meta/verse_%d.json", verseNum);

    FILE* f = fopen(path, "r");
    if (!f) { snprintf(state->quarter, sizeof(state->quarter), "META FILE NOT FOUND"); return; }

    char buf[2048];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f); buf[n] = 0;

    char* p = strstr(buf, "\"english\"");
    if (!p) { snprintf(state->quarter, sizeof(state->quarter), "MISSING KEY \"english\""); return; }
    p = strchr(p, ':');
    if (!p) { snprintf(state->quarter, sizeof(state->quarter), "MALFORMED METADATA"); return; }
    
    p++; while (*p == ' ' || *p == '"') p++; 
    int i = 0;
    while (*p && *p != '"' && i < (sizeof(state->quarter) - 1)) state->quarter[i++] = *p++;
    state->quarter[i] = 0;
}

void loadVerse(AppState* state, int verseNum) {
    memset(state->verse, 0, sizeof(state->verse));
    char path[256];
    snprintf(path, sizeof(path), "sdmc:/switch/Anukrama/verses/english/verse_%d.txt", verseNum);

    FILE* f = fopen(path, "r");
    if (!f) { snprintf(state->verse[0], MAX_LEN, "VERSE NOT FOUND"); return; }

    for (int i = 0; i < MAX_LINES; i++) {
        if (!fgets(state->verse[i], MAX_LEN, f)) break;
        state->verse[i][strcspn(state->verse[i], "\r\n")] = 0;
    }
    fclose(f);
}

void shuffleVerses(AppState* state) {
    for (int i = 0; i < MAX_VERSES; i++) state->verseOrder[i] = i + 1;
    for (int i = MAX_VERSES - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = state->verseOrder[i];
        state->verseOrder[i] = state->verseOrder[j];
        state->verseOrder[j] = tmp;
    }
    state->verseIndex = 0;
}

void chooseNewVerse(AppState* state) {
    if (state->verseIndex >= MAX_VERSES) shuffleVerses(state);
    state->currentVerse = state->verseOrder[state->verseIndex++];
    state->answerShown = false;
    loadQuarter(state, state->currentVerse);
    loadVerse(state, state->currentVerse);
}

void playPrompt(AppState* state) {
    char path[256];
    snprintf(path, sizeof(path), "sdmc:/switch/Anukrama/audio/chapter_15/v%02d_prompt.pcm", state->currentVerse);
    if (loadPcm(state, path)) playLoadedAudio(state);
}

void playAnswer(AppState* state) {
    char path[256];
    snprintf(path, sizeof(path), "sdmc:/switch/Anukrama/audio/chapter_15/v%02d.pcm", state->currentVerse);
    if (loadPcm(state, path)) playLoadedAudio(state);
}

void userAppInit(void) {
    plInitialize(PlServiceType_User);
    setInitialize();
    setGetSystemLanguage(&LanguageCode);
    setExit();
}

void userAppExit(void) { plExit(); }

int main(void) {
    padConfigureInput(1, HidNpadStyleSet_NpadStandard);
    PadState pad;
    padInitializeDefault(&pad);
    srand(time(NULL));

    AppState app = {0};
    app.currentVerse = 1;
    app.currentScene = SCENE_MENU;
    app.mempool_ptr = memalign(0x1000, MAX_AUDIO_SIZE);

    if (!app.mempool_ptr) return -1;

    PlFontData font;
    plGetSharedFontByType(&font, PlSharedFontType_Standard);
    FT_Library library;
    FT_Init_FreeType(&library);
    FT_Face face;
    FT_New_Memory_Face(library, font.address, font.size, 0, &face);
    FT_Set_Char_Size(face, 0, 26 * 64, 96, 96); 

    Framebuffer fb;
    framebufferCreate(&fb, nwindowGetDefault(), FB_WIDTH, FB_HEIGHT, PIXEL_FORMAT_RGBA_8888, 2);
    framebufferMakeLinear(&fb);

    static const AudioRendererConfig arConfig = {
        .output_rate = AudioRendererOutputRate_48kHz,
        .num_voices = 24, .num_sinks = 1, .num_mix_objs = 1, .num_mix_buffers = 2,
    };

    if (R_SUCCEEDED(audrenInitialize(&arConfig))) {
        app.initedAudren = true;
        if (R_SUCCEEDED(audrvCreate(&app.drv, &arConfig, 2))) {
            app.initedDriver = true;
            app.mempool_id = audrvMemPoolAdd(&app.drv, app.mempool_ptr, MAX_AUDIO_SIZE);
            audrvMemPoolAttach(&app.drv, app.mempool_id);

            static const u8 sink_channels[] = {0, 1};
            audrvDeviceSinkAdd(&app.drv, AUDREN_DEFAULT_DEVICE_NAME, 2, sink_channels);
            audrvUpdate(&app.drv); audrenStartAudioRenderer();

            audrvVoiceInit(&app.drv, 0, 1, PcmFormat_Int16, 48000);
            audrvVoiceSetDestinationMix(&app.drv, 0, AUDREN_FINAL_MIX_ID);
            audrvVoiceSetMixFactor(&app.drv, 0, 1.0f, 0, 0);
            audrvVoiceSetMixFactor(&app.drv, 0, 1.0f, 0, 1);
            audrvVoiceStart(&app.drv, 0);
        }
    }

    u32 global_frame = 0; 

    while (appletMainLoop()) {
        padUpdate(&pad);
        u64 kDown = padGetButtonsDown(&pad);
        global_frame++;

        if (kDown & HidNpadButton_Plus) break;

        if (app.initedDriver) {
            if (app.currentScene == SCENE_MENU) {
                if (kDown & HidNpadButton_A) {
                    app.isVsMode = false;
                    app.currentScene = SCENE_PRACTICE;
                    shuffleVerses(&app); chooseNewVerse(&app); playPrompt(&app);
                } else if (kDown & HidNpadButton_X) {
                    app.isVsMode = true;
                    app.playerTurn = 0;
                    app.scores[0] = 0; app.scores[1] = 0;
                    app.currentScene = SCENE_PRACTICE;
                    shuffleVerses(&app); chooseNewVerse(&app); playPrompt(&app);
                }
            } else if (app.currentScene == SCENE_PRACTICE) {
                if (!app.answerShown) {
                    if (kDown & HidNpadButton_A) {
                        app.answerShown = true;
                        playAnswer(&app);
                    }
                } else {
                    if (app.isVsMode) {
                        if (kDown & HidNpadButton_L) {
                            app.scores[app.playerTurn]++; 
                            app.playerTurn = !app.playerTurn; 
                            chooseNewVerse(&app); playPrompt(&app);
                        } else if (kDown & HidNpadButton_R) {
                            app.playerTurn = !app.playerTurn; 
                            chooseNewVerse(&app); playPrompt(&app);
                        }
                    } else {
                        if (kDown & HidNpadButton_A) {
                            chooseNewVerse(&app); playPrompt(&app);
                        }
                    }
                }
            }
            audrvUpdate(&app.drv);
        }

        u32 stride;
        u32* framebuf = (u32*)framebufferBegin(&fb, &stride);
        framebuf_width = stride / sizeof(u32);

        for (u32 y = 0; y < FB_HEIGHT; y++) {
            for (u32 x = 0; x < FB_WIDTH; x++) {
                u8 r = (x / 20 + global_frame / 4) % 20;     
                u8 g = (y / 20 + global_frame / 6) % 15;     
                u8 b = 25 + ((x + y) / 15 + global_frame) % 30; 
                framebuf[y * framebuf_width + x] = RGBA8_MAXALPHA(r, g, b);
            }
        }

        if (app.currentScene == SCENE_MENU) {
            draw_text_shadow(face, framebuf, 40, 200, "==================================================", COLOR_CYAN);
            draw_text_shadow(face, framebuf, 40, 240, "                ANUKRAMA PRACTICE                 ", COLOR_YELLOW);
            draw_text_shadow(face, framebuf, 40, 280, "==================================================", COLOR_CYAN);
            
            draw_text_shadow(face, framebuf, 100, 360, ">> Press [A] for Solo Practice", COLOR_GREEN);
            draw_text_shadow(face, framebuf, 100, 420, ">> Press [X] for VS Mode (Pass-and-Play)", COLOR_MAGENTA);
        } 
        else if (app.currentScene == SCENE_PRACTICE) {
            draw_text_shadow(face, framebuf, 40, 40, "==================================================", COLOR_CYAN);
            
            if (app.isVsMode) {
                char turnStr[128];
                snprintf(turnStr, sizeof(turnStr), "   PLAYER %d's TURN   (Score: P1: %d | P2: %d)   ", app.playerTurn + 1, app.scores[0], app.scores[1]);
                draw_text_shadow(face, framebuf, 40, 80, turnStr, (app.playerTurn == 0) ? COLOR_MAGENTA : COLOR_YELLOW);
            } else {
                draw_text_shadow(face, framebuf, 40, 80, "                SOLO PRACTICE MODE                ", COLOR_CYAN);
            }
            draw_text_shadow(face, framebuf, 40, 120, "==================================================", COLOR_CYAN);

            draw_text_shadow(face, framebuf, 40, 200, " [ PROMPT ] ", COLOR_YELLOW);
            char promptStr[300];
            snprintf(promptStr, sizeof(promptStr), " -> %s", app.quarter);
            draw_text_shadow(face, framebuf, 40, 240, promptStr, COLOR_WHITE);
            draw_text_shadow(face, framebuf, 40, 300, "--------------------------------------------------", COLOR_CYAN);

            if (!app.answerShown) {
                draw_text_shadow(face, framebuf, 60, 360, ">> Press [A] to Reveal the Target Verse", COLOR_GREEN);
            } else {
                char headerStr[64];
                snprintf(headerStr, sizeof(headerStr), " [ VERSE %d ] ", app.currentVerse);
                draw_text_shadow(face, framebuf, 40, 360, headerStr, COLOR_RED);

                u32 lineY = 400;
                for (int i = 0; i < MAX_LINES; i++) {
                    if (app.verse[i][0]) {
                        draw_text_shadow(face, framebuf, 60, lineY, app.verse[i], COLOR_WHITE);
                        lineY += 32;
                    }
                }
                
                if (app.isVsMode) {
                    draw_text_shadow(face, framebuf, 60, lineY + 30, ">> Press [L] if Correct (+1 Point)  |  Press [R] if Missed (Pass Turn)", COLOR_YELLOW);
                } else {
                    draw_text_shadow(face, framebuf, 60, lineY + 30, ">> Press [A] for Next Challenge", COLOR_GREEN);
                }
            }
        }

        draw_text_shadow(face, framebuf, 40, 660, "==================================================", COLOR_CYAN);
        draw_text_shadow(face, framebuf, 40, 690, " [+] Exit App", COLOR_BLUE);

        u32 active_solid_color = get_smooth_loop_color(global_frame);

        int bob_offset = (global_frame / 15) % 4;

        draw_text_shadow(face, framebuf, 700, 692 + bob_offset, "Animation by Aashvik Goel", active_solid_color);

        framebufferEnd(&fb);
    }

    framebufferClose(&fb);
    FT_Done_Face(face);
    FT_Done_FreeType(library);

    if (app.initedDriver) {
        audrvMemPoolDetach(&app.drv, app.mempool_id);
        audrvClose(&app.drv);
    }
    if (app.initedAudren) audrenExit();
    if (app.mempool_ptr) free(app.mempool_ptr);

    return 0;
}