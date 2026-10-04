// wiikit runtime — the renderer: a window, and the GX record drawn with OpenGL 4.5.
//
// It runs on the host's main thread, which owns the window and the GL
// context; the game runs on its own threads and hands over its GX record
// (video.h) in chunks, at most --frames-ahead frames ahead. The renderer keeps a mirror
// of the BP and XF registers from the record and turns each draw into GL
// state and a program generated from the TEV configuration (gxshader.cpp).
//
// The EFB is a framebuffer of 640 x 528 (times --scale), stored top row
// first as on the console. EFB copies stay on the host GPU: to the XFB they
// become the frames VI shows, by address; to textures they are converted to
// what the target format would decode to, and the game's later binds of that
// address find them (gx.cpp). On each VI retrace the XFB that VI's top-field
// register names is presented on a 4:3 or 16:9 screen (SYSCONF), as large as
// the window allows.
#include "video.h"
#include "gl.h"
#include "rt.h"
#include <SDL3/SDL.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define WIIKIT_GL_DEFINE(T, n) T n = nullptr;
WIIKIT_GL_FUNCS(WIIKIT_GL_DEFINE)
#undef WIIKIT_GL_DEFINE

std::string gx_shader_key(const uint32_t* bp, const uint32_t* xf, uint8_t vflags);
void gx_shader_gen(const uint32_t* bp, const uint32_t* xf, uint8_t vflags, std::string& vs, std::string& fs);

namespace {

VideoOptions opt;
constexpr int EFB_W = 640, EFB_H = 528;
using Clock = std::chrono::steady_clock;

// ---- the queue from the game ------------------------------------------------------------------
struct Chunk { std::vector<uint8_t> data; int frames; bool readback = false; uint32_t copy = 0; };   // copy: video_copy_read's
std::mutex qmx;
std::condition_variable q_space;
std::deque<Chunk> q;
std::vector<std::vector<uint8_t>> spare;           // record buffers to reuse (under qmx)
int q_frames = 0;
SDL_Semaphore* wake = nullptr;
std::atomic<uint32_t> xfb_addr{0}, retraces{0}, vi_lines{480};
// an EFB read-back asked for by the CPU (video_efb_read), done in the queue's order
std::mutex rb_mx;
std::condition_variable rb_cv;
bool rb_ready = false;
std::vector<uint32_t>*rb_argb = nullptr, *rb_z = nullptr;
std::vector<uint8_t>* rb_rgba = nullptr;           // video_copy_read's
int rb_w = 0, rb_h = 0;
std::mutex pad_mx;
PadState pad;
ClassicState classic[4];

// ---- GX state, as the record left it --------------------------------------------------------
uint32_t bp[0x100], xf[0x1058];
uint32_t xf_lo = 0, xf_hi = 0x1058;                // XF words not yet uploaded
int32_t tev_reg[4][4], tev_konst[4][4];

struct PSUniforms {                                // gxshader.cpp's uniform block PS (std140)
    int32_t reg[4][4], konst[4][4], alpha[4];
    float texdim[8][4];
    int32_t indscale[4][4], indmtx[6][4];
};

// ---- GL objects -------------------------------------------------------------------------------
SDL_Window* win = nullptr;
int S = 1;                                         // EFB scale
GLuint efb_fbo, efb_col, efb_dep, copy_fbo, vao, empty_vao, vbo, quad_ibo, xf_ubo, ps_ubo, samplers[8];
GLuint copy_prog;
GLint copy_rect_loc, copy_mode_loc;
constexpr size_t VBO_CAP = 64u << 20;
size_t vbo_off = 0;
// The vertex buffer is a ring mapped once, persistently: draws copy their
// vertices straight into it. glBufferSubData into a buffer the GPU is still
// reading makes the driver order a copy between draws, and at thousands of
// draws a frame the GPU spends its time waiting on those. The ring is fenced
// in quarters: a quarter is written again only when the GPU is done with it.
uint8_t* vbo_ptr = nullptr;
GLsync vbo_fence[4] = {};
constexpr size_t VBO_QUARTER = VBO_CAP / 4;
PSUniforms ps_last;
bool ps_valid = false;
uint32_t sampler_mode[8][2];
bool sampler_set[8];

struct Tex { GLuint name = 0; int w = 0, h = 0, levels = 0; };
std::unordered_map<uint32_t, Tex> texs;            // decoded textures, by id
std::unordered_map<uint32_t, Tex> efb_copies;      // EFB copies to texture, by address
std::unordered_map<uint32_t, Tex> xfbs;            // EFB copies to the XFB, by address
uint64_t map_src[8];                               // per map: texture id, or 1 << 32 | EFB copy address
uint32_t last_xfb = 0;
std::unordered_map<std::string, GLuint> programs;

struct Counters { uint64_t frames, presents, draws, programs; } cnt;
bool dump_ps = false;                              // WIIKIT_SHADERDUMP: log the next uniforms

int32_t s11(uint32_t v) { return (int32_t)((v & 0x7FF) << 21) >> 21; }
int32_t s10(uint32_t v) { return (int32_t)((v & 0x3FF) << 22) >> 22; }
float fx(uint32_t a) { float f; std::memcpy(&f, &xf[a], 4); return f; }

// ---- GL helpers ---------------------------------------------------------------------------------
bool gl_load() {
    bool ok = true;
#define WIIKIT_GL_LOAD(T, n)                                                   \
    n = reinterpret_cast<T>(SDL_GL_GetProcAddress(#n));                        \
    if (!n) { rt_log("video: no %s", #n); ok = false; }
    WIIKIT_GL_FUNCS(WIIKIT_GL_LOAD)
#undef WIIKIT_GL_LOAD
    return ok;
}

GLuint compile(GLenum type, const std::string& src) {
    GLuint sh = glCreateShader(type);
    const char* p = src.c_str();
    glShaderSource(sh, 1, &p, nullptr);
    glCompileShader(sh);
    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        rt_log("video: shader does not compile:\n%s\n%s", log, src.c_str());
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

GLuint link(const std::string& vs, const std::string& fs) {
    GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    if (!v || !f) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        rt_log("video: program does not link: %s", log);
        return 0;
    }
    return p;
}

void ensure_tex(Tex& t, int w, int h, int levels) {
    if (t.name && t.w == w && t.h == h && t.levels == levels) return;
    if (t.name) glDeleteTextures(1, &t.name);
    glCreateTextures(GL_TEXTURE_2D, 1, &t.name);
    glTextureStorage2D(t.name, levels, GL_RGBA8, w, h);
    t.w = w; t.h = h; t.levels = levels;
}

void APIENTRY gl_debug(GLenum, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar* msg, const void*) {
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    rt_log("gl: %s%s", type == GL_DEBUG_TYPE_ERROR ? "error: " : "", msg);
}

// ---- the EFB copy pass --------------------------------------------------------------------------
const char* FULLSCREEN_VS = R"(#version 450 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";

// what a copy to each texture copy format (Dolphin's EFBCopyFormat numbering)
// reads back as when the game samples it
const char* COPY_FS = R"(#version 450 core
layout(binding = 0) uniform sampler2D efb_color;
layout(binding = 1) uniform sampler2D efb_depth;
uniform ivec4 u_rect;   // source x, y (EFB pixels at scale), half size, -
uniform ivec4 u_mode;   // format, from depth, intensity, EFB has alpha
out vec4 o;
vec4 src(ivec2 p) {
    if (u_mode.y != 0) {
        uint z = uint(clamp(texelFetch(efb_depth, p, 0).r, 0.0, 1.0) * 16777215.0);
        return vec4(float(z >> 16), float((z >> 8) & 255u), float(z & 255u), 255.0) / 255.0;
    }
    vec4 c = texelFetch(efb_color, p, 0);
    if (u_mode.w == 0) c.a = 1.0;
    return c;
}
float q(float v, float n) { return floor(v * n + 0.5) / n; }
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 c;
    if (u_rect.z != 0) {
        ivec2 s = u_rect.xy + p * 2;
        c = (src(s) + src(s + ivec2(1, 0)) + src(s + ivec2(0, 1)) + src(s + ivec2(1, 1))) * 0.25;
    } else {
        c = src(u_rect.xy + p);
    }
    float i = (u_mode.z != 0 && u_mode.y == 0) ? clamp(0.257 * c.r + 0.504 * c.g + 0.098 * c.b + 16.0 / 255.0, 0.0, 1.0) : c.r;
    switch (u_mode.x) {
    case 0: o = vec4(q(i, 15.0)); break;                                  // R4 / Z4
    case 1: case 8: o = vec4(i); break;                                   // R8 / Z8
    case 2: o = vec4(vec3(q(i, 15.0)), q(c.a, 15.0)); break;              // RA4
    case 3: o = u_mode.y != 0 ? vec4(c.ggg, c.r) : vec4(vec3(i), c.a); break;   // RA8 / Z16
    case 4: o = vec4(q(c.r, 31.0), q(c.g, 63.0), q(c.b, 31.0), 1.0); break;
    case 5: o = c.a >= 0.875 ? vec4(q(c.r, 31.0), q(c.g, 31.0), q(c.b, 31.0), 1.0)
                             : vec4(q(c.r, 15.0), q(c.g, 15.0), q(c.b, 15.0), q(c.a, 7.0)); break;
    case 7: o = vec4(c.a); break;                                         // A8
    case 9: o = vec4(c.g); break;                                         // G8 / Z8M
    case 10: o = vec4(c.b); break;                                        // B8 / Z8L
    // RG8 and GB8 are two channels in an IA8-shaped texture: the second
    // channel first in memory, so it is the alpha (read as IA8: RG8 is
    // I = R, A = G; GB8 is I = G, A = B). A colour-grading pass reads G as
    // the intensity and B as the alpha of a GB8 copy, as Dolphin encodes it.
    case 11: o = vec4(c.rrr, c.g); break;                                 // RG8
    case 12: o = u_mode.y != 0 ? vec4(c.bbb, c.g) : vec4(c.ggg, c.b); break;   // Z16L / GB8
    default: o = c; break;                                                // RGBA8 / Z24X8
    }
}
)";

// ---- register side effects ----------------------------------------------------------------------
void efb_clear(int x, int y, int w, int h) {
    bool has_alpha = (bp[0x43] & 7) == 1;
    bool cu = bp[0x41] >> 3 & 1, au = bp[0x41] >> 4 & 1, zu = bp[0x40] >> 4 & 1;
    if (!cu && !au && !zu) return;
    glBindFramebuffer(GL_FRAMEBUFFER, efb_fbo);
    glEnable(GL_SCISSOR_TEST);
    glScissor(x * S, y * S, w * S, h * S);
    glColorMask(cu, cu, cu, au || !has_alpha);
    glDepthMask(zu);
    float a = (bp[0x4F] >> 8 & 255) / 255.0f, r = (bp[0x4F] & 255) / 255.0f;
    float g = (bp[0x50] >> 8 & 255) / 255.0f, b = (bp[0x50] & 255) / 255.0f;
    glClearColor(r, g, b, has_alpha ? a : 1.0f);
    glClearDepth((bp[0x51] & 0xFFFFFF) / 16777215.0);
    glClear((cu || au ? GL_COLOR_BUFFER_BIT : 0) | (zu ? GL_DEPTH_BUFFER_BIT : 0));
}

// WIIKIT_EFBDUMP=N: the EFB as PNGs during frame N (counted in XFB copies):
// before each EFB copy, and every WIIKIT_EFBDUMP_EVERY draws (debugging)
long efb_dump_frame = std::getenv("WIIKIT_EFBDUMP") ? std::atol(std::getenv("WIIKIT_EFBDUMP")) : -1;
long efb_dump_every = std::getenv("WIIKIT_EFBDUMP_EVERY") ? std::atol(std::getenv("WIIKIT_EFBDUMP_EVERY")) : 0;
bool efb_dumping() { return (long)cnt.frames == efb_dump_frame; }
// The scene's viewport and depth range: the last perspective draw's (2D drawn
// after it, a HUD, has its own); a port's model is drawn with them.
struct SceneVp { bool ok = false; float x, y, w, h, farz, zrange, sx, sy; } scene_vp;

// A port's 3D model (video_model_mesh, and each frame its VC_MODEL record).
std::mutex model_mx;
std::vector<float> model_verts;
std::vector<uint8_t> model_tex;
int model_n = 0, model_layers = 0, model_tw = 0, model_th = 0;
uint64_t model_serial = 0;
float model_view[12], model_proj[6];
bool model_visible = false;
std::vector<float> model_joints;                 // 3x4 rows per joint
uint64_t model_joints_serial = 0;
constexpr int kModelMaxVerts = 200000, kModelVertFloats = 13, kModelMaxJoints = 128;

// Into the EFB before it goes to the XFB: depth-tested against the scene, with
// GX's projection and the scene's viewport, as gxshader's own draws. Its own
// vertex array, buffer, texture unit (15) and uniform blocks (7, 8): the
// game's draws set everything else themselves. Skinned here: each vertex on
// one joint or two (video_model_mesh), the joints as the port last posed them.
void draw_model() {
    static GLuint prog = 0, vbuf = 0, mvao = 0, tex = 0, ubo = 0, jubo = 0;
    static uint64_t shown = 0, joints_shown = 0;
    static int n = 0, layers = 1;
    float ub[24];
    {
        std::lock_guard<std::mutex> lk(model_mx);
        if (!model_visible || !model_n || model_joints.empty() || !scene_vp.ok) return;
        if (!vbuf) {
            glCreateBuffers(1, &vbuf);
            glNamedBufferStorage(vbuf, (GLsizeiptr)kModelMaxVerts * kModelVertFloats * 4, nullptr, GL_DYNAMIC_STORAGE_BIT);
            glCreateBuffers(1, &ubo);
            glNamedBufferStorage(ubo, sizeof ub, nullptr, GL_DYNAMIC_STORAGE_BIT);
            glCreateBuffers(1, &jubo);
            glNamedBufferStorage(jubo, kModelMaxJoints * 12 * 4, nullptr, GL_DYNAMIC_STORAGE_BIT);
            glCreateVertexArrays(1, &mvao);
            glVertexArrayVertexBuffer(mvao, 0, vbuf, 0, kModelVertFloats * 4);
            const GLint size[4] = {3, 3, 4, 3};                  // p0, p1, joints and weights, u v layer
            const GLuint at[4] = {0, 12, 24, 40};
            for (GLuint a = 0; a < 4; ++a) {
                glEnableVertexArrayAttrib(mvao, a);
                glVertexArrayAttribFormat(mvao, a, size[a], GL_FLOAT, GL_FALSE, at[a]);
                glVertexArrayAttribBinding(mvao, a, 0);
            }
        }
        if (model_joints_serial != joints_shown) {
            joints_shown = model_joints_serial;
            glNamedBufferSubData(jubo, 0, (GLsizeiptr)model_joints.size() * 4, model_joints.data());
        }
        if (model_serial != shown) {
            n = std::min(model_n, kModelMaxVerts);
            glNamedBufferSubData(vbuf, 0, (GLsizeiptr)n * kModelVertFloats * 4, model_verts.data());
            if (tex) glDeleteTextures(1, &tex);
            layers = std::max(1, model_layers);
            // the layers as an array with mipmaps: seen from afar, a texture is
            // averaged rather than sampled sparsely (shimmering), and no layer
            // bleeds into the next
            int levels = 1;
            while ((std::max(model_tw, model_th) >> levels) > 0) ++levels;
            glCreateTextures(GL_TEXTURE_2D_ARRAY, 1, &tex);
            glTextureStorage3D(tex, levels, GL_RGBA8, model_tw, model_th, layers);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glTextureSubImage3D(tex, 0, 0, 0, 0, model_tw, model_th, layers, GL_RGBA, GL_UNSIGNED_BYTE, model_tex.data());
            glGenerateTextureMipmap(tex);
            glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
            glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_REPEAT);
            shown = model_serial;
        }
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c) ub[r * 4 + c] = model_view[r * 4 + c];
        ub[12] = model_proj[0]; ub[13] = model_proj[1]; ub[14] = model_proj[2]; ub[15] = model_proj[3];
        ub[16] = model_proj[4]; ub[17] = model_proj[5]; ub[18] = (float)layers; ub[19] = 0;
    }
    ub[20] = scene_vp.farz / 16777216.0f;
    ub[21] = scene_vp.zrange / 16777216.0f;
    ub[22] = scene_vp.sx;
    ub[23] = scene_vp.sy;
    if (!prog) {
        prog = link(
            "#version 450\n"
            "layout(location = 0) in vec3 a_p0;\n"
            "layout(location = 1) in vec3 a_p1;\n"
            "layout(location = 2) in vec4 a_jw;\n"                 // joint 0, joint 1 (-1: none), their weights
            "layout(location = 3) in vec3 a_uvl;\n"
            "layout(std140, binding = 7) uniform Model { vec4 mv[3]; vec4 pr0; vec4 pr1; vec4 vp; };\n"
            "layout(std140, binding = 8) uniform Joints { vec4 jm[384]; };\n"
            "out vec3 uvl;\n"
            // a joint's rows on a position already weighted: its turn, its offset by the weight
            "vec3 on(int j, vec3 q, float w) {\n"
            "    vec4 h = vec4(q, w);\n"
            "    return vec3(dot(jm[3 * j], h), dot(jm[3 * j + 1], h), dot(jm[3 * j + 2], h));\n"
            "}\n"
            "void main() {\n"
            "    int j0 = int(a_jw.x), j1 = int(a_jw.y);\n"
            "    vec3 m = on(j0, a_p0, a_jw.z);\n"
            "    if (j1 >= 0) m += on(j1, a_p1, a_jw.w);\n"
            "    vec4 p = vec4(m, 1.0);\n"
            "    vec3 pos = vec3(dot(mv[0], p), dot(mv[1], p), dot(mv[2], p));\n"
            "    vec4 clip = vec4(pr0.x * pos.x + pr0.y * pos.z, pr0.z * pos.y + pr0.w * pos.z, pr1.x * pos.z + pr1.y, -pos.z);\n"
            "    clip.z = clip.w * vp.x + clip.z * vp.y;\n"            // as gxshader: GX's depth range
            "    if (vp.z < 0.0) clip.x = -clip.x;\n"
            "    if (vp.w < 0.0) clip.y = -clip.y;\n"
            "    gl_Position = clip;\n"
            "    uvl = a_uvl;\n"
            "}\n",
            "#version 450\n"
            "in vec3 uvl;\n"
            "layout(std140, binding = 7) uniform Model { vec4 mv[3]; vec4 pr0; vec4 pr1; vec4 vp; };\n"
            "layout(binding = 15) uniform sampler2DArray tex;\n"
            "out vec4 col;\n"
            "void main() {\n"
            "    vec4 c = texture(tex, uvl);\n"                       // u, v, the layer
            "    if (c.a < 0.5) discard;\n"
            "    col = vec4(c.rgb, 1.0);\n"
            "}\n");
        if (!prog) { model_visible = false; return; }
    }
    static bool told = false;
    if (!told) {                                             // once: where it goes, for a port's diagnosis
        told = true;
        rt_log("video: a port's model: %d vertices, %d layers; the scene's viewport %.0f,%.0f %.0fx%.0f, depth %g/%g, flips %g %g",
               n, layers, scene_vp.x, scene_vp.y, scene_vp.w, scene_vp.h, ub[20], ub[21], ub[22], ub[23]);
    }
    glNamedBufferSubData(ubo, 0, sizeof ub, ub);
    glBindBufferBase(GL_UNIFORM_BUFFER, 7, ubo);
    glBindBufferBase(GL_UNIFORM_BUFFER, 8, jubo);
    glBindFramebuffer(GL_FRAMEBUFFER, efb_fbo);
    glViewportIndexedf(0, scene_vp.x, scene_vp.y, scene_vp.w, scene_vp.h);
    glDisable(GL_SCISSOR_TEST);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDepthMask(GL_TRUE);
    glColorMask(1, 1, 1, 1);
    glDisable(GL_BLEND);
    glDisable(GL_COLOR_LOGIC_OP);
    glDisable(GL_CULL_FACE);
    glUseProgram(prog);
    glBindTextureUnit(15, tex);
    glBindSampler(15, 0);
    glBindVertexArray(mvao);
    glDrawArrays(GL_TRIANGLES, 0, n);
    glBindVertexArray(vao);
}

void efb_dump(const char* tag) {
    static int k = 0;
    std::vector<uint8_t> px((size_t)EFB_W * S * EFB_H * S * 4);   // top row first, as the EFB is kept
    glGetTextureImage(efb_col, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLsizei)px.size(), px.data());
    char name[160];
    std::snprintf(name, sizeof name, "efb_%ld_%04d_%s.png", efb_dump_frame, k++, tag);
    write_png(name, EFB_W * S, EFB_H * S, px.data());
}

void efb_copy(uint32_t v) {
    int x = (int)(bp[0x49] & 0x3FF), y = (int)(bp[0x49] >> 10 & 0x3FF);
    int w = (int)(bp[0x4A] & 0x3FF) + 1, h = (int)(bp[0x4A] >> 10 & 0x3FF) + 1;
    uint32_t dest = (bp[0x4B] & 0xFFFFFF) << 5;
    glDisable(GL_SCISSOR_TEST);
    if (efb_dumping()) {
        char tag[64];
        std::snprintf(tag, sizeof tag, "copy_%06X_%d_%d_%dx%d", v, x, y, w, h);
        efb_dump(tag);
    }
    if (v >> 14 & 1) {                                           // to the XFB
        Tex& t = xfbs[dest];
        ensure_tex(t, w * S, h * S, 1);
        draw_model();                                            // a port's model, into the scene
        {                                                        // each frame its own: a frame with no
            std::lock_guard<std::mutex> lk(model_mx);            // model recorded, or no 3D scene, shows none
            model_visible = false;
        }
        scene_vp.ok = false;
        glDisable(GL_SCISSOR_TEST);
        glNamedFramebufferTexture(copy_fbo, GL_COLOR_ATTACHMENT0, t.name, 0);
        glBlitNamedFramebuffer(efb_fbo, copy_fbo, x * S, y * S, (x + w) * S, (y + h) * S, 0, 0, w * S, h * S,
                               GL_COLOR_BUFFER_BIT, GL_NEAREST);
        last_xfb = dest;
        ++cnt.frames;
    } else {                                                     // to a texture
        bool half = v >> 9 & 1;
        int tw = half ? (w + 1) / 2 : w, th = half ? (h + 1) / 2 : h;
        uint32_t tpf = v >> 3 & 15, fmt = tpf / 2 + (tpf & 1) * 8;
        Tex& t = efb_copies[dest];
        ensure_tex(t, tw * S, th * S, 1);
        glNamedFramebufferTexture(copy_fbo, GL_COLOR_ATTACHMENT0, t.name, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, copy_fbo);
        glViewport(0, 0, tw * S, th * S);
        glDisable(GL_BLEND);
        glDisable(GL_COLOR_LOGIC_OP);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_CULL_FACE);
        glColorMask(1, 1, 1, 1);
        glUseProgram(copy_prog);
        glProgramUniform4i(copy_prog, copy_rect_loc, x * S, y * S, half, 0);
        glProgramUniform4i(copy_prog, copy_mode_loc, (int)fmt, (bp[0x43] & 7) == 3, (int)(v >> 15 & 1),
                           (bp[0x43] & 7) == 1);
        glBindTextureUnit(0, efb_col);
        glBindTextureUnit(1, efb_dep);
        glBindSampler(0, 0);
        glBindSampler(1, 0);
        glBindVertexArray(empty_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(vao);
    }
    if (v >> 11 & 1) efb_clear(x, y, w, h);
}

void bp_set(uint32_t v) {
    uint32_t reg = v >> 24, val = v & 0xFFFFFF;
    bp[reg] = val;
    if (reg >= 0xE0 && reg < 0xE8) {                             // TEV colour and konst registers
        int32_t* r = (val >> 23 & 1) ? tev_konst[(reg - 0xE0) / 2] : tev_reg[(reg - 0xE0) / 2];
        if (!(reg & 1)) { r[0] = s11(val); r[3] = s11(val >> 12); }   // RA
        else { r[2] = s11(val); r[1] = s11(val >> 12); }              // BG
    }
    if (reg == 0x52) efb_copy(val);
}

// ---- draws --------------------------------------------------------------------------------------
GLuint program_for(uint8_t vflags) {
    std::string key = gx_shader_key(bp, xf, vflags);
    auto it = programs.find(key);
    if (it != programs.end()) return it->second;
    std::string vs, fs;
    gx_shader_gen(bp, xf, vflags, vs, fs);
    GLuint p = link(vs, fs);
    if (const char* dir = std::getenv("WIIKIT_SHADERDUMP")) {           // debugging: the sources
        std::string path = std::string(dir) + "/prog_" + std::to_string(cnt.programs) + ".glsl";
        if (FILE* f = std::fopen(path.c_str(), "w")) {
            std::fprintf(f, "%s\n// ---- fragment\n%s", vs.c_str(), fs.c_str());
            std::fclose(f);
        }
        dump_ps = true;
    }
    programs.emplace(std::move(key), p);
    ++cnt.programs;
    return p;
}

uint32_t used_maps() {
    uint32_t gen = bp[0x00], used = 0;
    int ntev = (int)(gen >> 10 & 15) + 1, nind = (int)(gen >> 16 & 7);
    for (int s = 0; s < ntev; ++s) {
        uint32_t t = bp[0x28 + s / 2] >> (12 * (s & 1));
        if (t & 0x40) used |= 1u << (t & 7);
    }
    for (int i = 0; i < nind && i < 4; ++i) used |= 1u << (bp[0x27] >> (6 * i) & 7);
    return used;
}

void set_ps_uniforms() {
    PSUniforms u{};
    std::memcpy(u.reg, tev_reg, sizeof u.reg);
    std::memcpy(u.konst, tev_konst, sizeof u.konst);
    u.alpha[0] = (int32_t)(bp[0xF3] & 255);
    u.alpha[1] = (int32_t)(bp[0xF3] >> 8 & 255);
    u.alpha[2] = (int32_t)(bp[0x42] & 255);
    for (int m = 0; m < 8; ++m) {
        uint32_t img0 = bp[0x88 + (m & 3) + (m >= 4 ? 0x20 : 0)];
        u.texdim[m][0] = 1.0f / (float)(((img0 & 0x3FF) + 1) * 128);
        u.texdim[m][1] = 1.0f / (float)(((img0 >> 10 & 0x3FF) + 1) * 128);
        u.texdim[m][2] = (float)(((bp[0x30 + 2 * m] & 0xFFFF) + 1) * 128);
        u.texdim[m][3] = (float)(((bp[0x31 + 2 * m] & 0xFFFF) + 1) * 128);
    }
    for (int i = 0; i < 4; ++i) {
        uint32_t r = bp[0x25 + i / 2] >> (8 * (i & 1));
        u.indscale[i][0] = (int32_t)(r & 15);
        u.indscale[i][1] = (int32_t)(r >> 4 & 15);
    }
    for (int m = 0; m < 3; ++m) {
        uint32_t c0 = bp[0x06 + 3 * m], c1 = bp[0x07 + 3 * m], c2 = bp[0x08 + 3 * m];
        int32_t scale = (int32_t)((c0 >> 22 & 3) | (c1 >> 22 & 3) << 2 | (c2 >> 22 & 3) << 4);
        int32_t r0[4] = {s11(c0), s11(c1), s11(c2), 17 - scale};
        int32_t r1[4] = {s11(c0 >> 11), s11(c1 >> 11), s11(c2 >> 11), 17 - scale};
        std::memcpy(u.indmtx[2 * m], r0, sizeof r0);
        std::memcpy(u.indmtx[2 * m + 1], r1, sizeof r1);
    }
    if (dump_ps) {
        dump_ps = false;
        std::string t = "video: program " + std::to_string(cnt.programs - 1) + " uniforms:";
        for (int i = 0; i < 4; ++i) t += " reg" + std::to_string(i) + "(" + std::to_string(u.reg[i][0]) + "," + std::to_string(u.reg[i][1]) + "," + std::to_string(u.reg[i][2]) + "," + std::to_string(u.reg[i][3]) + ")";
        for (int i = 0; i < 4; ++i) t += " k" + std::to_string(i) + "(" + std::to_string(u.konst[i][0]) + "," + std::to_string(u.konst[i][1]) + "," + std::to_string(u.konst[i][2]) + "," + std::to_string(u.konst[i][3]) + ")";
        for (int i = 0; i < 8; ++i) t += " td" + std::to_string(i) + "(" + std::to_string(1 / u.texdim[i][0] / 128) + "," + std::to_string(1 / u.texdim[i][1] / 128) + "," + std::to_string(u.texdim[i][2] / 128) + "," + std::to_string(u.texdim[i][3] / 128) + ")";
        for (int i = 0; i < 4; ++i) t += " is" + std::to_string(i) + "(" + std::to_string(u.indscale[i][0]) + "," + std::to_string(u.indscale[i][1]) + ")";
        for (int i = 0; i < 6; ++i) t += " im" + std::to_string(i) + "(" + std::to_string(u.indmtx[i][0]) + "," + std::to_string(u.indmtx[i][1]) + "," + std::to_string(u.indmtx[i][2]) + "," + std::to_string(u.indmtx[i][3]) + ")";
        rt_log("%s", t.c_str());
    }
    if (ps_valid && !std::memcmp(&u, &ps_last, sizeof u)) return;
    glNamedBufferSubData(ps_ubo, 0, sizeof u, &u);
    ps_last = u;
    ps_valid = true;
}

void bind_textures() {
    static const GLint wrap[4] = {GL_CLAMP_TO_EDGE, GL_REPEAT, GL_MIRRORED_REPEAT, GL_REPEAT};
    static const GLint minf[8] = {GL_NEAREST, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_LINEAR, GL_NEAREST,
                                  GL_LINEAR, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR};
    uint32_t used = used_maps();
    for (int m = 0; m < 8; ++m) {
        if (!(used >> m & 1)) continue;
        GLuint name = 0;
        if (map_src[m] >> 32) {
            auto it = efb_copies.find((uint32_t)map_src[m]);
            if (it != efb_copies.end()) name = it->second.name;
        } else {
            auto it = texs.find((uint32_t)map_src[m]);
            if (it != texs.end()) name = it->second.name;
        }
        glBindTextureUnit(m, name);
        int off = (m & 3) + (m >= 4 ? 0x20 : 0);
        uint32_t m0 = bp[0x80 + off], m1 = bp[0x84 + off];
        if (!sampler_set[m] || sampler_mode[m][0] != m0 || sampler_mode[m][1] != m1) {
            GLuint s = samplers[m];
            glSamplerParameteri(s, GL_TEXTURE_WRAP_S, wrap[m0 & 3]);
            glSamplerParameteri(s, GL_TEXTURE_WRAP_T, wrap[m0 >> 2 & 3]);
            glSamplerParameteri(s, GL_TEXTURE_MAG_FILTER, (m0 >> 4 & 1) ? GL_LINEAR : GL_NEAREST);
            glSamplerParameteri(s, GL_TEXTURE_MIN_FILTER, minf[m0 >> 5 & 7]);
            glSamplerParameterf(s, GL_TEXTURE_LOD_BIAS, (float)(int8_t)(m0 >> 9 & 0xFF) / 32.0f);
            glSamplerParameterf(s, GL_TEXTURE_MIN_LOD, (float)(m1 & 0xFF) / 16.0f);
            glSamplerParameterf(s, GL_TEXTURE_MAX_LOD, (float)(m1 >> 8 & 0xFF) / 16.0f);
            sampler_mode[m][0] = m0;
            sampler_mode[m][1] = m1;
            sampler_set[m] = true;
        }
        glBindSampler(m, samplers[m]);
    }
}

// A VC_DRAW's pieces (each n, then n vertices), drawn with one call.
void draw(uint8_t prim, uint8_t vflags, const uint8_t* pieces, uint32_t npieces) {
    uint32_t n = 0;
    const uint8_t* vtx = pieces + 4;                           // the first vertex
    for (uint32_t i = 0, k; i < npieces; ++i) {
        std::memcpy(&k, pieces + (size_t)n * sizeof(GVtx) + 4 * i, 4);
        n += k;
    }
    uint32_t cull = bp[0x00] >> 14 & 3;
    bool tri = prim < 0xA8;
    if (tri && cull == 3) return;
    GLuint prog = program_for(vflags);
    if (!prog) return;
    ++cnt.draws;
    glBindFramebuffer(GL_FRAMEBUFFER, efb_fbo);
    glUseProgram(prog);

    int offx = s10(bp[0x59]) * 2, offy = s10(bp[0x59] >> 10) * 2;
    float wd = std::fabs(fx(0x101A)), ht = std::fabs(fx(0x101B));
    glViewportIndexedf(0, (fx(0x101D) - wd - (float)offx) * (float)S, (fx(0x101E) - ht - (float)offy) * (float)S,
                       2 * wd * (float)S, 2 * ht * (float)S);
    if (xf[0x1026] == 0)                                         // a perspective draw: the scene's viewport
        scene_vp = {true, (fx(0x101D) - wd - (float)offx) * (float)S, (fx(0x101E) - ht - (float)offy) * (float)S,
                    2 * wd * (float)S, 2 * ht * (float)S, fx(0x101F), fx(0x101C), fx(0x101A), fx(0x101B)};
    int sx0 = (int)(bp[0x20] >> 12 & 0x7FF) - offx, sy0 = (int)(bp[0x20] & 0x7FF) - offy;
    int sx1 = (int)(bp[0x21] >> 12 & 0x7FF) - offx + 1, sy1 = (int)(bp[0x21] & 0x7FF) - offy + 1;
    sx0 = std::max(sx0, 0); sy0 = std::max(sy0, 0); sx1 = std::min(sx1, EFB_W); sy1 = std::min(sy1, EFB_H);
    if (sx1 <= sx0 || sy1 <= sy0) return;
    glEnable(GL_SCISSOR_TEST);
    glScissor(sx0 * S, sy0 * S, (sx1 - sx0) * S, (sy1 - sy0) * S);

    static const GLenum zf[8] = {GL_NEVER, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS};
    uint32_t zm = bp[0x40];
    if (zm & 1) { glEnable(GL_DEPTH_TEST); glDepthFunc(zf[zm >> 1 & 7]); }
    else glDisable(GL_DEPTH_TEST);
    glDepthMask(zm >> 4 & 1);

    uint32_t cm = bp[0x41];
    bool has_alpha = (bp[0x43] & 7) == 1;
    glColorMask(cm >> 3 & 1, cm >> 3 & 1, cm >> 3 & 1, (cm >> 4 & 1) && has_alpha);
    if (cm & 1) {
        glDisable(GL_COLOR_LOGIC_OP);
        glEnable(GL_BLEND);
        if (cm >> 11 & 1) {                                      // subtract: dst - src
            glBlendEquation(GL_FUNC_REVERSE_SUBTRACT);
            glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
        } else {
            GLenum da = has_alpha ? GL_DST_ALPHA : GL_ONE, ida = has_alpha ? GL_ONE_MINUS_DST_ALPHA : GL_ZERO;
            const GLenum src[8] = {GL_ZERO, GL_ONE, GL_DST_COLOR, GL_ONE_MINUS_DST_COLOR,
                                   GL_SRC1_ALPHA, GL_ONE_MINUS_SRC1_ALPHA, da, ida};
            const GLenum dst[8] = {GL_ZERO, GL_ONE, GL_SRC1_COLOR, GL_ONE_MINUS_SRC1_COLOR,
                                   GL_SRC1_ALPHA, GL_ONE_MINUS_SRC1_ALPHA, da, ida};
            GLenum sf = src[cm >> 8 & 7], df = dst[cm >> 5 & 7];
            glBlendEquation(GL_FUNC_ADD);
            glBlendFuncSeparate(sf, df, sf, df);
        }
    } else if (cm >> 1 & 1) {
        static const GLenum lo[16] = {GL_CLEAR, GL_AND, GL_AND_REVERSE, GL_COPY, GL_AND_INVERTED, GL_NOOP,
                                      GL_XOR, GL_OR, GL_NOR, GL_EQUIV, GL_INVERT, GL_OR_REVERSE,
                                      GL_COPY_INVERTED, GL_OR_INVERTED, GL_NAND, GL_SET};
        glDisable(GL_BLEND);
        glEnable(GL_COLOR_LOGIC_OP);
        glLogicOp(lo[cm >> 12 & 15]);
    } else {
        glDisable(GL_BLEND);
        glDisable(GL_COLOR_LOGIC_OP);
    }
    // GX's front faces are clockwise on screen: counter-clockwise for GL,
    // whose y runs the other way over the same rows
    if (tri && cull) { glEnable(GL_CULL_FACE); glCullFace(cull == 1 ? GL_BACK : GL_FRONT); }
    else glDisable(GL_CULL_FACE);

    if (xf_lo < xf_hi) {
        glNamedBufferSubData(xf_ubo, xf_lo * 4, (xf_hi - xf_lo) * 4, &xf[xf_lo]);
        xf_lo = 0x1058;
        xf_hi = 0;
    }
    set_ps_uniforms();
    bind_textures();

    if (efb_dumping() && std::getenv("WIIKIT_DRAWLOG")) {         // debugging: where each draw lands
        static FILE* lf = std::fopen("drawlog.txt", "w");
        const GVtx& v0 = *reinterpret_cast<const GVtx*>(vtx);
        uint32_t pm = v0.mtx[0] * 4u;
        float pos[3];
        for (int r = 0; r < 3; ++r)
            pos[r] = fx(pm + r * 4) * v0.pos[0] + fx(pm + r * 4 + 1) * v0.pos[1] + fx(pm + r * 4 + 2) * v0.pos[2] + fx(pm + r * 4 + 3);
        float c[4];
        if (xf[0x1026] == 0) { c[0] = fx(0x1020) * pos[0] + fx(0x1021) * pos[2]; c[1] = fx(0x1022) * pos[1] + fx(0x1023) * pos[2];
                               c[2] = fx(0x1024) * pos[2] + fx(0x1025); c[3] = -pos[2]; }
        else { c[0] = fx(0x1020) * pos[0] + fx(0x1021); c[1] = fx(0x1022) * pos[1] + fx(0x1023);
               c[2] = fx(0x1024) * pos[2] + fx(0x1025); c[3] = 1; }
        std::fprintf(lf, "%ld prim %02X n %u zm %X cm %X pix %X at %X ztex %X fog %X | vp %g %g %g %g %g %g | proj%u %g %g %g %g %g %g | "
                     "mtx %u obj %g %g %g eye %g %g %g ndc %g %g w %g | tev %u chans %u\n",
                     (long)cnt.draws, prim, n, bp[0x40], bp[0x41], bp[0x43], bp[0xF3], bp[0xF4], bp[0xF1],
                     fx(0x101A), fx(0x101B), fx(0x101C), fx(0x101D), fx(0x101E), fx(0x101F),
                     xf[0x1026], fx(0x1020), fx(0x1021), fx(0x1022), fx(0x1023), fx(0x1024), fx(0x1025),
                     v0.mtx[0], v0.pos[0], v0.pos[1], v0.pos[2], pos[0], pos[1], pos[2], c[0] / c[3], c[1] / c[3], c[3],
                     (bp[0x00] >> 10 & 15) + 1, xf[0x1009] & 3);
    }
    size_t bytes = (size_t)n * sizeof(GVtx);
    // a draw never straddles two quarters; offsets stay whole vertices
    size_t q = vbo_off / VBO_QUARTER;
    if ((vbo_off + bytes - 1) / VBO_QUARTER != q) {          // on to the next quarter: fence this one
        vbo_fence[q] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        q = (q + 1) % 4;
        vbo_off = (q * VBO_QUARTER + sizeof(GVtx) - 1) / sizeof(GVtx) * sizeof(GVtx);
        if (vbo_fence[q]) {
            glClientWaitSync(vbo_fence[q], GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
            glDeleteSync(vbo_fence[q]);
            vbo_fence[q] = nullptr;
        }
    }
    GLint base = (GLint)(vbo_off / sizeof(GVtx));
    static std::vector<GLint> firsts;
    static std::vector<GLsizei> counts;
    static std::vector<const void*> offsets;
    firsts.clear();
    counts.clear();
    for (uint32_t i = 0, at = 0, k; i < npieces; ++i) {       // the pieces' vertices, one after another
        std::memcpy(&k, pieces, 4);
        std::memcpy(vbo_ptr + vbo_off + (size_t)at * sizeof(GVtx), pieces + 4, (size_t)k * sizeof(GVtx));
        firsts.push_back(base + (GLint)at);
        counts.push_back((GLsizei)k);
        pieces += 4 + (size_t)k * sizeof(GVtx);
        at += k;
    }
    vbo_off += bytes;
    GLenum mode;
    switch (prim) {
    case 0x80: case 0x88:                                      // quads: a static index buffer of their triangles
        for (GLsizei& k : counts) k = k / 4 * 6;
        offsets.assign(npieces, nullptr);
        if (npieces == 1) glDrawElementsBaseVertex(GL_TRIANGLES, counts[0], GL_UNSIGNED_INT, nullptr, base);
        else glMultiDrawElementsBaseVertex(GL_TRIANGLES, counts.data(), GL_UNSIGNED_INT, offsets.data(),
                                           (GLsizei)npieces, firsts.data());
        mode = 0;
        break;
    case 0x90: mode = GL_TRIANGLES; break;
    case 0x98: mode = GL_TRIANGLE_STRIP; break;
    case 0xA0: mode = GL_TRIANGLE_FAN; break;
    case 0xA8: mode = GL_LINES; break;
    case 0xB0: mode = GL_LINE_STRIP; break;
    default: mode = GL_POINTS; break;
    }
    if (mode && npieces == 1) glDrawArrays(mode, base, counts[0]);
    else if (mode) glMultiDrawArrays(mode, firsts.data(), counts.data(), (GLsizei)npieces);
    if (efb_dump_every && efb_dumping()) {
        static long k = 0;
        if (++k % efb_dump_every == 0) {
            char tag[32];
            std::snprintf(tag, sizeof tag, "draw%05ld", k);
            efb_dump(tag);
        }
    }
}

// ---- the record -----------------------------------------------------------------------------------
template <class T> T rd(const uint8_t*& p) { T v; std::memcpy(&v, p, sizeof v); p += sizeof v; return v; }

// The EFB at its native size: the pixel at each native position's top-left
// sample of the scaled EFB (rows from the top, as the EFB copies read it).
void efb_readback() {
    int W = EFB_W * S, H = EFB_H * S;
    std::vector<uint8_t> px((size_t)W * H * 4);
    std::vector<float> dz((size_t)W * H);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, efb_fbo);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glReadPixels(0, 0, W, H, GL_DEPTH_COMPONENT, GL_FLOAT, dz.data());
    std::lock_guard<std::mutex> lk(rb_mx);
    rb_argb->resize((size_t)EFB_W * EFB_H);
    rb_z->resize((size_t)EFB_W * EFB_H);
    for (int y = 0; y < EFB_H; ++y)
        for (int x = 0; x < EFB_W; ++x) {
            size_t s = (size_t)y * S * W + (size_t)x * S, d = (size_t)y * EFB_W + x;
            const uint8_t* c = &px[s * 4];
            (*rb_argb)[d] = (uint32_t)c[3] << 24 | (uint32_t)c[0] << 16 | (uint32_t)c[1] << 8 | c[2];
            (*rb_z)[d] = (uint32_t)(std::clamp(dz[s], 0.0f, 1.0f) * 16777215.0f);
        }
    rb_ready = true;
    rb_cv.notify_all();
}

// An EFB copy to texture at its native size: the top-left sample of each
// native pixel of the scaled copy (rows from the top, as it was copied).
void copy_readback(uint32_t addr) {
    std::lock_guard<std::mutex> lk(rb_mx);
    rb_rgba->assign((size_t)rb_w * rb_h * 4, 0);
    auto it = efb_copies.find(addr);
    if (it != efb_copies.end() && it->second.name) {
        const Tex& t = it->second;
        std::vector<uint8_t> px((size_t)t.w * t.h * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glGetTextureImage(t.name, 0, GL_RGBA, GL_UNSIGNED_BYTE, (GLsizei)px.size(), px.data());
        for (int y = 0; y < rb_h && y * S < t.h; ++y)
            for (int x = 0; x < rb_w && x * S < t.w; ++x)
                std::memcpy(&(*rb_rgba)[((size_t)y * rb_w + x) * 4], &px[((size_t)y * S * t.w + (size_t)x * S) * 4], 4);
    }
    rb_ready = true;
    rb_cv.notify_all();
}

void exec(const std::vector<uint8_t>& data) {
    const uint8_t* p = data.data();
    const uint8_t* end = p + data.size();
    while (p < end) {
        switch (*p++) {
        case VC_BP: bp_set(rd<uint32_t>(p)); break;
        case VC_XF: {
            uint16_t a = rd<uint16_t>(p), n = rd<uint16_t>(p);
            std::memcpy(&xf[a], p, 4u * n);
            p += 4u * n;
            xf_lo = std::min<uint32_t>(xf_lo, a);
            xf_hi = std::max<uint32_t>(xf_hi, a + n);
            break;
        }
        case VC_DRAW: {
            uint8_t prim = rd<uint8_t>(p), fl = rd<uint8_t>(p);
            uint32_t pieces = rd<uint32_t>(p);
            draw(prim, fl, p, pieces);
            for (uint32_t i = 0; i < pieces; ++i) {
                uint32_t n = rd<uint32_t>(p);
                p += (size_t)n * sizeof(GVtx);
            }
            break;
        }
        case VC_TEXUP: {
            uint8_t m = rd<uint8_t>(p);
            uint32_t id = rd<uint32_t>(p);
            int w = rd<uint16_t>(p), h = rd<uint16_t>(p), levels = rd<uint8_t>(p);
            Tex& t = texs[id];
            ensure_tex(t, w, h, levels);
            for (int l = 0; l < levels; ++l) {
                int lw = std::max(1, w >> l), lh = std::max(1, h >> l);
                glTextureSubImage2D(t.name, l, 0, 0, lw, lh, GL_RGBA, GL_UNSIGNED_BYTE, p);
                p += (size_t)lw * lh * 4;
            }
            map_src[m] = id;
            break;
        }
        case VC_TEXBIND: { uint8_t m = rd<uint8_t>(p); map_src[m] = rd<uint32_t>(p); break; }
        case VC_TEXEFB: { uint8_t m = rd<uint8_t>(p); map_src[m] = 1ull << 32 | rd<uint32_t>(p); break; }
        case VC_FRAME: break;
        case VC_DRAWDONE: gx_draw_done_reached(); break;
        case VC_MODEL: {                                             // the port's model in this frame
            bool vis = rd<uint8_t>(p) != 0;
            int n = std::min<int>(rd<uint16_t>(p), kModelMaxJoints);
            std::lock_guard<std::mutex> lk(model_mx);
            std::memcpy(model_view, p, sizeof model_view);
            std::memcpy(model_proj, p + sizeof model_view, sizeof model_proj);
            p += sizeof model_view + sizeof model_proj;
            model_joints.assign(reinterpret_cast<const float*>(p), reinterpret_cast<const float*>(p) + (size_t)n * 12);
            p += (size_t)n * 12 * 4;
            ++model_joints_serial;
            model_visible = vis;
            break;
        }
        default: rt_die("video: bad record byte %02X", p[-1]);
        }
    }
}

// ---- presenting -----------------------------------------------------------------------------------
uint32_t crc_table[256];
uint32_t crc32(const uint8_t* p, size_t n, uint32_t c = 0xFFFFFFFFu) {
    if (!crc_table[1])
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t k = i;
            for (int j = 0; j < 8; ++j) k = k & 1 ? 0xEDB88320u ^ (k >> 1) : k >> 1;
            crc_table[i] = k;
        }
    for (size_t i = 0; i < n; ++i) c = crc_table[(c ^ p[i]) & 255] ^ (c >> 8);
    return c;
}

}  // namespace

// a PNG with stored (uncompressed) deflate blocks
void write_png(const std::string& path, int w, int h, const uint8_t* rgba) {
    std::vector<uint8_t> raw;
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba + (size_t)y * w * 4, rgba + (size_t)(y + 1) * w * 4);
    }
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (size_t off = 0; off < raw.size() || off == 0;) {
        size_t n = std::min<size_t>(65535, raw.size() - off);
        z.push_back(off + n == raw.size() ? 1 : 0);
        z.push_back((uint8_t)n); z.push_back((uint8_t)(n >> 8));
        z.push_back((uint8_t)~n); z.push_back((uint8_t)(~n >> 8));
        z.insert(z.end(), raw.begin() + (ptrdiff_t)off, raw.begin() + (ptrdiff_t)(off + n));
        off += n;
        if (off == raw.size()) break;
    }
    uint32_t ad = b << 16 | a;
    for (int i = 3; i >= 0; --i) z.push_back((uint8_t)(ad >> (8 * i)));
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    auto be = [&](uint32_t v) { uint8_t t[4] = {(uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v};
                                std::fwrite(t, 1, 4, f); };
    auto chunk = [&](const char* type, const std::vector<uint8_t>& d) {
        be((uint32_t)d.size());
        std::vector<uint8_t> td(type, type + 4);
        td.insert(td.end(), d.begin(), d.end());
        std::fwrite(td.data(), 1, td.size(), f);
        be(crc32(td.data(), td.size()) ^ 0xFFFFFFFFu);
    };
    std::fwrite("\x89PNG\r\n\x1a\n", 1, 8, f);
    std::vector<uint8_t> ihdr = {(uint8_t)(w >> 24), (uint8_t)(w >> 16), (uint8_t)(w >> 8), (uint8_t)w,
                                 (uint8_t)(h >> 24), (uint8_t)(h >> 16), (uint8_t)(h >> 8), (uint8_t)h,
                                 8, 6, 0, 0, 0};
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    std::fclose(f);
}

namespace {

// Where the picture sits in a window of w x h (pixels or window units): the
// TV's screen, 4:3 or 16:9 as SYSCONF says, as large as the window allows and
// centred; of its 480 lines (NTSC) VI scans the XFB out over its active ones,
// centred too (360 for a game letterboxing 16:9 on a 4:3 screen).
struct Rect { float x, y, w, h; };
Rect picture_rect(float w, float h) {
    float aspect = opt.widescreen ? 16.0f / 9 : 4.0f / 3;
    float sw = w, sh = w / aspect;
    if (sh > h) { sh = h; sw = h * aspect; }
    float ph = sh * (float)std::min<uint32_t>(vi_lines.load(), 480) / 480;
    return {(w - sw) / 2, (h - ph) / 2, sw, ph};
}

// The Wii Remote on the mouse and the keyboard. The mouse over the picture
// is the pointer; the buttons come from a key file (--keys, keys.txt next to
// the extracted disc by default), written with the defaults when there is
// none. Home has no key: Esc opens the port's own menu (video_run) instead
// of the Wii's; F11 and Alt+Enter switch fullscreen. The file's
// [Classic Controller] section does the same for the Classic Controller
// (video_classic); a file without one takes the defaults' for it.
// a key or a mouse button (mask); drag: only while the mouse moves fast with the button held;
// classic: bits are the Classic's (KPAD's bits, and the sticks' directions from kStick)
struct Binding { SDL_Scancode key; uint32_t mouse; uint32_t bits; bool shake; bool drag; bool classic; };
// not Remote buttons: the Remote raised, pointing up (PadState::tilt)
constexpr uint32_t kRaisePlus = 1u << 24, kRaiseMinus = 1u << 25;
// the Classic's buttons (KPAD_CL_*), and its sticks pushed to the full by a key
enum : uint32_t { CL_UP = 0x0001, CL_LEFT = 0x0002, CL_ZR = 0x0004, CL_X = 0x0008, CL_A = 0x0010,
                  CL_Y = 0x0020, CL_B = 0x0040, CL_ZL = 0x0080, CL_R = 0x0200, CL_PLUS = 0x0400,
                  CL_HOME = 0x0800, CL_MINUS = 0x1000, CL_L = 0x2000, CL_DOWN = 0x4000, CL_RIGHT = 0x8000 };
constexpr uint32_t kStick = 1u << 16;               // L up, down, left, right, then R's: kStick << 0..7
std::vector<Binding> bindings;
int keys_input = INPUT_MODE_AUTO;                        // the key file's Input
bool face_by_label = false;                         // the key file's Face Buttons = Label
float dead_zone = 0.15f;                            // the key file's Dead Zone: of the sticks' travel, radial

const char* DEFAULT_KEYS =
    "# wiiboot's keys: each Wii Remote button, then the keys and mouse buttons that press it.\n"
    "# Keys by SDL's names (https://wiki.libsdl.org/SDL3/SDL_Scancode): A..Z, 0..9, Return,\n"
    "# Space, Tab, Backspace, Left Shift, Up, Down, Left, Right, Keypad Enter, F1..F10...\n"
    "# Mouse buttons: Mouse Left, Mouse Right, Mouse Middle, Mouse X1, Mouse X2; Drag Left, Drag Right,\n"
    "# Drag Middle: the button held and the mouse moving fast (a swing: Shake = Drag Left).\n"
    "# Fixed: Esc (the pause box), F11 and Alt+Enter (fullscreen), F12 (trace a frame's GX commands). Delete the file for the defaults.\n"
    "A     = Return, Keypad Enter, Mouse Left\n"
    "B     = Backspace, Mouse Right\n"
    "Up    = W, Up\n"
    "Down  = S, Down\n"
    "Left  = A, Left\n"
    "Right = D, Right\n"
    "Plus  = Tab\n"
    "Minus = Q\n"
    "1     = 1\n"
    "2     = 2\n"
    "Shake = Space, Mouse Middle\n"
    "\n"
    "[Classic Controller]\n"
    "# For a game that plays with the Classic Controller: its buttons and sticks on keys and mouse buttons.\n"
    "# A gamepad is a Classic Controller too, on the channels in the order the pads are plugged in. Channel 1\n"
    "# takes the keys, the mouse and the first pad at once (Input = Pad or Keyboard: only one of them;\n"
    "# wiiboot --input says the same).\n"
    "# The pad: its triggers are ZL and ZR, its shoulders L and R, Start +, Back -, its guide button Home.\n"
    "# Face Buttons = Position: the right one is A, the bottom one B, the top one X, the left one Y (the\n"
    "# Classic's own layout); Label: the button labelled A (Cross) is A, B (Circle) B, X (Square) X, Y (Triangle) Y.\n"
    "# Dead Zone: how far the pad's sticks move before they count, of their travel.\n"
    "Input        = Auto\n"
    "Face Buttons = Position\n"
    "Dead Zone    = 0.15\n"
    "A     = Return, Space\n"
    "B     = Backspace, C\n"
    "X     = R\n"
    "Y     = F\n"
    "L     = Left Shift\n"
    "R     = E\n"
    "ZL    = Mouse Right\n"
    "ZR    = Mouse Left\n"
    "Plus  = Tab\n"
    "Minus = Q\n"
    "Home  = H\n"
    "Up    = Up\n"
    "Down  = Down\n"
    "Left  = Left\n"
    "Right = Right\n"
    "Left Stick Up    = W\n"
    "Left Stick Down  = S\n"
    "Left Stick Left  = A\n"
    "Left Stick Right = D\n"
    "Right Stick Up    =\n"
    "Right Stick Down  =\n"
    "Right Stick Left  =\n"
    "Right Stick Right =\n";

std::string trim(const std::string& t) {
    size_t a = t.find_first_not_of(" \t\r"), b = t.find_last_not_of(" \t\r");
    return a == std::string::npos ? std::string() : t.substr(a, b - a + 1);
}

// The bindings of a key file's text: the Remote's lines (before any section,
// or under [Wii Remote]) if remote, the [Classic Controller] section's if
// classic. True if the text has a Classic section.
bool parse_keys(const std::string& text, const std::string& path, bool remote, bool classic) {
    struct Btn { const char* name; uint32_t bits; };
    static const Btn buttons[] = {{"A", 0x0800}, {"B", 0x0400}, {"Up", 0x0008}, {"Down", 0x0004},
                                  {"Left", 0x0001}, {"Right", 0x0002}, {"Plus", 0x0010}, {"Minus", 0x1000},
                                  {"1", 0x0200}, {"2", 0x0100}, {"Shake", 0},
                                  {"Raise", kRaisePlus}, {"Raise Alt", kRaiseMinus}};
    static const Btn cl_buttons[] = {{"A", CL_A}, {"B", CL_B}, {"X", CL_X}, {"Y", CL_Y}, {"L", CL_L}, {"R", CL_R},
                                     {"ZL", CL_ZL}, {"ZR", CL_ZR}, {"Plus", CL_PLUS}, {"Minus", CL_MINUS},
                                     {"Home", CL_HOME}, {"Up", CL_UP}, {"Down", CL_DOWN}, {"Left", CL_LEFT},
                                     {"Right", CL_RIGHT},
                                     {"Left Stick Up", kStick << 0}, {"Left Stick Down", kStick << 1},
                                     {"Left Stick Left", kStick << 2}, {"Left Stick Right", kStick << 3},
                                     {"Right Stick Up", kStick << 4}, {"Right Stick Down", kStick << 5},
                                     {"Right Stick Left", kStick << 6}, {"Right Stick Right", kStick << 7}};
    static const Btn mice[] = {{"Mouse Left", SDL_BUTTON_LMASK}, {"Mouse Right", SDL_BUTTON_RMASK},
                               {"Mouse Middle", SDL_BUTTON_MMASK}, {"Mouse X1", SDL_BUTTON_X1MASK},
                               {"Mouse X2", SDL_BUTTON_X2MASK}};
    bool in_classic = false, has_classic = false;
    size_t start = 0;
    for (int line = 1; start < text.size(); ++line) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        std::string l = trim(text.substr(start, end - start));
        start = end + 1;
        if (l.empty() || l[0] == '#') continue;
        if (l[0] == '[') {
            std::string sec = trim(l.substr(1, l.find(']') - 1));
            in_classic = SDL_strcasecmp(sec.c_str(), "Classic Controller") == 0 || SDL_strcasecmp(sec.c_str(), "Classic") == 0;
            has_classic |= in_classic;
            if (!in_classic && SDL_strcasecmp(sec.c_str(), "Wii Remote") != 0 && SDL_strcasecmp(sec.c_str(), "Remote") != 0)
                rt_log("video: %s:%d: no section [%s] ([Wii Remote], [Classic Controller])", path.c_str(), line, sec.c_str());
            continue;
        }
        if (in_classic ? !classic : !remote) continue;
        size_t eq = l.find('=');
        std::string lhs = eq == std::string::npos ? l : trim(l.substr(0, eq));
        std::string rest = eq == std::string::npos ? std::string() : trim(l.substr(eq + 1));
        if (in_classic && eq != std::string::npos) {                   // the section's settings
            if (SDL_strcasecmp(lhs.c_str(), "Input") == 0) {
                static const char* names[] = {"Auto", "Pad", "Keyboard"};
                int v = -1;
                for (int i = 0; i < 3; ++i)
                    if (SDL_strcasecmp(rest.c_str(), names[i]) == 0) v = i;
                if (v < 0) rt_log("video: %s:%d: Input is Auto, Pad or Keyboard", path.c_str(), line);
                else keys_input = v;
                continue;
            }
            if (SDL_strcasecmp(lhs.c_str(), "Face Buttons") == 0) {
                if (SDL_strcasecmp(rest.c_str(), "Label") == 0) face_by_label = true;
                else if (SDL_strcasecmp(rest.c_str(), "Position") == 0) face_by_label = false;
                else rt_log("video: %s:%d: Face Buttons is Position or Label", path.c_str(), line);
                continue;
            }
            if (SDL_strcasecmp(lhs.c_str(), "Dead Zone") == 0) {
                dead_zone = std::clamp((float)std::atof(rest.c_str()), 0.0f, 0.9f);
                continue;
            }
        }
        const Btn* b = nullptr;
        if (eq != std::string::npos) {
            if (in_classic) {
                for (const Btn& c : cl_buttons)
                    if (SDL_strcasecmp(lhs.c_str(), c.name) == 0) b = &c;
            } else {
                for (const Btn& c : buttons)
                    if (SDL_strcasecmp(lhs.c_str(), c.name) == 0) b = &c;
            }
        }
        if (!b) {
            if (in_classic) rt_log("video: %s:%d: not a Classic Controller button (A B X Y L R ZL ZR Plus Minus Home Up Down Left Right, "
                                   "Left Stick Up..., Right Stick Up...) or setting (Input, Face Buttons, Dead Zone)", path.c_str(), line);
            else rt_log("video: %s:%d: not a Remote button (A B Up Down Left Right Plus Minus 1 2 Shake Raise)", path.c_str(), line);
            continue;
        }
        for (size_t p = 0; p <= rest.size();) {
            size_t q = rest.find(',', p);
            if (q == std::string::npos) q = rest.size();
            std::string name = trim(rest.substr(p, q - p));
            p = q + 1;
            if (name.empty()) continue;
            Binding k{SDL_SCANCODE_UNKNOWN, 0, b->bits, !in_classic && b->bits == 0, false, in_classic};
            for (const Btn& m : mice)
                if (SDL_strcasecmp(name.c_str(), m.name) == 0) k.mouse = m.bits;
            static const Btn drags[] = {{"Drag Left", SDL_BUTTON_LMASK}, {"Drag Right", SDL_BUTTON_RMASK},
                                        {"Drag Middle", SDL_BUTTON_MMASK}};
            for (const Btn& m : drags)
                if (SDL_strcasecmp(name.c_str(), m.name) == 0) { k.mouse = m.bits; k.drag = true; }
            if (!k.mouse) k.key = SDL_GetScancodeFromName(name.c_str());
            if (!k.mouse && k.key == SDL_SCANCODE_UNKNOWN) { rt_log("video: %s:%d: no key named \"%s\"", path.c_str(), line, name.c_str()); continue; }
            bindings.push_back(k);
        }
    }
    return has_classic;
}

void load_keys(const std::string& path) {
    std::string text;
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
        char buf[4096];
        for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
        std::fclose(f);
    } else {
        text = DEFAULT_KEYS;
        if (FILE* out = std::fopen(path.c_str(), "wb")) {
            std::fputs(DEFAULT_KEYS, out);
            std::fclose(out);
            rt_log("video: wrote the default keys to %s", path.c_str());
        }
    }
    bindings.clear();
    if (!parse_keys(text, path, true, true)) parse_keys(DEFAULT_KEYS, "(the default keys)", false, true);
}

// ---- gamepads, as Classic Controllers ----------------------------------------------------------
// A pad keeps its channel while it stays plugged in; a new one takes the
// lowest free channel (from channel 2 with Input = Keyboard, channel 1 being
// the keyboard's alone).
SDL_Gamepad* chan_pad[4] = {};
std::atomic<bool> rumble_want[4];
int input_mode() { return opt.input >= 0 ? opt.input : keys_input; }

void pad_added(SDL_JoystickID id) {
    for (SDL_Gamepad* g : chan_pad)
        if (g && SDL_GetGamepadID(g) == id) return;
    int first = input_mode() == INPUT_MODE_KEYBOARD ? 1 : 0, chan = -1;
    for (int i = first; i < 4 && chan < 0; ++i)
        if (!chan_pad[i]) chan = i;
    if (chan < 0) { rt_log("video: a fifth pad (%s): no channel for it", SDL_GetGamepadNameForID(id)); return; }
    SDL_Gamepad* g = SDL_OpenGamepad(id);
    if (!g) { rt_log("video: pad %s: %s", SDL_GetGamepadNameForID(id), SDL_GetError()); return; }
    chan_pad[chan] = g;
    rt_log("video: pad \"%s\" (%s) on channel %d%s", SDL_GetGamepadName(g),
           SDL_GetGamepadStringForType(SDL_GetGamepadType(g)), chan + 1,
           chan == 0 && input_mode() == INPUT_MODE_AUTO ? ", with the keyboard and the mouse" : "");
}

void pad_removed(SDL_JoystickID id) {
    for (int i = 0; i < 4; ++i)
        if (chan_pad[i] && SDL_GetGamepadID(chan_pad[i]) == id) {
            rt_log("video: pad \"%s\" off channel %d", SDL_GetGamepadName(chan_pad[i]), i + 1);
            SDL_CloseGamepad(chan_pad[i]);
            chan_pad[i] = nullptr;
        }
}

float axis(SDL_Gamepad* g, SDL_GamepadAxis a) { return std::clamp(SDL_GetGamepadAxis(g, a) / 32767.0f, -1.0f, 1.0f); }

// a stick through the dead zone (radial, the rest of the travel rescaled to
// 0..1), y up
void stick(SDL_Gamepad* g, SDL_GamepadAxis ax, SDL_GamepadAxis ay, float& x, float& y) {
    x = axis(g, ax);
    y = -axis(g, ay);
    float m = std::hypot(x, y);
    float k = m <= dead_zone ? 0.0f : std::min(1.0f, (m - dead_zone) / (1 - dead_zone)) / m;
    x *= k;
    y *= k;
}

ClassicState classic_from_pad(SDL_Gamepad* g) {
    ClassicState s;
    s.connected = true;
    static const struct { SDL_GamepadButton b; uint32_t bit; } fixed[] = {
        {SDL_GAMEPAD_BUTTON_DPAD_UP, CL_UP}, {SDL_GAMEPAD_BUTTON_DPAD_DOWN, CL_DOWN},
        {SDL_GAMEPAD_BUTTON_DPAD_LEFT, CL_LEFT}, {SDL_GAMEPAD_BUTTON_DPAD_RIGHT, CL_RIGHT},
        {SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, CL_L}, {SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, CL_R},
        {SDL_GAMEPAD_BUTTON_START, CL_PLUS}, {SDL_GAMEPAD_BUTTON_BACK, CL_MINUS}, {SDL_GAMEPAD_BUTTON_GUIDE, CL_HOME}};
    for (auto& f : fixed)
        if (SDL_GetGamepadButton(g, f.b)) s.buttons |= f.bit;
    // the face buttons: by position, the Classic's own layout (A right, B
    // bottom, X top, Y left), or by the pad's labels
    static const struct { SDL_GamepadButton b; uint32_t pos; } face[] = {
        {SDL_GAMEPAD_BUTTON_EAST, CL_A}, {SDL_GAMEPAD_BUTTON_SOUTH, CL_B},
        {SDL_GAMEPAD_BUTTON_NORTH, CL_X}, {SDL_GAMEPAD_BUTTON_WEST, CL_Y}};
    for (auto& f : face) {
        if (!SDL_GetGamepadButton(g, f.b)) continue;
        uint32_t bit = f.pos;
        if (face_by_label) switch (SDL_GetGamepadButtonLabel(g, f.b)) {
            case SDL_GAMEPAD_BUTTON_LABEL_A: case SDL_GAMEPAD_BUTTON_LABEL_CROSS: bit = CL_A; break;
            case SDL_GAMEPAD_BUTTON_LABEL_B: case SDL_GAMEPAD_BUTTON_LABEL_CIRCLE: bit = CL_B; break;
            case SDL_GAMEPAD_BUTTON_LABEL_X: case SDL_GAMEPAD_BUTTON_LABEL_SQUARE: bit = CL_X; break;
            case SDL_GAMEPAD_BUTTON_LABEL_Y: case SDL_GAMEPAD_BUTTON_LABEL_TRIANGLE: bit = CL_Y; break;
            default: break;
        }
        s.buttons |= bit;
    }
    // the triggers are ZL and ZR (the Classic Controller Pro's place for
    // them); L and R, digital on the Pro, report their analog value full
    if (axis(g, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) > 0.5f) s.buttons |= CL_ZL;
    if (axis(g, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) > 0.5f) s.buttons |= CL_ZR;
    s.lt = s.buttons & CL_L ? 1.0f : 0.0f;
    s.rt = s.buttons & CL_R ? 1.0f : 0.0f;
    stick(g, SDL_GAMEPAD_AXIS_LEFTX, SDL_GAMEPAD_AXIS_LEFTY, s.lx, s.ly);
    stick(g, SDL_GAMEPAD_AXIS_RIGHTX, SDL_GAMEPAD_AXIS_RIGHTY, s.rx, s.ry);
    return s;
}

// channel 1's two sources as one: buttons OR'd, each stick the one deflected
// more (never summed: WASD and the stick together are no faster)
void merge(ClassicState& a, const ClassicState& b) {
    a.connected |= b.connected;
    a.buttons |= b.buttons;
    if (std::hypot(b.lx, b.ly) > std::hypot(a.lx, a.ly)) { a.lx = b.lx; a.ly = b.ly; }
    if (std::hypot(b.rx, b.ry) > std::hypot(a.rx, a.ry)) { a.rx = b.rx; a.ry = b.ry; }
    a.lt = std::max(a.lt, b.lt);
    a.rt = std::max(a.rt, b.rt);
}

// the Remote's motor on the channel's pad: on and off as the game asks,
// renewed while it stays on (SDL's rumble lasts as long as it is told)
void update_rumble() {
    static bool on[4];
    static Clock::time_point renewed[4];
    auto now = Clock::now();
    for (int i = 0; i < 4; ++i) {
        bool want = rumble_want[i].load() && chan_pad[i];
        if (want && (!on[i] || now - renewed[i] > std::chrono::milliseconds(500))) {
            SDL_RumbleGamepad(chan_pad[i], 0x6000, 0xA000, 1000);
            renewed[i] = now;
        } else if (!want && on[i] && chan_pad[i]) {
            SDL_RumbleGamepad(chan_pad[i], 0, 0, 0);
        }
        on[i] = want;
    }
}

void update_pad() {
    const bool* ks = SDL_GetKeyboardState(nullptr);
    PadState p;
    bool alt = SDL_GetModState() & SDL_KMOD_ALT;           // Alt+Enter is fullscreen, nothing else
    float mx = 0, my = 0;
    SDL_MouseButtonFlags mb = SDL_GetMouseState(&mx, &my);
    // the mouse's speed, in window heights a second over the last 40 ms or so;
    // a drag counts from kDragSpeed, and lasts kDragHold after it slows
    // down, so that a quick flick still gives the game a few samples
    constexpr float kDragSpeed = 1.5f;
    constexpr double kDragHold = 0.12;
    static float lx = mx, ly = my, speed = 0;
    static Clock::time_point lt = Clock::now(), drag_until{};
    {
        auto now = Clock::now();
        double dt = std::chrono::duration<double>(now - lt).count();
        int ww0 = 0, wh0 = 0;
        SDL_GetWindowSize(win, &ww0, &wh0);
        if (dt > 0.002 && wh0 > 0) {
            float v = std::hypot(mx - lx, my - ly) / (float)wh0 / (float)dt;
            float a = (float)std::min(1.0, dt / 0.04);
            speed += (v - speed) * a;
            lx = mx; ly = my; lt = now;
            if (speed > kDragSpeed)
                drag_until = now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(kDragHold));
        }
    }
    bool dragging = Clock::now() < drag_until;
    uint32_t cl = 0;                                          // the Classic's buttons and stick directions from keys
    for (const Binding& k : bindings) {
        bool down = k.mouse ? (mb & k.mouse) != 0 && (!k.drag || dragging)
                            : ks[k.key] && !(alt && (k.key == SDL_SCANCODE_RETURN || k.key == SDL_SCANCODE_KP_ENTER));
        if (!down) continue;
        if (k.classic) { cl |= k.bits; continue; }
        p.buttons |= k.bits;
        if (k.shake) p.shake = true;
        if (k.bits & kRaisePlus) p.tilt = 1;
        if (k.bits & kRaiseMinus) p.tilt = -1;
    }
    p.buttons &= ~(kRaisePlus | kRaiseMinus);
    int ww = 0, wh = 0;
    SDL_GetWindowSize(win, &ww, &wh);                        // mouse coordinates are in window units
    if (ww > 0 && wh > 0 && (SDL_GetWindowFlags(win) & SDL_WINDOW_MOUSE_FOCUS)) {
        // -1..1 spans the picture as present() shows it, so that the game's
        // cursor lands under the mouse
        Rect r = picture_rect((float)ww, (float)wh);
        p.x = (mx - r.x) / r.w * 2 - 1;
        p.y = (my - r.y) / r.h * 2 - 1;
        p.pointer = p.x >= -1 && p.x <= 1 && p.y >= -1 && p.y <= 1;
    }
    // over the picture the game draws its own cursor, as on the Wii
    static bool hidden = false;
    if (p.pointer != hidden) {
        hidden = p.pointer;
        if (hidden) SDL_HideCursor(); else SDL_ShowCursor();
    }
    // WIIKIT_PAD="45:A 50.5:@0.2,-0.1 51:A": at each time (seconds from
    // start) press buttons for 150 ms (A B 1 2 + - H U D L R, X = shake), or
    // move the pointer, which stays: reproducible runs for debugging
    static const char* script = std::getenv("WIIKIT_PAD");
    uint32_t scripted = 0;
    static const Clock::time_point t0 = Clock::now();
    static float sx = 0, sy = 0;
    static bool spointer = false;
    if (script) {
        double t = std::chrono::duration<double>(Clock::now() - t0).count();
        for (const char* q = script; *q;) {
            char* e;
            double at = std::strtod(q, &e);
            if (e == q || *e != ':') break;
            q = e + 1;
            if (*q == '@') {
                float x = std::strtof(q + 1, &e), y = std::strtof(e + 1, &e);
                if (t >= at) { sx = x; sy = y; spointer = true; }
                q = e;
            } else {
                for (; *q && *q != ' '; ++q) {
                    static const char names[] = "AB12+-HUDLR";
                    static const uint32_t bits[] = {0x0800, 0x0400, 0x0200, 0x0100, 0x0010, 0x1000,
                                                    0x8000, 0x0008, 0x0004, 0x0001, 0x0002};
                    const char* n = std::strchr(names, *q);
                    if (n && t >= at && t < at + 0.15) scripted |= bits[n - names];
                    if (*q == 'X' && t >= at && t < at + 0.15) p.shake = true;
                }
            }
            while (*q == ' ') ++q;
        }
        if (spointer && !p.pointer) { p.x = sx; p.y = sy; p.pointer = true; }
    }
    p.buttons |= scripted;
    // the Classic Controllers: channel 1 the keys, the mouse's buttons and
    // the script (its Remote buttons as the Classic's: 1 is X, 2 is Y),
    // merged with its pad unless Input says one of them; the others their pads
    ClassicState c[4];
    int mode = input_mode();
    if (mode != INPUT_MODE_PAD) {
        static const uint32_t from_remote[][2] = {{0x0800, CL_A}, {0x0400, CL_B}, {0x0200, CL_X}, {0x0100, CL_Y},
                                                  {0x0010, CL_PLUS}, {0x1000, CL_MINUS}, {0x8000, CL_HOME},
                                                  {0x0008, CL_UP}, {0x0004, CL_DOWN}, {0x0001, CL_LEFT}, {0x0002, CL_RIGHT}};
        for (auto& m : from_remote)
            if (scripted & m[0]) cl |= m[1];
        ClassicState& k = c[0];
        k.connected = true;
        k.buttons = cl & 0xFFFF;
        auto dir = [&](int i) { return cl & (kStick << i) ? 1.0f : 0.0f; };
        k.lx = dir(3) - dir(2);
        k.ly = dir(0) - dir(1);
        k.rx = dir(7) - dir(6);
        k.ry = dir(4) - dir(5);
        if (k.lx && k.ly) { k.lx *= 0.7071f; k.ly *= 0.7071f; }
        if (k.rx && k.ry) { k.rx *= 0.7071f; k.ry *= 0.7071f; }
        k.lt = k.buttons & CL_L ? 1.0f : 0.0f;
        k.rt = k.buttons & CL_R ? 1.0f : 0.0f;
    }
    for (int i = 0; i < 4; ++i)
        if (chan_pad[i] && !(i == 0 && mode == INPUT_MODE_KEYBOARD)) merge(c[i], classic_from_pad(chan_pad[i]));
    update_rumble();
    std::lock_guard<std::mutex> lk(pad_mx);
    pad = p;
    for (int i = 0; i < 4; ++i) classic[i] = c[i];
}

// The port's overlay (video_overlay_update): handed over on any thread, uploaded
// and drawn over the picture here, on the renderer's.
std::mutex overlay_mx;
std::vector<uint8_t> overlay_px;
int overlay_w = 0, overlay_h = 0;
uint64_t overlay_serial = 0;

void draw_overlay(const Rect& r, int wh) {
    static GLuint tex = 0, prog = 0;
    static int tw = 0, th = 0;
    static uint64_t shown = 0;
    {
        std::lock_guard<std::mutex> lk(overlay_mx);
        if (!overlay_w || !overlay_h) return;
        if (overlay_serial != shown) {
            if (!tex || tw != overlay_w || th != overlay_h) {
                if (tex) glDeleteTextures(1, &tex);
                glCreateTextures(GL_TEXTURE_2D, 1, &tex);
                glTextureStorage2D(tex, 1, GL_RGBA8, overlay_w, overlay_h);
                glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                tw = overlay_w;
                th = overlay_h;
            }
            glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
            glTextureSubImage2D(tex, 0, 0, 0, tw, th, GL_RGBA, GL_UNSIGNED_BYTE, overlay_px.data());
            shown = overlay_serial;
        }
    }
    if (!prog) {
        prog = link(
            "#version 450\n"
            "out vec2 uv;\n"
            "void main() {\n"
            "    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);\n"   // a triangle over the viewport
            "    uv = vec2(p.x, 1.0 - p.y);\n"                                   // rows top first
            "    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n"
            "}\n",
            "#version 450\n"
            "in vec2 uv;\n"
            "layout(binding = 0) uniform sampler2D overlay;\n"
            "out vec4 col;\n"
            "void main() { col = texture(overlay, uv); }\n");
        if (!prog) return;
    }
    // The game's draws set their viewport, program and blending each time: nothing to restore but
    // blending off and the usual vertex array. GL's rows count from the bottom; the picture is
    // centred, so its rectangle reads the same either way.
    glViewport((GLint)std::lround(r.x), wh - (GLint)std::lround(r.y + r.h), (GLsizei)std::lround(r.w), (GLsizei)std::lround(r.h));
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_COLOR_LOGIC_OP);
    glUseProgram(prog);
    glBindTextureUnit(0, tex);
    glBindSampler(0, 0);
    glBindVertexArray(empty_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(vao);
    glDisable(GL_BLEND);
}

void present() {
    ++cnt.presents;
    int ww = 0, wh = 0;
    SDL_GetWindowSizeInPixels(win, &ww, &wh);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_SCISSOR_TEST);
    glColorMask(1, 1, 1, 1);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    auto it = xfbs.find(xfb_addr.load());
    if (it == xfbs.end()) it = xfbs.find(last_xfb);
    if (it != xfbs.end() && ww > 0 && wh > 0) {
        const Tex& t = it->second;
        Rect r = picture_rect((float)ww, (float)wh);
        int dx = (int)std::lround(r.x), dy = (int)std::lround(r.y);
        int dw = (int)std::lround(r.w), dh = (int)std::lround(r.h);
        glNamedFramebufferTexture(copy_fbo, GL_COLOR_ATTACHMENT0, t.name, 0);
        glBlitNamedFramebuffer(copy_fbo, 0, 0, 0, t.w, t.h, dx, dy + dh, dx + dw, dy,   // top row first
                               GL_COLOR_BUFFER_BIT, GL_LINEAR);
        draw_overlay(r, wh);
    }
    if (opt.dump_dir && cnt.presents % (uint64_t)opt.dump_every == 0 && ww > 0 && wh > 0) {
        std::vector<uint8_t> px((size_t)ww * wh * 4), flip(px.size());   // the window, as shown
        glReadPixels(0, 0, ww, wh, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        for (int y = 0; y < wh; ++y)
            std::memcpy(&flip[(size_t)y * ww * 4], &px[(size_t)(wh - 1 - y) * ww * 4], (size_t)ww * 4);
        char name[64];
        std::snprintf(name, sizeof name, "/frame_%06llu.png", (unsigned long long)cnt.presents);
        write_png(std::string(opt.dump_dir) + name, ww, wh, flip.data());
    }
    SDL_GL_SwapWindow(win);
}

void setup() {
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glDisable(GL_DITHER);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    glCreateTextures(GL_TEXTURE_2D, 1, &efb_col);
    glTextureStorage2D(efb_col, 1, GL_RGBA8, EFB_W * S, EFB_H * S);
    glCreateTextures(GL_TEXTURE_2D, 1, &efb_dep);
    glTextureStorage2D(efb_dep, 1, GL_DEPTH_COMPONENT32F, EFB_W * S, EFB_H * S);
    for (GLuint t : {efb_col, efb_dep}) {
        glTextureParameteri(t, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTextureParameteri(t, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    glCreateFramebuffers(1, &efb_fbo);
    glNamedFramebufferTexture(efb_fbo, GL_COLOR_ATTACHMENT0, efb_col, 0);
    glNamedFramebufferTexture(efb_fbo, GL_DEPTH_ATTACHMENT, efb_dep, 0);
    if (glCheckNamedFramebufferStatus(efb_fbo, GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        rt_die("video: the EFB framebuffer is incomplete");
    glCreateFramebuffers(1, &copy_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, efb_fbo);
    glClearColor(0, 0, 0, 1);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glCreateBuffers(1, &vbo);
    const GLbitfield map = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    glNamedBufferStorage(vbo, VBO_CAP, nullptr, map);
    vbo_ptr = static_cast<uint8_t*>(glMapNamedBufferRange(vbo, 0, VBO_CAP, map));
    if (!vbo_ptr) rt_die("video: the vertex buffer cannot be mapped");
    std::vector<uint32_t> qi;
    for (uint32_t k = 0; k < 0x10000 / 4; ++k)
        for (uint32_t d : {0u, 1u, 2u, 0u, 2u, 3u}) qi.push_back(4 * k + d);
    glCreateBuffers(1, &quad_ibo);
    glNamedBufferStorage(quad_ibo, qi.size() * 4, qi.data(), 0);
    glCreateVertexArrays(1, &vao);
    glCreateVertexArrays(1, &empty_vao);
    glVertexArrayVertexBuffer(vao, 0, vbo, 0, sizeof(GVtx));
    glVertexArrayElementBuffer(vao, quad_ibo);
    struct A { GLuint loc; int n; GLenum type; bool norm, integer; size_t off; };
    const A attrs[] = {{0, 3, GL_FLOAT, false, false, offsetof(GVtx, pos)},
                       {1, 3, GL_FLOAT, false, false, offsetof(GVtx, nrm)},
                       {2, 3, GL_FLOAT, false, false, offsetof(GVtx, bin)},
                       {3, 3, GL_FLOAT, false, false, offsetof(GVtx, tan)},
                       {4, 4, GL_UNSIGNED_BYTE, true, false, offsetof(GVtx, col)},
                       {5, 4, GL_UNSIGNED_BYTE, true, false, offsetof(GVtx, col) + 4},
                       {6, 4, GL_FLOAT, false, false, offsetof(GVtx, tc)},
                       {7, 4, GL_FLOAT, false, false, offsetof(GVtx, tc) + 16},
                       {8, 4, GL_FLOAT, false, false, offsetof(GVtx, tc) + 32},
                       {9, 4, GL_FLOAT, false, false, offsetof(GVtx, tc) + 48},
                       {10, 4, GL_UNSIGNED_BYTE, false, true, offsetof(GVtx, mtx)},
                       {11, 4, GL_UNSIGNED_BYTE, false, true, offsetof(GVtx, mtx) + 4},
                       {12, 4, GL_UNSIGNED_BYTE, false, true, offsetof(GVtx, mtx) + 8}};
    for (const A& a : attrs) {
        glEnableVertexArrayAttrib(vao, a.loc);
        if (a.integer) glVertexArrayAttribIFormat(vao, a.loc, a.n, a.type, (GLuint)a.off);
        else glVertexArrayAttribFormat(vao, a.loc, a.n, a.type, a.norm, (GLuint)a.off);
        glVertexArrayAttribBinding(vao, a.loc, 0);
    }
    glBindVertexArray(vao);

    glCreateBuffers(1, &xf_ubo);
    glNamedBufferStorage(xf_ubo, sizeof xf, nullptr, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, xf_ubo);
    glCreateBuffers(1, &ps_ubo);
    glNamedBufferStorage(ps_ubo, sizeof(PSUniforms), nullptr, GL_DYNAMIC_STORAGE_BIT);
    glBindBufferBase(GL_UNIFORM_BUFFER, 1, ps_ubo);
    glCreateSamplers(8, samplers);

    copy_prog = link(FULLSCREEN_VS, COPY_FS);
    if (!copy_prog) rt_die("video: the EFB copy program does not build");
    copy_rect_loc = glGetUniformLocation(copy_prog, "u_rect");
    copy_mode_loc = glGetUniformLocation(copy_prog, "u_mode");
}

}  // namespace

// ---- the game's side ------------------------------------------------------------------------------
bool video_enabled() { return opt.enabled; }
void video_configure(const VideoOptions& o) { opt = o; S = std::max(1, o.scale); }

VideoPerf g_vperf;
bool g_vperf_on = std::getenv("WIIKIT_PERF") != nullptr;

bool video_submit(std::vector<uint8_t>& rec, int frames, int wait_ms) {
    auto t0 = Clock::now();
    std::unique_lock<std::mutex> lk(qmx);
    // frames in flight: each one queued is 33 ms more between what the game
    // decides and what is seen, and one less the game and the renderer can
    // work on side by side
    auto room = [] { return q_frames < opt.frames_ahead && q.size() < 256; };
    bool ok = true;
    if (wait_ms < 0) q_space.wait(lk, room);
    else ok = q_space.wait_for(lk, std::chrono::milliseconds(wait_ms), room);
    if (g_vperf_on) g_vperf.wait += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
    if (!ok) return false;
    q.push_back(Chunk{std::move(rec), frames});
    q_frames += frames;
    // a buffer the renderer is done with, capacity and pages kept: fresh
    // allocations of megabytes a frame cost page faults and copies as they grow
    if (!spare.empty()) {
        rec = std::move(spare.back());
        spare.pop_back();
    } else {
        rec = std::vector<uint8_t>();
        rec.reserve(2u << 20);
    }
    lk.unlock();
    if (wake) SDL_SignalSemaphore(wake);
    return true;
}

bool video_efb_read(std::vector<uint32_t>& argb, std::vector<uint32_t>& z) {
    if (!opt.enabled) return false;
    std::unique_lock<std::mutex> lk(rb_mx);
    rb_argb = &argb;
    rb_z = &z;
    rb_ready = false;
    {
        std::lock_guard<std::mutex> ql(qmx);
        q.push_back(Chunk{{}, 0, true});
    }
    if (wake) SDL_SignalSemaphore(wake);
    // the console's CPU takes its interrupts while it waits: so does this wait
    while (!rb_ready) {
        rb_cv.wait_for(lk, std::chrono::milliseconds(1));
        if (!rb_ready && t_ppc && g_ppc_pending.load(std::memory_order_relaxed)) {
            lk.unlock();
            ppc_poll(*t_ppc);
            lk.lock();
        }
    }
    return true;
}

bool video_copy_read(uint32_t addr, int w, int h, std::vector<uint8_t>& rgba) {
    if (!opt.enabled) return false;
    std::unique_lock<std::mutex> lk(rb_mx);
    rb_rgba = &rgba;
    rb_w = w;
    rb_h = h;
    rb_ready = false;
    {
        std::lock_guard<std::mutex> ql(qmx);
        q.push_back(Chunk{{}, 0, true, addr});
    }
    if (wake) SDL_SignalSemaphore(wake);
    while (!rb_ready) {                              // taking interrupts meanwhile, as video_efb_read
        rb_cv.wait_for(lk, std::chrono::milliseconds(1));
        if (!rb_ready && t_ppc && g_ppc_pending.load(std::memory_order_relaxed)) {
            lk.unlock();
            ppc_poll(*t_ppc);
            lk.lock();
        }
    }
    return true;
}

void video_set_xfb(uint32_t a) { xfb_addr.store(a); }
void video_set_lines(uint32_t n) { vi_lines.store(n); }
std::atomic<bool> want_relative{false};
std::mutex motion_mx;
float motion_x = 0, motion_y = 0;
void video_set_relative_mouse(bool on) { want_relative = on; }
void video_take_mouse_motion(float& dx, float& dy) {
    std::lock_guard<std::mutex> lk(motion_mx);
    dx = motion_x; dy = motion_y;
    motion_x = motion_y = 0;
}

void video_model_mesh(const float* verts, int n, const uint8_t* rgba, int layers, int w, int h) {
    std::lock_guard<std::mutex> lk(model_mx);
    if (!verts || n <= 0 || !rgba || layers <= 0 || w <= 0 || h <= 0) {
        model_n = 0;
        return;
    }
    n = std::min(n, kModelMaxVerts);
    model_verts.assign(verts, verts + (size_t)n * kModelVertFloats);
    model_tex.assign(rgba, rgba + (size_t)w * h * layers * 4);
    model_n = n;
    model_layers = layers;
    model_tw = w;
    model_th = h;
    ++model_serial;
}


void video_overlay_update(const uint8_t* rgba, int w, int h) {
    std::lock_guard<std::mutex> lk(overlay_mx);
    if (!rgba || w <= 0 || h <= 0) {
        overlay_w = overlay_h = 0;
        return;
    }
    overlay_px.assign(rgba, rgba + (size_t)w * h * 4);
    overlay_w = w;
    overlay_h = h;
    ++overlay_serial;
}

PadState video_pad() {
    std::lock_guard<std::mutex> lk(pad_mx);
    return pad;
}
ClassicState video_classic(int chan) {
    std::lock_guard<std::mutex> lk(pad_mx);
    return chan >= 0 && chan < 4 ? classic[chan] : ClassicState{};
}
void video_set_rumble(int chan, bool on) {
    if (chan >= 0 && chan < 4) rumble_want[chan] = on;
}
void video_retrace() {
    retraces.fetch_add(1);
    if (wake) SDL_SignalSemaphore(wake);
}

// ---- the window --------------------------------------------------------------------------------
void video_run(const char* title) {
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) rt_die("video: SDL_Init: %s", SDL_GetError());
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    bool debug = std::getenv("WIIKIT_GLDEBUG") != nullptr;
    if (debug) SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
    // 720 lines at the screen's shape unless --window says otherwise;
    // fullscreen is SDL's borderless one at the desktop's mode: the other
    // windows stay, and the picture keeps its shape with bars at the sides
    int w0 = opt.window_w ? opt.window_w : (opt.widescreen ? 1280 : 960), h0 = opt.window_h ? opt.window_h : 720;
    SDL_WindowFlags wflags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE;
    if (opt.fullscreen) wflags |= SDL_WINDOW_FULLSCREEN;
    win = SDL_CreateWindow(title, w0, h0, wflags);
    if (!win) rt_die("video: SDL_CreateWindow: %s", SDL_GetError());
    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) rt_die("video: no OpenGL 4.5 context: %s", SDL_GetError());
    SDL_GL_MakeCurrent(win, ctx);
    SDL_GL_SetSwapInterval(0);                   // VI paces the presents
    if (!gl_load()) rt_die("video: OpenGL functions missing");
    rt_log("video: %s, %s", (const char*)glGetString(GL_RENDERER), (const char*)glGetString(GL_VERSION));
    if (debug) {
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(gl_debug, nullptr);
    }
    wake = SDL_CreateSemaphore(0);
    load_keys(opt.keys.empty() ? std::string("keys.txt") : opt.keys);
    if (opt.scale <= 0) {
        // enough EFB lines for the screen's height at the window's first
        // size: 3 for 1080 or 1440 lines, 2 for 720
        int ww = 0, wh = 0;
        SDL_GetWindowSizeInPixels(win, &ww, &wh);
        float aspect = opt.widescreen ? 16.0f / 9 : 4.0f / 3;
        float sh = std::min((float)wh, (float)ww / aspect);
        S = std::clamp((int)std::ceil(sh / 480 - 0.01f), 1, 4);
    }
    rt_log("video: internal resolution x%d (EFB %d x %d)", S, EFB_W * S, EFB_H * S);
    setup();

    auto t0 = Clock::now(), t_title = t0;
    uint64_t frames_then = 0;
    uint32_t seen = retraces.load();
    std::string base_title = title;
    for (;;) {
        SDL_Event e;
        bool quit = false;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) quit = true;
            if (e.type == SDL_EVENT_GAMEPAD_ADDED) pad_added(e.gdevice.which);
            if (e.type == SDL_EVENT_GAMEPAD_REMOVED) pad_removed(e.gdevice.which);
            // F11 or Alt+Enter: fullscreen and back
            if (e.type == SDL_EVENT_KEY_DOWN && !e.key.repeat &&
                (e.key.scancode == SDL_SCANCODE_F11 ||
                 ((e.key.scancode == SDL_SCANCODE_RETURN || e.key.scancode == SDL_SCANCODE_KP_ENTER) &&
                  (e.key.mod & SDL_KMOD_ALT))))
                SDL_SetWindowFullscreen(win, !(SDL_GetWindowFlags(win) & SDL_WINDOW_FULLSCREEN));
            // F12: the next frame's GX commands to a file (debugging, as WIIKIT_GXTRACE)
            if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_F12 && !e.key.repeat)
                gx_trace_next_frame();
            // Esc: the port's menu in place of the Wii's Home Button menu.
            // While it is open the renderer stops, and the game with it as
            // soon as its FIFO record queue is full.
            if (e.type == SDL_EVENT_KEY_DOWN && e.key.scancode == SDL_SCANCODE_ESCAPE && !e.key.repeat) {
                const SDL_MessageBoxButtonData buttons[] = {
                    {SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT | SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT, 0, "Resume"},
                    {0, 1, "Quit"}};
                SDL_MessageBoxData box = {SDL_MESSAGEBOX_INFORMATION, win, base_title.c_str(), "Paused.",
                                          2, buttons, nullptr};
                int choice = 0;
                SDL_SetWindowRelativeMouseMode(win, false);
                if (SDL_ShowMessageBox(&box, &choice) && choice == 1) quit = true;
            }
        }
        auto now = Clock::now();
        if (opt.quit_after > 0 && std::chrono::duration<double>(now - t0).count() > opt.quit_after) quit = true;
        if (quit) {
            rt_log("video: %llu frames, %llu presents, %llu draws, %llu programs",
                   (unsigned long long)cnt.frames, (unsigned long long)cnt.presents,
                   (unsigned long long)cnt.draws, (unsigned long long)cnt.programs);
            gx_report();
            std::fflush(stdout);
            std::_Exit(0);
        }
        update_pad();
        {
            bool rel = want_relative && (SDL_GetWindowFlags(win) & SDL_WINDOW_INPUT_FOCUS);
            if (rel != SDL_GetWindowRelativeMouseMode(win)) SDL_SetWindowRelativeMouseMode(win, rel);
            float dx = 0, dy = 0;
            SDL_GetRelativeMouseState(&dx, &dy);
            if (rel) {
                std::lock_guard<std::mutex> lk(motion_mx);
                motion_x += dx; motion_y += dy;
            }
        }
        if (now - t_title >= std::chrono::seconds(1)) {
            double s = std::chrono::duration<double>(now - t_title).count();
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s  |  %.1f fps", base_title.c_str(), (double)(cnt.frames - frames_then) / s);
            SDL_SetWindowTitle(win, buf);
            if (g_vperf_on && cnt.frames > frames_then) {
                double f = (double)(cnt.frames - frames_then);
                auto ms = [&](std::atomic<uint64_t>& a) { return (double)a.exchange(0) / 1e6 / f; };
                double vtx = ms(g_vperf.vtx), tex = ms(g_vperf.tex), wait = ms(g_vperf.wait), draw = ms(g_vperf.draw),
                       pres = ms(g_vperf.present);
                rt_log("perf: %.1f fps; frame %.1f ms; game thread: vertices %.1f, textures %.1f, waiting for the renderer %.1f; "
                       "renderer %.1f, presenting %.1f ms", f / s, 1000.0 * s / f, vtx, tex, wait, draw, pres);
            }
            frames_then = cnt.frames;
            t_title = now;
        }
        bool busy = false;
        for (int k = 0; k < 16; ++k) {
            Chunk c;
            {
                std::lock_guard<std::mutex> lk(qmx);
                if (q.empty()) break;
                c = std::move(q.front());
                q.pop_front();
            }
            if (c.readback) { if (c.copy) copy_readback(c.copy); else efb_readback(); busy = true; continue; }
            auto te = Clock::now();
            exec(c.data);
            if (g_vperf_on) g_vperf.draw += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - te).count();
            {
                std::lock_guard<std::mutex> lk(qmx);
                q_frames -= c.frames;
                c.data.clear();
                if (spare.size() < 16) spare.push_back(std::move(c.data));
            }
            q_space.notify_all();
            busy = true;
            if (retraces.load() != seen) break;
        }
        uint32_t r = retraces.load();
        if (r != seen) {
            seen = r;
            auto tp = Clock::now();
            present();
            if (g_vperf_on) g_vperf.present += (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - tp).count();
            busy = true;
        }
        if (!busy) SDL_WaitSemaphoreTimeout(wake, 5);
    }
}
