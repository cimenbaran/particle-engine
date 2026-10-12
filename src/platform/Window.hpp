#pragma once
#include <SDL3/SDL.h>
#include <bgfx/bgfx.h>
#include <stdexcept>
#include <string>
namespace pw {
class Window {
public:
    Window() {
        if (!SDL_Init(SDL_INIT_VIDEO)) throw std::runtime_error(SDL_GetError());
        handle_ = SDL_CreateWindow("Particle World | Sprint 1", 1280, 720,
            SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
        if (!handle_) { const std::string error = SDL_GetError(); SDL_Quit(); throw std::runtime_error(error); }
    }
    ~Window() { SDL_DestroyWindow(handle_); SDL_Quit(); }
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    SDL_Window* get() const { return handle_; }
    void drawable(int& w, int& h) const { SDL_GetWindowSizeInPixels(handle_, &w, &h); }
    void* nativeHandle() const {
        void* handle = nullptr;
        auto properties = SDL_GetWindowProperties(handle_);
#if defined(__APPLE__)
        // bgfx owns the CAMetalLayer on the SDL NSWindow. Do not create an SDL Metal view.
        handle = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr);
#elif defined(_WIN32)
        handle = SDL_GetPointerProperty(properties, SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr);
#endif
        if (!handle) throw std::runtime_error("SDL native window handle is unavailable");
        return handle;
    }
private:
    SDL_Window* handle_{};
};
}
