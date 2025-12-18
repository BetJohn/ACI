#pragma once

#include "components/simple_scene.h"
#include <optional>
#include <array>

static constexpr int GRID_W = 160*2;
static constexpr int GRID_H = 90*2;
template <typename T>
using Pixels = std::array<std::array<T, GRID_W>, GRID_H>;

namespace m1
{
    class GPUWorld : public gfxc::SimpleScene
    {
    public:
        GPUWorld();
        ~GPUWorld();
        void RegenerateTrackBorder(float limit);
        void Init() override;

    private:
        void FrameStart() override;
        void Update(float deltaTimeSeconds) override;
        void FrameEnd() override;

      //  void OnInputUpdate(float deltaTime, int mods) override;
        void OnKeyPress(int key, int mods) override;
       // void OnKeyRelease(int key, int mods) override;
      //  void OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY) override;
       // void OnMouseBtnPress(int mouseX, int mouseY, int button, int mods) override;
      //  void OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods) override;
      //  void OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY) override;
      //  void OnWindowResize(int width, int height) override;

        void SetFrame();
        void ControlImGuiWindow();
        
        std::optional<glm::ivec2> MouseToGrid(int mouseX, int mouseY);



        void DrawTemporaryLine(glm::vec3 p1, glm::vec3 p2, glm::vec3 color1, glm::vec3 color2, gfxc::SimpleScene* ss);


    private:




        


    };
}   // namespace m1
