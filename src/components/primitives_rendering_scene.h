#pragma once

#include <string>
#include <vector>

#include "components/simple_scene.h"
#include "components/text_renderer.h"

namespace gfxc
{
    class PrimitivesRenderingScene : public gfxc::SimpleScene
    {
     private:
        struct ViewportSpace
        {
            ViewportSpace () : x (0), y (0), width (1), height (1) {}
            ViewportSpace (int x, int y, int width, int height)
                : x (x), y (y), width (width), height (height) {}
            int x;
            int y;
            int width;
            int height;
        };

        struct LogicSpace
        {
            LogicSpace () : x (0), y (0), pixel_count (1) {}
            LogicSpace (float x, float y, unsigned int pixel_count)
                : x (x), y (y), pixel_count (pixel_count) {}
            float x;
            float y;
            unsigned int pixel_count;
        };

        struct TextElement
        {
            TextElement () : position (0), text (), size(1), color(0) {}
            TextElement (const glm::vec2 &position, const std::string &text,
                const glm::vec3 &color, float size)
                : position (position), text (text), color (color), size (size) {}
            glm::vec2 position;
            std::string text;
            glm::vec3 color;
            float size;
        };

        struct ExercisePanel
        {
            LogicSpace logic_space;
            ViewportSpace viewport_space;
            glm::mat3 viewport_transformation;

            std::vector<VertexFormat> vertices;
            std::vector<unsigned int> indices;

            std::vector<TextElement> text_elements;

            Mesh* mesh;
        };

     public:
        PrimitivesRenderingScene();
        ~PrimitivesRenderingScene();

        void Init() override;

     protected:
        virtual void Initialize() = 0;

        void FrameStart() override;
        void Update(float deltaTimeSeconds) override;
        void FrameEnd() override;

        void OnInputUpdate(float deltaTime, int mods) override;
        void OnKeyPress(int key, int mods) override;
        void OnKeyRelease(int key, int mods) override;
        void OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY) override;
        void OnMouseBtnPress(int mouseX, int mouseY, int button, int mods) override;
        void OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods) override;
        void OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY) override;
        void OnWindowResize(int width, int height) override;

        void SetExercise();
        void SetPanel(const std::string &label, const LogicSpace&, const ViewportSpace&);

        // Draw an arrow that represents the vector on the canvas
        // @param v - the vector being drawn
        // @param color - the drawing color of the vector
        // @param label - optional! the label of the vector to be drawn, the default value is std::string()
        // @param p - optional! The point to which the tail of the arrow is attached in the process
        //          - of drawing the vector. the default value is glm::vec3(0.0f)
        void RenderVector(const glm::vec2 &v, const glm::vec3 &color,
            const std::string &label = std::string(),
            const glm::vec2 &p = glm::vec2(0.0f));

        // Draw a small circle disc at the position of a point on the canvas
        // @param p - the point at which the circle disc is drawn
        // @param color - the color with which the circle disc is drawn
        // @param label - optional! the label with which the circle disc is drawn, the default value is std::string()
        void RenderPoint(const glm::vec2 &p, const glm::vec3 &color,
            const std::string &label = std::string());

        // Draw an arc of a circle in the counterclockwise direction on the canvas.
        //     - the arc angle is calculated with the Ox axis
        // @param start_angle - the angle from which the circular arc begins.
        // @param angle - arc angle in radians
        // @param color - the color with which the circle arc is drawn
        // @param radius - optional! radius of the circular arc, the default value is 3
        // @param center - optional! the center of the circular arc, the default value is glm::vec3(0.0f)
        void RenderArc(float start_angle, float angle, const glm::vec3 &color,
            float radius = 3.0f, const glm::vec2 &center = glm::vec2(0.0f));

        // Draw an unfilled triangle, visualized only through its sides on the canvas
        // @param p1 - the first point of the triangle to be drawn
        // @param p2 - the second point of the triangle that is being drawn
        // @param p3 - the third point of the triangle that is being drawn
        // @param color - the color with which the triangle is drawn
        void RenderTriangle(const glm::vec2 &p1, const glm::vec2 &p2,
            const glm::vec2 &p3, const glm::vec3 &color);

        // Draw a text on the canvas
        // @param position - the position at which the text is drawn
        // @param text - the text that is drawn
        // @param color - the color with which the text is drawn
        // @param label - optional! the size at which the text is drawn, the default value is 1
        void RenderText(const glm::vec2 &position, const std::string &text,
            const glm::vec3 &color, float size = 1.0f);

     private:
        void RenderArrow(const glm::vec2 &p1, const glm::vec2 &p2,
            const glm::vec3 &color, const std::string &label = std::string(),
            unsigned int width = 5, unsigned int head_size = 10);
        void RenderCircle(const glm::vec2 &center, float radius,
            const glm::vec3 &color);
        void RenderLine(const glm::vec2 &p1, const glm::vec2 &p2,
            const glm::vec3 &color, unsigned int width = 5);
        void RenderLineViewportSpace(const glm::vec2 &p1, const glm::vec2 &p2,
            const glm::vec3 &color, unsigned int width = 5);
        void RenderFilledTriangle(const glm::vec2 &p1, const glm::vec2 &p2,
            const glm::vec2 &p3, const glm::vec3 &color);
        void RenderFilledTriangleViewportSpace(const glm::vec2 &p1, const glm::vec2 &p2,
            const glm::vec2 &p3, const glm::vec3 &color);
        std::string GetLabel(const std::string &label, const glm::vec2 &vec, bool is_point = true);

     private:
         unsigned int exercise_count;
         unsigned int exercise_index;

         glm::mat3 viewport_transformation;
         float pixel_count;
         float depth;

         std::vector<std::vector<ExercisePanel>> exercises;

         // The actual renderer object
         gfxc::TextRenderer* textRenderer;
    };
}   // namespace m2
