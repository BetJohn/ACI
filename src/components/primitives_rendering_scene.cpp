#include "components/primitives_rendering_scene.h"

#include <vector>
#include <iostream>

using namespace std;

using namespace gfxc;


/*
 *  To find out more about `FrameStart`, `Update`, `FrameEnd`
 *  and the order in which they are called, see `world.cpp`.
 */


PrimitivesRenderingScene::PrimitivesRenderingScene()
{
    window->SetSize(900, 900);
}


PrimitivesRenderingScene::~PrimitivesRenderingScene()
{

}

void PrimitivesRenderingScene::Init()
{
    glm::ivec2 resolution = window->GetResolution();
    auto camera = GetSceneCamera();
    camera->SetPosition(glm::vec3(0, 0, 50));
    camera->SetRotation(glm::vec3(0, 0, 0));
    camera->Update();
    GetCameraInput()->SetActive(false);

    textRenderer = new gfxc::TextRenderer(window->props.selfDir, resolution.x, resolution.y);

    textRenderer->Load(PATH_JOIN(window->props.selfDir, RESOURCE_PATH::FONTS, "Hack-Bold.ttf"), 18);

    Initialize();

    for (unsigned int exercise_index = 0; exercise_index < exercises.size(); exercise_index++) {
        for (unsigned int panel_index = 0; panel_index < exercises[exercise_index].size(); panel_index++) {

            auto &exercise_panel = exercises[exercise_index][panel_index];

            Mesh* mesh = new Mesh(std::string());
            mesh->InitFromData(
                exercise_panel.vertices,
                exercise_panel.indices
            );

            exercise_panel.mesh = mesh;
        }
    }

    exercise_index = 0;
}

void PrimitivesRenderingScene::FrameStart()
{

}

void PrimitivesRenderingScene::Update(float deltaTimeSeconds)
{
    ClearScreen(glm::vec3(1.0f));

    for (auto panel : exercises[exercise_index]) {
        const auto &viewport = panel.viewport_space;

        RenderMesh2D(panel.mesh, shaders["Simple2D"], glm::ivec2(1280, 720),
            glm::ivec4(viewport.x, viewport.y, viewport.width, viewport.height));

        for (auto text_element : panel.text_elements) {
            textRenderer->RenderText(text_element.text, text_element.position.x, text_element.position.y, text_element.size, text_element.color);
        }
    }
}


void PrimitivesRenderingScene::FrameEnd()
{
    //DrawCoordinateSystem();
}

/*
 *  These are callback functions. To find more about callbacks and
 *  how they behave, see `input_controller.h`.
 */


void PrimitivesRenderingScene::OnInputUpdate(float deltaTime, int mods)
{
    // Treat continuous update based on input
}


void PrimitivesRenderingScene::OnKeyPress(int key, int mods)
{
    if (key - GLFW_KEY_1 < 9 && key - GLFW_KEY_1 < exercises.size())
    {
        exercise_index = key - GLFW_KEY_1;
    }
}


void PrimitivesRenderingScene::OnKeyRelease(int key, int mods)
{
    // Add key release event
}


void PrimitivesRenderingScene::OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY)
{
    // Add mouse move event
}


void PrimitivesRenderingScene::OnMouseBtnPress(int mouseX, int mouseY, int button, int mods)
{
    // Add mouse button press event
}


void PrimitivesRenderingScene::OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods)
{
    // Add mouse button release event
}


void PrimitivesRenderingScene::OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY)
{
    // Treat mouse scroll event
}


void PrimitivesRenderingScene::OnWindowResize(int width, int height)
{
    // Treat window resize event
}

void PrimitivesRenderingScene::SetExercise()
{
    exercises.push_back(std::vector<ExercisePanel>());
}

void PrimitivesRenderingScene::SetPanel(const std::string &label,
    const LogicSpace &logic_space, const ViewportSpace &viewport_space)
{
    float tx, ty;
    tx = viewport_space.x - logic_space.pixel_count * logic_space.x;
    ty = viewport_space.y - logic_space.pixel_count * logic_space.y;

    viewport_transformation = glm::transpose(glm::mat3(
        logic_space.pixel_count, 0.0f, tx,
        0.0f, logic_space.pixel_count, ty,
        0.0f, 0.0f, 1.0f));

    pixel_count = logic_space.pixel_count;

    float width = (float)viewport_space.width / logic_space.pixel_count;
    float height = (float)viewport_space.height / logic_space.pixel_count;

    depth = 1.0f;

    exercises.back().push_back(ExercisePanel());

    exercises.back().back().logic_space = logic_space;
    exercises.back().back().viewport_space = viewport_space;

    for (int index = (int)logic_space.y; index <= height; index++) {
        RenderLine(glm::vec2(logic_space.x, index), glm::vec2(width, index), glm::vec3(0.25f), 1);
        RenderText(glm::vec2(0.1, index), std::to_string(index), glm::vec3(0.0f), 1);
    }

    for (int index = logic_space.x; index <= width; index++) {
        RenderLine(glm::vec2(index, logic_space.y), glm::vec2(index, height), glm::vec3(0.25f), 1);
        RenderText(glm::vec2(index, -0.1), std::to_string(index), glm::vec3(0.0f), 1);
    }

    RenderArrow(glm::vec2(logic_space.x, 0), glm::vec2(logic_space.x + width, 0), glm::vec3(0.1f), "", 3, 25);
    RenderArrow(glm::vec2(0, logic_space.y), glm::vec2(0, logic_space.y + height), glm::vec3(0.1f), "", 3, 25);
}

void PrimitivesRenderingScene::RenderVector(const glm::vec2 &v, const glm::vec3 &color,
    const std::string &label, const glm::vec2 &p)
{
    RenderArrow(p, p + v, color, label != std::string() ? GetLabel(label, v, false) : std::string(), 5, 25);
}

void PrimitivesRenderingScene::RenderPoint(const glm::vec2 &position, const glm::vec3 &color,
    const std::string &label)
{
    RenderCircle(position, 10, color);

    if (label != std::string()) {
        RenderText(position + glm::vec2(0.25), GetLabel(label, position), color, 1);
    }
}

void PrimitivesRenderingScene::RenderArc(float start_angle, float angle, const glm::vec3 &color,
    float radius, const glm::vec2 &center)
{
    const unsigned int segment_count = 100;

    for (unsigned int segment_index = 0; segment_index < segment_count; segment_index++) {
        glm::vec2 p1 = center + radius * glm::vec2(
            cos(start_angle + angle / segment_count * segment_index),
            sin(start_angle + angle / segment_count * segment_index)
        );

        glm::vec2 p2 = center + radius * glm::vec2(
            cos(start_angle + angle / segment_count * (segment_index + 1)),
            sin(start_angle + angle / segment_count * (segment_index + 1))
        );

        RenderLine(p1, p2, color, 5);
    }
}

void PrimitivesRenderingScene::RenderTriangle(const glm::vec2 &p1, const glm::vec2 &p2,
    const glm::vec2 &p3, const glm::vec3 &color)
{
    RenderLine(p1, p2, color);
    RenderLine(p1, p3, color);
    RenderLine(p2, p3, color);
}

void PrimitivesRenderingScene::RenderText(const glm::vec2 &pos, const std::string &text,
    const glm::vec3 &color, float size)
{
    glm::vec2 viewport_space_position = viewport_transformation * glm::vec3(pos, 1);
    //TODO: Change this
    viewport_space_position.y = window->props.resolution.y - viewport_space_position.y;

    exercises.back().back().text_elements.push_back(TextElement(viewport_space_position, text, color, size));
}

/// Internal methods!

void PrimitivesRenderingScene::RenderArrow(const glm::vec2 &p1, const glm::vec2 &p2,
    const glm::vec3 &color, const std::string &label, unsigned int width, unsigned int head_size)
{
    glm::vec2 v = p2 - p1;
    glm::vec2 vp = glm::cross(glm::vec3(glm::normalize(v), 1), glm::vec3(0, 0, 1));

    glm::vec2 p_internal = p2 - glm::normalize(v) * (float)head_size / pixel_count;
    glm::vec2 p3 = p_internal + vp * (head_size / pixel_count / 2.0f);
    glm::vec2 p4 = p_internal - vp * (head_size / pixel_count / 2.0f);

    RenderLine(p1, p2 - glm::normalize(v) * 0.1f, color, width);
    RenderFilledTriangle(p_internal, p2, p3, color);
    RenderFilledTriangle(p_internal, p2, p4, color);

    glm::vec2 label_position = (p1 + p2) / 2.0f;
    glm::vec2 dir = glm::cross(glm::vec3(glm::normalize(v), 0), glm::vec3(0, 0, 1));
    //label_position += atan2(v.x, v.y) > 0 ? dir * 0.1f : dir * 0.1f;
    label_position += (v.x >= 0 && v.y >= 0) ? dir * 0.1f
                    : (v.x < 0 && v.y > 0) ? dir * 0.2f
                    : (v.x < 0 && v.y < 0) ? -dir * 0.1f
                    : -dir * 0.2f;

    RenderText(label_position, label, color);
}

void PrimitivesRenderingScene::RenderLine(const glm::vec2 &p1, const glm::vec2 &p2,
    const glm::vec3 &color, unsigned int width)
{
    glm::vec2 viewport_space_p1 = viewport_transformation * glm::vec3(p1, 1);
    glm::vec2 viewport_space_p2 = viewport_transformation * glm::vec3(p2, 1);

    RenderLineViewportSpace(viewport_space_p1, viewport_space_p2, color, width);
}

void PrimitivesRenderingScene::RenderLineViewportSpace(const glm::vec2 &p1, const glm::vec2 &p2,
    const glm::vec3 &color, unsigned int width)
{
    glm::vec2 v = glm::normalize(p2 - p1);
    glm::vec2 vp = glm::cross(glm::vec3(v, 0), glm::vec3(0, 0, 1));

    glm::vec2 corner1 = p1 + vp * (float)width / 2.0f;
    glm::vec2 corner2 = p1 - vp * (float)width / 2.0f;
    glm::vec2 corner3 = p2 + vp * (float)width / 2.0f;
    glm::vec2 corner4 = p2 - vp * (float)width / 2.0f;

    RenderFilledTriangleViewportSpace(corner1, corner2, corner3, color);
    RenderFilledTriangleViewportSpace(corner2, corner3, corner4, color);
}

void PrimitivesRenderingScene::RenderCircle(const glm::vec2 &center, float radius,
    const glm::vec3 &color)
{
    unsigned int start_index = exercises.back().back().vertices.size();

    glm::vec2 viewport_space_center = viewport_transformation * glm::vec3(center, 1);
    
    depth -= 0.001f;

    exercises.back().back().vertices.push_back(VertexFormat(glm::vec3(viewport_space_center, depth), color));

    unsigned int count = 10;
    for (int index = 0; index < count; index++) {
        glm::vec2 p = viewport_space_center + radius * glm::vec2(cos(2 * 3.14 / count * index), sin(2 * 3.14 / count * index));

        exercises.back().back().vertices.push_back(VertexFormat(glm::vec3(p, depth), color));
    }

    for (int index = 2; index <= count; index++) {
        exercises.back().back().indices.push_back(start_index);
        exercises.back().back().indices.push_back(start_index + index - 1);
        exercises.back().back().indices.push_back(start_index + index);
    }

    exercises.back().back().indices.push_back(start_index);
    exercises.back().back().indices.push_back(start_index + count);
    exercises.back().back().indices.push_back(start_index + 1);
}

void PrimitivesRenderingScene::RenderFilledTriangle(const glm::vec2 &p1, const glm::vec2 &p2,
    const glm::vec2 &p3, const glm::vec3 &color)
{
    glm::vec2 viewport_space_p1 = viewport_transformation * glm::vec3(p1, 1);
    glm::vec2 viewport_space_p2 = viewport_transformation * glm::vec3(p2, 1);
    glm::vec2 viewport_space_p3 = viewport_transformation * glm::vec3(p3, 1);

    RenderFilledTriangleViewportSpace(viewport_space_p1, viewport_space_p2, viewport_space_p3, color);
}

void PrimitivesRenderingScene::RenderFilledTriangleViewportSpace(const glm::vec2 &p1, const glm::vec2 &p2,
    const glm::vec2 &p3, const glm::vec3 &color)
{
    unsigned int start_index = exercises.back().back().vertices.size();

    depth -= 0.001f;

    exercises.back().back().vertices.push_back(VertexFormat(glm::vec3(p1, depth), color));
    exercises.back().back().vertices.push_back(VertexFormat(glm::vec3(p2, depth), color));
    exercises.back().back().vertices.push_back(VertexFormat(glm::vec3(p3, depth), color));

    exercises.back().back().indices.push_back(start_index);
    exercises.back().back().indices.push_back(start_index + 1);
    exercises.back().back().indices.push_back(start_index + 2);
}

inline std::string PrimitivesRenderingScene::GetLabel(const std::string &label, const glm::vec2 &vec, bool is_point)
{
    std::ostringstream out;

    out << label << (is_point == true ? "(" : "[");

    if (vec.x == (int)vec.x) {
        out.precision(0);
    } else {
        out.precision(2);
    }

    out << std::fixed << vec.x << (is_point == true ? "," : " ");
    
    if (vec.y == (int)vec.y) {
        out.precision(0);
    } else {
        out.precision(2);
    }

    out << std::fixed << vec.y << (is_point == true ? ")" : "]");

    return std::move(out).str();
}

