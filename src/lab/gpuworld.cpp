#include "lab/gpuworld.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <optional>
#include <variant>
#include <array>
#include <glm/gtc/quaternion.hpp>
#include <random>        
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp> 
#include <execution>
#include <imgui/imgui.h>
#include <imgui/imgui_impl_glfw.h>
#include <imgui/imgui_impl_opengl3.h>
#include <algorithm>

#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>
#include <cmath>

using namespace std;
using namespace m1;


GPUWorld::GPUWorld()
{
}

GPUWorld::~GPUWorld()
{
}
static const int INTERPOLATION_SEGMENTS = 2; // Number of physics/logic segments per visual segment

// --- GLOBAL DATA CONTAINERS ---
std::vector<std::array<glm::vec2, 2>> normalizedSegments;
static std::vector<float> laneInterpolations;
static std::vector<glm::vec2> trackPath;
static std::vector<float> pathCurvature;

// --- VELOCITY PROFILE DATA ---
static std::vector<float> speedProfileCurvature; // Limit from lateral Gs
static std::vector<float> speedProfileBraking;   // Limit after braking pass
static std::vector<float> speedProfileFinal;     // Final limit after acceleration pass

// --- SIMULATION PARAMETERS ---
static int currentPathIndex = 0;
static glm::vec2 carPosition = glm::vec2(0.5f);
static float globalLaneOffset = 0.5f;
static float totalTimeNow = 0.0f;

// Note: carSpeed is now derived from the profile, but we keep a scalar for the slider override if needed, 
// though the prompt implies we calculate it now. We will use the calculated one.
static float maxLateralAcc = 20.0f;
static float brakeAcc = 10.0f;      // Deceleration magnitude
static float forwardAcc = 5.f;    // Acceleration magnitude
static float drag = 0.0f;    // Acceleration magnitude

static float SPEED_LIMIT = 100.0f; // Max speed for straight lines
static float SPEED_LIMIT_FOR_VISUALIZATION = 100.0f; // Max speed for straight lines
static float MIN_SPEED_FOR_VISUALIZATION = -1.0f; // Max speed for straight lines
// Change this line near the top:
static float trackLimit = 0.1f; // Now a variable, not const
// Range [0.01f, 0.45f] is enforced in ImGui
// --------------------------
// --- OPTIMIZATION & UI STATE ---
static bool useGlobalLaneOffset = true;
static bool isOptimizing = false;
static float mutationMagnitude = 0.01f;
static int mutationBatchSize = 1;
static float mutationPropagateChance = 0.9f;
static float mutationPropagateMagnitude = 0.9f;

static int optimizationIterations = 500; // <--- NEW: Number of variations to test per frame
static std::mt19937 rng(std::random_device{}());

// 1. Helper: Build Track Path
static void BuildTrackPath(const std::vector<std::array<glm::vec2, 2>>& segments,
    const std::vector<float>& interpolations,
    std::vector<glm::vec2>& outPath)
{
    outPath.clear();
    if (segments.size() != interpolations.size()) return;

    for (size_t i = 0; i < segments.size(); ++i) {
        glm::vec2 p = glm::mix(segments[i][0], segments[i][1], interpolations[i]);
        outPath.push_back(p);
    }
}

static std::vector<float> ComputeCurvature(const std::vector<glm::vec2>& path) {
    std::vector<float> curvatures;
    if (path.size() < 3) return curvatures;
    curvatures.resize(path.size());

    for (size_t i = 0; i < path.size(); ++i) {
        size_t prevIdx = (i - 1 + path.size()) % path.size();
        size_t nextIdx = (i + 1) % path.size();

        glm::vec2 pPrev = path[prevIdx];
        glm::vec2 pCurr = path[i];
        glm::vec2 pNext = path[nextIdx];

        glm::vec2 vIn = pCurr - pPrev;
        glm::vec2 vOut = pNext - pCurr;

        float lenIn = glm::length(vIn);
        float lenOut = glm::length(vOut);

        if (lenIn < 1e-6f || lenOut < 1e-6f) {
            curvatures[i] = 0.0f;
            continue;
        }

        vIn /= lenIn;
        vOut /= lenOut;

        float dot = glm::clamp(glm::dot(vIn, vOut), -1.0f, 1.0f);
        float angle = std::acos(dot);
        float dist = std::min(lenIn, lenOut);

        if (dist < 1e-6f) curvatures[i] = 0.0f;
        else curvatures[i] = angle / dist;
    }
    return curvatures;
}

// ---------------------------------------------------------
// 2. Physics: Get Static Max Speeds (The "Ceiling")
// ---------------------------------------------------------
static std::vector<float> GetMaxSpeedsFromCurvature(const std::vector<float>& curvatures, float maxLatAcc) {
    std::vector<float> speeds;
    speeds.resize(curvatures.size());

    for (size_t i = 0; i < curvatures.size(); ++i) {
        float k = std::abs(curvatures[i]);
        if (k < 1e-6f) {
            speeds[i] = SPEED_LIMIT;
        }
        else {
            // v = sqrt(a_lat_max / k)
            float v = std::sqrt(maxLatAcc / k);
            speeds[i] = std::min(v, SPEED_LIMIT);
        }
    }
    return speeds;
}

// ---------------------------------------------------------
// 3. Physics: Friction Circle Braking (Backwards Pass)
// ---------------------------------------------------------
static std::vector<float> GetMaxSpeedsFromBraking(const std::vector<float>& curvatureSpeeds,
    const std::vector<glm::vec2>& path,
    const std::vector<float>& curvatures,
    float maxBrakeAccel,
    float maxLatAccel)
{
    std::vector<float> speeds = curvatureSpeeds;
    size_t count = speeds.size();
    if (count < 2) return speeds;

    // We use a small epsilon to avoid division by zero in ellipse calc
    float maxLatSq = maxLatAccel * maxLatAccel;

    for (int pass = 0; pass < 2; ++pass) {
        for (int i = (int)count - 1; i >= 0; --i) {
            int nextIdx = (i + 1) % count;
            float dist = glm::distance(path[i], path[nextIdx]);

            // We are integrating BACKWARDS from 'next' to 'current'.
            // The constraint is determined by the state at 'next'.
            // If we are fast at 'next', we are using tires for turning there,
            // so we have less grip available to have slowed down to reach it.

            float vNext = speeds[nextIdx];
            float kNext = std::abs(curvatures[i]);

            // 1. Calculate Lateral Accel used at the next point
            float aLat = vNext * vNext * kNext;

            // 2. Calculate Available Longitudinal Decel using Ellipse
            // Formula: a_long = a_long_max * sqrt(1 - (a_lat / a_lat_max)^2)
            float availableDecel = 0.0f;

            if (aLat < maxLatAccel) {
                float ratio = aLat / maxLatAccel;
                // Ratio squared is (aLat^2 / maxLatSq)
                float gripRemaining = std::sqrt(std::max(0.0f, 1.0f - (ratio * ratio)));
                availableDecel = maxBrakeAccel * gripRemaining;
            }
            else {
                // If we are already at max lateral limit, we can't brake at all.
                availableDecel = 0.0f;
            }
            availableDecel += vNext * vNext * drag;

            // 3. Kinematic update: v_curr = sqrt(v_next^2 + 2 * a * d)
            float maxV = std::sqrt(vNext * vNext + 2.0f * availableDecel * dist);

            // Clamp to the static limit (the ceiling)
            speeds[i] = std::min(speeds[i], maxV);
        }
    }
    return speeds;
}

// ---------------------------------------------------------
// 4. Physics: Friction Circle Acceleration (Forward Pass)
// ---------------------------------------------------------
static std::vector<float> GetActualSpeedFromAcceleration(const std::vector<float>& brakingSpeeds,
    const std::vector<glm::vec2>& path,
    const std::vector<float>& curvatures,
    float maxFwdAccel,
    float maxLatAccel)
{
    std::vector<float> speeds = brakingSpeeds;
    size_t count = speeds.size();
    if (count < 2) return speeds;

    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < count; ++i) {
            size_t nextIdx = (i + 1) % count;
            float dist = glm::distance(path[i], path[nextIdx]);

            // We are integrating FORWARDS from 'current' to 'next'.
            // The constraint is determined by the state at 'current'.

            float vCurr = speeds[i];
            float kCurr = std::abs(curvatures[i]);

            // 1. Calculate Lateral Accel used at current point
            float aLat = vCurr * vCurr * kCurr;

            // 2. Calculate Available Longitudinal Accel using Ellipse
            float availableAccel = 0.0f;

            if (aLat < maxLatAccel) {
                float ratio = aLat / maxLatAccel;
                float gripRemaining = std::sqrt(std::max(0.0f, 1.0f - (ratio * ratio)));
                availableAccel = std::max(0.0f,maxFwdAccel * gripRemaining - vCurr * vCurr * drag);
            }
            else {
                availableAccel = 0.0f;
            }

            // 3. Kinematic update: v_next = sqrt(v_curr^2 + 2 * a * d)
            float maxAchievable = std::sqrt(vCurr * vCurr + 2.0f * availableAccel * dist);

            // Clamp to the braking/curvature limit calculated previously
            speeds[nextIdx] = std::min(speeds[nextIdx], maxAchievable);
        }
    }
    return speeds;
}

// ---------------------------------------------------------
// 5. Compute Time (Unchanged)
// ---------------------------------------------------------
static void ComputeTotalTime(const std::vector<glm::vec2>& path, const std::vector<float>& speeds) {
    if (path.size() != speeds.size() || path.empty()) return;

    float totalTime = 0.0f;
    for (size_t i = 0; i < path.size(); ++i) {
        size_t nextIdx = (i + 1) % path.size();
        float dist = glm::distance(path[i], path[nextIdx]);
        float avgSpeed = (speeds[i] + speeds[nextIdx]) * 0.5f;

        if (avgSpeed > 1e-6f)
            totalTime += dist / avgSpeed;
        else 
            totalTime += 1e6f;
    }
    totalTimeNow = totalTime;
}

// ---------------------------------------------------------
// Wrapper
// ---------------------------------------------------------
static void RecalculatePhysics() {
    if (trackPath.empty()) return;

    // 1. Curvature
    pathCurvature = ComputeCurvature(trackPath);

    // 2. Max speeds based on lateral Gs (Static Ceiling)
    speedProfileCurvature = GetMaxSpeedsFromCurvature(pathCurvature, maxLateralAcc);

    // 3. Max speeds based on Braking (Backwards with Ellipse)
    // Note: We pass pathCurvature to calculate lateral Gs dynamically
    speedProfileBraking = GetMaxSpeedsFromBraking(speedProfileCurvature, trackPath, pathCurvature, brakeAcc, maxLateralAcc);

    // 4. Actual speeds based on Acceleration (Forwards with Ellipse)
    speedProfileFinal = GetActualSpeedFromAcceleration(speedProfileBraking, trackPath, pathCurvature, forwardAcc, maxLateralAcc);


    // 5. Log Time
    ComputeTotalTime(trackPath, speedProfileFinal);
}

void GPUWorld::RegenerateTrackBorder(float limit)
{
    // Safety check
    if (normalizedSegments.empty()) return;

    // Remove old safety meshes if they exist to prevent memory leaks
    if (meshes.find("safety_border_left") != meshes.end()) {
        delete meshes["safety_border_left"];
        meshes.erase("safety_border_left");
    }
    if (meshes.find("safety_border_right") != meshes.end()) {
        delete meshes["safety_border_right"];
        meshes.erase("safety_border_right");
    }

    glm::vec3 safeColor(0.2f, 0.2f, 0.2f); // Yellow color for safety limit

    std::vector<VertexFormat> leftVertices, rightVertices;
    std::vector<unsigned int> leftIndices, rightIndices;

    for (size_t i = 0; i < normalizedSegments.size(); ++i) {
        glm::vec2 p1 = normalizedSegments[i][0]; // Outer Left
        glm::vec2 p2 = normalizedSegments[i][1]; // Outer Right

        // Interpolate to find the inner safety limit
        // Left inner = mix(p1, p2, limit)
        glm::vec2 innerLeft = glm::mix(p1, p2, limit);

        // Right inner = mix(p1, p2, 1.0 - limit)
        glm::vec2 innerRight = glm::mix(p1, p2, 1.0f - limit);

        // --- Build Left Safety Mesh ---
        leftVertices.emplace_back(glm::vec3(innerLeft.x, innerLeft.y, 0.0f), safeColor);
        leftIndices.push_back(static_cast<unsigned int>(i));

        // --- Build Right Safety Mesh ---
        rightVertices.emplace_back(glm::vec3(innerRight.x, innerRight.y, 0.0f), safeColor);
        rightIndices.push_back(static_cast<unsigned int>(i));
    }

    // Create and Store Left Mesh
    Mesh* safeLeft = new Mesh("safety_border_left");
    safeLeft->SetDrawMode(GL_LINE_LOOP);
    safeLeft->InitFromData(leftVertices, leftIndices);
    meshes[safeLeft->GetMeshID()] = safeLeft;

    // Create and Store Right Mesh
    Mesh* safeRight = new Mesh("safety_border_right");
    safeRight->SetDrawMode(GL_LINE_LOOP);
    safeRight->InitFromData(rightVertices, rightIndices);
    meshes[safeRight->GetMeshID()] = safeRight;
}
void GPUWorld::Init()
{
#pragma region Load meshes
    {
        // Create a simple 1x1 square in the XY plane
        std::vector<VertexFormat> vertices =
        {
            VertexFormat(glm::vec3(0.f, 0.f, 0.f), glm::vec3(1.f), glm::vec3(1.f), glm::vec2(0.f, 0.f)), // bottom-left
            VertexFormat(glm::vec3(1.f, 0.f, 0.f), glm::vec3(1.f), glm::vec3(1.f), glm::vec2(1.f, 0.f)), // bottom-right
            VertexFormat(glm::vec3(1.f, 1.f, 0.f), glm::vec3(1.f), glm::vec3(1.f), glm::vec2(1.f, 1.f)), // top-right
            VertexFormat(glm::vec3(0.f, 1.f, 0.f), glm::vec3(1.f), glm::vec3(1.f), glm::vec2(0.f, 1.f))  // top-left
        };

        std::vector<unsigned int> indices = {
            0, 1, 2,
            0, 2, 3
        };

        Mesh* mesh = new Mesh("cell");
        mesh->InitFromData(vertices, indices);
        meshes[mesh->GetMeshID()] = mesh;
    }
    {
        // Create a simple 1x1 square in the XY plane
        std::vector<VertexFormat> vertices =
        {
            VertexFormat(glm::vec3(0.f, 0.f, 0.f), glm::vec3(1.f), glm::vec3(1.f), glm::vec2(0.f, 0.f))
        };

        std::vector<unsigned int> indices = {
            0 };

        Mesh* mesh = new Mesh("car");
        mesh->SetDrawMode(GL_POINTS);

        mesh->InitFromData(vertices, indices);
        meshes[mesh->GetMeshID()] = mesh;
    }
#pragma endregion

#pragma region CSV Track Loading
    {
        std::vector<VertexFormat> trackVertices;
        std::vector<unsigned int> trackIndices;
        std::vector<glm::vec4> rawSegments;

        // Temporary storage for the original CSV segments (used for blue lines only)
        std::vector<std::array<glm::vec2, 2>> visualSegments;

        std::ifstream file("track_data.csv");

        float minX = std::numeric_limits<float>::max();
        float minY = std::numeric_limits<float>::max();
        float maxX = std::numeric_limits<float>::lowest();
        float maxY = std::numeric_limits<float>::lowest();

        if (file.is_open()) {
            std::string line;
            std::getline(file, line); // Skip Header

            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string cell;
                std::vector<float> values;

                while (std::getline(ss, cell, ',')) {
                    try { values.push_back(std::stof(cell)); }
                    catch (...) { values.push_back(0.0f); }
                }

                if (values.size() >= 7) {
                    float p1x = values[3];
                    float p1y = values[4];
                    float p2x = values[5];
                    float p2y = values[6];

                    rawSegments.push_back(glm::vec4(p1x, p1y, p2x, p2y));

                    minX = std::min({ minX, p1x, p2x });
                    minY = std::min({ minY, p1y, p2y });
                    maxX = std::max({ maxX, p1x, p2x });
                    maxY = std::max({ maxY, p1y, p2y });
                }
            }
            file.close();

            float width = maxX - minX;
            float height = maxY - minY;
            if (width == 0) width = 1.0f;
            if (height == 0) height = 1.0f;

            // 1. Fill the VISUAL segments (1:1 with CSV)
            for (const auto& seg : rawSegments) {
                float nP1x = (seg.x - minX) / width;
                float nP1y = (seg.y - minY) / height;
                float nP2x = (seg.z - minX) / width;
                float nP2y = (seg.w - minY) / height;

                visualSegments.push_back({ glm::vec2(nP1x, nP1y) * 1000.f, glm::vec2(nP2x, nP2y) * 1000.f });
            }

            // 2. Build the VISUAL Mesh (Blue Lines) from visualSegments
            unsigned int indexCounter = 0;
            glm::vec3 blueColor(0.1f, 0.1f, 0.5f);

            for (const auto& line : visualSegments) {
                trackVertices.emplace_back(glm::vec3(line[0].x, line[0].y, 0.0f), blueColor);
                trackVertices.emplace_back(glm::vec3(line[1].x, line[1].y, 0.0f), blueColor);
                trackIndices.push_back(indexCounter++);
                trackIndices.push_back(indexCounter++);
            }

            Mesh* trackMesh = new Mesh("track_lines");
            trackMesh->SetDrawMode(GL_LINES);
            trackMesh->InitFromData(trackVertices, trackIndices);
            meshes["track_lines"] = trackMesh;

            // 3. Generate SIMULATION Data (Interpolated)
            // This fills 'normalizedSegments' which is used by physics, borders, and the optimizer
            // 3. Generate SIMULATION Data (Interpolated)
            normalizedSegments.clear();
            int segmentsPerLine = std::max(1, INTERPOLATION_SEGMENTS);
            size_t count = visualSegments.size();

            for (size_t i = 0; i < count; ++i) {
                // Get current perpendicular (Left side, Right side)
                glm::vec2 left_curr = visualSegments[i][0];
                glm::vec2 right_curr = visualSegments[i][1];

                // Get next perpendicular (wrapping around to 0 for the last one)
                size_t nextIndex = (i + 1) % count;
                glm::vec2 left_next = visualSegments[nextIndex][0];
                glm::vec2 right_next = visualSegments[nextIndex][1];

                // Generate intermediate perpendiculars between 'i' and 'next'
                for (int k = 0; k < segmentsPerLine; ++k) {
                    float t = (float)k / segmentsPerLine;

                    // Interpolate the Left wall point longitudinally
                    glm::vec2 subLeft = glm::mix(left_curr, left_next, t);

                    // Interpolate the Right wall point longitudinally
                    glm::vec2 subRight = glm::mix(right_curr, right_next, t);

                    normalizedSegments.push_back({ subLeft, subRight });
                }
            }

            std::cout << "Loaded " << visualSegments.size() << " visual segments." << std::endl;
            std::cout << "Generated " << normalizedSegments.size() << " simulation segments." << std::endl;
        }
        else {
            std::cerr << "Could not open track.csv" << std::endl;
        }
    }
#pragma endregion
#pragma region Track Borders Creation
    // Ensure we have segments to connect
    if (!normalizedSegments.empty()) {
        glm::vec3 borderColor(0.5f);

        // --- Left Border Mesh (connecting all P1 endpoints) ---
        {
            std::vector<VertexFormat> leftVertices;
            std::vector<unsigned int> leftIndices;

            for (size_t i = 0; i < normalizedSegments.size(); ++i) {
                // normalizedSegments[i][0] is P1 of current segment
                leftVertices.emplace_back(glm::vec3(normalizedSegments[i][0].x, normalizedSegments[i][0].y, 0.0f), borderColor);
                leftIndices.push_back(static_cast<unsigned int>(i));
            }

            Mesh* leftBorderMesh = new Mesh("track_border_left");
            // Use GL_LINE_STRIP to connect points 0-1-2-3... continuously
            leftBorderMesh->SetDrawMode(GL_LINE_LOOP);
            leftBorderMesh->InitFromData(leftVertices, leftIndices);
            meshes[leftBorderMesh->GetMeshID()] = leftBorderMesh;
        }

        // --- Right Border Mesh (connecting all P2 endpoints) ---
        {
            std::vector<VertexFormat> rightVertices;
            std::vector<unsigned int> rightIndices;

            for (size_t i = 0; i < normalizedSegments.size(); ++i) {
                // normalizedSegments[i][1] is P2 of current segment
                rightVertices.emplace_back(glm::vec3(normalizedSegments[i][1].x, normalizedSegments[i][1].y, 0.0f), borderColor);
                rightIndices.push_back(static_cast<unsigned int>(i));
            }

            Mesh* rightBorderMesh = new Mesh("track_border_right");
            rightBorderMesh->SetDrawMode(GL_LINE_LOOP);
            rightBorderMesh->InitFromData(rightVertices, rightIndices);
            meshes[rightBorderMesh->GetMeshID()] = rightBorderMesh;
        }
    }
#pragma endregion

#pragma region ImGui mennu

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    // Connect ImGui to GLFW and OpenGL3
    ImGui_ImplGlfw_InitForOpenGL(window->getWindowHandle(), true);
    ImGui_ImplOpenGL3_Init("#version 150");
    io.FontGlobalScale = 1.5f; // increase global scale (1.0f = normal size)
#pragma endregion


    {
        // ---------------------------------------------------------
// LOCK CAMERA AND SWITCH TO 2D ORTHOGRAPHIC VIEW
// ---------------------------------------------------------

// Disable all built-in camera movement (WASD + mouse look)
        GetCameraInput()->SetActive(false);
        auto camera = GetSceneCamera();
        // Optional: remove FPS mouse-lock behavior
       // window->DisablePointer();

        // Force camera position and orientation (look straight at -Z)

        // Orthographic projection covering the entire window
        float w = (float)window->GetResolution().x;
        float h = (float)window->GetResolution().y;
        camera->SetPositionAndRotation(
            glm::vec3(0.f, 0.f, 1.f), // position
            glm::quat(glm::vec3(0.f, 0.f, 0.f)) // rotation
        );
        camera->SetOrthographic(-0.1f * 1000.f, 1.4f * 1000.f, -0.25f * 1000.f, 1.25f * 1000.f, 0.01f, 400.f);


        // ---------------------------------------------------------
        // END 2D CAMERA BLOCK
        // ---------------------------------------------------------

    }

    // Initialize Simulation Data
    if (!normalizedSegments.empty()) {
        laneInterpolations.resize(normalizedSegments.size(), 0.5f); // Default middle
        // >>> Use new helper <<<
        BuildTrackPath(normalizedSegments, laneInterpolations, trackPath);

        // >>> Compute initial physics <<<
        RecalculatePhysics();

        // Set initial car position
        if (!trackPath.empty()) {
            carPosition = trackPath[0];
        }

        RegenerateTrackBorder(trackLimit);
    }
}

std::optional<glm::ivec2> GPUWorld::MouseToGrid(int mouseX, int mouseY)
{
    int w = window->GetResolution().x;
    int h = window->GetResolution().y;

    float cellW = (float)w / GRID_W;
    float cellH = (float)h / GRID_H;

    int gx = mouseX / cellW;
    int gy = (h - mouseY) / cellH; // invert Y

    if (gx < 0 || gx >= GRID_W || gy < 0 || gy >= GRID_H)
        return std::nullopt;

    return glm::ivec2(gx, gy);
}


void GPUWorld::FrameStart()
{

}

void GPUWorld::DrawTemporaryLine(glm::vec3 p1, glm::vec3 p2, glm::vec3 color1, glm::vec3 color2, gfxc::SimpleScene* ss)
{
    // 1. Create a new mesh instance (temporary)
    Mesh* lineMesh = new Mesh("temp_line");
    lineMesh->SetDrawMode(GL_LINES);
    lineMesh->UseMaterials(false); // no textures or materials

    // 2. Create vertex and index data
    std::vector<VertexFormat> vertices = {
        VertexFormat(p1, color1),
        VertexFormat(p2, color2)
    };
    std::vector<unsigned int> indices = { 0, 1 };

    // 3. Upload to GPU
    lineMesh->InitFromData(vertices, indices);

    // 4. Render using the same shader/camera pipeline as your scene
    // Bind your shader before rendering!
    ss->RenderMesh(lineMesh, ss->shaders["VertexColor"], glm::mat4(1));
    lineMesh->Render();
    lineMesh->ClearData();
    // 5. Delete immediately after drawing
    delete lineMesh;
}

void GPUWorld::SetFrame() {
    glm::ivec2 resolution = window->props.resolution;

    // Sets the clear color for the color buffer
    glClearColor(0, 0, 0, 1);

    // Clears the color buffer (using the previously set color) and depth buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Sets the screen area where to draw
    glViewport(0, 0, resolution.x, resolution.y);
}


void GPUWorld::Update(float deltaTimeSeconds)
{
    static float slowMo = 1.0f;
    static bool showAcceleration = true;
    deltaTimeSeconds *= slowMo;
    // --- IMGUI START ---
    {
        ImGuiIO& io = ImGui::GetIO();
        io.DisplaySize = ImVec2((float)window->GetResolution().x, (float)window->GetResolution().y);
        io.DeltaTime = deltaTimeSeconds > 0 ? deltaTimeSeconds : 1.0f / 60.0f;
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    SetFrame();
    glLineWidth(1.0f);

    // --- RENDER TRACK ---
    glm::mat4 modelMatrix = glm::mat4(1);
    if (meshes.find("track_lines") != meshes.end())
        RenderMesh(meshes["track_lines"], shaders["VertexColor"], modelMatrix);

    glLineWidth(2.0f);

    // Original borders (Grey)
    if (meshes.find("track_border_left") != meshes.end())
        RenderMesh(meshes["track_border_left"], shaders["VertexColor"], modelMatrix);
    if (meshes.find("track_border_right") != meshes.end())
        RenderMesh(meshes["track_border_right"], shaders["VertexColor"], modelMatrix);

    // >>> NEW: Safety Borders (YELLOW) <<<
    if (meshes.find("safety_border_left") != meshes.end())
        RenderMesh(meshes["safety_border_left"], shaders["VertexColor"], modelMatrix);
    if (meshes.find("safety_border_right") != meshes.end())
        RenderMesh(meshes["safety_border_right"], shaders["VertexColor"], modelMatrix);

	glLineWidth(4.0f);
    // --- RENDER CHOSEN PATH ---
    if (!trackPath.empty()) {
        // ... (existing check for physics data) ...

        if (!showAcceleration) {
            for (size_t i = 0; i < trackPath.size(); ++i) {
                glm::vec2 p1 = trackPath[i];
                glm::vec2 p2 = trackPath[(i + 1) % trackPath.size()];

                // >>> REPLACE THE OLD vNorm CALCULATION WITH THIS BLOCK <<<
                float range = SPEED_LIMIT_FOR_VISUALIZATION - MIN_SPEED_FOR_VISUALIZATION;
                if (range < 0.001f) range = 0.001f; // Avoid division by zero

                // Remap speed from [min, max] to [0, 1]
                float v1 = speedProfileFinal[i];
                float v2 = speedProfileFinal[(i + 1) % trackPath.size()];

                float vNorm1 = glm::clamp((v1 - MIN_SPEED_FOR_VISUALIZATION) / range, 0.0f, 1.0f);
                float vNorm2 = glm::clamp((v2 - MIN_SPEED_FOR_VISUALIZATION) / range, 0.0f, 1.0f);
                // >>> END REPLACEMENT <<<

                glm::vec3 colorSlow(1.0f, 0.0f, 0.0f);   // Red
                glm::vec3 colorFast(0.0f, 1.0f, 0.0f); // Green

                glm::vec3 finalColor = glm::mix(colorSlow, colorFast, vNorm1);
                glm::vec3 finalColor2 = glm::mix(colorSlow, colorFast, vNorm2);

                DrawTemporaryLine(
                    glm::vec3(p1.x, p1.y, 0.0f),
                    glm::vec3(p2.x, p2.y, 0.0f),
                    finalColor,
                    finalColor2, this
                );
            }
        }
        else
        {
            for (size_t i = 0; i < trackPath.size(); ++i) {
                glm::vec2 p1 = trackPath[i];
                glm::vec2 p2 = trackPath[(i + 1) % trackPath.size()];

                // >>> REPLACE THE OLD vNorm CALCULATION WITH THIS BLOCK <<<
                float vNow = speedProfileFinal[i];
                float vNext = speedProfileFinal[(i + 1) % trackPath.size()];
                float vAfterNext = speedProfileFinal[(i + 2) % trackPath.size()];

                // Get distance for current segment
                glm::vec2 pNext = trackPath[(i + 1) % trackPath.size()];
                float distNow = glm::distance(p1, pNext);

                // Get distance for next segment
                glm::vec2 pAfterNext = trackPath[(i + 2) % trackPath.size()];
                float distNext = glm::distance(pNext, pAfterNext);

                float a1 = 0.0f;
                if (distNow > 0.001f) {
                    a1 = (vNext * vNext - vNow * vNow) / (2.0f * distNow);
                }

                float a2 = 0.0f;
                if (distNext > 0.001f) {
                    a2 = (vAfterNext * vAfterNext - vNext * vNext) / (2.0f * distNext);
                }

                glm::vec3 colorBrake(1.0f, 0.0f, 0.0f);
                glm::vec3 colorNeutral(0.5f, 0.5f, 0.5f);
                glm::vec3 colorAccel(0.0f, 1.0f, 0.0f);

                glm::vec3 finalColor;
                if (a1 < 0.0f) {
                    float t = glm::clamp(-a1 / brakeAcc, 0.0f, 1.0f);
                    finalColor = glm::mix(colorNeutral, colorBrake, t);
                }
                else {
                    float t = glm::clamp(a1 / forwardAcc, 0.0f, 1.0f);
                    finalColor = glm::mix(colorNeutral, colorAccel, t);
                }

                glm::vec3 finalColor2;
                if (a2 < 0.0f) {
                    float t = glm::clamp(-a2 / brakeAcc, 0.0f, 1.0f);
                    finalColor2 = glm::mix(colorNeutral, colorBrake, t);
                }
                else {
                    float t = glm::clamp(a2 / forwardAcc, 0.0f, 1.0f);
                    finalColor2 = glm::mix(colorNeutral, colorAccel, t);
                }
                // >>> END REPLACEMENT <<<

                DrawTemporaryLine(
                    glm::vec3(p1.x, p1.y, 0.0f),
                    glm::vec3(p2.x, p2.y, 0.0f),
                    finalColor,
                    finalColor2, this
                );
            }
        }
    }
    float t = 0.0f;

    // --- SIMULATION LOGIC ---
    if (!trackPath.empty() && !speedProfileFinal.empty()) {

        // 1. Calculate Current Speed based on interpolation between nodes
        int nextIdx = (currentPathIndex + 1) % trackPath.size();
        float distSegment = glm::distance(trackPath[currentPathIndex], trackPath[nextIdx]);
        float distToCurrent = glm::distance(carPosition, trackPath[currentPathIndex]);

        // Factor t [0, 1] along the segment
        t = distToCurrent / distSegment;

        // Linear interpolate speed
        float currentSpeed = glm::mix(speedProfileFinal[currentPathIndex], speedProfileFinal[nextIdx], t);

        // 2. Move Car
        float distanceToTravel = currentSpeed * deltaTimeSeconds;

        while (distanceToTravel > 0) {
            nextIdx = (currentPathIndex + 1) % trackPath.size();
            glm::vec2 targetPoint = trackPath[nextIdx];

            float distToNext = glm::distance(carPosition, targetPoint);

            if (distanceToTravel >= distToNext) {
                // Reach node
                carPosition = targetPoint;
                distanceToTravel -= distToNext;
                currentPathIndex = nextIdx;
            }
            else {
                // Interpolate segment
                glm::vec2 dir = glm::normalize(targetPoint - carPosition);
                carPosition += dir * distanceToTravel;
                distanceToTravel = 0;
            }
        }
    }
    {
        int nextIdx = (currentPathIndex + 1) % trackPath.size();
        float distSegment = glm::distance(trackPath[currentPathIndex], trackPath[nextIdx]);
        float distToCurrent = glm::distance(carPosition, trackPath[currentPathIndex]);

        // Factor t [0, 1] along the segment
        t = distToCurrent / distSegment;
    }

    // --- RENDER CAR ---
    // Using the 'cell' mesh (1x1 square) scaled down to look like a point
    {
        glPointSize(10); // Made it bigger to see
        glm::mat4 carModel = glm::mat4(1);
        carModel = glm::translate(carModel, glm::vec3(carPosition.x, carPosition.y, 0.0f));
        // Use a different color shader if possible, or VertexColor uses mesh colors
        RenderMesh(meshes["car"], shaders["VertexColor"], carModel);
    }
    if (!trackPath.empty() && !speedProfileFinal.empty())
    {
        // 1. Settings
        glm::vec3 hudOrigin(1.20f, 0.9f, 0.0f);
		hudOrigin *= 1000.0f; // Scale to match world coords
        float hudScale = 1000 * 0.2f * 1.0f / std::max({ forwardAcc,brakeAcc,maxLateralAcc });
        glm::vec3 limitColor(0.5f);
        glm::vec3 vecColor(1.0f, 0.0f, 0.0f);

        auto getAccel = [&](int currentIndex) {
            // [Existing acceleration calculation logic...]
            int idxNext = (currentIndex + 1) % trackPath.size();
            int idxPrev = (currentIndex - 1 + trackPath.size()) % trackPath.size();
            float dist = glm::distance(trackPath[currentIndex], trackPath[idxNext]);
            float vNow = speedProfileFinal[currentIndex];
            float vNext = speedProfileFinal[idxNext];
            float vLat = std::min(vNow,vNext);
            float accLong = 0.0f;
            if (dist > 1e-6f) accLong = (vNext * vNext - vNow * vNow) / (2.0f * dist);

            glm::vec2 dirIn = glm::normalize(trackPath[currentIndex] - trackPath[idxPrev]);
            glm::vec2 dirOut = glm::normalize(trackPath[idxNext] - trackPath[currentIndex]);
            float crossZ = dirIn.x * dirOut.y - dirIn.y * dirOut.x;
            float turnSign = (crossZ >= 0.0f) ? 1.0f : -1.0f;
            float accLat = vLat * vLat * pathCurvature[currentIndex] * turnSign;
            return glm::vec3(accLat * hudScale, accLong * hudScale, 0.1f);
            };

        // --- 3. Draw Limits Ellipse (G-Circle) ---
        // Instead of a box, we draw a loop of lines to approximate an ellipse.
        // Note: We handle asymmetry (Forward vs Braking limits).

        int segments = 36; // Higher number = smoother circle
        float twoPi = 6.283185307f;

        for (int i = 0; i < segments; ++i)
        {
            // Calculate angles for current segment
            float angle1 = (float)i / (float)segments * twoPi;
            float angle2 = (float)(i + 1) / (float)segments * twoPi;

            auto getEllipsePoint = [&](float theta) -> glm::vec3 {
                float sinA = sin(theta);
                float cosA = cos(theta);

                // X is simple: scaled by max lateral
                float xVal = cosA * maxLateralAcc;

                // Y depends on if we are accelerating (Top) or braking (Bottom)
                float yLimit = (sinA >= 0.0f) ? forwardAcc : brakeAcc;
                float yVal = sinA * yLimit;

                return hudOrigin + glm::vec3(xVal * hudScale, yVal * hudScale, 0.0f);
                };

            glm::vec3 p1 = getEllipsePoint(angle1);
            glm::vec3 p2 = getEllipsePoint(angle2);

            DrawTemporaryLine(p1, p2, limitColor, limitColor, this);
        }

        // Center Crosshair (Axes)
        // We limit the crosshair length to the max dimensions of the ellipse
        glm::vec3 grey(0.3f);
        float xMax = maxLateralAcc * hudScale;
        float yUp = forwardAcc * hudScale;
        float yDown = -brakeAcc * hudScale;

        DrawTemporaryLine(hudOrigin + glm::vec3(-xMax, 0, 0), hudOrigin + glm::vec3(xMax, 0, 0), grey, grey, this);
        DrawTemporaryLine(hudOrigin + glm::vec3(0, yDown, 0), hudOrigin + glm::vec3(0, yUp, 0), grey, grey, this);

        // 4. Draw Live Vector
        glm::vec3 currentAccel = getAccel(currentPathIndex);
        glm::vec3 nextAccel = getAccel((currentPathIndex + 1) % trackPath.size());
        glm::vec3 interpAccel = glm::mix(currentAccel, nextAccel, t);

        glm::vec3 vecTip = hudOrigin + interpAccel;

        DrawTemporaryLine(hudOrigin, vecTip, vecColor, vecColor, this);

        }
    // --- IMGUI CONTROLS ---
   // ControlImGuiWindow(); // Standard window

    ImGui::Begin("Simulation Controls");

    bool paramsChanged = false;

    // 1. General  Parameters

   {
		ImGui::Separator();
        ImGui::Text("Track params");
		static float speedLimit = 400;
        paramsChanged |= ImGui::SliderFloat("Speed Limit", &speedLimit, 0.1f, 800.0f);
		SPEED_LIMIT = speedLimit / 3.6f;
        // >>> NEW SLIDER FOR TRACK LIMIT <<<
        if (ImGui::SliderFloat("Safe Zone Limit", &trackLimit, 0.05f, 0.45f)) {
            // 1. Rebuild the visual mesh
            RegenerateTrackBorder(trackLimit);

            // 2. Clamp existing lane data so points don't stay "off track" if we narrow the track
            for (auto& val : laneInterpolations) {
                val = std::clamp(val, trackLimit, 1.0f - trackLimit);
            }
            paramsChanged = true;
        }
    }
    //car params
    {

        ImGui::Separator();
        ImGui::Text("Car Parameters");
        static float bAcc = 2;
        static float lAcc = 2;
        static float fAcc = 1;

        paramsChanged |= ImGui::SliderFloat("Brake Decel", &bAcc, 0.1f, 10.0f);
        paramsChanged |= ImGui::SliderFloat("Lat Acc Limit", &lAcc, 0.1f, 10.0f);
        paramsChanged |= ImGui::SliderFloat("Fwd Acc", &fAcc, 0.1f, 10.0f);
		brakeAcc = bAcc * 9.81f;
		maxLateralAcc = lAcc * 9.81f;
		forwardAcc = fAcc * 9.81f;
        paramsChanged |= ImGui::SliderFloat("Drag", &drag, 0.0f, 0.005f);

    }
    //ellipseparams
    {

        ImGui::Separator();
        ImGui::Text("Car action visualization");
        ImGui::Checkbox("Show Acceleration", &showAcceleration);
        
		ImGui::SliderFloat("SlowMo", &slowMo, 0.1f, 5.0f);
    }
    ImGui::Separator();

    // 2. Lane Interpolation Control
    ImGui::Text("Racing Line Optimizer");

    // Checkbox to toggle mode
    if (ImGui::Checkbox("Use Global Lane Offset", &useGlobalLaneOffset)) {
        if (useGlobalLaneOffset) {
            std::fill(laneInterpolations.begin(), laneInterpolations.end(), globalLaneOffset);
            isOptimizing = false;
            paramsChanged = true;
        }
    }

    if (useGlobalLaneOffset) {
        // --- MODE A: Global Slider ---
        if (ImGui::SliderFloat("Global Offset", &globalLaneOffset, trackLimit, 1-trackLimit)) {
            std::fill(laneInterpolations.begin(), laneInterpolations.end(), globalLaneOffset);
            paramsChanged = true;
        }
    }
    else {
        // --- MODE B: Optimization ---

        // Manual Randomize
        if (ImGui::Button("Randomize Line")) {
            std::uniform_real_distribution<float> dist(0.1f, 0.9f);
            for (auto& val : laneInterpolations) {
                val = dist(rng);
            }
            paramsChanged = true;
        }

        ImGui::SameLine();
        ImGui::Checkbox("Auto-Optimize", &isOptimizing);

        if (isOptimizing) {
            // 1. Mutation Power (How far points move)
            static float logMutationMagnitude = -2.2f; // Default ~0.05
            ImGui::SliderFloat("Mut Magnitude (log)", &logMutationMagnitude, -3.0f, -0.0f, "%.2f");
            mutationMagnitude = std::pow(10.0f, logMutationMagnitude);
            

            ImGui::Indent();
            ImGui::SliderFloat("Expand chance", &mutationPropagateChance, 0,1.0f);

            ImGui::SliderFloat("Maintain amplitude", &mutationPropagateMagnitude, 0, 1.0f);
            ImGui::Unindent();

            // Store this as a static or member variable to pass to the logic below
            // For this snippet, I'll calculate it right here, but you might want to store it in the class
            // if you need it elsewhere.
            // (Note: mutationBatchSize is no longer used)

            ImGui::SliderInt("Variations/Frame", &optimizationIterations, 1, 1000);

        }
    }

    // 3. Optimization Logic (Parallel / Batch Hill Climbing)
    if (isOptimizing && !laneInterpolations.empty()) {

        // Store the baseline (current accepted state)
        std::vector<float> bestStateSoFar = laneInterpolations;
        float bestTimeSoFar = totalTimeNow;
        bool foundBetterThisFrame = false;

        // Distributions for mutation
        std::normal_distribution<float> normalDist(0.0f, mutationMagnitude);
        std::uniform_int_distribution<int> indexDist(0, laneInterpolations.size()-1);
        std::uniform_real_distribution<float> chanceDist(0, 1.0f);

        // Try N different variations
        for (int k = 0; k < optimizationIterations; ++k) {

            // A. Reset to baseline before mutating
            laneInterpolations = bestStateSoFar;

            // B. Apply Mutation
            {
                int idx = indexDist(rng);
                float noise = normalDist(rng);
                laneInterpolations[idx] += noise;
                laneInterpolations[idx] = std::clamp(laneInterpolations[idx], trackLimit, 1 - trackLimit);
				float multiplier = 1.0f;
                for (size_t j = 0; j < laneInterpolations.size(); ++j) {
                    // Small chance to nudge other points slightly for more exploration
					idx = (idx + 1) % laneInterpolations.size();
                    if (chanceDist(rng) > mutationPropagateChance)
                        break; 
                    multiplier *= 1.0f - mutationPropagateMagnitude;
                    noise += normalDist(rng) * multiplier;
                    laneInterpolations[idx] += noise;
                    laneInterpolations[idx] = std::clamp(laneInterpolations[idx], trackLimit, 1 - trackLimit);
				}

            }

            // C. Calculate Physics for this variation
            BuildTrackPath(normalizedSegments, laneInterpolations, trackPath);
            RecalculatePhysics(); // This updates totalTimeNow

            // D. Check if this specific variation is the new best
            if (totalTimeNow < bestTimeSoFar) {
                bestTimeSoFar = totalTimeNow;
                // Temporarily store this as the new baseline for subsequent checks 
                // (or strictly keep it as candidate)
                bestStateSoFar = laneInterpolations;
                foundBetterThisFrame = true;
            }
        }

        // 4. Final Apply
        // If we found a better time, 'bestStateSoFar' holds the winner.
        // If we didn't, 'bestStateSoFar' holds the original state.
        // We set it and run physics one last time to ensure visuals match the data.
        laneInterpolations = bestStateSoFar;
        BuildTrackPath(normalizedSegments, laneInterpolations, trackPath);
        RecalculatePhysics();

        if (foundBetterThisFrame) {
            auto result = std::minmax_element(speedProfileFinal.begin(), speedProfileFinal.end());
            MIN_SPEED_FOR_VISUALIZATION = *result.first;
            SPEED_LIMIT_FOR_VISUALIZATION = *result.second;
        }
        else {

        }
    }
    // Handle manual parameter changes
    else if (paramsChanged && !normalizedSegments.empty() ||MIN_SPEED_FOR_VISUALIZATION == -1) {
        BuildTrackPath(normalizedSegments, laneInterpolations, trackPath);
        RecalculatePhysics();
        auto result = std::minmax_element(speedProfileFinal.begin(), speedProfileFinal.end());
        MIN_SPEED_FOR_VISUALIZATION = *result.first;
        SPEED_LIMIT_FOR_VISUALIZATION = *result.second;
    }

    // --- Statistics ---
    ImGui::Separator();
    ImGui::Text("Lap Time: %.4f s", totalTimeNow);
    //make these a bit more grey
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.7f, 0.7f, 0.7f, 1.0f));
	ImGui::Text("Max Speed: %.2f", SPEED_LIMIT_FOR_VISUALIZATION * 3.6f);
	ImGui::Text("Min Speed: %.2f", MIN_SPEED_FOR_VISUALIZATION * 3.6f);
	ImGui::PopStyleColor();

    ImGui::End();
    // --- SPEED PROFILE GRAPH ---
    // --- TELEMETRY ANALYSIS WINDOW ---
    // --- PHYSICS ANALYSIS WINDOW ---
    // --- PHYSICS ANALYSIS WINDOW ---
    if (!speedProfileFinal.empty() && !trackPath.empty())
    {
        ImGui::Begin("Physics Analysis");

        // Common Layout Data
        ImVec2 regionAvail = ImGui::GetContentRegionAvail();
        float windowWidth = regionAvail.x;
        float graphHeight = 140.0f*2;
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        int count = (int)speedProfileFinal.size();
        float x_step = windowWidth / (float)(count > 1 ? count - 1 : 1);

        // ---------------------------------------------------------
        // 1. SPEED PROFILE GRAPH
        // ---------------------------------------------------------
        ImGui::Text("Speed Profiles (m/s)");
        {
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImVec2 p1 = ImVec2(p0.x + windowWidth, p0.y + graphHeight);

            // Determine Scale (Max of all profiles + 10% padding)
            float maxVal = 0.1f;
            for (float v : speedProfileCurvature) maxVal = std::max(maxVal, v);
            for (float v : speedProfileBraking) maxVal = std::max(maxVal, v);
            for (float v : speedProfileFinal) maxVal = std::max(maxVal, v);
            maxVal *= 1.1f;

            // Background
            draw_list->AddRectFilled(p0, p1, IM_COL32(30, 30, 30, 255));
            draw_list->AddRect(p0, p1, IM_COL32(100, 100, 100, 255));

            // >>> GRID LINES (Subtle Horizontal Lines) <<<
            // Draw a line for every 1.0 m/s
            for (float v = 0.0f; v < maxVal; v += 50.0f / 3.6f) {
                float y = p0.y + graphHeight * (1.0f - v / maxVal);
                // Subtle white line
                draw_list->AddLine(ImVec2(p0.x, y), ImVec2(p1.x, y), IM_COL32(255, 255, 255, 30));

                // Small label on the left
                char buf[16];
                snprintf(buf, sizeof(buf), "%.0f", v * 3.6f);
                draw_list->AddText(ImVec2(p0.x + 2, y - 13), IM_COL32(200, 200, 200, 100), buf);
            }
            // >>> END GRID LINES <<<

            auto PlotLine = [&](const std::vector<float>& data, ImU32 color, float thick) {
                if (data.size() != count) return;
                for (int i = 0; i < count - 1; ++i) {
                    float x1 = p0.x + i * x_step;
                    float x2 = p0.x + (i + 1) * x_step;
                    // Invert Y: 0 at bottom
                    float y1 = p0.y + graphHeight * (1.0f - data[i] / maxVal);
                    float y2 = p0.y + graphHeight * (1.0f - data[i + 1] / maxVal);
                    draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), color, thick);
                }
                };

            // Draw Lines
            PlotLine(speedProfileCurvature, IM_COL32(80, 80, 220, 150), 2.0f); // Blue (Static Limit)
            PlotLine(speedProfileBraking, IM_COL32(220, 80, 80, 150), 2.0f);   // Red (Braking Limit)
            PlotLine(speedProfileFinal, IM_COL32(50, 255, 50, 200), 3.0f);     // Green (Actual)

            // Cursor
            float carX = p0.x + currentPathIndex * x_step;
            draw_list->AddLine(ImVec2(carX, p0.y), ImVec2(carX, p1.y), IM_COL32(255, 255, 0, 200), 2.0f);

            // Reserve space
            ImGui::Dummy(ImVec2(windowWidth, graphHeight));
        }
        ImGui::Spacing();

        ImGui::Separator();
        ImGui::Spacing();

        graphHeight = 140.0f;

        // ---------------------------------------------------------
        // 2. ACCELERATION GRAPH
        // ---------------------------------------------------------
        ImGui::Text("G-Forces (Orange=Longitudinal, Cyan=Lateral)");
        {
            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImVec2 p1 = ImVec2(p0.x + windowWidth, p0.y + graphHeight);

            // Determine Scale based on car limits (Symmetric +/-)
            float scaleLimit = std::max({ maxLateralAcc, brakeAcc, forwardAcc });
            scaleLimit *= 1.2f; // 20% padding

            // Background
            draw_list->AddRectFilled(p0, p1, IM_COL32(30, 30, 30, 255));
            draw_list->AddRect(p0, p1, IM_COL32(100, 100, 100, 255));

            // Helper to map Value -> Y coordinate
            auto MapY = [&](float v) {
                // v=scaleLimit -> Top (0.0)
                // v=0          -> Middle (0.5)
                // v=-scaleLimit-> Bottom (1.0)
                float norm = v / scaleLimit;
                return p0.y + graphHeight * 0.5f * (1.0f - norm);
                };

            // Zero Line (Middle)
            float yZero = MapY(0.0f);
            draw_list->AddLine(ImVec2(p0.x, yZero), ImVec2(p1.x, yZero), IM_COL32(150, 150, 150, 255));

            // >>> GRID LINES (Subtle Horizontal Lines) <<<
            // Draw a line for every 1 unit of acceleration (+1, +2... and -1, -2...)
            for (float g = 0.0f; g < scaleLimit; g += 9.81f) {
                // Positive G
                float yPos = MapY(g);
                draw_list->AddLine(ImVec2(p0.x, yPos), ImVec2(p1.x, yPos), IM_COL32(255, 255, 255, 30));

                // Negative G
                float yNeg = MapY(-g);
                draw_list->AddLine(ImVec2(p0.x, yNeg), ImVec2(p1.x, yNeg), IM_COL32(255, 255, 255, 30));

                // Labels
                char buf[16];
                snprintf(buf, sizeof(buf), "%.0f", g / 9.81f);
                draw_list->AddText(ImVec2(p0.x + 2, yPos - 13), IM_COL32(200, 200, 200, 100), buf);
                snprintf(buf, sizeof(buf), "-%.0f", g / 9.81f);
                draw_list->AddText(ImVec2(p0.x + 2, yNeg - 13), IM_COL32(200, 200, 200, 100), buf);
            }
            // >>> END GRID LINES <<<

            // Plot Longitudinal (Acceleration / Braking)
            for (int i = 0; i < count - 1; ++i) {
                int next = (i + 1) % count;
                float vNow = speedProfileFinal[i];
                float vNext = speedProfileFinal[next];
                float dist = glm::distance(trackPath[i], trackPath[next]);

                float accLong = 0.0f;
                if (dist > 1e-5f) accLong = (vNext * vNext - vNow * vNow) / (2.0f * dist);

                int next2 = (next + 1) % count;
                float vNext2 = speedProfileFinal[next2];
                float dist2 = glm::distance(trackPath[next], trackPath[next2]);
                float accLongNext = 0.0f;
                if (dist2 > 1e-5f) accLongNext = (vNext2 * vNext2 - vNext * vNext) / (2.0f * dist2);

                float x1 = p0.x + i * x_step;
                float x2 = p0.x + (i + 1) * x_step;
                draw_list->AddLine(ImVec2(x1, MapY(accLong)), ImVec2(x2, MapY(accLongNext)), IM_COL32(200, 120, 0, 200), 2.0f);
            }

            // Plot Lateral (Turning Gs)
            // Helper: Calculate Signed G (Positive = Left Turn, Negative = Right Turn)
            auto GetSignedLatG = [&](int idx) {
                float v = speedProfileFinal[idx];
                float k = pathCurvature[idx];
                float mag = v * v * k;

                // Determine direction using Cross Product
                int prev = (idx - 1 + count) % count;
                int next = (idx + 1) % count;
                glm::vec2 d1 = trackPath[idx] - trackPath[prev];
                glm::vec2 d2 = trackPath[next] - trackPath[idx];
                float cross = d1.x * d2.y - d1.y * d2.x;

                // Apply sign
                return mag * ((cross >= 0.0f) ? 1.0f : -1.0f);
                };

            for (int i = 0; i < count - 1; ++i) {
                float latG = GetSignedLatG(i);
                float latG2 = GetSignedLatG(i + 1);

                float x1 = p0.x + i * x_step;
                float x2 = p0.x + (i + 1) * x_step;
                draw_list->AddLine(ImVec2(x1, MapY(latG)), ImVec2(x2, MapY(latG2)), IM_COL32(0, 200, 200, 200), 2.0f);
            }

            // Cursor
            float carX = p0.x + currentPathIndex * x_step;
            draw_list->AddLine(ImVec2(carX, p0.y), ImVec2(carX, p1.y), IM_COL32(200, 200, 0, 200), 2.0f);

            ImGui::Dummy(ImVec2(windowWidth, graphHeight));
        }

        ImGui::End();
    }// --- TIME EVOLUTION HISTORY WINDOW ---
    // --- OPTIMIZATION HISTORY WINDOW ---
    {
        // Static buffers to store history
        static std::vector<float> timeHistory;
        static std::vector<float> lengthHistory;
        static std::vector<float> velocityHistory;

        // Only update if we have valid data
        if (totalTimeNow > 0.001f && !trackPath.empty())
        {
            // 1. Calculate new metrics for this frame
            float currentLength = 0.0f;
            for (size_t i = 0; i < trackPath.size(); ++i) {
                currentLength += glm::distance(trackPath[i], trackPath[(i + 1) % trackPath.size()]);
            }
            float currentAvgVel = currentLength / totalTimeNow;

            // 2. Push to history
            timeHistory.push_back(totalTimeNow);
            lengthHistory.push_back(currentLength);
            velocityHistory.push_back(currentAvgVel);

            // 3. Maintain fixed buffer size (1000 frames)
            if (timeHistory.size() > 250) {
                timeHistory.erase(timeHistory.begin());
                lengthHistory.erase(lengthHistory.begin());
                velocityHistory.erase(velocityHistory.begin());
            }

            // --- DRAW WINDOW ---
            ImGui::Begin("Optimization History (Last 250 Frames)");

            ImVec2 p0 = ImGui::GetCursorScreenPos();
            ImVec2 contentSize = ImGui::GetContentRegionAvail();
            if (contentSize.y < 150.0f) contentSize.y = 150.0f;
            ImVec2 p1 = ImVec2(p0.x + contentSize.x, p0.y + contentSize.y);
            ImDrawList* draw_list = ImGui::GetWindowDrawList();

            // Background
            draw_list->AddRectFilled(p0, p1, IM_COL32(20, 20, 20, 255));
            draw_list->AddRect(p0, p1, IM_COL32(100, 100, 100, 255));

            int count = (int)timeHistory.size();
            float xStep = contentSize.x / (float)(count > 1 ? count - 1 : 1);

            // Helper to normalize and plot a dataset
            auto PlotData = [&](const std::vector<float>& data, ImU32 color, float& outMin, float& outMax)
                {
                    if (data.empty()) return;

                    // Find Range
                    outMin = data[0];
                    outMax = data[0];
                    for (float v : data) {
                        if (v < outMin) outMin = v;
                        if (v > outMax) outMax = v;
                    }
                    // Avoid divide-by-zero
                    float range = outMax - outMin;
                    if (range < 0.0001f) range = 1.0f;

                    for (int i = 0; i < count - 1; ++i) {
                        float v1 = data[i];
                        float v2 = data[i + 1];

                        float x1 = p0.x + i * xStep;
                        float x2 = p0.x + (i + 1) * xStep;

                        // Normalize to [0, 1]
                        float n1 = (v1 - outMin) / range;
                        float n2 = (v2 - outMin) / range;

                        // Map to Screen Y 
                        // Lower Value (0.0) -> Bottom (p1.y)
                        // Higher Value (1.0) -> Top (p0.y)
                        float y1 = p1.y - n1 * contentSize.y;
                        float y2 = p1.y - n2 * contentSize.y;

                        draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), color, 2.0f);
                    }
                };

            // Variables to capture min/max for the legend
            float minT, maxT, minL, maxL, minV, maxV;

            // 4. Plot Graphs
            // Track Length (Yellow) - Drawn first (background layer)
            PlotData(lengthHistory, IM_COL32(200, 200, 50, 100), minL, maxL);

            // Avg Velocity (Magenta)
            PlotData(velocityHistory, IM_COL32(200, 50, 200, 100), minV, maxV);

            // Lap Time (Cyan) - Drawn last (top layer)
            PlotData(timeHistory, IM_COL32(255, 255, 255, 255), minT, maxT);

            // 5. Draw Legends
            float latestT = timeHistory.back();
            float latestL = lengthHistory.back();
            float latestV = velocityHistory.back();

            ImGui::SetCursorPos(ImVec2(10, 50));
            ImGui::TextColored(ImVec4(1, 1, 1, 1), "Lap Time: %.3f", latestT);

            ImGui::SetCursorPos(ImVec2(10, 70));
            ImGui::TextColored(ImVec4(0.8, 0.5, 0.8, 1), "Avg Vel:  %.2f", latestV * 3.6f);

            ImGui::SetCursorPos(ImVec2(10, 90));
            ImGui::TextColored(ImVec4(0.8f, 0.8f, 0.5, 1), "Length:   %.1f", latestL);

            ImGui::Dummy(contentSize);
            ImGui::End();
        }
    }
    FrameEnd();
}

void GPUWorld::ControlImGuiWindow() {

}


void GPUWorld::FrameEnd()
{
    //  DrawCoordinateSystem();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
void GPUWorld::OnKeyPress(int key, int mods) {
    if (key == GLFW_KEY_R) {
        // Reset
        currentPathIndex = 0;
        if (!trackPath.empty()) carPosition = trackPath[0];
    }

}