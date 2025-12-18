#include "components/software_rendering_scene.h"

#include <vector>
#include <iostream>

#include "pfd/portable-file-dialogs.h"

using namespace std;

using namespace gfxc;


/*
 *  To find out more about `FrameStart`, `Update`, `FrameEnd`
 *  and the order in which they are called, see `world.cpp`.
 */


SoftwareRenderScene::SoftwareRenderScene() :
    image (nullptr),
    depthImage (nullptr)
{
    saveScreenToImage = false;
    window->SetSize(900, 900);
}


SoftwareRenderScene::~SoftwareRenderScene()
{
    delete depthImage;
    delete image;
}

void SoftwareRenderScene::Init()
{
    GetCameraInput()->SetActive (false);

    {
        Mesh* mesh = new Mesh("quad");
        mesh->LoadMesh(PATH_JOIN(window->props.selfDir, RESOURCE_PATH::MODELS, "primitives"), "quad.obj");
        mesh->UseMaterials(false);
        meshes[mesh->GetMeshID()] = mesh;
    }

    //std::string shaderPath = PATH_JOIN(window->props.selfDir, SOURCE_PATH::M2, "lab7", "shaders");

    //// Create a shader program for particle system
    //{
    //    Shader *shader = new Shader("ImageProcessing");
    //    shader->AddShader(PATH_JOIN(shaderPath, "VertexShader.glsl"), GL_VERTEX_SHADER);
    //    shader->AddShader(PATH_JOIN(shaderPath, "FragmentShader.glsl"), GL_FRAGMENT_SHADER);

    //    shader->CreateAndLink();
    //    shaders[shader->GetName()] = shader;
    //}

    image = new Image();
    depthImage = new DepthImage();

    Initialize();

    image->UpdateInternalData();
    //depthImage->UpdateInternalData();
}

void SoftwareRenderScene::FrameStart()
{

}

void SoftwareRenderScene::Update(float deltaTimeSeconds)
{
    ClearScreen();

    auto shader = shaders["Screen"];
    shader->Use();

    //if (saveScreenToImage)
    //{
    //    window->SetSize(originalImage->GetWidth(), originalImage->GetHeight());
    //}

    //int screenSize_loc = shader->GetUniformLocation("screenSize");
    //glm::ivec2 resolution = window->GetResolution();
    //glUniform2i(screenSize_loc, resolution.x, resolution.y);

    int locTexture = shader->GetUniformLocation("u_texture_0");
    glUniform1i(locTexture, 0);

    image->BindToTextureUnit(GL_TEXTURE0);
    //depthImage->BindToTextureUnit(GL_TEXTURE0);

    RenderMesh(meshes["quad"], shader, glm::mat4(1));

    //if (saveScreenToImage)
    //{
    //    saveScreenToImage = false;

    //    GLenum format = GL_RGB;
    //    if (originalImage->GetNrChannels() == 4)
    //    {
    //        format = GL_RGBA;
    //    }

    //    glReadPixels(0, 0, originalImage->GetWidth(), originalImage->GetHeight(), format, GL_UNSIGNED_BYTE, processedImage->GetImageData());
    //    processedImage->UploadNewData(processedImage->GetImageData());
    //    SaveImage("shader_processing_" + std::to_string(outputMode));

    //    float aspectRatio = static_cast<float>(originalImage->GetWidth()) / originalImage->GetHeight();
    //    window->SetSize(static_cast<int>(600 * aspectRatio), 600);
    //}
}


void SoftwareRenderScene::FrameEnd()
{
    DrawCoordinateSystem();
}


//void Lab7::OnFileSelected(const std::string &fileName)
//{
//    if (fileName.size())
//    {
//        std::cout << fileName << endl;
//        originalImage = TextureManager::LoadTexture(fileName, nullptr, "image", true, true);
//        processedImage = TextureManager::LoadTexture(fileName, nullptr, "newImage", true, true);
//
//        float aspectRatio = static_cast<float>(originalImage->GetWidth()) / originalImage->GetHeight();
//        window->SetSize(static_cast<int>(600 * aspectRatio), 600);
//    }
//}
//
//void Lab7::SaveImage(const std::string &fileName)
//{
//    cout << "Saving image! ";
//    processedImage->SaveToFile((fileName + ".png").c_str());
//    cout << "[Done]" << endl;
//}

//
//void Lab7::OpenDialog()
//{
//    std::vector<std::string> filters =
//    {
//        "Image Files", "*.png *.jpg *.jpeg *.bmp",
//        "All Files", "*"
//    };
//
//    auto selection = pfd::open_file("Select a file", ".", filters).result();
//    if (!selection.empty())
//    {
//        std::cout << "User selected file " << selection[0] << "\n";
//        OnFileSelected(selection[0]);
//    }
//}


/*
 *  These are callback functions. To find more about callbacks and
 *  how they behave, see `input_controller.h`.
 */


void SoftwareRenderScene::OnInputUpdate(float deltaTime, int mods)
{
    // Treat continuous update based on input
}


void SoftwareRenderScene::OnKeyPress(int key, int mods)
{
    // Add key press event
    if (key == GLFW_KEY_F || key == GLFW_KEY_ENTER || key == GLFW_KEY_SPACE)
    {
        //OpenDialog();
    }

    //if (key == GLFW_KEY_S && mods & GLFW_MOD_CONTROL)
    //{
    //    if (!gpuProcessing)
    //    {
    //        SaveImage("processCPU_" + std::to_string(outputMode));
    //    } else {
    //        saveScreenToImage = true;
    //    }
    //}
}


void SoftwareRenderScene::OnKeyRelease(int key, int mods)
{
    // Add key release event
}


void SoftwareRenderScene::OnMouseMove(int mouseX, int mouseY, int deltaX, int deltaY)
{
    // Add mouse move event
}


void SoftwareRenderScene::OnMouseBtnPress(int mouseX, int mouseY, int button, int mods)
{
    // Add mouse button press event
}


void SoftwareRenderScene::OnMouseBtnRelease(int mouseX, int mouseY, int button, int mods)
{
    // Add mouse button release event
}


void SoftwareRenderScene::OnMouseScroll(int mouseX, int mouseY, int offsetX, int offsetY)
{
    // Treat mouse scroll event
}


void SoftwareRenderScene::OnWindowResize(int width, int height)
{
    // Treat window resize event
}
