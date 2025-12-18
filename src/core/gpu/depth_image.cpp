#include "core/gpu/depth_image.h"

#include <vector>

using namespace gfxc;

void DepthImage::Init (unsigned int width, unsigned int height)
{
    this->width = width;
    this->height = height;

    unsigned int size = width * height;

    imageData = new unsigned int[size];

    Clear();
}

void DepthImage::Set(unsigned int row, unsigned int column, glm::vec3 depth)
{
    imageData[width * row + column] = depth.x * INT_MAX;
}

glm::vec3 DepthImage::Get(unsigned int row, unsigned int column)
{
    return glm::vec3((float)imageData[width * row + column] / INT_MAX, 0, 0);
}

void DepthImage::Clear()
{
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            Set(i, j, glm::vec3 (1, 0, 0));
        }
    }
}
