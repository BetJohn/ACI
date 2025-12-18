#pragma once

#include "core/gpu/image.h"

#include <glm/glm.hpp>

namespace gfxc
{
    class DepthImage : public Image
    {
    public:
        void Init(unsigned int width, unsigned int height);

        void Set(unsigned int row, unsigned int column, glm::vec3 data);
        glm::vec3 Get(unsigned int row, unsigned int column);

        void Clear();

    private:
        unsigned int *imageData;
    };
}
