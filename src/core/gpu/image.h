#pragma once

#include "core/gpu/texture2D.h"

#include <glm/glm.hpp>

namespace gfxc
{
    class Image : public Texture2D
    {
    public:
        virtual void Init(unsigned int width, unsigned int height,
            unsigned int channels, const glm::vec3 initial_value = glm::vec3(0));

        virtual void UpdateInternalData();

        virtual void Set(unsigned int row, unsigned int column, glm::vec3 data);
        virtual glm::vec3 Get(unsigned int row, unsigned int column);

        virtual void Clear(const glm::vec3 &color);
    };
}
