#version 330

// Uniform properties
uniform vec3 color;

// Output
layout(location = 0) out vec4 out_color;
in vec3 local_pos;

void main() {
    out_color = vec4(color, 1.0);
}
