#version 330

// Input
layout(location = 0) in vec3 v_position;
layout(location = 3) in vec3 v_color;

// Uniform properties
uniform ivec2 window_resolution;

// Output
out vec3 frag_color;


void main()
{
    frag_color = v_color;

    vec2 p = (v_position.xy / window_resolution) * 2 - 1;

    gl_Position = vec4(p, v_position.z, 1.0);
}
