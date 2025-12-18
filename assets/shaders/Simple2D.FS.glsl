#version 330

// Input
in vec3 frag_color;

// Uniform properties
uniform ivec4 viewport;

// Output
layout(location = 0) out vec4 out_color;


void main()
{
//    if (gl_FragCoord.x < viewport.x || gl_FragCoord.x > viewport.x + viewport.z
//        || gl_FragCoord.y < viewport.y || gl_FragCoord.y > viewport.y + viewport.w) {
//        discard;
//    }

    out_color = vec4(frag_color, 1);
}
