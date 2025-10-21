# visualizer.py
"""
Modern OpenGL 3.3+ Visualizer for Naomi SOL Hub
================================================
Uses Core Profile with VAO/VBO and GLSL shaders for maximum performance
Includes fancy 3D effects, bloom, particles, and post-processing
"""

import sys
import math
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from ctypes import c_float, c_uint, c_void_p, sizeof

# Graphics libraries
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

logger = logging.getLogger("ModernVisualizer")


# ══════════════════════════════════════════════════════════════════════════════
# MATRIX MATH (for Modern OpenGL without GLU)
# ══════════════════════════════════════════════════════════════════════════════

def perspective(fov, aspect, near, far):
    """Create perspective projection matrix"""
    f = 1.0 / math.tan(fov / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)


def lookAt(eye, center, up):
    """Create view matrix"""
    f = np.array(center) - np.array(eye)
    f = f / np.linalg.norm(f)
    
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    
    u = np.cross(s, f)
    
    result = np.identity(4, dtype=np.float32)
    result[0, :3] = s
    result[1, :3] = u
    result[2, :3] = -f
    result[3, :3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    
    return result.T


def translate(x, y, z):
    """Create translation matrix"""
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def rotate_y(angle):
    """Create Y-axis rotation matrix"""
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def scale(x, y, z):
    """Create scale matrix"""
    return np.array([
        [x, 0, 0, 0],
        [0, y, 0, 0],
        [0, 0, z, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SHADERS
# ══════════════════════════════════════════════════════════════════════════════

# Main 3D Object Shader (with lighting and glow)
VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in vec3 aNormal;

out vec3 FragPos;
out vec3 Color;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float time;

void main() {
    // Subtle wave animation
    vec3 pos = aPos;
    pos.z += sin(time * 2.0 + aPos.x * 0.5) * 0.05;
    
    FragPos = vec3(model * vec4(pos, 1.0));
    Color = aColor;
    Normal = mat3(transpose(inverse(model))) * aNormal;
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec3 FragPos;
in vec3 Color;
in vec3 Normal;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;
uniform float time;
uniform float glowStrength;

void main() {
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * Color;
    
    // Diffuse
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * Color;
    
    // Specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0);
    
    // Glow effect (pulsing)
    float glow = sin(time * 3.0) * 0.5 + 0.5;
    vec3 glowColor = Color * glowStrength * glow;
    
    vec3 result = ambient + diffuse + specular + glowColor;
    FragColor = vec4(result, 1.0);
}
"""

# Particle Shader (for effects)
PARTICLE_VERTEX = """
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec4 aColor;
layout(location = 2) in float aSize;

out vec4 ParticleColor;

uniform mat4 view;
uniform mat4 projection;
uniform float time;

void main() {
    ParticleColor = aColor;
    gl_Position = projection * view * vec4(aPos, 1.0);
    gl_PointSize = aSize;
}
"""

PARTICLE_FRAGMENT = """
#version 330 core

in vec4 ParticleColor;
out vec4 FragColor;

void main() {
    // Circular particles
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    
    float alpha = 1.0 - (dist * 2.0);
    FragColor = vec4(ParticleColor.rgb, ParticleColor.a * alpha);
}
"""

# Grid Shader (for shape logic visualization)
GRID_VERTEX = """
#version 330 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aColor;

out vec3 GridColor;

uniform mat4 projection;
uniform float time;

void main() {
    GridColor = aColor;
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
}
"""

GRID_FRAGMENT = """
#version 330 core

in vec3 GridColor;
out vec4 FragColor;

uniform float time;

void main() {
    // Pulsing grid effect
    float pulse = sin(time * 2.0) * 0.2 + 0.8;
    FragColor = vec4(GridColor * pulse, 0.8);
}
"""

# Post-Processing Shader (bloom effect)
POSTPROCESS_VERTEX = """
#version 330 core

layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    TexCoord = aTexCoord;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

POSTPROCESS_FRAGMENT = """
#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform float bloomStrength;
uniform float time;

void main() {
    vec4 color = texture(screenTexture, TexCoord);
    
    // Simple bloom effect
    vec3 bloom = vec3(0.0);
    float weight[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    
    vec2 tex_offset = 1.0 / textureSize(screenTexture, 0);
    bloom += color.rgb * weight[0];
    
    for(int i = 1; i < 5; ++i) {
        bloom += texture(screenTexture, TexCoord + vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
        bloom += texture(screenTexture, TexCoord - vec2(tex_offset.x * i, 0.0)).rgb * weight[i];
        bloom += texture(screenTexture, TexCoord + vec2(0.0, tex_offset.y * i)).rgb * weight[i];
        bloom += texture(screenTexture, TexCoord - vec2(0.0, tex_offset.y * i)).rgb * weight[i];
    }
    
    vec3 result = color.rgb + bloom * bloomStrength;
    
    // Vignette effect
    float dist = length(TexCoord - vec2(0.5));
    float vignette = 1.0 - dist * 0.5;
    result *= vignette;
    
    FragColor = vec4(result, 1.0);
}
"""

# Text Rendering Shader
TEXT_VERTEX = """
#version 330 core
layout(location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>

out vec2 TexCoords;

uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}
"""

TEXT_FRAGMENT = """
#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D text;
uniform vec3 textColor;

void main() {
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
    FragColor = vec4(textColor, 1.0) * sampled;
}
"""


# ══════════════════════════════════════════════════════════════════════════════
# MODERN OPENGL VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

class ModernVisualizer:
    """Modern OpenGL 3.3+ Core Profile Visualizer"""
    
    def __init__(self, width=1280, height=720, title="Naomi SOL Hub - Modern OpenGL"):
        self.width = width
        self.height = height
        self.title = title
        
        # State
        self.running = False
        self.clock = None
        self.start_time = time.time()
        
        # Camera
        self.camera_distance = 5.0
        self.camera_angle = 0.0
        self.camera_height = 2.0
        
        # OpenGL objects
        self.shader_program = None
        self.particle_program = None
        self.grid_program = None
        self.postprocess_program = None
        
        self.vao_cube = None
        self.vbo_cube = None
        self.vao_particles = None
        self.vbo_particles = None
        self.vao_grid = None
        self.vbo_grid = None
        
        # Framebuffer for post-processing
        self.fbo = None
        self.fbo_texture = None
        self.rbo = None
        
        # Data
        self.particles = []
        self.grid_data = None
        
        # UI
        self.font = None
        self.show_info = True
        
        logger.info("Initializing Modern OpenGL Visualizer...")
        self._initialize()
    
    def _initialize(self):
        """Initialize Pygame and OpenGL"""
        pygame.init()
        self.clock = pygame.time.Clock()
        
        # Request OpenGL 3.3 Core Profile
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, 
                                       pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        
        # Create window
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption(self.title)
        
        # Initialize OpenGL
        self._init_opengl()
        
        # Initialize text rendering
        self._init_text_rendering()  # Add this line
    
        # Create font for UI
        self.font = pygame.font.SysFont("Segoe UI", 14)
        
        logger.info("✓ Modern OpenGL visualizer initialized")
    
    def _init_opengl(self):
        """Initialize Modern OpenGL"""
        # Enable features
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        
        # Clear color
        glClearColor(0.05, 0.05, 0.1, 1.0)
        
        # Compile shaders
        self._compile_shaders()
        
        # Create geometry
        self._create_cube()
        self._create_particles(1000)
        self._create_grid()
        
        # Create framebuffer for post-processing
        self._create_framebuffer()
        
        # Initialize text rendering
        self._init_text_rendering()
        
        logger.info("✓ OpenGL initialized (Core Profile 3.3+)")
    
    def _compile_shaders(self):
        """Compile all shader programs"""
        try:
            # Main shader
            self.shader_program = compileProgram(
                compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
                compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
            )
            
            # Particle shader
            self.particle_program = compileProgram(
                compileShader(PARTICLE_VERTEX, GL_VERTEX_SHADER),
                compileShader(PARTICLE_FRAGMENT, GL_FRAGMENT_SHADER)
            )
            
            # Grid shader
            self.grid_program = compileProgram(
                compileShader(GRID_VERTEX, GL_VERTEX_SHADER),
                compileShader(GRID_FRAGMENT, GL_FRAGMENT_SHADER)
            )
            
            # Post-process shader
            self.postprocess_program = compileProgram(
                compileShader(POSTPROCESS_VERTEX, GL_VERTEX_SHADER),
                compileShader(POSTPROCESS_FRAGMENT, GL_FRAGMENT_SHADER)
            )
            
            # Text shader
            self.text_program = compileProgram(
                compileShader(TEXT_VERTEX, GL_VERTEX_SHADER),
                compileShader(TEXT_FRAGMENT, GL_FRAGMENT_SHADER)
            )
            
            logger.info("✓ Shaders compiled successfully")
            
        except Exception as e:
            logger.error(f"Shader compilation failed: {e}")
            raise
    
    def _create_cube(self):
        """Create cube with VAO/VBO"""
        # Cube vertices (position, color, normal)
        vertices = np.array([
            # Front face (red)
            -0.5, -0.5,  0.5,  1.0, 0.3, 0.3,  0, 0, 1,
             0.5, -0.5,  0.5,  1.0, 0.3, 0.3,  0, 0, 1,
             0.5,  0.5,  0.5,  1.0, 0.3, 0.3,  0, 0, 1,
            -0.5,  0.5,  0.5,  1.0, 0.3, 0.3,  0, 0, 1,
            # Back face (blue)
            -0.5, -0.5, -0.5,  0.3, 0.3, 1.0,  0, 0, -1,
             0.5, -0.5, -0.5,  0.3, 0.3, 1.0,  0, 0, -1,
             0.5,  0.5, -0.5,  0.3, 0.3, 1.0,  0, 0, -1,
            -0.5,  0.5, -0.5,  0.3, 0.3, 1.0,  0, 0, -1,
            # Top face (green)
            -0.5,  0.5, -0.5,  0.3, 1.0, 0.3,  0, 1, 0,
             0.5,  0.5, -0.5,  0.3, 1.0, 0.3,  0, 1, 0,
             0.5,  0.5,  0.5,  0.3, 1.0, 0.3,  0, 1, 0,
            -0.5,  0.5,  0.5,  0.3, 1.0, 0.3,  0, 1, 0,
            # Bottom face (yellow)
            -0.5, -0.5, -0.5,  1.0, 1.0, 0.3,  0, -1, 0,
             0.5, -0.5, -0.5,  1.0, 1.0, 0.3,  0, -1, 0,
             0.5, -0.5,  0.5,  1.0, 1.0, 0.3,  0, -1, 0,
            -0.5, -0.5,  0.5,  1.0, 1.0, 0.3,  0, -1, 0,
            # Right face (magenta)
             0.5, -0.5, -0.5,  1.0, 0.3, 1.0,  1, 0, 0,
             0.5, -0.5,  0.5,  1.0, 0.3, 1.0,  1, 0, 0,
             0.5,  0.5,  0.5,  1.0, 0.3, 1.0,  1, 0, 0,
             0.5,  0.5, -0.5,  1.0, 0.3, 1.0,  1, 0, 0,
            # Left face (cyan)
            -0.5, -0.5, -0.5,  0.3, 1.0, 1.0,  -1, 0, 0,
            -0.5, -0.5,  0.5,  0.3, 1.0, 1.0,  -1, 0, 0,
            -0.5,  0.5,  0.5,  0.3, 1.0, 1.0,  -1, 0, 0,
            -0.5,  0.5, -0.5,  0.3, 1.0, 1.0,  -1, 0, 0,
        ], dtype=np.float32)
        
        indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front
            4, 5, 6, 6, 7, 4,  # Back
            8, 9, 10, 10, 11, 8,  # Top
            12, 13, 14, 14, 15, 12,  # Bottom
            16, 17, 18, 18, 19, 16,  # Right
            20, 21, 22, 22, 23, 20,  # Left
        ], dtype=np.uint32)
        
        # Create VAO
        self.vao_cube = glGenVertexArrays(1)
        glBindVertexArray(self.vao_cube)
        
        # Create VBO
        self.vbo_cube = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_cube)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        # Create EBO
        self.ebo_cube = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_cube)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 36, c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 36, c_void_p(12))
        glEnableVertexAttribArray(1)
        
        # Normal attribute
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 36, c_void_p(24))
        glEnableVertexAttribArray(2)
        
        glBindVertexArray(0)
        
        self.cube_indices_count = len(indices)
    
    def _create_particles(self, count):
        """Create particle system"""
        particles = []
        for _ in range(count):
            # Random position in sphere
            theta = np.random.random() * 2 * np.pi
            phi = np.random.random() * np.pi
            r = np.random.random() * 3.0
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Random color
            color = [np.random.random(), np.random.random(), np.random.random(), 0.6]
            
            # Random size
            size = np.random.uniform(2.0, 8.0)
            
            particles.extend([x, y, z] + color + [size])
        
        vertices = np.array(particles, dtype=np.float32)
        
        # Create VAO
        self.vao_particles = glGenVertexArrays(1)
        glBindVertexArray(self.vao_particles)
        
        # Create VBO
        self.vbo_particles = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_particles)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        
        # Position attribute
        stride = 8 * sizeof(c_float)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, c_void_p(12))
        glEnableVertexAttribArray(1)
        
        # Size attribute
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, c_void_p(28))
        glEnableVertexAttribArray(2)
        
        glBindVertexArray(0)
        
        self.particle_count = count
    
    def _create_grid(self):
        """Create 2D grid for shape logic visualization"""
        # Simple quad for now (will be updated with real grid data)
        vertices = np.array([
            # Position, Color
            -0.9, 0.7, 0.3, 0.3, 0.3,
             0.9, 0.7, 0.3, 0.3, 0.3,
             0.9, -0.7, 0.3, 0.3, 0.3,
            -0.9, -0.7, 0.3, 0.3, 0.3,
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VAO
        self.vao_grid = glGenVertexArrays(1)
        glBindVertexArray(self.vao_grid)
        
        # Create VBO
        self.vbo_grid = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_grid)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_DYNAMIC_DRAW)
        
        # Create EBO
        self.ebo_grid = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo_grid)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 20, c_void_p(0))
        glEnableVertexAttribArray(0)
        
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 20, c_void_p(8))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    def _create_framebuffer(self):
        """Create framebuffer for post-processing"""
        # Generate framebuffer
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        
        # Create texture
        self.fbo_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.fbo_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, 
                               GL_TEXTURE_2D, self.fbo_texture, 0)
        
        # Create renderbuffer for depth
        self.rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, 
                             self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, 
                                 GL_RENDERBUFFER, self.rbo)
        
        # Check framebuffer complete
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            logger.error("Framebuffer not complete!")
        
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # Create quad for post-processing
        quad_vertices = np.array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
             1,  1, 1, 1,
            -1,  1, 0, 1,
        ], dtype=np.float32)
        
        quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        self.vao_quad = glGenVertexArrays(1)
        glBindVertexArray(self.vao_quad)
        
        vbo_quad = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_quad)
        glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)
        
        ebo_quad = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_quad)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, c_void_p(0))
        glEnableVertexAttribArray(0)
        
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, c_void_p(8))
        glEnableVertexAttribArray(1)
        
        glBindVertexArray(0)
    
    def _init_text_rendering(self):
        """Initialize text rendering"""
        # Compile text shader
        self.text_shader = compileProgram(
            compileShader(TEXT_VERTEX, GL_VERTEX_SHADER),
            compileShader(TEXT_FRAGMENT, GL_FRAGMENT_SHADER)
        )
        
        # Create VAO/VBO for text quads
        self.text_vao = glGenVertexArrays(1)
        self.text_vbo = glGenBuffers(1)
        glBindVertexArray(self.text_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.text_vbo)
        glBufferData(GL_ARRAY_BUFFER, sizeof(c_float) * 6 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
    
    def _handle_input(self):
        """Handle keyboard and mouse input"""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.running = False
                elif event.key == K_SPACE:
                    logger.info("Pause/Resume (not implemented in demo)")
                elif event.key == K_i:
                    self.show_info = not self.show_info
        
        # Mouse for camera control
        if pygame.mouse.get_pressed()[0]:  # Left button
            dx, dy = pygame.mouse.get_rel()
            self.camera_angle += dx * 0.01
            self.camera_height += dy * 0.01
            self.camera_height = np.clip(self.camera_height, 0.5, 5.0)
        else:
            pygame.mouse.get_rel()  # Clear delta
        
        # Keyboard for camera distance
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            self.camera_distance -= 0.1
        if keys[K_s]:
            self.camera_distance += 0.1
        self.camera_distance = np.clip(self.camera_distance, 2.0, 15.0)
    
    def render(self):
        """Main render loop"""
        current_time = time.time() - self.start_time
        
        # === PASS 1: Render to framebuffer ===
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Setup camera
        camera_x = self.camera_distance * math.cos(self.camera_angle)
        camera_z = self.camera_distance * math.sin(self.camera_angle)
        
        view = lookAt(
            [camera_x, self.camera_height, camera_z],
            [0, 0, 0],
            [0, 1, 0]
        )
        
        projection = perspective(
            math.radians(45),
            self.width / self.height,
            0.1,
            100.0
        )
        
        # Render 3D cube
        glUseProgram(self.shader_program)
        
        # Set uniforms
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "view"), 
                          1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "projection"), 
                          1, GL_FALSE, projection)
        glUniform1f(glGetUniformLocation(self.shader_program, "time"), current_time)
        glUniform3f(glGetUniformLocation(self.shader_program, "lightPos"), 5, 5, 5)
        glUniform3f(glGetUniformLocation(self.shader_program, "viewPos"), 
                   camera_x, self.camera_height, camera_z)
        glUniform1f(glGetUniformLocation(self.shader_program, "glowStrength"), 0.3)
        
        # Render multiple cubes in a pattern
        for i in range(-2, 3):
            for j in range(-2, 3):
                model = translate(i * 1.5, 0, j * 1.5)
                model = model @ rotate_y(current_time + i * 0.5 + j * 0.3)
                model = model @ scale(0.5, 0.5, 0.5)
                
                glUniformMatrix4fv(glGetUniformLocation(self.shader_program, "model"), 
                                  1, GL_FALSE, model)
                
                glBindVertexArray(self.vao_cube)
                glDrawElements(GL_TRIANGLES, self.cube_indices_count, GL_UNSIGNED_INT, None)
        
        # Render particles
        glUseProgram(self.particle_program)
        glUniformMatrix4fv(glGetUniformLocation(self.particle_program, "view"), 
                          1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.particle_program, "projection"), 
                          1, GL_FALSE, projection)
        glUniform1f(glGetUniformLocation(self.particle_program, "time"), current_time)
        
        glBindVertexArray(self.vao_particles)
        glDrawArrays(GL_POINTS, 0, self.particle_count)
        
        # === PASS 2: Post-processing ===
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(self.postprocess_program)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.fbo_texture)
        glUniform1i(glGetUniformLocation(self.postprocess_program, "screenTexture"), 0)
        glUniform1f(glGetUniformLocation(self.postprocess_program, "bloomStrength"), 0.3)
        glUniform1f(glGetUniformLocation(self.postprocess_program, "time"), current_time)
        
        glBindVertexArray(self.vao_quad)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
        
        # === UI Overlay ===
        if self.show_info:
            self._render_ui(current_time)
    
    def _render_ui(self, current_time):
        """Render UI overlay with modern OpenGL"""
        info_text = [
            f"FPS: {int(self.clock.get_fps())}",
            f"Time: {current_time:.1f}s",
            f"Camera: [{self.camera_distance:.1f}, {self.camera_height:.1f}, {math.degrees(self.camera_angle):.0f}°]",
            "",
            "Controls:",
            "  Mouse drag - Rotate camera",
            "  W/S - Zoom in/out",
            "  I - Toggle info",
            "  ESC - Exit"
        ]
        
        # Use text shader
        glUseProgram(self.text_shader)
        ortho = np.array([
            [2/self.width, 0, 0, -1],
            [0, 2/self.height, 0, -1],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        glUniformMatrix4fv(
            glGetUniformLocation(self.text_shader, "projection"),
            1, GL_FALSE, ortho
        )
        
        glUniform3f(
            glGetUniformLocation(self.text_shader, "textColor"),
            1.0, 1.0, 1.0
        )
        
        glActiveTexture(GL_TEXTURE0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        x, y = 10, 10
        for text in info_text:
            surface = self.font.render(text, True, (255, 255, 255))
            texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture)
            
            text_data = pygame.image.tostring(surface, "RGBA", True)
            w, h = surface.get_size();
            
            glTexImage2D(
                GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, text_data
            )
            
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            
            # Render text quad
            vertices = np.array([
                [x, y + h, 0.0, 0.0],
                [x + w, y, 1.0, 1.0],
                [x, y, 0.0, 1.0],
                [x, y + h, 0.0, 0.0],
                [x + w, y + h, 1.0, 0.0],
                [x + w, y, 1.0, 1.0]
            ], dtype=np.float32).flatten()
            
            glBindVertexArray(self.text_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.text_vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            
            glDrawArrays(GL_TRIANGLES, 0, 6)
            
            y += 25
            
            # Cleanup texture
            glDeleteTextures(1, [texture])
    
    def run(self):
        """Main loop"""
        self.running = True
        logger.info("Starting render loop...")
        
        while self.running:
            self._handle_input()
            self.render()
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        self.shutdown()
    
    def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down visualizer...")
        
        # Cleanup OpenGL objects
        if self.vao_cube:
            glDeleteVertexArrays(1, [self.vao_cube])
        if self.vbo_cube:
            glDeleteBuffers(1, [self.vbo_cube])
        
        pygame.quit()
        logger.info("✓ Shutdown complete")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN / TESTING
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Test the visualizer"""
    logging.basicConfig(level=logging.INFO)
    
    try:
        viz = ModernVisualizer(width=1280, height=720)
        viz.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()