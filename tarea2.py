import pyglet
from OpenGL import GL
import numpy as np
import trimesh as tm
import networkx as nx
import os
from pathlib import Path

import auxiliares.utils.shapes as shapes
from auxiliares.utils.camera import OrbitCamera, FreeCamera
from auxiliares.utils.scene_graph import SceneGraph
from auxiliares.utils.drawables import Model, Texture, DirectionalLight, PointLight, SpotLight, Material
from auxiliares.utils.helpers import init_axis, init_pipeline, mesh_from_file, get_path

WIDTH = 640
HEIGHT = 640

class Controller(pyglet.window.Window):
    def __init__(self, title, *args, **kargs):
        super().__init__(*args, **kargs)
        self.set_minimum_size(240, 240)
        self.set_caption(title)
        self.key_handler = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.key_handler)
        self.program_state = {"total_time": 0.0, "camera": None}
        self.init()

    def init(self):
        GL.glClearColor(1, 1, 1, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)

    def is_key_pressed(self, key):
        return self.key_handler[key]

if __name__ == "__main__":
    # Instancia del controller
    controller = Controller("Tarea 2", width=WIDTH, height=HEIGHT, resizable=True)

    controller.program_state["camera"] = FreeCamera([4, 5, 0], "perspective")
    controller.program_state["camera"].yaw = np.pi
    controller.program_state["camera"].pitch = -np.pi / 4

    axis_scene = init_axis(controller)

    color_mesh_pipeline = init_pipeline(
        get_path("auxiliares/shaders/color_mesh.vert"),
        get_path("auxiliares/shaders/color_mesh.frag"))
    
    textured_mesh_pipeline = init_pipeline(
        get_path("auxiliares/shaders/textured_mesh.vert"),
        get_path("auxiliares/shaders/textured_mesh.frag"))
    
    color_mesh_lit_pipeline = init_pipeline(
        get_path("auxiliares/shaders/color_mesh_lit.vert"),
        get_path("auxiliares/shaders/color_mesh_lit.frag"))
    
    textured_mesh_lit_pipeline = init_pipeline(
        get_path("auxiliares/shaders/textured_mesh_lit.vert"),
        get_path("auxiliares/shaders/textured_mesh_lit.frag"))
    
    square = Model(shapes.Square["position"], shapes.Square["uv"], shapes.Square["normal"], index_data=shapes.Square["indices"])
    arrow = mesh_from_file("assets/arrow.off")[0]["mesh"]

    wall5 = Texture("assets/wall5.jpg")

    graph = SceneGraph(controller)

    graph.add_node("sun",
                   pipeline=[color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                   position=[0, 7, 0],
                   light=DirectionalLight(diffuse = [0.5, 0.5, 0.5], specular = [0.5, 0.5, 0.5], ambient = [0.15, 0.15, 0.15]))
    graph.add_node("sun_arrow",
                   attach_to="sun",
                   mesh=arrow,
                   position=[0, 0, -0.5],
                   rotation=[np.pi, np.pi, 0],
                   scale=[0.5, 0.5, 0.5],
                   color=[1, 1, 0],
                   pipeline=color_mesh_pipeline)
    
    graph.add_node("garage")
    graph.add_node("rear_wall",
                   attach_to="garage",
                   mesh = square,
                   pipeline = textured_mesh_lit_pipeline,
                   position = [-7, 3, 0], 
                   rotation = [0, np.pi/2, 0],
                   scale = [14, 6, 1],
                   texture=wall5,
                   material = Material(
                       diffuse = [1, 1, 1],
                       specular = [0.1, 0.1, 0.1],
                       ambient = [0.8, 0.8, 0.8],
                       shininess = 256))
    
    graph.add_node("left_wall",
                   attach_to="garage",
                   mesh = square,
                   pipeline = textured_mesh_lit_pipeline,
                   position = [0, 3, 7], 
                   rotation = [0, np.pi, 0],
                   scale = [14, 6, 1],
                   texture=wall5,
                   material = Material(
                       diffuse = [1, 1, 1],
                       specular = [0.1, 0.1, 0.1],
                       ambient = [0.5, 0.5, 0.5],
                       shininess = 256))
    
    graph.add_node("right_wall",
                   attach_to="garage",
                   mesh = square,
                   pipeline = textured_mesh_lit_pipeline,
                   position = [0, 3, -7], 
                   rotation = [0, 0, 0],
                   scale = [14, 6, 1],
                   texture=wall5,
                   cull_face=False,
                   material = Material(
                       diffuse = [1, 1, 1],
                       specular = [0.1, 0.1, 0.1],
                       ambient = [0.5, 0.5, 0.5],
                       shininess = 256),)
    
    graph.add_node("floor",
                   attach_to="garage",
                   mesh = square,
                   pipeline = textured_mesh_lit_pipeline,
                   position = [0, 0, 0],
                   rotation = [-np.pi/2, 0, 0],
                   scale = [14, 14, 1],
                   texture=wall5,
                   material = Material(
                       diffuse = [1, 1, 1],
                       specular = [0.1, 0.1, 0.1],
                       ambient = [0.5, 0.5, 0.5],
                       shininess = 256))
    
    graph.add_node("spotlight",
                    pipeline=[color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                    position=[1.5, 3.5, 0],
                    rotation=[-np.pi/4, np.pi/2, np.pi/2],
                    light=SpotLight(diffuse = [1, 1, 1],
                          specular = [0, 1, 0],
                          ambient = [0.4, 0.4, 0.4]))

    
    graph.add_node("turntable")
    graph.add_node("car_A", attach_to="turntable")
    graph.add_node("car_B", attach_to="turntable")
    graph.add_node("car_C", attach_to="turntable")
    graph.add_node("car_D", attach_to="turntable")

    auto = mesh_from_file("assets/auto.off")
    graph.add_node("autoA",
                attach_to="car_A",
                mesh=auto[0]["mesh"],
                pipeline=color_mesh_lit_pipeline,
                position=[-3.5, 0.48, 0],
                scale=[2, 2, 2],
                rotation=[0, np.pi/2, 0],
                material=Material(
                          diffuse = [0, 0, 1],
                          specular = [0, 1, 0],
                          ambient = [0.1, 0, 0],
                          shininess = 2),
                cull_face=False)
    graph.add_node("autoB",
                attach_to="car_B",
                mesh=auto[0]["mesh"],
                pipeline=color_mesh_lit_pipeline,
                position=[0, 0.48, 3.5],
                scale=[2, 2, 2],
                rotation=[0, np.pi, 0],
                material=Material(
                          diffuse = [0, 0, 1],
                          specular = [0, 1, 0],
                          ambient = [0.1, 0, 0],
                          shininess = 16),
                cull_face=False)
    graph.add_node("autoC",
                attach_to="car_C",
                mesh=auto[0]["mesh"],
                pipeline=color_mesh_lit_pipeline,
                position=[3.5, 0.48, 0],
                scale=[2, 2, 2],
                rotation=[0, -np.pi/2, 0],
                material=Material(
                          diffuse = [0, 0, 1],
                          specular = [0, 1, 0],
                          ambient = [0.1, 0, 0],
                          shininess = 256),
                cull_face=False)
    graph.add_node("autoD",
                attach_to="car_D",
                mesh=auto[0]["mesh"],
                pipeline=color_mesh_lit_pipeline,
                position=[0, 0.48, -3.5],
                scale=[2, 2, 2],
                rotation=[0, 0, 0],
                material=Material(
                          diffuse = [1, 0, 0],
                          specular = [1, 0, 1],
                          ambient = [0.1, 0.1, 0.1],
                          shininess = 512),
                cull_face=False)

    print("Controles Cámara:\n\tA/D: Izquierda/Derecha\n\tQ/E: Bajar/Subir\n\t W/S: Acercar/Alejar\n\t1/2: Cambiar tipo de cámara")
    print("Espacio (mantener): Cambiar auto bajo el foco")
    print("Mouse:\n\tClic izquierdo para mover la cámara\n\tClic derecho para mover la escena")
    print("Controles luz central: flechas")
    def update(dt):
        controller.program_state["total_time"] += dt
        camera = controller.program_state["camera"]
        sun = graph["sun"]
        if controller.is_key_pressed(pyglet.window.key.A):
            camera.position -= camera.right * 5*dt
        if controller.is_key_pressed(pyglet.window.key.D):
            camera.position += camera.right * 5*dt
        if controller.is_key_pressed(pyglet.window.key.W):
            camera.position += camera.forward * 5*dt
        if controller.is_key_pressed(pyglet.window.key.S):
            camera.position -= camera.forward * 5*dt
        if controller.is_key_pressed(pyglet.window.key.Q):
            camera.position[1] -= 5*dt
        if controller.is_key_pressed(pyglet.window.key.E):
            camera.position[1] += 5*dt
        if controller.is_key_pressed(pyglet.window.key._1):
            camera.type = "perspective"
        if controller.is_key_pressed(pyglet.window.key._2):
            camera.type = "orthographic"
        camera.update()
        
        if controller.is_key_pressed(pyglet.window.key.UP):
            sun["rotation"][0] -= 2 * dt
        if controller.is_key_pressed(pyglet.window.key.DOWN):
            sun["rotation"][0] += 2 * dt
        if controller.is_key_pressed(pyglet.window.key.LEFT):
            sun["rotation"][1] -= 2 * dt
        if controller.is_key_pressed(pyglet.window.key.RIGHT):
            sun["rotation"][1] += 2 * dt

        turntable = graph["turntable"]
        if controller.is_key_pressed(pyglet.window.key.SPACE):
            turntable["rotation"][1] += np.pi/128


    @controller.event
    def on_resize(width, height):
        controller.program_state["camera"].resize(width, height)

    @controller.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        #if buttons & pyglet.window.mouse.RIGHT:
        if buttons & pyglet.window.mouse.LEFT:
            controller.program_state["camera"].yaw += dx * 0.01
            controller.program_state["camera"].pitch += dy * 0.01
        
        #if buttons & pyglet.window.mouse.LEFT:
        if buttons & pyglet.window.mouse.RIGHT:
            graph["root"]["rotation"][0] += dy * 0.01
            graph["root"]["rotation"][1] += dx * 0.01

    @controller.event
    def on_draw():
        controller.clear()
        axis_scene.draw()
        graph.draw()
    
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()