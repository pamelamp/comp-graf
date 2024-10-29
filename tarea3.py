import pyglet
from OpenGL import GL
import numpy as np
import trimesh as tm
import networkx as nx
import os
from pathlib import Path
from Box2D import b2PolygonShape, b2World
import grafica.transformations as tr

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
        self.keys_state = {}
        self.push_handlers(self.key_handler)
        self.program_state = {"total_time": 0.0,
                              "camera": None,
                              "bodies": {},
                              "world": None,
                              # parámetros para el integrador
                              "vel_iters": 6,
                              "pos_iters": 2,
                              # parámetros extra para el correcto funcionamiento
                              "graph": None,
                              "car": None,
                              "car_code": None,
                              "chose": False}
        self.init()

    def init(self):
        GL.glClearColor(1, 1, 1, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)

    def is_key_pressed(self, key):
        return self.keys_state.get(key ,False)
    
    def on_key_press(self, symbol, modifiers):
        controller.keys_state[symbol] = True
        super().on_key_press(symbol, modifiers)
    
    def on_key_release(self, symbol, modifiers):
        controller.keys_state[symbol] = False

if __name__ == "__main__":
    # Instancia del controller
    controller = Controller("Tarea 3", width=WIDTH, height=HEIGHT, resizable=True)

    controller.program_state["camera"] = FreeCamera([-2, 1, 1.5], "perspective")
    controller.program_state["camera"].yaw = -3*np.pi/4
    controller.program_state["camera"].pitch = -0.5*np.pi/4

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
    
    # Figuras
    square = Model(shapes.Square["position"], shapes.Square["uv"], shapes.Square["normal"], index_data=shapes.Square["indices"])
    arrow = mesh_from_file("assets/arrow.off")[0]["mesh"]

    # Texturas usadas
    wall5 = Texture("assets/wall5.jpg")
    track = Texture("tarea_3/track.jpg")
    sand = Texture("assets/Sand 002_COLOR.jpg")

    matA = Material(diffuse=[0.6, 0, 0.6], specular=[1, 1, 0], ambient=[0.1, 0.5, 1], shininess=2)
    matB = Material(diffuse=[1, 0.3, 0], specular=[1, 0, 1], ambient=[1, 1, 1], shininess=512)
    matC = Material(diffuse=[0, 0.3, 1], specular=[0, 1, 0], ambient=[0.1, 0, 0], shininess=2)

    # Grafos
    graph_world = SceneGraph(controller)
    graph_car_A = SceneGraph(controller)
    graph_car_B = SceneGraph(controller)
    graph_car_C = SceneGraph(controller)
    # 'Sol' dentro del garaje
    graph_world.add_node("sun",
                   pipeline=[color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                   position=[0, 7, 0],
                   light=DirectionalLight(
                       diffuse = [0.5, 0.5, 0.5],
                       specular = [0.5, 0.25, 0.5],
                       ambient = [0.15, 0.15, 0.15]))
    graph_world.add_node("sun_arrow",
                   attach_to="sun",
                   mesh=arrow,
                   position=[0, 0, -0.5],
                   rotation=[-np.pi, 0, 0],
                   scale=[0.5, 0.5, 0.5],
                   color=[1, 1, 0],
                   pipeline=color_mesh_pipeline)
    # Garaje + Autos en el garaje + Pista + 'base' del mundo
    graph_world.add_node("garage")
    graph_world.add_node("rear_wall",
                   attach_to="garage",
                   mesh = square,
                   pipeline = textured_mesh_lit_pipeline,
                   position = [-7, 3, 0], 
                   rotation = [0, np.pi/2, 0],
                   scale = [14, 6, 1],
                   texture=wall5,
                   cull_face=False,
                   material = Material(
                       diffuse = [1, 1, 1],
                       specular = [0.1, 0.1, 0.1],
                       ambient = [0.8, 0.8, 0.8],
                       shininess = 256))
    graph_world.add_node("left_wall",
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
    graph_world.add_node("right_wall",
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
    graph_world.add_node("floor",
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
    graph_world.add_node("roof",
                   attach_to="garage",
                   mesh = square,
                   pipeline = textured_mesh_lit_pipeline,
                   position = [0, 6, 0],
                   rotation = [-np.pi/2, 0, 0],
                   scale = [14, 14, 1],
                   texture = wall5,
                   material = Material(
                       diffuse = [0.1, 0.1, 0.1],
                       specular = [0.1, 0.1, 0.1],
                       ambient = [0.5, 0.5, 0.5],
                       shininess = 256))
    graph_world.add_node("world_base",
                   mesh = square,
                   pipeline = textured_mesh_lit_pipeline,
                   position = [0, -0.01, 0],
                   rotation = [-np.pi/2, 0, 0],
                   scale = [500, 500, 1],
                   texture = sand,
                   material = Material(
                       diffuse = [1, 1, 1],
                       specular = [0.1, 0.1, 0.1],
                       ambient = [1, 1, 1],
                       shininess = 64))
    graph_world.add_node("track",
                   mesh = square,
                   pipeline = textured_mesh_lit_pipeline,
                   position = [-50, 0, 0],
                   rotation = [-np.pi/2, 0, 0],
                   scale = [70, 70, 1],
                   texture = track,
                   material = Material(
                       diffuse = [1, 1, 1],
                       specular = [0.1, 0.1, 0.1],
                       ambient = [1, 1, 1],
                       shininess = 2))

    # Luz spotlight en el garaje
    graph_world.add_node("spotlight",
                    pipeline = [color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                    position = [1.5, 3.5, 0],
                    rotation = [-np.pi/4, np.pi/2, np.pi/2],
                    light=SpotLight(diffuse = [1, 1, 1],
                          specular = [0, 1, 0],
                          ambient = [0.4, 0.4, 0.4]))
    # Luces en la pista
    graph_world.add_node("track_light_1",
                   pipeline = [color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                   position = [-30, 10, 0],
                   rotation = [-np.pi/2, 0, 0],
                   light = SpotLight(diffuse = [1, 1, 1],
                                     specular = [1, 1, 1],
                                     ambient = [0.15, 0.15, 0.15]))
    '''graph.add_node("track_sun",
                   pipeline = [color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                   position = [-30, 10, 0],
                   rotation = [0,-np.pi/2, 0],
                   light = DirectionalLight(
                       diffuse = [1, 1, 1],
                       specular = [1, 1, 1],
                       ambient = [0.15, 0.15, 0.15]))
    graph.add_node("track_sun_arrow",
                   attach_to="track_sun",
                   mesh=arrow,
                   position = [0, 0, 0],
                   #rotation=[np.pi/6, -3*np.pi/4, 0],
                   scale = [0.5, 0.5, 0.5],
                   color = [1, 1, 0],
                   pipeline=color_mesh_pipeline)'''
    # Autos de exhibición
    autoA = mesh_from_file("assets/auto.off")
    autoB = mesh_from_file("assets/auto.off")
    autoC = mesh_from_file("assets/auto.off")
    graph_world.add_node("car_A")
    graph_world.add_node("autoA",
                attach_to="car_A",
                mesh=autoA[0]["mesh"],
                pipeline=color_mesh_lit_pipeline,
                position=[0, 0.48, -3.5],
                scale=[2, 2, 2],
                rotation=[0, 0, 0],
                material=matA,
                cull_face=False)
    graph_world.add_node("car_B")
    graph_world.add_node("autoB",
                attach_to="car_B",
                mesh=autoB[0]["mesh"],
                pipeline=color_mesh_lit_pipeline,
                position=[0, 0.48, 3.5],
                scale=[2, 2, 2],
                rotation=[0, np.pi, 0],
                material=matB,
                cull_face=False)
    graph_world.add_node("car_C")
    graph_world.add_node("autoC",
                attach_to="car_C",
                mesh=autoC[0]["mesh"],
                pipeline=color_mesh_lit_pipeline,
                position=[3.5, 0.48, 0],
                scale=[2, 2, 2],
                rotation=[0, -np.pi/2, 0],
                material=matC,
                cull_face=False)
    # Autos de prueba
    graph_car_A.add_node("mov_car_A")
    for i in range(len(autoA)):
        graph_car_A.add_node(autoA[i]["id"],
                    attach_to="mov_car_A",
                    mesh=autoA[i]["mesh"],
                    pipeline=color_mesh_lit_pipeline,
                    position=[0, 0.48, 0],
                    scale=[2, 2, 2],
                    material=matA,
                    cull_face=False)
    graph_car_A.add_node("car_A_light",
                        attach_to="mov_car_A",
                        pipeline=[color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                        light=PointLight(diffuse=[0, 1, 1], specular=[1, 1, 0], ambient=[0.1, 0.1, 0.1]))
    graph_car_B.add_node("mov_car_B")
    for i in range(len(autoB)):
        graph_car_B.add_node(autoB[i]["id"],
                    attach_to="mov_car_B",
                    mesh=autoB[i]["mesh"],
                    pipeline=color_mesh_lit_pipeline,
                    position=[0, 0.48, 0],
                    scale=[2, 2, 2],
                    material=matB,
                    cull_face=False)
    graph_car_B.add_node("car_B_light",
                        attach_to="mov_car_B",
                        pipeline=[color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                        light=PointLight(diffuse=[0, 1, 1], specular=[1, 1, 0], ambient=[0.1, 0.1, 0.1]))
    graph_car_C.add_node("mov_car_C")
    for i in range(len(autoC)):
        graph_car_C.add_node(autoC[i]["id"],
                    attach_to="mov_car_C",
                    mesh=autoC[i]["mesh"],
                    pipeline=color_mesh_lit_pipeline,
                    position=[0, 0.48, 0],
                    scale=[2, 2, 2],
                    material=matC,
                    cull_face=False)
    graph_car_C.add_node("car_C_light",
                        attach_to="mov_car_C",
                        pipeline=[color_mesh_lit_pipeline, textured_mesh_lit_pipeline],
                        light=PointLight(diffuse=[0, 1, 1], specular=[1, 1, 0], ambient=[0.1, 0.1, 0.1]))

    world = b2World(gravity=(0, 0))
    controller.program_state["world"] = world

    track = world.CreateStaticBody(position=(-50, 0))
    track.CreatePolygonFixture(box=(1/1, 1/1), density=1, friction=10)

    dynamic_car_A = world.CreateDynamicBody(position=(-26, -20))
    dynamic_car_A.CreatePolygonFixture(box=(1, 1), density=4, friction=2)
    controller.program_state["bodies"]["mov_car_A"] = dynamic_car_A

    dynamic_car_B = world.CreateDynamicBody(position=(-30, -20))
    dynamic_car_B.CreatePolygonFixture(box=(1, 1), density=4, friction=2)
    controller.program_state["bodies"]["mov_car_B"] = dynamic_car_B

    dynamic_car_C = world.CreateDynamicBody(position=(-34, -20))
    dynamic_car_C.CreatePolygonFixture(box=(1, 1), density=4, friction=2)
    controller.program_state["bodies"]["mov_car_C"] = dynamic_car_C

    # Extra: canciones
    fever = pyglet.media.load("tarea_3/audio/fever.mp3", streaming=False)
    knock = pyglet.media.load("tarea_3/audio/knock.mp3", streaming=False)
    badger = pyglet.media.load("tarea_3/audio/badger.mp3", streaming=False)
    # Reproductores
    fever_player = pyglet.media.Player()
    fever_player.queue(fever)
    fever_player.pause()
    fever_player.loop = True
    knock_player = pyglet.media.Player()
    knock_player.queue(knock)
    knock_player.pause()
    knock_player.loop = True
    badger_player = pyglet.media.Player()
    badger_player.queue(badger)
    badger_player.pause()
    badger_player.loop = True

    ######### CONTROLES ######################################################################
    print("~~~~~ Controles globales ~~~~~")
    print("Click derecho y mover: Rotar la cámara")
    print("O/P: Cambiar tipo de cámara")
    print("N: Detener la canción")
    print("~~~~~ Controles fase de selección ~~~~~")
    print("1/2/3: Cambiar auto en pantalla")
    print("Flechas izq/der: Rotar el auto")
    print("IJKL: Luz direccional del garaje")
    print("Mouse: Clic izquierdo para mover la luz spotlight")
    print("ENTER: Seleccionar auto en escena y cambiar a la fase de conducción")
    print("~~~~~ Controles fase de conducción ~~~~~")
    print("WASD: Mover el auto")
    print("ENTER: Regresar a la fase de selección")
    
    def update_world(dt):
        controller.program_state["total_time"] += dt
        controller.program_state["world"].Step(dt, controller.program_state["vel_iters"], controller.program_state["pos_iters"])
        controller.program_state["world"].ClearForces()

        graph_car_A["mov_car_A"]["position"][0] = dynamic_car_A.position[0]
        graph_car_A["mov_car_A"]["position"][2] = dynamic_car_A.position[1]
        graph_car_A["mov_car_A"]["rotation"][1] = -dynamic_car_A.angle

        graph_car_B["mov_car_B"]["position"][0] = dynamic_car_B.position[0]
        graph_car_B["mov_car_B"]["position"][2] = dynamic_car_B.position[1]
        graph_car_B["mov_car_B"]["rotation"][1] = -dynamic_car_B.angle
        
        graph_car_C["mov_car_C"]["position"][0] = dynamic_car_C.position[0]
        graph_car_C["mov_car_C"]["position"][2] = dynamic_car_C.position[1]
        graph_car_C["mov_car_C"]["rotation"][1] = -dynamic_car_C.angle

    def update(dt):
        update_world(dt)
        camera = controller.program_state["camera"]
        sun = graph_world["sun"]
        graph = controller.program_state["graph"]
        car = controller.program_state["car"]
        car_code = controller.program_state["car_code"]
        chose = controller.program_state["chose"]
        if controller.is_key_pressed(pyglet.window.key.P):
            camera.type = "perspective"
        if controller.is_key_pressed(pyglet.window.key.O):
            camera.type = "orthographic"
        if chose == False:
            if controller.is_key_pressed(pyglet.window.key.LEFT):
                graph_world["autoA"]["rotation"][1] += dt*0.6
                graph_world["autoB"]["rotation"][1] += dt*0.6
                graph_world["autoC"]["rotation"][1] += dt*0.6
            if controller.is_key_pressed(pyglet.window.key.RIGHT):
                graph_world["autoA"]["rotation"][1] -= dt*0.6
                graph_world["autoB"]["rotation"][1] -= dt*0.6
                graph_world["autoC"]["rotation"][1] -= dt*0.6
            if controller.is_key_pressed(pyglet.window.key.I):
                sun["rotation"][0] -= 2 * dt
            if controller.is_key_pressed(pyglet.window.key.K):
                sun["rotation"][0] += 2 * dt
            if controller.is_key_pressed(pyglet.window.key.J):
                sun["rotation"][1] -= 2 * dt
            if controller.is_key_pressed(pyglet.window.key.L):
                sun["rotation"][1] += 2 * dt
            camera.update()
        if chose == True:
            forward = graph.get_forward(car_code)
            if controller.is_key_pressed(pyglet.window.key.A):
                car.ApplyTorque(-1, True)
            if controller.is_key_pressed(pyglet.window.key.D):
                car.ApplyTorque(1, True)
            if controller.is_key_pressed(pyglet.window.key.W):
                car.ApplyForceToCenter((forward[0]*10, forward[2]*10), True)
            if controller.is_key_pressed(pyglet.window.key.S):
                car.ApplyForceToCenter((-forward[0]*10, -forward[2]*10), True)
            camera.position[0] = car.position[0] + 2 * np.sin(car.angle)
            camera.position[1] = 2
            camera.position[2] = car.position[1] - 2 * np.cos(car.angle)
            camera.yaw = car.angle + np.pi/2
            camera.update()

    @controller.event
    def on_resize(width, height):
        controller.program_state["camera"].resize(width, height)

    @controller.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        chose = controller.program_state["chose"]
        if buttons & pyglet.window.mouse.RIGHT:
            controller.program_state["camera"].yaw += dx * 0.01
            controller.program_state["camera"].pitch += dy * 0.01
        if chose == False & buttons & pyglet.window.mouse.LEFT:
            graph_world["spotlight"]["rotation"][0] += dy * 0.01
            graph_world["spotlight"]["rotation"][1] += dx * 0.01

    @controller.event
    def on_key_press(symbol, modifiers):
        camera = controller.program_state["camera"]
        car_A = graph_world["car_A"]
        car_B = graph_world["car_B"]
        car_C = graph_world["car_C"]
        chose = controller.program_state["chose"]
        if chose == False:
            if symbol == pyglet.window.key._1:
                print("Mostrando auto 1")
                graph_world["autoA"]["rotation"][1] = 0
                graph_world["autoB"]["rotation"][1] = np.pi
                graph_world["autoC"]["rotation"][1] = -np.pi/2
                car_A["rotation"][1] = np.pi/2
                car_B["rotation"][1] = np.pi/2
                car_C["rotation"][1] = np.pi/2
                controller.program_state["graph"] = graph_car_A
                controller.program_state["car"] = dynamic_car_A
                controller.program_state["car_code"] = "mov_car_A"
                fever_player.pause()
                knock_player.pause()
                badger_player.pause()
                fever_player.seek(0.0)
                fever_player.play()
            if symbol == pyglet.window.key._2:
                print("Mostrando auto 2")
                graph_world["autoA"]["rotation"][1] = 0
                graph_world["autoB"]["rotation"][1] = np.pi
                graph_world["autoC"]["rotation"][1] = -np.pi/2
                car_A["rotation"][1] = -np.pi/2
                car_B["rotation"][1] = -np.pi/2
                car_C["rotation"][1] = -np.pi/2
                controller.program_state["graph"] = graph_car_B
                controller.program_state["car"] = dynamic_car_B
                controller.program_state["car_code"] = "mov_car_B"
                fever_player.pause()
                knock_player.pause()
                badger_player.pause()
                knock_player.seek(0.0)
                knock_player.play()
            if symbol == pyglet.window.key._3:
                print("Mostrando auto 3")
                graph_world["autoA"]["rotation"][1] = 0
                graph_world["autoB"]["rotation"][1] = np.pi
                graph_world["autoC"]["rotation"][1] = -np.pi/2
                car_A["rotation"][1] = np.pi
                car_B["rotation"][1] = np.pi
                car_C["rotation"][1] = np.pi
                controller.program_state["graph"] = graph_car_C
                controller.program_state["car"] = dynamic_car_C
                controller.program_state["car_code"] = "mov_car_C"
                fever_player.pause()
                knock_player.pause()
                badger_player.pause()
                badger_player.seek(0.0)
                badger_player.play()
        if chose == False and symbol == pyglet.window.key.ENTER:
            controller.program_state["chose"] = True
        if chose == True and symbol == pyglet.window.key.ENTER:
            print("Volviendo a la pantalla de selección")
            controller.program_state["chose"] = False
            fever_player.pause()
            knock_player.pause()
            badger_player.pause()
            fever_player.seek(0.0)
            knock_player.seek(0.0)
            badger_player.seek(0.0)
            camera.position = [-2, 1, 1.5]
            camera.yaw = -3*np.pi/4
            camera.pitch = -0.5*np.pi/4
            camera.type = "perspective"
        if symbol == pyglet.window.key.N:
            print("Deteniendo la música")
            fever_player.pause()
            knock_player.pause()
            badger_player.pause()
            fever_player.seek(0.0)
            knock_player.seek(0.0)
            badger_player.seek(0.0)
    
    @controller.event
    def on_draw():
        controller.clear()
        axis_scene.draw()
        graph_world.draw()
        graph_car_A.draw()
        graph_car_B.draw()
        graph_car_C.draw()
    
    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()