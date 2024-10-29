import pyglet
from OpenGL import GL
import numpy as np
import trimesh as tm
import networkx as nx
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
import grafica.transformations as tr
import auxiliares.utils.shapes as shapes

WIDTH = 640
HEIGHT = 640

class Controller(pyglet.window.Window):
    def __init__(self, title, *args, **kargs):
        super().__init__(*args, **kargs)
        self.set_minimum_size(240, 240)
        self.set_caption(title)
        self.key_handler = pyglet.window.key.KeyStateHandler()
        self.push_handlers(self.key_handler)
        self.init()

    def init(self):
        GL.glClearColor(1, 1, 1, 1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        #GL.glEnable(GL.GL_CULL_FACE) # Línea comentada para evitar problemas al ejecutar el programa
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)

    def is_key_pressed(self, key):
        return self.key_handler[key]

class Model():
    def __init__(self, position_data, index_data=None):
        self.position_data = position_data

        self.index_data = index_data
        if index_data is not None:
            self.index_data = np.array(index_data, dtype=np.uint32)

        self.gpu_data = None

    def init_gpu_data(self, pipeline):
        self.pipeline = pipeline
        if self.index_data is not None:
            self.gpu_data = pipeline.vertex_list_indexed(len(self.position_data) // 3, GL.GL_TRIANGLES, self.index_data)
        else:
            self.gpu_data = pipeline.vertex_list(len(self.position_data) // 3, GL.GL_TRIANGLES)
        
        self.gpu_data.position[:] = self.position_data

    def draw(self, mode = GL.GL_TRIANGLES):
        self.gpu_data.draw(mode)

    def get_transform(self):
        translation_matrix = tr.translate(self.position[0], self.position[1], self.position[2])
        rotation_matrix = tr.rotationX(self.rotation[0]) @ tr.rotationY(self.rotation[1]) @ tr.rotationZ(self.rotation[2])
        scale_matrix = tr.scale(self.scale[0], self.scale[1], self.scale[2])
        transformation = translation_matrix @ rotation_matrix @ scale_matrix
        return np.reshape(transformation, (16, 1), order = "F")

class Mesh(Model):
    def __init__(self, asset_path):
        mesh_data = tm.load(asset_path)
        mesh_scale = tr.uniformScale(2.0 / mesh_data.scale)
        mesh_translate = tr.translate(*-mesh_data.centroid)
        mesh_data.apply_transform(mesh_scale @ mesh_translate)
        vertex_data = tm.rendering.mesh_to_vertexlist(mesh_data)
        indices = vertex_data[3]
        positions = vertex_data[4][1]

        super().__init__(positions, indices)

class Camera():
    def __init__(self, camera_type = "perspective"):
        self.position = np.array([1, 0, 0], dtype=np.float32)
        self.focus = np.array([0, 0, 0], dtype=np.float32)
        self.type = camera_type
        self.width = WIDTH
        self.height = HEIGHT

    def update(self):
            pass

    def get_view(self):
        lookAt_matrix = tr.lookAt(self.position, self.focus, np.array([0, 1, 0], dtype=np.float32))
        return np.reshape(lookAt_matrix, (16, 1), order="F")
        
    def get_projection(self):
        if self.type == "perspective":
            perspective_matrix = tr.perspective(90, self.width / self.height, 0.01, 100)
        elif self.type == "orthographic":
            depth = self.position - self.focus
            depth = np.linalg.norm(depth)
            perspective_matrix = tr.ortho(-(self.width/self.height)*depth, (self.width/self.height)*depth, -1*depth, 1*depth, 0.01, 100)
        return np.reshape(perspective_matrix, (16, 1), order="F")
    
    def resize(self, width, height):
        self.width = width
        self.height = height

class OrbitCamera(Camera):
    def __init__(self, distance, camera_type = "perspective"):
        super().__init__(camera_type)
        self.distance = distance
        self.phi = 0
        self.theta = np.pi/2
        self.update()

    def update(self):
        if self.theta > np.pi:
            self.theta = np.pi
        elif self.theta < 0:
            self.theta = 0.0001
        
        self.position[0] = self.distance * np.sin(self.theta) * np.sin(self.phi)
        self.position[1] = self.distance * np.cos(self.theta)
        self.position[2] = self.distance * np.sin(self.theta) * np.cos(self.phi)

class SceneGraph():
    def __init__(self, camera=None):
        self.graph = nx.DiGraph(root="root")
        self.add_node("root")
        self.camera = camera

    def add_node(self, 
                 name, 
                 attach_to=None, 
                 mesh=None, 
                 color=[1, 1, 1], 
                 transform=tr.identity(),
                 position=[0, 0, 0], 
                 rotation=[0, 0, 0], 
                 scale=[1, 1, 1], 
                 mode=GL.GL_TRIANGLES):
        self.graph.add_node(
            name,
            mesh=mesh,
            color=color, 
            transform=transform,
            position=np.array(position, dtype=np.float32),
            rotation=np.array(rotation, dtype=np.float32),
            scale=np.array(scale, dtype=np.float32), 
            mode=mode)
        if attach_to is None:
            attach_to = "root"
        
        self.graph.add_edge(attach_to, name)

    def __getitem__(self, name):
        if name not in self.graph.nodes:
            raise KeyError(f"Node {name} not in graph")
        
        return self.graph.nodes[name]
    
    def __setitem__(self, name, value):
        if name not in self.graph.nodes:
            raise KeyError(f"Node {name} not in graph")
        
        self.graph.nodes[name] = value
    
    def get_transform(self, node):
        node = self.graph.nodes[node]
        transform = node["transform"]
        translation_matrix = tr.translate(node["position"][0], node["position"][1], node["position"][2])
        rotation_matrix = tr.rotationX(node["rotation"][0]) @ tr.rotationY(node["rotation"][1]) @ tr.rotationZ(node["rotation"][2])
        scale_matrix = tr.scale(node["scale"][0], node["scale"][1], node["scale"][2])
        return transform @ translation_matrix @ rotation_matrix @ scale_matrix
    
    def draw(self):
        root_key = self.graph.graph["root"]
        edges = list(nx.edge_dfs(self.graph, source=root_key))
        transformations = {root_key: self.get_transform(root_key)}
        
        for src, dst in edges:
            current_node = self.graph.nodes[dst]

            if not dst in transformations:
                transformations[dst] = transformations[src] @ self.get_transform(dst)
            
            if current_node["mesh"] is not None:
                current_pipeline = current_node["mesh"].pipeline
                current_pipeline.use()

                if self.camera is not None:
                    if "u_view" in current_pipeline.uniforms:
                        current_pipeline["u_view"] = self.camera.get_view()
                    
                    if "u_projection" in current_pipeline.uniforms:
                        current_pipeline["u_projection"] = self.camera.get_projection()

                current_pipeline["u_model"] = np.reshape(transformations[dst], (16, 1), order="F")

                if "u_color" in current_pipeline.uniforms:
                    current_pipeline["u_color"] = np.array(current_node["color"], dtype=np.float32)
                current_node["mesh"].draw(current_node["mode"])

# La clase Garage da forma al hangar/garaje en el que se encuentra el auto
class Garage():
    def __init__(self, room_mesh, camera):
        garage = SceneGraph(camera)
        garage.add_node("rear_wall", mesh=room_mesh, color=shapes.BLACK, scale=[7, 4, 1], position=[-4, 2, 0], rotation=[0, np.pi/2, 0])
        garage.add_node("left_wall", mesh=room_mesh, color=shapes.DARK_BLUE, scale=[8, 4, 1], position=[0, 2, 3.5], rotation=[0, np.pi, 0])
        garage.add_node("right_wall", mesh=room_mesh, color=shapes.DARK_BLUE, scale=[8, 4, 1], position=[0, 2, -3.5], rotation=[0, -np.pi, 0])
        garage.add_node("roof", mesh=room_mesh, color=shapes.GRAY, scale=[8, 7, 1], position=[0, 4, 0], rotation=[np.pi/2, 0, 0])
        garage.add_node("floor", mesh=room_mesh, color=shapes.GRAY, scale=[8, 7, 1], position=[0, 0, 0], rotation=[np.pi/2, 0, 0])

        self.graph = garage

    def draw(self):
        self.graph.draw()

# La clase Car da forma al auto completo (chasis y ruedas)
class Car():
    def __init__(self, mesh1, mesh2, camera):
        car = SceneGraph(camera)
        car.add_node("body", mesh=mesh1, color=shapes.BROWN, scale=[2, 2, 2], position=[0, 0.582 , 0], rotation=[-np.pi/2, 0, np.pi/2])
        # Para poder ver lo mejor posible el auto, la línea siguiente crea nuevamente en cuerpo del auto pero ligeramente más pequeño (ver README.md)
        car.add_node("filler_body", mesh=mesh1, color=shapes.BLACK, scale=[1.95, 1.95, 1.95], position=[0, 0.583, 0], rotation=[-np.pi/2, 0, np.pi/2])

        car.add_node("front_left_wheel", mesh=mesh2, color=shapes.BLACK, scale=[0.408, 0.408, 0.408], position=[1.13, 0.33, 0.6], rotation=[0, -np.pi/2, 0])
        car.add_node("front_right_wheel", mesh=mesh2, color=shapes.BLACK, scale=[0.408, 0.408, 0.408], position=[1.13, 0.33, -0.6], rotation=[0, np.pi/2, 0])
        car.add_node("rear_left_wheel", mesh=mesh2, color=shapes.BLACK, scale=[0.408, 0.408, 0.408], position=[-0.95, 0.33, 0.6], rotation=[0, -np.pi/2, 0])
        car.add_node("rear_right_wheel", mesh=mesh2, color=shapes.BLACK, scale=[0.408, 0.408, 0.408], position=[-0.95, 0.33, -0.6], rotation=[0, np.pi/2, 0])

        self.graph = car

    def draw(self):
        self.graph.draw()

if __name__ == "__main__":
    # Instancia del controller
    controller = Controller("Tarea 1", width=WIDTH, height=HEIGHT, resizable=True)

    with open(Path(os.path.dirname(__file__)) / "transform.vert") as f:
        color_vertex_source_code = f.read()

    with open(Path(os.path.dirname(__file__)) / "color.frag") as f:
        color_fragment_source_code = f.read()

    color_pipeline = pyglet.graphics.shader.ShaderProgram(
        pyglet.graphics.shader.Shader(color_vertex_source_code, "vertex"),
        pyglet.graphics.shader.Shader(color_fragment_source_code, "fragment"))
    

    camera = OrbitCamera(5, "perspective")
    camera.phi = np.pi / 4
    camera.theta = np.pi / 4

    # Ejes
    axes = Model(shapes.Axes["position"])
    axes.init_gpu_data(color_pipeline)
    axes.gpu_data.color[:] = shapes.Axes["color"]

    with open(Path(os.path.dirname(__file__)) / "color_mesh.vert") as f:
        vertex_source_code = f.read()
    
    with open(Path(os.path.dirname(__file__)) / "color_mesh.frag") as f:
        fragment_source_code = f.read()
    
    mesh_pipeline = pyglet.graphics.shader.ShaderProgram(
        pyglet.graphics.shader.Shader(vertex_source_code, "vertex"),
        pyglet.graphics.shader.Shader(fragment_source_code, "fragment"))
    
    axis_scene = SceneGraph(camera)
    axis_scene.add_node("axes", attach_to="root", mesh=axes, mode=GL.GL_LINES)

    # Importación de la figura de cilindro para crear la plataforma sobre la que se encuentra el auto
    cylinder = Mesh("tarea_1/cylinder.off")
    cylinder.init_gpu_data(mesh_pipeline)

    platform = SceneGraph(camera)
    platform.add_node("platform", attach_to="root", mesh=cylinder, color=shapes.RED, scale=[4, 0.05, 4], position=[0, 0.0295, 0], mode=GL.GL_TRIANGLES)

    square = Model(shapes.Square["position"], index_data=shapes.Square["indices"])
    square.init_gpu_data(mesh_pipeline)    

    garage = Garage(square, camera)

    # Importación del archivo que forma el cuerpo/chasis del auto
    car_body = Mesh("tarea_1/Dodge Challenger STL.stl")
    car_body.init_gpu_data(mesh_pipeline)

    # Importación del archivo que forma las ruedas del auto
    car_wheel = Mesh("tarea_1/Dodge Challenger wheel STL.stl")
    car_wheel.init_gpu_data(mesh_pipeline)

    car = Car(car_body, car_wheel, camera)

    print("Controles Cámara:\n\tWASD: Rotar\n\t Q/E: Alejar/Acercar\n\t1/2: Cambiar tipo")
    def update(dt):
        if controller.is_key_pressed(pyglet.window.key.A):
            camera.phi -= dt
        if controller.is_key_pressed(pyglet.window.key.D):
            camera.phi += dt
        if controller.is_key_pressed(pyglet.window.key.W):
            camera.theta -= dt
        if controller.is_key_pressed(pyglet.window.key.S):
            camera.theta += dt
        if controller.is_key_pressed(pyglet.window.key.Q):
            camera.distance += 2*dt
        if controller.is_key_pressed(pyglet.window.key.E):
            camera.distance -= 2*dt
        if controller.is_key_pressed(pyglet.window.key._1):
            camera.type = "perspective"
        if controller.is_key_pressed(pyglet.window.key._2):
            camera.type = "orthographic"        

        camera.update()
        
    @controller.event
    def on_resize(width, height):
        camera.resize(width, height)

    # draw loop
    @controller.event
    def on_draw():
        controller.clear()
        axis_scene.draw()

        platform.draw()
        garage.draw()
        car.draw()

    pyglet.clock.schedule_interval(update, 1/60)
    pyglet.app.run()