import matplotlib.pyplot as plt
from stl import mesh
import numpy as np


def magnitude(vector):
    sum = 0
    for column in np.array(vector):
        for x in column:
            sum += x ** 2
    return np.sqrt(sum)


def normalize(vector):
    return vector / magnitude(vector)


class Triangle:
    def __init__(self, point1, point2, point3):
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3

        self.vector1 = normalize(point2 - point1)
        self.vector2 = normalize(point3 - point1)
        self.normal = normalize(np.cross(self.vector1, self.vector2))

        self.vector3 = normalize(point2 - point3)


class Ball:
    def __init__(self, radius, mass, position=np.matrix([0.0, 0.0, 0.0]), velocity=np.matrix([0.0, 0.0, 0.0])):
        self.radius = radius
        self.mass = mass
        self.position = position
        self.velocity = velocity

    def offset(self, triangle):
        return np.dot(np.array(self.position)[0], np.array(triangle.normal)[0]) - np.dot(np.array(triangle.point1)[0], np.array(triangle.normal)[0])

    def collision(self, triangle):
        offset = self.offset(triangle)
        if abs(offset) > self.radius:
            return None

        position = self.position - triangle.normal * offset

        edges = [triangle.vector2, -triangle.vector1, triangle.vector3]
        offsets = [position - triangle.point1, position -
                   triangle.point2, position - triangle.point3]
        for i, edge in enumerate(edges):
            if np.dot(np.cross(np.array(offsets[i])[0], np.array(edge)[0]), np.array(triangle.normal)[0]) <= 0:
                return None

        return position


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)

    scene_x = []
    scene_y = []
    scene_z = []
    triangles = []

    scene = mesh.Mesh.from_file("scene.stl")
    scale = 0.001
    for triangle in scene.vectors:
        for point in triangle:
            scene_x.append(point[0] * scale)
            scene_y.append(point[1] * scale)
            scene_z.append(point[2] * scale)

        triangles.append(Triangle(np.matrix([scene_x[-3], scene_y[-3], scene_z[-3]]), np.matrix(
            [scene_x[-2], scene_y[-2], scene_z[-2]]), np.matrix([scene_x[-1], scene_y[-1], scene_z[-1]])))
    ax.scatter(scene_x, scene_y, scene_z, s=1, c="b")

    ball = Ball(0.0065, 0.0065, np.matrix(
        [0.9, 0.0, 1.0]), np.matrix([0.0, -1.0, 0.0]))

    dt = 0.001
    coefficient_of_restitution = 0.658
    rolling_resistance_coefficient = 0.15

    x_points = []
    y_points = []
    z_points = []

    time = 0

    while time <= 30:
        if np.array(ball.position)[0][2] < -5:
            break

        ball.velocity += np.matrix([0.0, 0.0, -9.81]) * dt

        for triangle in triangles:
            collision = ball.collision(triangle)
            if collision is not None:
                ball.velocity += triangle.normal * np.dot(np.array(ball.velocity)
                                                          [0], np.array(triangle.normal)[0]) * -(1 + coefficient_of_restitution)

                normal_force = np.dot(np.array(triangle.normal)[
                                      0], np.array([0, 0, 1])) * ball.mass * 9.81
                rolling_resistance = normal_force * rolling_resistance_coefficient
                planar_velocity = ball.velocity - \
                    np.dot(np.array(triangle.normal)[0], np.array(ball.velocity)[0])
                planar_velocity_direction = normalize(planar_velocity)
                rolling_resistance_direction = -planar_velocity_direction
                rolling_resistance_dv = rolling_resistance_direction * \
                    rolling_resistance / ball.mass * dt
                ball.velocity += rolling_resistance_dv

                break

        ball.position += ball.velocity * dt
        time += dt
        print(f"{time}, {ball.position}, {ball.velocity}")

        current_position = np.array(ball.position)[0]
        x_points.append(current_position[0])
        y_points.append(current_position[1])
        z_points.append(current_position[2])

    ax.scatter(x_points, y_points, z_points, s=1, c="g")
    plt.show()
