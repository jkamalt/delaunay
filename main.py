import numpy as np
from numpy.typing import NDArray

from delaunay import DelaunayBuilder


def get_test_points_1() -> NDArray[int]:
    """Возвращает набор тестовых точек"""
    points = np.array(
        [[103, 265],
         [142, 180],
         [194, 128],
         [264, 88],
         [336, 76],
         [425, 75],
         [574, 87],
         [641, 132],
         [679, 245],
         [669, 365],
         [613, 491],
         [469, 582],
         [213, 590],
         [68, 450],
         [437, 269]]
    )
    return points


def get_test_points_2() -> NDArray[float]:
    """Возвращает тестовый набор точек"""
    return np.loadtxt('test/points2.txt')


if __name__ == '__main__':
    points = get_test_points_2()

    # Основной объект построителя триангуляции
    builder = DelaunayBuilder(points)
    # Построение сетки
    builder.build()

    # Отображение построенной сетки на графике
    builder.draw_triangulation()
    # Отображение сетки, построенной с помощью функции из библиотеки scipy
    builder.draw_triangulation_scipy()
