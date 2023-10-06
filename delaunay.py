from collections import defaultdict
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from numpy.typing import NDArray

from helper import TwoVertices, ListNode, Edge


class DelaunayBuilder:
    """Класс описывает объект-построитель триангуляции Делоне"""

    def __init__(self, points: NDArray[float]):
        # Граф, соответствующий построенной сетке
        self.graph = defaultdict(TwoVertices)
        # Точки, отсортированные по координате x, по которым будет построена сетка
        self.points: NDArray[float] = np.array(sorted(points, key=lambda p: p[0]))
        # Минимальная выпуклая оболочка
        self.convex_hull: List[ListNode] = [ListNode() for _ in points]
        # Стек рекурсии
        self.stack: List[Edge] = [Edge() for _ in points]
        # Точность расчета
        self.eps: float = 1e-9

    def build(self):
        """Основной метод: строит триангуляцию Делоне по заданным точкам"""
        # Если точек меньше 3, то триангуляция пустая
        if len(self.points) < 3:
            return

        # Инициализация первого треугольника
        self.convex_hull[0] = ListNode(1, 1)
        self.graph[Edge(0, 1)].insert(2)

        for i in range(2, len(self.points)):
            self._add_point_to_triangulation(i)

    def _add_point_to_triangulation(self, i: int):
        """
        Добавляет точку с индексом i в триангуляцию
        :param i: индекс точки
        """
        hull_pt = i - 1
        last_vector = self.points[hull_pt] - self.points[i]
        next_hull_pt = self.convex_hull[hull_pt].right
        new_vector = self.points[next_hull_pt] - self.points[i]

        # Пока при движении вправо от точки ребра видимые, выполняется рекурсивное перестроение
        while np.cross(last_vector, new_vector) > -self.eps:
            self._fix_triangulation(hull_pt, next_hull_pt, i)
            hull_pt = next_hull_pt
            last_vector = new_vector
            next_hull_pt = self.convex_hull[hull_pt].right
            new_vector = self.points[next_hull_pt] - self.points[i]
        self.convex_hull[i].right = hull_pt

        # Аналогично при движении влево от точки
        hull_pt = i - 1
        last_vector = self.points[hull_pt] - self.points[i]
        next_hull_pt = self.convex_hull[hull_pt].left
        new_vector = self.points[next_hull_pt] - self.points[i]

        while np.cross(last_vector, new_vector) < self.eps:
            self._fix_triangulation(next_hull_pt, hull_pt, i)
            hull_pt = next_hull_pt
            last_vector = new_vector
            next_hull_pt = self.convex_hull[hull_pt].left
            new_vector = self.points[next_hull_pt] - self.points[i]
        self.convex_hull[i].left = hull_pt

        self.convex_hull[self.convex_hull[i].right].left = i
        self.convex_hull[hull_pt].right = i

    def _fix_triangulation(self, left: int, right: int, outer: int):
        """
        Перестраивает треугольники, не удовлетворяющие условию Делоне
        :param left: смежная с outer точка слева
        :param right: смежная с outer точка справа
        :param outer: добавляемая точка
        """
        self.stack[0] = Edge(left, right)
        stack_size = 1
        while stack_size > 0:
            last_edge = self.stack[stack_size - 1]
            left = last_edge.v1
            right = last_edge.v2
            stack_size -= 1

            # Находим inner из ребра (left, right)
            inner = self.graph[Edge(min(left, right), max(left, right))].min()

            # Если условие Делоне выполнено, то перестраивать треугольники не нужно,
            # просто добавляем недостающие ребра и выходим
            if self._check_delaunay_condition(left, right, outer, inner):
                self.graph[Edge(right, outer)].insert(left)

                self.graph[Edge(left, outer)].insert(right)
                if right < left:
                    right, left = left, right
                self.graph[Edge(left, right)].insert(outer)
                continue

            # Иначе перестраиваем триангуляцию в четырехугольнике
            self.graph[Edge(right, outer)].replace(left, inner)
            self.graph[Edge(left, outer)].replace(right, inner)
            self.graph[Edge(min(inner, left), max(inner, left))].replace(right, outer)
            self.graph[Edge(min(inner, right), max(inner, right))].replace(left, outer)

            del self.graph[Edge(min(left, right), max(left, right))]

            # И добавляем 2 новых рекурсивных вызова
            self.stack[stack_size] = Edge(left, inner)
            stack_size += 1
            self.stack[stack_size] = Edge(inner, right)
            stack_size += 1

    def _check_delaunay_condition(self, left: int, right: int, outer: int, inner: int):
        """
        Проверяет условие Делоне
        :param left:
        :param right:
        :param outer:
        :param inner:
        """
        l = self.points[left]
        r = self.points[right]
        t = self.points[outer]
        b = self.points[inner]

        if outer == inner:
            return True

        # Проверка на выпуклость
        if np.cross(l - t, b - t) < 0 or np.cross(r - t, b - t) > 0:
            return True

        # Модифицированная проверка суммы противолежащих углов
        s_a = (t[0] - r[0]) * (t[0] - l[0]) + (t[1] - r[1]) * (t[1] - l[1])
        s_b = (b[0] - r[0]) * (b[0] - l[0]) + (b[1] - r[1]) * (b[1] - l[1])

        # В этом случае условие выполняется
        if s_a > -self.eps and s_b > -self.eps:
            return True

        # В этом случае нужны полные вычисления по формуле
        if s_a >= 0 or s_b >= 0:
            s_c = abs(np.cross(t - r, t - l))
            s_d = abs(np.cross(b - r, b - l))
            if s_c * s_b + s_a * s_d > -self.eps:
                return True

        return False

    def draw_triangulation(self):
        """Отображает построенную триангуляцию"""
        # Вершины построенного графа
        vertices = [e.get_vertices() for e in self.graph.keys()]
        # Отображает ребра построенного графа
        for v1, v2 in vertices:
            coords = np.vstack((self.points[v1], self.points[v2]))
            plt.plot(coords[:, 0], coords[:, 1], color='royalblue')

        # Отображает исходные точки
        plt.plot(self.points[:, 0], self.points[:, 1], 'o', color='deeppink')
        plt.show()

    def draw_triangulation_scipy(self):
        """Отображает триангуляцию, построенную с помощью функции из scipy, на заданном наборе точек"""
        tri = Delaunay(self.points)
        plt.triplot(self.points[:, 0], self.points[:, 1], tri.simplices.copy())
        plt.plot(self.points[:, 0], self.points[:, 1], 'o')
        plt.show()
