from typing import List


class Edge:
    """Класс описывает объект ребра графа"""

    def __init__(self, v1: int = -1, v2: int = -1):
        self.v1 = v1
        self.v2 = v2

    def __eq__(self, other):
        return self.v1 == other.v1 and self.v2 == other.v2

    def __hash__(self):
        v1_hash = hash(self.v1)
        v2_hash = hash(self.v2)
        v1_hash ^= v2_hash + 0x9e3779b9 + (v1_hash << 6) + (v2_hash >> 2)
        return v1_hash

    def __str__(self):
        return f'{self.v1}, {self.v2}'

    def get_vertices(self) -> List[int]:
        """
        Возвращает вершины ребра в виде списка
        """
        return [self.v1, self.v2]


class TwoVertices:
    """Класс описывает подобие ребра с вершинами - вспомогательный объект для построения сетки"""

    def __init__(self, v1: int = -1, v2: int = -1):
        self.v1 = v1
        self.v2 = v2

    def __str__(self):
        return f'{self.v1}, {self.v2}'

    def insert(self, v: int):
        """
        Задает новое значение вершины
        :param v: новое значение
        """
        if self.v1 == v or self.v2 == v:
            return
        if self.v1 == -1:
            self.v1 = v
        else:
            self.v2 = v

    def replace(self, u: int, v: int):
        """
        Заменяет значение вершины u на v
        :param u: начальное значение вершины, которое будет заменено
        :param v: новое значение вершины
        """
        if self.v1 == u:
            self.v1 = v
        elif self.v2 == u:
            self.v2 = v
        else:
            self.insert(v)

    def max(self) -> int:
        """
        Возвращает максимальное значение вершины
        """
        return max(self.v1, self.v2)

    def min(self) -> int:
        """
        Возвращает минимальное значение вершины
        """
        if self.v1 != -1 and self.v2 != -1:
            return min(self.v1, self.v2)
        return self.v1 if self.v1 != -1 else self.v2


class ListNode:
    """Класс описывает вспомогательный объект для построения сетки"""

    def __init__(self, left: int = 0, right: int = 0):
        self.left = left
        self.right = right

    def __str__(self):
        return f'{self.left}, {self.right}'
