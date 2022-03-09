import numpy as np
import matplotlib.pyplot as plt
import random
import time


class Map():
    def __init__(self, num_mine, num_hero):
        self.num_mine = num_mine
        self.num_hero = num_hero
        self.mines = []
        self.heros = []
        self.init_map()

    def init_map(self):
        for i in range(self.num_mine):
            x = random.random()
            y = random.random()
            self.mines.append([x, y])
        for j in range(self.num_hero):
            x1 = random.random()
            y1 = random.random()
            self.heros.append([x1, y1])

    def hero_move(self):
        # 重新计算质心点
        for center in self.result_c:
            self.heros[center] = np.average(self.result_c[center], axis=0)

    def mine_align(self):
        # 当前计算当前质心点所属的分类点有那些
        self.result_c = {}
        for i in range(self.num_hero):
            self.result_c[i] = []
        for it in self.mines:
            distance = []
            for center in self.heros:
                distance.append(np.linalg.norm(np.array(it) - np.array(center)))
            classification = distance.index(min(distance))
            self.result_c[classification].append(it)

    def map_visualization(self):
        tmp = np.array(self.mines)
        X = tmp[:, 0]
        Y = tmp[:, 1]
        plt.plot(X, Y, 'b*')
        tmp = np.array(self.heros)
        X = tmp[:, 0]
        Y = tmp[:, 1]
        plt.plot(X, Y, 'ro')

        plt.show()


def main():
    map = Map(num_mine=100, num_hero=3)
    while True:
        map.mine_align()
        map.hero_move()
        map.map_visualization()
        time.sleep(3)


if __name__ == '__main__':
    main()
