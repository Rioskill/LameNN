import tkinter as tk
import numpy as np


class Paint():
    def __init__(self, func, vals=None):
        self.func = func

        self.window = tk.Tk()
        self.sizeX = 560
        self.sizeY = 560
        self.canvas = tk.Canvas(self.window, width=self.sizeX,
                                height=self.sizeY, highlightthickness=0)
        # Set canvas background color so you can see it
        self.canvas.config(bg="white")
        self.canvas.pack()

        if vals is None:
            self.vals = np.zeros(784)
        else:
            self.vals = vals
        self.rects = list()

        for i, val in enumerate(self.vals):
            y = int(i // 28) * 20
            x = (i % 28) * 20

            fill = "white" if self.vals[i] == 0 else "black"
            self.rects.append(self.canvas.create_rectangle(x, y, x + 20, y + 20, fill=fill))

        self.canvas.bind("<B1-Motion>", self.color_in)
        self.canvas.bind("<Button-3>", self.export_data)

    def color_in(self, event):
        x, y = event.x, event.y

        x = int(x // 20)
        y = int(y // 20)

        # self.img.put("black", to=(x-2, y-2, x+2, y+2))

        size = 1
        for i in range(x - size, x + size):
            for j in range(y - size, y + size):

                self.vals[j * 28 + i] = 1
                self.canvas.itemconfigure(self.rects[j * 28 + i], fill="black")

    def export_data(self, event):
        data = self.vals
        self.vals = np.zeros(784)
        for i in range(len(self.rects)):
            self.canvas.itemconfigure(self.rects[i], fill="white")
        self.func(data)


if __name__ == '__main__':
    file = open('./mnist_test/mnist_train.csv')
    line = file.readlines()[26]
    file.close()
    paint = Paint(print, vals=np.array(list(map(int, line.split(',')))))
    paint.window.mainloop()
