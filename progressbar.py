import sys
import time


class ProgressBar:
    def __init__(self, real_width, data_width):
        self.width = real_width
        self.data_batch_width = data_width // real_width
        self.it = 0

    def begin(self):
        self.it = 0
        sys.stdout.write("[%s]" % (" " * self.width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.width + 1))

    def update(self):
        if self.it % self.data_batch_width == 0:
            sys.stdout.write("-")

            number_of_lines = int(self.it // self.data_batch_width)
            number_of_spaces = self.width - number_of_lines - 1

            counter = '[' + str(self.it + self.data_batch_width) + '/' + str(self.width * self.data_batch_width) + ']'

            sys.stdout.write(" " * number_of_spaces + ']' + counter)
            sys.stdout.write("\b" * (self.width - number_of_lines + len(counter)))
            sys.stdout.flush()
        self.it += 1

    def close(self):
        sys.stdout.write("]\n")  # this ends the progress bar


if __name__ == '__main__':
    bar = ProgressBar(20, 100)
    bar.begin()

    for i in range(100):
        bar.update()
        time.sleep(0.01)
    bar.close()
