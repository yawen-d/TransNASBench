import time


class Logger(object):
    def __init__(self, path, mode='w+'):
        self.log_file = open(path, mode)
        self.write(f"logger initiated!")
        self.write(f"logger_path {path}")

    def write(self, message):
        self.log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M')}] " + str(message) + '\n')
        print(f"[{time.strftime('%Y-%m-%d %H:%M')}] " + str(message))
        self.log_file.flush()

    def close(self):
        self.log_file.close()
