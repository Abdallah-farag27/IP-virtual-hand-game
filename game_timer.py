import time

class GameTimer:
    def __init__(self, total_time=90):
        self.total_time = total_time
        self.start_time = None
        self.paused = False
        self.pause_time = 0
    
    def start(self):
        self.start_time = time.time()
        self.paused = False
    
    def time_left(self):
        if self.start_time is None:
            return self.total_time
        
        if self.paused:
            return max(0, self.total_time - self.pause_time)
        
        elapsed = time.time() - self.start_time
        return max(0, self.total_time - elapsed)
    
    def is_over(self):
        return self.time_left() <= 0
    
    def pause(self):
        if not self.paused and self.start_time is not None:
            self.pause_time = time.time() - self.start_time
            self.paused = True
    
    def resume(self):
        if self.paused:
            self.start_time = time.time() - self.pause_time
            self.paused = False
    
    def reset(self):
        self.start_time = None
        self.paused = False
        self.pause_time = 0