class ColumnMapper:
    def __init__(self, num_cols=3, smooth=False, max_jump=1):
        self.num_cols = num_cols
        self.smooth = smooth
        self.max_jump = max_jump
        self.prev_col = 0
    
    def map_to_column(self, x, frame_width):
        # Handle None input
        if x is None:
            return self.prev_col
        
        # Calculate which column x belongs to
        col_width = frame_width / self.num_cols
        col = int(x / col_width)
        col = max(0, min(self.num_cols - 1, col))
        
        # Smooth movement to prevent jitter
        if self.smooth:
            jump = col - self.prev_col
            if abs(jump) > self.max_jump:
                col = self.prev_col + (self.max_jump if jump > 0 else -self.max_jump)
        
        self.prev_col = col
        return col
    
    def reset(self):
        self.prev_col = 0