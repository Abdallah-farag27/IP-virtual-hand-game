import cv2
import time
from column_mapper import ColumnMapper
from renderer import Renderer
from game_timer import GameTimer

# TODO: Import teammate modules when ready
# from preprocess import preprocess
# from segment_hand import segment_hand
# from get_hand_centroid import get_hand_centroid
# from update_balls import update_balls

class VirtualHandGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Tony's modules
        self.mapper = ColumnMapper(num_cols=4)
        self.renderer = Renderer(width=1920, height=1080)
        self.timer = GameTimer(total_time=90)
        
        # Game state
        self.score = 0
        self.lives = 3
        self.balls = []
        self.basket_col = 0
        self.game_over = False
    
    def game_loop(self):
        self.timer.start()
        
        while True:
            # 1. Capture frame (Wafa)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1920, 1080))
            
            if not self.game_over:
                # 2. Preprocess (Hassan)
                # preprocessed = preprocess(frame)
                
                # 3. Segment hand (Hassan)
                # mask = segment_hand(preprocessed)
                
                # 4. Get centroid (Wafa)
                # centroid = get_hand_centroid(mask)
                
                centroid = {'x': 480, 'y': 360}  # TODO: Replace
                
                # 5. Map to column (Tony)
                self.basket_col = self.mapper.map_to_column(centroid['x'], 1920)
                
                # 6. Update balls (Fararg)
                # update_balls(self.balls, self.basket_col)
                
                # Check game over
                if self.timer.is_over() or self.lives <= 0:
                    self.game_over = True
                
                # 7. Render (Tony)
                self.renderer.render_ui(frame, self.balls, self.basket_col,
                                       self.score, self.lives, self.timer.time_left())
            else:
                self.renderer.draw_game_over(frame, self.score)
            
            cv2.imshow('Virtual Hand Basketball Game', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.game_over:
                self.restart()
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def restart(self):
        self.score = 0
        self.lives = 3
        self.balls = []
        self.game_over = False
        self.timer.reset()
        self.timer.start()
        self.mapper.reset()

if __name__ == '__main__':
    game = VirtualHandGame()
    game.game_loop()