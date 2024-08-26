import pygame
import numpy as np
import random
import math
import collections

'''
Explaining some terms 
1. Reward: this is the immediate reward after taking an action in some state
2. self.gamma is the discount factor which determines the importance of future rewards compared to immediate rewards. so close to 1 means future reward is high, otherwise immediate reward is prioritized
3. self.alpha is the learning rate which controls how much the q-value is updated based on the new info. 1 means we fully replace the q-value and 0 means no update.
'''

# initialize pygame
pygame.init()

# define constants
WIDTH, HEIGHT = 800, 800
CAR_WIDTH, CAR_HEIGHT = 7, 10  
MAZE_SIZE = 40  
CELL_SIZE = WIDTH // MAZE_SIZE
FPS = 60
NUM_PARKED_CARS = 5

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

# set up the screen and window title
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Car Simulation")
# start the timer
clock = pygame.time.Clock()

# maze class
class Maze:
    def __init__(self):
        # 2d array representing the maze where 0 is a wall and 1 is a path
        self.grid = [[0] * MAZE_SIZE for _ in range(MAZE_SIZE)]
        # 2d array to track which cells have been visited during the maze generation
        self.visited = [[False] * MAZE_SIZE for _ in range(MAZE_SIZE)]
        self.generate_maze()
        # bottom left corner as starting point
        self.start = (1, 1)  
        # define end point as the top right corner ish
        self.end = (MAZE_SIZE - 2, MAZE_SIZE - 2)  

    # depth first search!!
    def generate_maze(self):
        # start from one cell and then explore the neighbors
        stack = [(1, 1)]
        # visited edges
        self.visited[1][1] = True
        while stack:
            x, y = stack[-1]
            # lets explore the neighbors!
            neighbors = []
            if x > 1 and not self.visited[x - 2][y]:  
                neighbors.append((x - 2, y))
            if x < MAZE_SIZE - 2 and not self.visited[x + 2][y]: 
                neighbors.append((x + 2, y))
            if y > 1 and not self.visited[x][y - 2]: 
                neighbors.append((x, y - 2))
            if y < MAZE_SIZE - 2 and not self.visited[x][y + 2]:  
                neighbors.append((x, y + 2))
            if neighbors:
                # if a neighbor has not been visited, we add it to the stack and mark it as visited
                # if there are neighbors, we randomly select one of them -- let nx and ny be these coordinates of the neighbor
                nx, ny = random.choice(neighbors)
                # mark this node as visited
                self.visited[nx][ny] = True
                # if the neighbor is two left up, removes the wall directly right of it
                if nx == x - 2:
                    self.grid[nx + 1][ny] = 1
                # two cells right -> remove wall to the left of it
                elif nx == x + 2:
                    self.grid[nx - 1][ny] = 1
                # two cells up  -> remove wall directly below
                elif ny == y - 2:
                    self.grid[nx][ny + 1] = 1
                # two cells down, remove directly above it
                elif ny == y + 2:
                    self.grid[nx][ny - 1] = 1
                # now mark this neighbor as a path so its part of the maze
                self.grid[nx][ny] = 1
                # add to the stack which allows for backtracking if we reach a dead end
                stack.append((nx, ny))
            else:
                # no valid neighbors left (all adjacent have been visited) we pop the last cell from the stack to explore other potential paths
                stack.pop()

    def draw(self):
        # draw a rectangle for each x, y pair in the grid
        for x in range(MAZE_SIZE):
            for y in range(MAZE_SIZE):
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                # if its a wall, make it black rectangle
                if self.grid[x][y] == 0:
                    pygame.draw.rect(screen, BLACK, rect)
                # else white for the path
                pygame.draw.rect(screen, WHITE, rect, 1)

    # checks if the position is a wall or out of bounds
    def is_wall(self, x, y):
        grid_x, grid_y = int(x // CELL_SIZE), int(y // CELL_SIZE)
        if grid_x < 0 or grid_x >= MAZE_SIZE or grid_y < 0 or grid_y >= MAZE_SIZE:
            return True
        # its a wall then
        return self.grid[grid_y][grid_x] == 0

    # check if we reached the goal position which is the top right (defined in maze class)
    def is_goal(self, x, y):
        grid_x, grid_y = int(x // CELL_SIZE), int(y // CELL_SIZE)
        return (grid_x, grid_y) == self.end

# define the car
class Car:
    def __init__(self, x, y, angle, is_player=True):
        self.x = x
        self.y = y
        # speed and angle the car is facing
        self.speed = 2
        self.angle = angle
        # car is not fixed then its a player car
        self.is_player = is_player
        # make it red -- no other cars for now
        self.color = RED if is_player else GREEN

    # update car position based on angle and speed
    def move(self, action=None):
        if self.is_player and action is not None:
            # car turns left
            if action == 0: 
                self.angle -= 5
            # car turns right
            elif action == 2: 
                self.angle += 5

        # calculate the coordinates based on the speed and the facing angle 
        if self.is_player:
            self.x += self.speed * math.sin(math.radians(self.angle))
            self.y -= self.speed * math.cos(math.radians(self.angle))

    # draw the car
    def draw(self):
        # rectangle car
        rotated_car = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
        pygame.draw.rect(rotated_car, self.color, (0, 0, CAR_WIDTH, CAR_HEIGHT))
        rotated_car = pygame.transform.rotate(rotated_car, self.angle)
        screen.blit(rotated_car, (self.x - rotated_car.get_width() // 2, self.y - rotated_car.get_height() // 2))

# agents class
class QLearningAgent:
    def __init__(self, state_size, action_size):
        # num possible states
        self.state_size = state_size
        # num possible actions
        self.action_size = action_size
        # stores q values for state-action pairs
        self.q_table = np.zeros((state_size, action_size))
        # exploration rate -- need enough exploration initially then it should get better at learning
        self.epsilon = 0.1
        # learning rate
        self.alpha = 0.1
        # discount factor
        self.gamma = 0.9

    # decide action based on epsilon-greedy policy
    def get_action(self, state):
        # if were still exploring
        if random.random() < self.epsilon:
            # choose a random action
            return random.randint(0, self.action_size - 1)
        else:
            # select the action with the highest q-value for the current state 
            # this is exploitation
            return np.argmax(self.q_table[state])

    # update function for q-value
    def update(self, state, action, reward, next_state):
        # get the current q value from the q table
        current_q = self.q_table[state, action]
        # calc the max q-value for the next state
            # this is the best expected future reward from this current state
        next_max_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state, action] = new_q

# convert car position to a state index based on its grid location in the maze
def get_state(car, maze):
    grid_x = int(car.x // CELL_SIZE)
    grid_y = int(car.y // CELL_SIZE)
    return grid_x + grid_y * MAZE_SIZE

def reset_game():
    global car, maze, score
    start_x, start_y = 1, MAZE_SIZE - 2  
    while maze.is_wall(start_x * CELL_SIZE + CELL_SIZE // 2, start_y * CELL_SIZE + CELL_SIZE // 2):
        # while were at a wall move to the next cell until a valid positon is found
        start_x += 1  
        # reset to beginning if out of bounds
        if start_x >= MAZE_SIZE - 1:  
            start_x = 1
            start_y -= 1
            # reset if out of bounds
            if start_y < 1:  
                start_y = MAZE_SIZE - 2
    car = Car(start_x * CELL_SIZE + CELL_SIZE // 2, start_y * CELL_SIZE + CELL_SIZE // 2, 0, is_player=True)
    score = 0

# initialize everything
high_score = 0
maze = Maze()

# state size based on maze grid 
# 3 actions -- left, right, forward
agent = QLearningAgent(MAZE_SIZE * MAZE_SIZE, 3)  
reset_game()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # increment score for each frame the game runs
    score += 1

    # get the state and action
    state = get_state(car, maze)
    action = agent.get_action(state)
    
    # move player car
    car.move(action)
    
    # if theres a wall we penalize
    if maze.is_wall(car.x, car.y):
        reward = -10
        reset_game()
    # if not a wall, increase the reward
    elif maze.is_goal(car.x, car.y):
        reward = 100
        score += 1
        reset_game()
    else:
        reward = 1
    
    # next state
    next_state = get_state(car, maze)

    # update the q table
    agent.update(state, action, reward, next_state)
    
    # update high score
    if score > high_score:
        high_score = score
    
    # draw everything
    screen.fill(WHITE)
    maze.draw()
    car.draw()
    
    # draw score/high score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, GREEN)
    high_score_text = font.render(f"High Score: {high_score}", True, GREEN)
    score_pos = (10, 10)
    high_score_pos = (10, 50)
    pygame.draw.rect(screen, BLACK, (score_pos[0] - 2, score_pos[1] - 2, score_text.get_width() + 4, score_text.get_height() + 4))
    pygame.draw.rect(screen, BLACK, (high_score_pos[0] - 2, high_score_pos[1] - 2, high_score_text.get_width() + 4, high_score_text.get_height() + 4))

    screen.blit(score_text, score_pos)
    screen.blit(high_score_text, high_score_pos)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()