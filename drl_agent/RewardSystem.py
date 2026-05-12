
class Reward:
    def __init__(self):
        self.goal_reached = 0
        self.out_of_bound = 0
        self.connection_lost = 0

    def reset(self):
        self.goal_reached = 0
        self.out_of_bound = 0
        self.connection_lost = 0

    def set_goal_reached(self):
        self.goal_reached = 1

    def set_out_of_bound(self):
        self.out_of_bound = 1

    def set_connection_lost(self):
        self.connection_lost = 1