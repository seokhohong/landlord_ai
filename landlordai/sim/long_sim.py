from landlordai.game.player import LearningPlayer_v1
from landlordai.sim.simulate import Simulator

if __name__ == "__main__":
    for i in range(100):
        players = [LearningPlayer_v1(name='random', use_montecarlo_random=True, random_mc_num_explorations=30) for _ in range(5)]
        simulator = Simulator(10, players)
        simulator.play_rounds()