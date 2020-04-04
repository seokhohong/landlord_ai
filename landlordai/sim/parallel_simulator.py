import multiprocessing

from landlordai.game.player import LearningPlayer
from landlordai.sim.simulate import Simulator


def run_simulator(player_pool):
    print('launch Task')
    sim = Simulator(1, player_pool)
    sim.play_rounds(debug=True)
    print('Done')
    return sim.get_sparse_game_data()


def run_parallel():
    player_pool = []
    for i in range(10):
        player_pool.append(LearningPlayer(name='random'))

    p = multiprocessing.Pool(2)
    print('Launch Map')
    results = p.map(run_simulator, [player_pool] * 2)
    print(results)


if __name__ == '__main__':
    run_parallel()

