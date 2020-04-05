from landlordai.game.landlord import LandlordGame
from landlordai.game.move import KittyReveal
from landlordai.game.player import LearningPlayer, TurnPosition, HumanPlayer

ref_net = '4_4_consensus1_model5'
reference_player = LearningPlayer(name=ref_net, net_dir='../models/' + ref_net, estimation_mode=LearningPlayer.CONSENSUS_Q)

def load_net(net):
    return LearningPlayer(name=net, net_dir='../models/' + net, estimation_mode=LearningPlayer.CONSENSUS_Q)


def play_against_two(players):
    game = LandlordGame(players)
    while not game.is_round_over():
        current_player = game.get_current_player()
        current_position = game.get_current_position()

        best_move, best_move_q = current_player.decide_best_move(game)

        print(current_player.get_name(), "(" + game.get_position_role_name(current_position) + ", " \
              + str(len(game.get_hand(current_position))) + "):", best_move, '(' + str(best_move_q) + ')')
        game.play_move(best_move)

        if type(game.get_last_played()) == KittyReveal:
            print(game.get_last_played())

    if game.has_winners():
        for winner in game.get_winners():
            print('WINNERS:', game.get_ai_players()[winner].get_name(), game.get_scores()[winner])

if __name__ == "__main__":
    play_against_two([load_net('4_2_sim4_model15'), load_net('4_4_consensus1_model5'),
                      HumanPlayer(name='human', reference_player=reference_player)])
