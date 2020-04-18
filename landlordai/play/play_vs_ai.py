from landlordai.game.landlord import LandlordGame
from landlordai.game.move import KittyReveal
from landlordai.game.player import LearningPlayer, HumanPlayer, LearningPlayer_v2

ref_net = '4_13_stream2_model2_134'
reference_player = LearningPlayer_v2(name=ref_net, net_dir='../stream_models/' + ref_net, estimation_mode=LearningPlayer.ACTUAL_Q)

def load_net(net):
    return LearningPlayer(name=net, net_dir='../models/' + net, estimation_mode=LearningPlayer.ACTUAL_Q)

def load_v2_net(net, models_dir='../stream_models/'):
    return LearningPlayer_v2(name=net, net_dir=models_dir + net,
                          estimation_mode=LearningPlayer.ACTUAL_Q,
                          estimation_depth=7,
                          discount_factor=1,
                          epsilon=0,
                          learning_rate=0.3)

def play_against_two(players, show_q=True):
    game = LandlordGame(players)
    while not game.is_round_over():
        current_player = game.get_current_player()
        current_position = game.get_current_position()

        best_move, best_move_q = current_player.decide_best_move(game)

        if show_q:
            best_move_q_str = '(' + str(best_move_q) + ')'
        else:
            best_move_q_str = ''

        print(current_player.get_name(), "(" + game.get_position_role_name(current_position) + ", " \
                  + str(len(game.get_hand(current_position))) + "):", best_move, best_move_q_str)
        game.play_move(best_move)

        if type(game.get_last_played()) == KittyReveal:
            print(game.get_last_played())

    if game.has_winners():
        for winner in game.get_winners():
            print('WINNERS:', game.get_ai_players()[winner].get_name())

if __name__ == "__main__":
    play_against_two([load_v2_net('4_13_stream2_model2_134', '../stream_models/'),
                      load_v2_net('4_13_stream2_model3_119', '../stream_models/'),
                      HumanPlayer(name='human', reference_player=reference_player, known_hand=True, ai_before=False)],
                     show_q=False)
