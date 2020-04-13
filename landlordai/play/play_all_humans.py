from landlordai.game.landlord import LandlordGame
from landlordai.game.move import KittyReveal
from landlordai.game.player import LearningPlayer, TurnPosition, HumanPlayer

ref_net = '4_10_actualq1_model19'
reference_player = LearningPlayer(name=ref_net, net_dir='../models/' + ref_net, estimation_mode=LearningPlayer.ACTUAL_Q)

def load_net(net):
    return LearningPlayer(name=net, net_dir='../models/' + net, estimation_mode=LearningPlayer.ACTUAL_Q)

def parse_cardlist(statement):
    print(statement)
    inp = input(">").strip()
    return HumanPlayer.parse_input_for_cardset(inp)

def manual_kitty():
    return parse_cardlist("Please enter the Cards in the Kitty")

def manual_hand():
    return parse_cardlist("Please enter the Cards in your hand")

def human_game(player_names, perspective):
    perspective_hand = None
    players = []
    for player_name in player_names:
        if player_name == perspective:
            perspective_hand = manual_hand()
        player_is_perspective = (player_name == perspective)
        players.append(HumanPlayer(name=player_name,
                                   reference_player=reference_player,
                                   known_hand=player_is_perspective,
                                   ai_before=player_is_perspective))

    game = LandlordGame(players, kitty_callback=manual_kitty)
    game.force_current_position(TurnPosition.FIRST)
    game.force_hand(TurnPosition.FIRST, perspective_hand)

    while not game.is_round_over():
        current_player = game.get_current_player()
        current_position = game.get_current_position()

        best_move, best_move_q = current_player.decide_best_move(game)

        print(current_player.get_name(), "(" + game.get_position_role_name(current_position) + ", " \
              + str(len(game.get_hand(current_position))) + "):", best_move, '(' + str(best_move_q) + ')')

        # play with known hand if it matches perspective
        game.play_move(best_move, hand_known=current_player.get_name() == perspective)

        if type(game.get_last_played()) == KittyReveal:
            print(game.get_last_played())

    if game.has_winners():
        for winner in game.get_winners():
            print('WINNERS:', game.get_ai_players()[winner].get_name())

if __name__ == "__main__":
    player_names = ['Seokho', 'Clare', 'Jon']
    perspective = player_names[0]
    human_game(player_names, perspective)
