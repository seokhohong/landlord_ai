from landlordai.game.landlord import LandlordGame
from landlordai.game.move import KittyReveal
from landlordai.game.player import TurnPosition, HumanPlayer, LearningPlayer_v2, LearningPlayer

ref_net = '4_13_stream2_model2_134'
reference_player = LearningPlayer_v2(name=ref_net, net_dir='../stream_models/' + ref_net, estimation_mode=LearningPlayer.ACTUAL_Q)

def parse_cardlist(statement):
    print(statement)
    inp = input(">").strip()
    return HumanPlayer.parse_input_for_cardset(inp)

def manual_kitty():
    return parse_cardlist("Please enter the Cards in the Kitty")

def manual_hand():
    return parse_cardlist("Please enter the Cards in your hand")

def get_first_player(game):
    while True:
        first_player = input("First Player Name >").strip()
        first_turn = None
        for turn in list(TurnPosition):
            if game.get_ai_players()[turn].get_name() == first_player:
                first_turn = turn

        if first_turn is not None:
            return first_turn

def perspective_position(game, perspective):
    for turn in list(TurnPosition):
        if game.get_ai_players()[turn].get_name() == perspective:
            return turn
    assert False


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

    first_player = get_first_player(game)

    game.force_current_position(first_player)
    game.force_hand(perspective_position(game, perspective), perspective_hand)

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
    player_names = ['Clare', 'Seokho', 'Jon']
    human_game(player_names, 'Seokho')
