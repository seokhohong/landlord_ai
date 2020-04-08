import unittest
from copy import copy

import numpy as np

from landlordai.game.card import Card
from landlordai.game.landlord import LandlordGame
from landlordai.game.move import BetMove, MoveType, RankedMoveType, SpecificMove
from landlordai.game.player import LearningPlayer, TurnPosition, HumanPlayer, TypoError, InvalidMoveError
from collections import Counter

class TestLandlordMethods(unittest.TestCase):
    def test_player_move(self):
        players = [LearningPlayer(name='random')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4 + [Card.FOUR] * 4 + [Card.SIX] * 4 + [Card.TWO] * 4 + [Card.THREE] * 2 + [Card.LITTLE_JOKER] + [Card.BIG_JOKER]
        }
        game.betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 3)
        game2 = copy(game)
        game.play_move(SpecificMove(RankedMoveType(MoveType.BOMB, Card.FIVE), Counter({Card.FIVE: 4})))
        self.assertNotEqual(game2.get_hand(TurnPosition.THIRD), game.get_hand(TurnPosition.THIRD))

    def test_player_game(self):
        players = [LearningPlayer(name='random', estimation_mode=LearningPlayer.ACTUAL_Q)] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4 + [Card.FOUR] * 4 + [Card.SIX] * 4 + [Card.TWO] * 4 + [Card.THREE] * 2 + [Card.LITTLE_JOKER] + [Card.BIG_JOKER]
        }
        game.betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 3)
        game.main_game()
        players[0].compute_future_q(game)
        self.assertTrue(np.sum(np.abs(game.get_scores())) > 0)
        # game is over
        self.assertTrue(np.abs(players[0]._record_future_q[-1]) > 0.5)

        features = players[0].derive_features(game)
        self.assertTrue(np.sum(features[:, players[0].get_feature_index('I_AM_LANDLORD')]) != 0)
        # it is possible this guy never plays, eventually
        self.assertTrue(np.sum(features[:, players[0].get_feature_index('I_AM_BEFORE_LANDLORD')]) != 0)

    def test_full_game(self):
        players = [LearningPlayer(name='random') for _ in range(3)]
        game = LandlordGame(players=players)
        game.play_round()

        while np.sum(np.abs(game.get_scores())) == 0:
            players = [LearningPlayer(name='random') for _ in range(3)]
            game = LandlordGame(players=players)
            game.play_round()

        # game is over
        for i in range(3):
            #print(players[i].record_future_q[-1])
            #self.assertTrue(np.abs(players[i].record_future_q[-1]) > 0.5)

            features = players[i].derive_features(game)
            self.assertTrue(np.sum(features[:, players[i].get_feature_index('I_AM_LANDLORD')]) != 0)
            # it is possible this guy never plays, eventually
            self.assertTrue(np.sum(features[:, players[i].get_feature_index('I_AM_BEFORE_LANDLORD')]) != 0)

    def test_llord_winning(self):
        players = [LearningPlayer(name='random')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4 + [Card.KING] * 4 + [Card.QUEEN] * 4 + [Card.JACK] * 4 + [Card.THREE],
            TurnPosition.SECOND: [Card.TEN] * 4 + [Card.NINE] * 4 + [Card.EIGHT] * 4 + [Card.SEVEN] * 4 + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 4
        }
        game.betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 3)
        game.main_game()
        self.assertTrue(TurnPosition.THIRD in game.get_winners())
        self.assertTrue(len(game.get_move_logs()) == 1)

    def test_peasant_winning(self):
        players = [LearningPlayer(name='random')] * 3
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4,
            TurnPosition.SECOND: [Card.TEN] + [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 3 + [Card.THREE] + [Card.FOUR]
        }
        game.betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 3)
        hand_vector = players[0].get_hand_vector(game, TurnPosition.FIRST)
        self.assertTrue(hand_vector[11] == 4)
        self.assertTrue(hand_vector[-2] == 2)
        self.assertTrue(hand_vector[-3] == 5)
        self.assertTrue(hand_vector[-1] == 4)
        #self.assertTrue(np.sum(hand_vector) == 4)
        game.main_game()
        self.assertTrue(TurnPosition.THIRD not in game.get_winners())
        self.assertTrue(TurnPosition.SECOND in game.get_winners())
        self.assertTrue(TurnPosition.FIRST in game.get_winners())
        self.assertTrue(len(game.get_move_logs()) == 2)

    def test_best_montecarlo(self):
        players = [LearningPlayer(name='random', estimation_mode=LearningPlayer.MONTECARLO_RANDOM)] * 3
        game = LandlordGame(players=players)
        game.play_round(debug=False)

    def test_features(self):
        players = [LearningPlayer(name='random', estimation_mode=LearningPlayer.ACTUAL_Q) for _ in range(3)]
        game = LandlordGame(players=players)
        while not game.is_round_over():
            curr_player = game.get_current_player()
            curr_features = curr_player.derive_features_bridge(game, game.get_current_position())
            curr_hand_vector = game.get_current_player().get_hand_vector(game, game.get_current_position())
            move = game.get_current_player().make_move(game, game.get_current_position())
            curr_move_vector = game.get_current_player().compute_move_vector(game.get_current_position(),
                                                                             game.get_landlord_position(), move)

            game.play_move(move)

            self.assertTrue(np.allclose(curr_features, curr_player.record_history_matrices[-1]))
            self.assertTrue(np.allclose(curr_move_vector, curr_player.record_move_vectors[-1]))
            self.assertTrue(np.allclose(curr_hand_vector, curr_player.record_hand_vectors[-1]))

    def test_features_bridge(self):
        players = [LearningPlayer(name='random', estimation_mode=LearningPlayer.MONTECARLO_RANDOM) for _ in range(3)]
        game = LandlordGame(players=players)

        while not game.is_round_over():
            curr_player = game.get_current_player()
            curr_features = curr_player.derive_features_bridge(game, game.get_current_position())
            curr_hand_vector = game.get_current_player().get_hand_vector(game, game.get_current_position())

            move = game.get_current_player().make_move(game, game.get_current_position())

            game.play_move(move)

            self.assertTrue(np.allclose(curr_features[0][:LearningPlayer.HAND_FEATURES], curr_hand_vector))

    # checks that the recorded q for event replay is within expected bounds
    '''
    def test_estimation(self):
        def load_best_sim_net(net):
            return LearningPlayer(name=net, net_dir='../models/' + net,
                                  estimation_mode=LearningPlayer.BEST_SIMULATION,
                                  estimation_depth=4,
                                  epsilon=0,
                                  discount_factor=1)

        players = [load_best_sim_net('4_1_sim3_model1') for i in range(3)]

        game = LandlordGame(players=players)
        while not game.is_round_over():
            curr_player = game.get_current_player()

            best_move, best_move_q = curr_player.decide_best_move(game)
            curr_player.make_move(game)

            copy_game = copy(game)
            if copy_game.move_ends_game(best_move):
                break

            copy_game.play_move(best_move)

            next_best_move, next_best_move_q = curr_player.decide_best_move(copy_game)

            recorded_q = curr_player._record_future_q[-1]

            if next_best_move_q > best_move_q:
                self.assertTrue(next_best_move_q > recorded_q > best_move_q)
            else:
                self.assertTrue(next_best_move_q < recorded_q < best_move_q)

            game.play_move(best_move)
    '''
    def test_record_actual_q(self):
        def load_best_sim_net(net):
            return LearningPlayer(name=net, net_dir='../models/' + net,
                                  estimation_mode=LearningPlayer.ACTUAL_Q,
                                  epsilon=0,
                                  discount_factor=1)

        players = [load_best_sim_net('4_2_sim4_model10') for i in range(3)]
        player_0_scores = []
        game = LandlordGame(players=players)
        while not game.is_round_over():
            curr_player = game.get_current_player()

            best_move, best_move_q = curr_player.decide_best_move(game)
            if curr_player == players[0]:
                player_0_scores.append(best_move_q)

            curr_player.make_move(game)

            game.play_move(best_move)

        for player in players:
            player.compute_future_q(game)

        record_state = players[0]._record_state_q
        future_q = players[0].get_estimated_q()
        # assert in bounds based on update function
        for i, val in enumerate(record_state):
            if i != len(record_state) - 1:
                if record_state[i + 1] < record_state[i]:
                    self.assertTrue(record_state[i + 1] < future_q[i] < record_state[i])
                elif record_state[i + 1] > record_state[i]:
                    self.assertTrue(record_state[i + 1] > future_q[i] > record_state[i])

        self.assertEqual(len(players[0].get_record_hand_vectors()), len(players[0].get_estimated_q()))
    '''
    def test_record_bomb_usage(self):
        def load_net(net):
            return LearningPlayer_v1(name=net, net_dir='../models/' + net,
                                     use_montecarlo_random=False,
                                     mc_best_move_depth=1,
                                     epsilon=0,
                                     discount_factor=1)

        players = [load_net('4_1_sim3_model1') for i in range(3)]
        game = LandlordGame(players=players)
        hands = {
            TurnPosition.FIRST: [Card.ACE] * 4,
            TurnPosition.SECOND: [Card.THREE],
            TurnPosition.THIRD: [Card.FIVE] * 3 + [Card.THREE] + [Card.FOUR]
        }
        game.betting_complete = True
        game.force_setup(TurnPosition.THIRD, hands, 3)
        game.main_game()
        self.assertTrue(players[0].get_future_q()[0] < -4)
    '''

    def test_human(self):
        human = HumanPlayer(name='human')
        self.assertTrue(human.parse_input('nine nine nine six six'), SpecificMove(RankedMoveType(MoveType.TRIPLE_PAIR_KICKER, Card.NINE), cards=Counter({Card.NINE: 3, Card.SIX: 2})))
        self.assertRaises(TypoError, human.parse_input, 'ni')
        self.assertRaises(InvalidMoveError, human.parse_input, 'nine 10')


if __name__ == '__main__':
    unittest.main()

