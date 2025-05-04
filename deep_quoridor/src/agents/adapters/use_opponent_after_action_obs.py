from agents.adapters.base import BaseTrainableAgentAdapter


class UseOpponentsAfterActionObsAdapter(BaseTrainableAgentAdapter):
    def handle_opponent_step_outcome(
        self,
        opponent_observation_before_action,
        my_observation_after_opponent_action,
        opponent_observation_after_action,
        opponent_reward,
        opponent_action,
        done,
    ):
        return self._agent().handle_opponent_step_outcome(
            opponent_observation_before_action,
            my_observation_after_opponent_action,
            opponent_observation_after_action=my_observation_after_opponent_action,
            opponent_reward=opponent_reward,
            opponent_action=opponent_action,
            done=done,
        )

    def handle_step_outcome(
        self,
        observation_before_action,
        opponent_observation_after_action,
        observation_after_action,
        reward,
        action,
        done,
    ):
        return self._agent().handle_step_outcome(
            observation_before_action,
            opponent_observation_after_action,
            observation_after_action=opponent_observation_after_action,
            reward=reward,
            action=action,
            done=done,
        )
