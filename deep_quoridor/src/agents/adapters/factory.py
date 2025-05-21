def create_agent_with_adapters(adapter_factories, agent_factory, observation_space, action_space, **kwargs):
    new_action_space = action_space
    new_observation_space = observation_space

    for factory in reversed(adapter_factories):
        new_observation_space = factory.get_observation_space(new_observation_space)
        new_action_space = factory.get_action_space(new_action_space)

    agent = agent_factory(
        observation_space=new_observation_space,
        action_space=new_action_space,
        **kwargs,
    )
    for factory in adapter_factories:
        agent = factory(agent, **kwargs)

    return agent
