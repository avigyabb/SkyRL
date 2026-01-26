AGENT_GENERATOR_REGISTRY = {
    "skyrl_agent.agents.oh_codeact.OHCodeActAgent": "skyrl_agent.agents.base.AgentRunner",
    "skyrl_agent.agents.react.ReActAgent": "skyrl_agent.agents.base.AgentRunner",
    "skyrl_agent.agents.biomni_codeact.BiomniCodeActAgent": "skyrl_agent.agents.base.AgentRunner",
    "skyrl_agent.agents.biomni_codeact.BiomniCodeActRubricAgent": "skyrl_agent.agents.base.AgentRunner",
}

AGENT_TRAJECTORY_REGISTRY = {
    "skyrl_agent.agents.oh_codeact.OHCodeActAgent": "skyrl_agent.agents.oh_codeact.CodeActTrajectory",
    "skyrl_agent.agents.react.ReActAgent": "skyrl_agent.agents.react.ReActTrajectory",
    "skyrl_agent.agents.biomni_codeact.BiomniCodeActAgent": "skyrl_agent.agents.biomni_codeact.biomni_codeact_runner.BiomniCodeActTrajectory",
    "skyrl_agent.agents.biomni_codeact.BiomniCodeActRubricAgent": "skyrl_agent.agents.biomni_codeact.biomni_codeact_rubric_runner.BiomniCodeActRubricTrajectory",
}
