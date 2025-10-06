import pandas as pd

legend_kwargs = {
    # 'bbox_to_anchor': (1.05, 1),
    'loc': 'best',
    'frameon': True,
    'framealpha': 0.95,
    'edgecolor': 'lightgray'
}
# eval_colors_list = ['#949494', '#DE8F05', '#029E73', '#D55E00', '#0173B2', '#CA9161', '#FBAFE4']
eval_colors_list = ['#FF69B4', '#DE8F05', '#029E73', '#D55E00', '#0173B2', '#CA9161', '#FBAFE4']

# Coloutblind codes
# #0173B2	#DE8F05	#029E73	#D55E00	#CC78BC	#CA9161	#FBAFE4	#949494	#ECE133	#56B4E9

run_name_mapping = {
    'FL_GRPO_0pt5B': 'Frozen Lake GRPO 0.5B',
    'FL_GRPO_3B': 'Frozen Lake GRPO 3B',
    'FL_PPO_0pt5B': 'Frozen Lake PPO 0.5B',
    'FL_PPO_3B': 'Frozen Lake PPO 3B',
    'FL_PPO_0pt5B': 'Frozen Lake PPO 0.5B',
    'FL_GRPO_0pt5B': 'Frozen Lake GRPO 0.5B',
    'FS_GRPO_0pt5B': 'Snake GRPO 0.5B',
    'FS_GRPO_3B': 'Snake GRPO 3B',
    'FS_PPO_0pt5B': 'Snake PPO 0.5B',
    'FS_PPO_3B': 'Snake PPO 3B',
    'FS_PPO_7B': 'Snake PPO 7B',
}

run_env_mapping = {
    'Frozen Lake GRPO 3B': 'Frozen Lake',
    'Frozen Lake PPO 3B': 'Frozen Lake',
    'Frozen Lake PPO 0.5B': 'Frozen Lake',
    'Frozen Lake GRPO 0.5B': 'Frozen Lake',
    'Snake GRPO 3B': 'Snake',
    'Snake PPO 3B': 'Snake',
    'Snake PPO 7B': 'Snake',
}

run_alg_mapping = {
    'Frozen Lake GRPO 3B': 'GRPO',
    'Frozen Lake PPO 3B': 'PPO',
    'Frozen Lake PPO 0.5B': 'PPO',
    'Frozen Lake GRPO 0.5B': 'GRPO',
    'Snake GRPO 3B': 'GRPO',
    'Snake PPO 3B': 'PPO',
    'Snake PPO 7B': 'PPO',
}

run_params_mapping = {
    'Frozen Lake GRPO 3B': '3B',
    'Frozen Lake PPO 3B': '3B',
    'Frozen Lake PPO 0.5B': '0.5B',
    'Frozen Lake GRPO 0.5B': '0.5B',
    'Snake GRPO 3B': '3B',
    'Snake PPO 3B': '3B',
    'Snake PPO 7B': '7B',
}

run_name_colors = {
    'Frozen Lake GRPO 3B': '#949494',
    'Frozen Lake PPO 3B': '#DE8F05',
    'Frozen Lake PPO 0.5B': '#029E73',
    'Frozen Lake GRPO 0.5B': '#D55E00',
    'Snake GRPO 3B': '#0173B2',
    'Snake PPO 3B': '#CA9161',
    'Snake PPO 7B': '#000000',
    'Snake PPO 0.5B': '#029E73',
    'Snake GRPO 0.5B': '#FBAFE4', # New colour
}

# Dictionary mapping evaluation environment names to themselves for customization
# You can modify the values to change how environment names appear in plots
eval_env_name_mapping = {
    'BabyAI-Pickup': 'BabyAI - Pickup',
    'BabyAI-PickupDist': 'BabyAI - Pickup Distractors',
    'BabyAI-GoToRedBall': 'BabyAI - Go To Red Ball',
    'BabyAI-GoToRedBallNoDists': 'BabyAI - Go To Red Ball No Distractors',
    'FrozenLake-BigSlippery': 'FrozenLake - Big Slippery',
    'FrozenLake-Slippery': 'FrozenLake - Slippery',
    'FrozenLake-NoSlip': 'FrozenLake - Not Slippery',
    'Snake-20Step': 'Snake - 20 Steps',
    'Snake-20Steps-PoisonAppleAndBanana': 'Snake - Poison Apple, Healthy Banana',
    'FastSnake-Default': 'Snake - 8 Steps',
    'val': 'Training Environment',
}
