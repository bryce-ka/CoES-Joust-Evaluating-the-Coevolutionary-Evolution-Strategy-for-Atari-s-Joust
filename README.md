# README

atari_joust_ces.py is a Python script that implements the coevolutionary evolution strategy from [Klijn et al.](https://arxiv.org/pdf/2104.05610.pdf). Ultimately this program performs neuroevolution and saves a model that can control both players in a single game of joust. Our research paper [Exploring the Effectiveness of Klijn's Coevolutionary Evolution Strategy for Multi-Agent Reinforcement Learning in Atari's Joust.pdf](https://github.com/bryce-ka/CoES-Joust-Evaluating-the-Coevolutionary-Evolution-Strategy-for-Atari-s-Joust/blob/main/Exploring%20the%20Effectiveness%20of%20Klijn's%20Coevolutionary%20Evolution%20Strategy%20for%20Multi-Agent%20Reinforcement%20Learning%20in%20Atari's%20Joust.pdf) explains our implementation of the coevolutionary evolution strategy along with our results. 
 
## Prerequisites

install the ROM for Joust using [AutoROM](https://github.com/Farama-Foundation/AutoROM)

Make sure you have the following libraries installed:

- `scikit-image`
- `matplotlib`
- `numpy`
- `gym`
- `PIL`
- `cv2`
- `tensorflow`
- `pettingzoo`

You can install them using `pip`:

```shell
pip install scikit-image matplotlib numpy gym Pillow opencv-python tensorflow pettingzoo
```

## Usage

1. Clone the repository and navigate to the directory containing the script.

2. Adjust hyperparameters: Suggested hyperparameters would be a population= 200 and max_rounds= 500, however this requires a substantial amount of computing power and memory and we were only able to use a population of size 3 for 3 rounds.

3.  name the model by changing the global name variable.

4. Run the script using Python:

```shell
python3 atari_joust_ces.py
```

## Description

The script uses the `skimage`, `matplotlib`, `numpy`, `gym`, `PIL`, `cv2`, `tensorflow`, and `pettingzoo` libraries. It defines a genetic programming approach to train a deep reinforcement learning agent to play the game Joust.

The script contains the following main components:

- Model Definition: Two convolutional neural network models (`model1` and `model2`) are defined using the Keras API. These models serve as the basis for generating new policies in the coevolutionary evolution strategy.

- Environment Initialization: The Joust environment (`joust_v3`) is initialized using the PettingZoo library. The environment is set to grayscale image observation type and a maximum of 10000 cycles.

- Image Preprocessing Functions: The script includes a function `standardize_img` to preprocess the game frames by normalizing the pixel values. Another function `player_stack` stacks the preprocessed frames to create the input for the models.

- Game Playing Functions: The script includes two functions `playGame1` and `playGame2` that simulate the game playing process. These functions iterate over the game agents and make decisions based on the current policy. The game frames are preprocessed and passed to the models for action selection.

- Genetic Programming: The `main` function implements the genetic programming approach. It defines the number of generations (`max_rounds`) and population size (`population`). The function generates initial policies and evaluates them using the game playing functions. The best scores from each generation are recorded and used to generate the next policy. The process continues for the specified number of generations.

- Training Progress Visualization: After the genetic programming process is completed, the script plots a graph showing the training progression, displaying the best score achieved in each generation.

- Model Saving: The evolved model is saved as final_model_"name".h5

