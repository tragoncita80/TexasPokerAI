# TexasPokerAI
NYCU Spring 2025 Intro AI final project

This is the project for 2025 spring Artificial Intelligence course in National Yang Ming Chao Tung University

We develop a **TexasPoker AI agent**.

## Project Structure

```text
TexasPokerAI
├── README.md
├── main.py
├── consol.py
├── poker_conf.yaml
├── test.py
├── ai
│   ├── deep_cfr_player.py
│   ├── regret_net.py
│   └── strategy_net.py
├── engine
│   ├── runner.py
|   └── state_encoder.py
├── memory
│   ├── regret_memory.py
|   └── strategy_memory.py
├── traverse
│   └── traverse.py
├── models
│   └── strategy_net.pt (After training)
├── requirements.txt
└── players
    ├── console_player.py
    ├── fish_player.py
    ├── random_player.py
    └── strategy_player.py

```

## Install
First, execute the following instruction to install the required packages.

```
pip3 install -r requirements.txt
git clone https://github.com/CubeMan940331/PyPokerEngine.git
pip install -e .
```

For the agent training , execute the game by typing
```
python3 main.py
```

You will get model/strategy_net.pt and the plot of acc/miss.

Then, execute the game by typing
```
python3 test.py
```

## GUI support


```
git clone https://github.com/DoobieD00/PyPokerGUI.git
pip install -e .
python3 -m pypokergui serve ./poker_conf.yaml --port 8000 --speed fast
```

fish player : always call

strategy player : our agent

## Reference
[Pypokerengine](https://github.com/ishikota/PyPokerEngine)
[PyPokerGUI](https://github.com/DoobieD00/PyPokerGUI)