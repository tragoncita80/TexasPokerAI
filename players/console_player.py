from pypokerengine.players import BasePokerPlayer

class ConsolePlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    def __init__(self, name="ConsolePlayer"):
        self.name=name
    
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        
        '''
        valid_actions = [
            { "action" : "fold" , "amount" : 0 },
            { "action" : "call" , "amount" : agree_amount},
            { "action" : "raise", "amount" : raise_to_amount}
        ]
        '''
        print(f"console player: {self.name} acting")
        print(hole_card)
        for action in valid_actions:
            if action["action"]=="call": call_amount=action["amount"]
            elif action["action"]=="raise": raise_amount=action["amount"]
        
        while True:
            op = input("choose an action: ")
            if op=='r':
                if raise_amount<0: print("invalid raise")
                else: return "raise", raise_amount
            elif op=='c': return "call", call_amount
            elif op=='f': return "fold", 0
            else: print("invalid action")

    def receive_game_start_message(self, game_info): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

def setup_ai():
    return ConsolePlayer()