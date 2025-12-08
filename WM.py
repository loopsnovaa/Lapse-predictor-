import time
class WashingMachineDFA:
    def __init__(self):
        self.states = {'Idle', 'Fill', 'Wash', 'Rinse', 'Spin'}
        self.current_state = 'Idle'
        self.finished = False
    def transition(self, input_signal):
        """
        The Transition Function (delta)
        """
        previous_state = self.current_state
        if self.current_state == 'Idle' and input_signal == 'START':
            self.current_state = 'Fill'
        elif self.current_state == 'Fill' and input_signal == 'WATER_FULL':
            self.current_state = 'Wash'
        elif self.current_state == 'Wash' and input_signal == 'TIMER_WASH':
            self.current_state = 'Rinse'
        elif self.current_state == 'Rinse' and input_signal == 'TIMER_RINSE':
            self.current_state = 'Spin'
        elif self.current_state == 'Spin' and input_signal == 'TIMER_SPIN':
            self.current_state = 'Idle'
            self.finished = True
        else:
            print(f"Invalid transition from {self.current_state} with input {input_signal}")
            return
        print(f"State Change: {previous_state} -> {self.current_state}")
        
machine = WashingMachineDFA()
inputs = ['START', 'WATER_FULL', 'TIMER_WASH', 'TIMER_RINSE', 'TIMER_SPIN']
print("--- Washing Machine Cycle Start ---")
for signal in inputs:
    time.sleep(1) 
    print(f"Sensor Input: {signal}")
    machine.transition(signal)
print("--- Cycle Complete ---")