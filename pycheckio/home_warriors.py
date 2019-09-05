class Warrior:
    def __init__(self, health = 50, attack=5):
        self.health = health
        self.attack = attack

    @property
    def is_alive(self):
        return self.health > 0

class Knight(Warrior):
    def __init__(self):
        super().__init__(attack=7)

def fight(unit_1, unit_2):

    while True:
        # unit 1 attacks first
        unit_2.health -= unit_1.attack
        if unit_1.is_alive and not unit_2.is_alive:
            return True

        # unit 2 attacks
        unit_1.health -= unit_2.attack
        if unit_2.is_alive and not unit_1.is_alive:
            return False

if __name__ == '__main__':
    # These "asserts" using only for self-checking and not necessary for auto-testing

    chuck = Warrior()
    bruce = Warrior()
    carl = Knight()
    dave = Warrior()
    mark = Warrior()

    assert fight(chuck, bruce) == True
    assert fight(dave, carl) == False
    assert chuck.is_alive == True
    assert bruce.is_alive == False
    assert carl.is_alive == True
    assert dave.is_alive == False
    assert fight(carl, mark) == False
    assert carl.is_alive == False

    print("Coding complete? Let's try tests!")
