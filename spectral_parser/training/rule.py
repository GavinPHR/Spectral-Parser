import config


def hash_forward(a, b, c):
    """
    20 bits for each position, should be more than enough for any grammar
    """
    return (a << 40) ^ (b << 20) ^ c


class Rule3:
    """
    Encapsulation for a 3rd order rule
    """
    def __init__(self, a, b, c):
        """
        a -> b c where a, b, c are non-terminals
        """
        self.a, self.b, self.c = a, b, c

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash_forward(self.a, self.b, self.c)

    def __repr__(self):
        nmap = config.nonterminal_map
        return nmap[self.a] + '->' + nmap[self.b] + ' ' + nmap[self.c]


class Rule2:
    """
    Encapsulation for a 1st order rule
    """
    def __init__(self, a, b):
        """
        a -> b where a, b are a non-terminal
        """
        self.a, self.b = a, b

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash_forward(self.a, self.b, 0)

    def __repr__(self):
        nmap = config.nonterminal_map
        return nmap[self.a] + '->' + nmap[self.b]


class Rule1:
    """
    Encapsulation for a 1st order rule
    """
    def __init__(self, a, x):
        """
        a -> x where a is a non-terminal and x is a terminal
        """
        self.a, self.x = a, x

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash_forward(self.a, self.x, 0)

    def __repr__(self):
        nmap = config.nonterminal_map
        tmap = config.terminal_map
        return nmap[self.a] + '->' + tmap[self.x]
