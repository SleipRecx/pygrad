import random

random.seed(1400)


class Value:
    def __init__(self, scalar):
        self.scalar = scalar
        self._parents = ()
        self._op = None

    def __repr__(self) -> str:
        return f"Value({self.scalar})"

    def relu(self):
        return self if self.scalar > 0 else Value(0)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.scalar * other.scalar)
        out._op = "*"
        out._parents = (self, other)
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.scalar + other.scalar)
        out._op = "+"
        out._parents = (self, other)
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.scalar**other.scalar)
        out._op = "**"
        out._parents = (self, other)
        return out

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1

    def tree(value):
        nodes, edges = set(), set()

        queue = [value]

        while queue:
            current = queue.pop()
            if current not in nodes:
                nodes.add(current)
                for parent in current._parents:
                    edges.add((parent, current))
                    queue.append(parent)
        return nodes, edges


class Neuron:
    def __init__(self, size, linear):
        self.weights: list[Value] = [Value(random.uniform(-1, 1)) for _ in range(size)]
        self.bias: Value = Value(0)
        self.linear = linear

    def __call__(self, _inputs: list[Value]):
        assert len(_inputs) == len(self.weights)

        act = sum(
            (weight_i * input_i for weight_i, input_i in zip(self.weights, _inputs)),
            self.bias,
        )
        return act if not self.linear else act.relu()

    def __repr__(self):
        return f"{'Linear' if self.linear else "ReLU" }-Neuron({len(self.weights)})"


class Layer:
    def __init__(self, input_size: int, output_size: int, linear: bool):
        self.neurons = [Neuron(input_size, linear) for _ in range(output_size)]

    def __call__(self, _inputs: list[Value]):
        return [n(_inputs) for n in self.neurons]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP:
    def __init__(self, _input, dimensions):
        structure = [_input] + dimensions
        self.layers = []
        for i in range(len(dimensions)):
            input_dimension = structure[i]
            output_dimension = structure[i + 1]
            is_final_layer = i == (len(dimensions) - 1)
            self.layers.append(
                Layer(input_dimension, output_dimension, linear=is_final_layer)
            )

        self.layers = [
            Layer(structure[i], structure[i + 1], i == len(dimensions) - 1)
            for i in range(len(dimensions))
        ]

    def __call__(self, _inputs: list[Value]):
        current_output = _inputs
        for layer in self.layers:
            current_output = layer(current_output)
        return current_output

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


mlp = MLP(2, [16, 16, 1])
print(mlp)
print(mlp([Value(0.22), Value(0.233)]))


def test_math():
    def graph(a, b):
        c = a * b
        d = -c
        y = d + c - b**5
        return y

    assert graph(5, 6) == graph(Value(5), Value(6)).scalar


test_math()


w1 = Value(5)
w2 = Value(6)
y = (w1 + w2 - w2 * 2).relu()
print(y)
