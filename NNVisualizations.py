from manimlib.imports import *
from .NeuralNetwork import *


class Neuron(Scene):
    def construct(self):
        EDGE_COLOR = MAROON_E
        TEXT_COLOR = BLUE_A
        SCALE_SIZE = 0.75
        neuron = Circle(color=BLUE_C, radius=1)
        previous_layer = TextMobject("Previous Layer", color=GOLD_E)
        next_layer = TextMobject("Next Layer", color=GOLD_E)

        nn_input_line = Line(3*LEFT, color=EDGE_COLOR)
        nn_output_line = Line(3*LEFT, color=EDGE_COLOR)
        nn_input_line.next_to(neuron, LEFT, buff=0)
        nn_output_line.next_to(neuron, RIGHT, buff=0)
        w_and_b = TexMobject('w ', ',', 'b', color=TEXT_COLOR)
        w_and_b.scale(SCALE_SIZE)
        activation1 = TexMobject('Activation', color=TEXT_COLOR)
        activation2 = TexMobject('Function\,\,', 'a', '= g(z)', color=TEXT_COLOR)
        activation2.next_to(neuron, UP)
        activation1.next_to(activation2, UP)
        activation1.scale(SCALE_SIZE)
        activation2.scale(SCALE_SIZE)
        input_data = TexMobject('x', color=TEXT_COLOR)
        input_data.scale(SCALE_SIZE)
        input_data.next_to(nn_input_line, LEFT)
        input_data_legend = TextMobject("x: Input Data")
        input_data_legend.scale(SCALE_SIZE)
        input_data_legend.to_corner(LEFT+DOWN, buff=0)
        input_data_updated = TextMobject("x", color=TEXT_COLOR)
        input_data_updated.next_to(nn_input_line, LEFT)
        output_data = TexMobject('a', color=TEXT_COLOR)
        output_data.scale(SCALE_SIZE)
        '''
        output_data_legend = TextMobject("a: Output data")
        output_data_legend.scale(SCALE_SIZE)
        output_data_legend.to_corner(RIGHT+DOWN, buff=0)
        '''
        output_data.next_to(nn_output_line, RIGHT)
        nn_equation = TexMobject('z = w*x+')
        nn_equation.scale(SCALE_SIZE)
        nn_equation.next_to(neuron, DOWN*2)
        nn_equation_updated = TexMobject('z = {w}_{1}*{x}_{1} + {w}_{2}*{x}_{2} + {w}_{3}*{x}_{3} + b', color=TEXT_COLOR)
        nn_equation_updated.next_to(neuron, DOWN*2)
        nn_equation_updated.scale(SCALE_SIZE)
        nn_equation_updated.shift(RIGHT)


        input1 = Line((-5.73, 1.73, 0), (0, 0, 0), buff=1, color=EDGE_COLOR)
        input1_text = TexMobject("{x}_{1}")
        input1_text.scale(SCALE_SIZE)
        input1_text.next_to(input1, UP+LEFT)
        input1_parameter = TexMobject("{w}_{1}")
        input1_parameter.scale(SCALE_SIZE)
        input1_parameter.next_to(input1, UP, buff=0)


        input2 = Line((-6, 0, 0), (0, 0, 0), buff=1, color=EDGE_COLOR)
        input2_text = TexMobject("{x}_{2}")
        input2_text.scale(SCALE_SIZE)
        input2_text.next_to(input2, LEFT)
        input2_parameter = TexMobject("{w}_{2}")
        input2_parameter.next_to(input2, UP)
        input2_parameter.scale(SCALE_SIZE)

        input3 = Line((-5.73, -1.73, 0), (0, 0, 0), buff=1, color=EDGE_COLOR)
        input3_text = TexMobject("{x}_{3}")
        input3_text.scale(SCALE_SIZE)
        input3_text.next_to(input3, DOWN+LEFT)
        input3_parameter = TexMobject("{w}_{3}")
        input3_parameter.next_to(input3, UP,  buff=0)
        input3_parameter.scale(SCALE_SIZE)
        input3_parameter.shift(0.4*DOWN)

        inputs = VGroup(input1, input2, input3)
        previous_layer.next_to(inputs, DOWN)

        output1 = Line((5.73, 1.73, 0), (0, 0, 0), buff=1, color=EDGE_COLOR)
        output2 = Line((6, 0, 0), (0, 0, 0), buff=1, color=EDGE_COLOR)
        output3 = Line((5.73, -1.73, 0), (0, 0, 0), buff=1, color=EDGE_COLOR)
        outputs = VGroup(output1, output2, output3)
        next_layer.next_to(outputs, DOWN)

        parameters = TexMobject(r"w=\begin{bmatrix}w_{1}\\w_{2}\\ w_{3}\end{bmatrix}", color=GREEN_SCREEN)
        parameters.scale(SCALE_SIZE)
        parameters.to_edge(DOWN, buff=0)
        parameters.shift(0.5*LEFT)
        features = TexMobject(r"x=\begin{bmatrix}x_{1}\\x_{2}\\ x_{3}\end{bmatrix}", color=GREEN_SCREEN)
        features.scale(SCALE_SIZE)
        features.next_to(parameters, RIGHT)
        


        nn_equation_matrix = TexMobject("z = ","w^{T}", "*", "x",  "+ b")
        nn_equation_matrix.scale(SCALE_SIZE)
        nn_equation_matrix.next_to(neuron, DOWN * 2 )
        nn_equation_matrix[1:4:2].set_color(GREEN_SCREEN)
        

        self.add(neuron, inputs, outputs)
        self.wait(2)
        self.play(Write(previous_layer))
        self.wait(2)
        self.play(Write(next_layer))
        self.wait(2)
        self.play(FadeOut(previous_layer), FadeOut(next_layer))
        
        self.play(FadeOut(inputs), FadeOut(outputs))
        self.wait(2)

        self.play(Write(w_and_b))
        self.wait(3)
        self.play(Write(activation1), Write(activation2))
        self.wait(2)
        self.play(Write(input_data), GrowArrow(nn_input_line), GrowArrow(nn_output_line), Write(input_data_legend), run_time=2)
        self.play(Write(output_data), run_time=2)
        self.wait(12)
        self.play(ApplyMethod(w_and_b[0].next_to, nn_input_line, UP), FadeOut(w_and_b[1]))
        self.play(Write(nn_equation), ApplyMethod(w_and_b[2].next_to, nn_equation, RIGHT))
       
        self.wait(4)
        self.play(FadeOut(nn_input_line), FadeOut(input_data), FadeOut(w_and_b[0]), )
        self.play(FadeOut(activation1), FadeOut(activation2[0]), ApplyMethod(activation2[1:].next_to, nn_equation, DOWN*1.5))
        self.wait(5)

        self.play(GrowArrow(input1), Write(input1_text), Write(input1_parameter))
        self.play(GrowArrow(input2), Write(input2_text), Write(input2_parameter))
        self.play(GrowArrow(input3),  Write(input3_text), Write(input3_parameter))
        self.wait(12)
        self.play(Transform(nn_equation, nn_equation_updated), FadeOut(w_and_b[2]))
        self.play(ApplyMethod(activation2[1:].next_to, nn_output_line, RIGHT), FadeOut(output_data))
        self.wait(2)
        self.play(Transform(nn_equation, nn_equation_matrix))
        self.play(Write(parameters), Write(features))
        self.play(Transform(nn_output_line, outputs), FadeOut(activation2[1:]))
        
        self.wait(2)



class NeuronDoubled(Scene):
    CONFIG = {
        "edge_color" : MAROON_E,
        "text_color" : BLUE_A,
        "neuron_radius" : 0.6,
        "neuron_color" : BLUE_C,
        "neuron_neuron_buff" : 2,
        "layer_layer_buff": 3,
    }

    def construct(self):
        neurons1 = VGroup(
            Circle(radius=self.neuron_radius, color=self.neuron_color),
            Circle(radius=self.neuron_radius, color=self.neuron_color),
            Circle(radius=self.neuron_radius, color=self.neuron_color),
        )
        neurons2 = VGroup(
            Circle(radius=self.neuron_radius, color=self.neuron_color),
            Circle(radius=self.neuron_radius, color=self.neuron_color)
        )
        layers = VGroup(neurons1, neurons2)
        neurons1.arrange(DOWN, buff=self.neuron_neuron_buff)
        neurons2.arrange(DOWN, buff=self.neuron_neuron_buff)
        layers.arrange(RIGHT, buff=self.layer_layer_buff)
        """parameters = VGroup(
            TexMobject("w", "\,\,", "b", color=self.text_color, ApplyMethod(shift, neuron2[0], get_center)),
            TexMobject("w", "\,\,", "b", color=self.text_color).shift(neurons2[1].get_center())
        )"""

        self.add(layers)
        #self.play(Write(parameters[1]))
        
        self.wait(2)




class NeuralNetworkAnime(Scene):
    def construct(self):
        network_mob = NetworkMobject(Network(
            sizes = [6, 4, 5, 4, 3, 5, 2]
        ))
        network_mob.scale(0.8)
        network_mob.to_edge(UP, buff = MED_SMALL_BUFF)
        network_mob.shift(RIGHT)
        edge_update = ContinualEdgeUpdate(
            network_mob, stroke_width_exp = 1,
        )
        self.play(FadeIn(network_mob))



class NetworkScene(Scene):
        CONFIG = {
        "layer_sizes" : [3, 7, 3, 1],
        "network_mob_config" : {},
        }

        def setup(self):
            self.add_network()

        def add_network(self):
            self.network = Network(sizes = self.layer_sizes)
            self.network_mob = NetworkMobject(
                self.network,
                **self.network_mob_config
            )
            self.add(self.network_mob)

        def construct(self):
            self.play(ShowCreation(
            self.network_mob.edge_groups,
            lag_ratio = 0.5,
            run_time = 5,
            rate_func=linear,
            ))
            self.wait(2)


class NetworkSceneZoom(MovingCameraScene):
        CONFIG = {
        "layer_sizes" : [3, 7, 3, 1],
        "network_mob_config" : {},
        "camera_class": MovingCamera
        }

        def setup(self):
            self.add_network()
            Scene.setup(self)
            assert(isinstance(self.camera, MovingCamera))
            self.camera_frame = self.camera.frame
            return self   

        def add_network(self):
            self.network = Network(sizes = self.layer_sizes)
            self.network_mob = NetworkMobject(
                self.network,
                **self.network_mob_config
            )
            self.add(self.network_mob)

        def construct(self):
            self.camera_frame.save_state()

            self.play(ShowCreation(
            self.network_mob.edge_groups,
            lag_ratio = 0.5,
            run_time = 5,
            rate_func=linear,
            ))
            print(self.network_mob.layer_sizes[1])
            print(self.network_mob.get_layer(self.network_mob.layer_sizes[1]).neurons)
            self.play(
                self.camera_frame.set_height, self.network_mob.get_layer(self.network_mob.layer_sizes),
                self.camera_frame.move_to, self.layer_sizes[1][3].get_center())

            self.wait(2)



"""
class NeuralNetwork(Scene):
    def construct(self):
        layer_1 = []
        for i in range(3):
            neuron = Circle(radius=0.16)
            layer_1.append(neuron)

        layer_2 = []
        for i in range(5):
            neuron = Circle(radius=0.16)
            layer_2.append(neuron)

        layer_3 = []
        for i in range(3):
            neuron = Circle(radius=0.16)
            layer_3.append(neuron)


        neural_network = []
        neural_network.append(layer_1)
        neural_network.append(layer_2)
        neural_network.append(layer_3)
        print(neural_network)
        offset_horizontal = 3
        for layer_id, layer in enumerate(neural_network):
            offset_vertical = 2
            for neuron in layer:
                if layer_id is 0 or layer_id is 2:
                    neuron.shift(0.5*DOWN)
                else:
                    neuron.shift(0.5*UP)
                neuron.shift(offset_horizontal*LEFT)
                neuron.shift(DOWN)
                neuron.shift(offset_vertical*UP)
                self.add(neuron)
                offset_vertical -= 1
            offset_horizontal -= 2

        stroke1 = Line()
        stroke1.next_to(neural_network[0][0])

        
        n1 = Circle(radius=0.25)
        n1.shift(UP*2)
        n2 = Circle(radius=0.25)
        n2.shift(UP)
        n3 = Circle(radius=0.25)
        n4 = Circle(radius=0.25)
        n4.shift(DOWN*1)
        n5 = Circle(radius=0.25)
        n5.shift(DOWN*2)
        


        #self.play(FadeIn(n1), FadeIn(n2), FadeIn(n3), FadeIn(n4), FadeIn(n5))
        self.wait(3)
"""